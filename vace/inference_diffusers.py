import cv2
import imageio
import os
import sys
import torch
import torch.multiprocessing as mp
import PIL.Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

# Add Video-Depth-Anything to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Video-Depth-Anything'))
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Depth model configuration
DEPTH_ENCODER = 'vitl'  # Options: 'vits', 'vitb', 'vitl'
DEPTH_CHECKPOINT_PATH = 'Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth'

def initialize_depth_model(encoder='vitl', device='cuda'):
    """
    Initialize Video-Depth-Anything model directly in Python.
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    checkpoint_path = f'Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth'
    
    print(f"Loading depth model: {encoder}")
    model = VideoDepthAnything(**model_configs[encoder], metric=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    
    return model

def load_and_sample_frames(video_path, num_frames, max_res=1280, from_middle=True):
    """
    Load video and sample frames from the middle section.
    If from_middle=True, extracts consecutive frames from the center of the video.
    This reads ONLY the specific frames needed, not the entire video.
    """
    print(f"  → Reading {num_frames} frames from {Path(video_path).name}...")
    
    try:
        from decord import VideoReader, cpu
        DECORD_AVAILABLE = True
    except:
        DECORD_AVAILABLE = False
    
    if DECORD_AVAILABLE:
        # Use decord for efficient random frame access
        vid = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vid)
        fps = vid.get_avg_fps()
        
        # Calculate frame indices to sample
        if total_frames > num_frames:
            if from_middle:
                # Extract consecutive frames from the middle
                start_idx = (total_frames - num_frames) // 2
                end_idx = start_idx + num_frames
                indices = list(range(start_idx, end_idx))
                print(f"  → Extracting frames {start_idx}-{end_idx} from middle of {total_frames} total frames")
            else:
                # Uniform sampling across entire video
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
                print(f"  → Sampling {num_frames} frames from {total_frames} total frames")
        else:
            indices = list(range(total_frames))
            print(f"  → Video has {total_frames} frames (≤ target {num_frames})")
        
        # Get dimensions for resizing
        original_height, original_width = vid[0].shape[:2]
        height, width = original_height, original_width
        
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale))
            width = int(round(original_width * scale))
            # Ensure even dimensions
            height = height if height % 2 == 0 else height + 1
            width = width if width % 2 == 0 else width + 1
            
            # Reload with target resolution
            vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        
        # Read ONLY the selected frames (much faster!)
        sampled_frames = vid.get_batch(indices).asnumpy()
        
    else:
        # Fallback to cv2 - less efficient but works
        print("  ⚠ Decord not available, using cv2 (slower)")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample
        if total_frames > num_frames:
            if from_middle:
                # Extract consecutive frames from the middle
                start_idx = (total_frames - num_frames) // 2
                end_idx = start_idx + num_frames
                indices = np.arange(start_idx, end_idx)
                print(f"  → Extracting frames {start_idx}-{end_idx} from middle of {total_frames} total frames")
            else:
                # Uniform sampling across entire video
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                print(f"  → Sampling {num_frames} frames from {total_frames} total frames")
        else:
            indices = np.arange(total_frames)
            print(f"  → Video has {total_frames} frames (≤ target {num_frames})")
        
        # Get dimensions
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height, width = original_height, original_width
        
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale))
            width = int(round(original_width * scale))
        
        # Read only specific frames
        sampled_frames = []
        indices_set = set(indices)
        frame_idx = 0
        
        while cap.isOpened() and len(sampled_frames) < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in indices_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))
                sampled_frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        sampled_frames = np.stack(sampled_frames, axis=0)
    
    return sampled_frames, fps

def estimate_depth(depth_model, frames, fps, input_size=518, device='cuda'):
    """
    Estimate depth maps for pre-loaded frames using Video-Depth-Anything model.
    Returns numpy array of depth maps [T, H, W] and fps.
    """
    print(f"  → Estimating depth for {len(frames)} frames...")
    with torch.no_grad():
        depths, fps = depth_model.infer_video_depth(
            frames, 
            fps, 
            input_size=input_size, 
            device=device, 
            fp32=False
        )
    
    return depths, fps

def resample_depth_maps(depth_maps, num_frames):
    """
    Resample depth maps to match target number of frames.
    """
    if len(depth_maps) == num_frames:
        return depth_maps
    
    print(f"  → Resampling depth maps from {len(depth_maps)} to {num_frames} frames")
    indices = np.linspace(0, len(depth_maps)-1, num_frames, dtype=int)
    return depth_maps[indices]

def get_video_paths(dataset_dir):
    """Get all video file paths from dataset directory."""
    video_paths = []
    for dir in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mov'):
                    video_paths.append(os.path.join(dir_path, file))
    return video_paths

def frames_to_pil(frames_array, height, width, num_frames):
    """
    Convert numpy frames array to PIL Images and resample to target dimensions.
    Args:
        frames_array: numpy array of shape [T, H, W, C]
        height: target height
        width: target width  
        num_frames: target number of frames
    """
    # Resample frames if needed
    if len(frames_array) != num_frames:
        indices = np.linspace(0, len(frames_array)-1, num_frames, dtype=int)
        frames_array = frames_array[indices]
    
    pil_frames = []
    for frame in frames_array:
        # Resize if needed
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        pil_frames.append(PIL.Image.fromarray(frame))
    
    return pil_frames

def save_pil_to_video(frames, output_path, fps=16):
    imageio.mimwrite(output_path, [np.array(frame) for frame in frames], fps=fps)

def rgb2gray(video: list[PIL.Image.Image]) -> list[PIL.Image.Image]:
    gray_video = []
    for frame in video:
        gray_frame = frame.convert("L").convert("RGB")
        gray_video.append(gray_frame)
    return gray_video

def depth_to_rgb(depth_maps):
    """Convert depth maps to RGB images for visualization (with colormap)."""
    depth_frames = []
    for depth in depth_maps:
        # Normalize depth to 0-255
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        # Apply colormap for better visualization
        depth_rgb = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
        depth_frames.append(PIL.Image.fromarray(depth_rgb))
    return depth_frames

def depth_to_grayscale_pil(depth_maps):
    """
    Convert raw depth maps to grayscale PIL images for model conditioning.
    This preserves the actual depth information without color mapping.
    """
    depth_frames = []
    for depth in depth_maps:
        # Normalize depth to 0-255 range
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # Convert to PIL grayscale, then to RGB (3 channels with same values)
        depth_gray = PIL.Image.fromarray(depth_uint8, mode='L')
        depth_rgb = depth_gray.convert('RGB')
        depth_frames.append(depth_rgb)
    
    return depth_frames

def initialize_vace_pipeline(device='cuda'):
    """
    Initialize VACE pipeline for style transfer.
    """
    print(f"Loading VACE model on {device}...")
    model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )

    flow_shift = 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to(device)
    
    print(f"VACE model loaded successfully on {device}!")
    return pipe

def process_video_on_gpu(gpu_id, video_info_list, all_depths, all_frames, all_fps, 
                         output_dirs, prompt, negative_prompt, height, width, num_frames):
    """
    Worker function to process videos on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        video_info_list: List of (video_path, video_name) tuples to process
        all_depths: Dictionary of pre-computed depth maps
        all_frames: Dictionary of pre-computed frames
        all_fps: Dictionary of FPS values
        output_dirs: Dictionary with 'output', 'input', 'depth' paths
        prompt: Generation prompt
        negative_prompt: Negative prompt
        height: Output height
        width: Output width
        num_frames: Number of frames
    """
    try:
        # Set device for this process
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        
        print(f"[GPU {gpu_id}] Initializing VACE pipeline...")
        pipe = initialize_vace_pipeline(device=device)
        
        print(f"[GPU {gpu_id}] Processing {len(video_info_list)} videos...")
        
        for video_path, video_name in video_info_list:
            print(f"[GPU {gpu_id}] Style transfer: {video_name}")
            
            try:
                # Get pre-computed depth and frames
                depths = all_depths[video_name]
                frames = all_frames[video_name]
                fps = all_fps[video_name]
                
                # Resample depth maps to target num_frames
                depth_resampled = resample_depth_maps(depths, num_frames)
                
                # Convert frames to PIL and resample
                video_frames = frames_to_pil(frames, height, width, num_frames)
                
                # Convert depth to grayscale PIL for model conditioning
                depth_condition_frames = depth_to_grayscale_pil(depth_resampled)
                
                # Resize depth frames to match target dimensions
                depth_condition_frames = [frame.resize((width, height)) for frame in depth_condition_frames]
                
                # Convert depth to colormap for visualization/saving
                depth_rgb_frames = depth_to_rgb(depth_resampled)
                
                # Run style transfer with depth as input
                print(f"[GPU {gpu_id}] Running inference on {video_name}...")
                torch.compiler.cudagraph_mark_step_begin()
                
                output = pipe(
                    video=depth_condition_frames,  # Use grayscale depth as conditioning
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=5.0,
                    generator=torch.Generator(device=device).manual_seed(42),
                ).frames[0]
                
                # Save outputs to organized directories
                print(f"[GPU {gpu_id}] Saving outputs for {video_name}...")
                export_to_video(output, os.path.join(output_dirs['output'], f"{video_name}.mp4"), fps=16)
                save_pil_to_video(video_frames, os.path.join(output_dirs['input'], f"{video_name}.mp4"), fps=16)
                save_pil_to_video(depth_rgb_frames, os.path.join(output_dirs['depth'], f"{video_name}.mp4"), fps=16)
                
                print(f"[GPU {gpu_id}] ✓ Completed {video_name}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ✗ Error processing {video_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[GPU {gpu_id}] Finished all assigned videos!")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()

# Global constants
prompt = "First-person perspective, realistic photorealistic footage, natural camera movement, real-world environment, cinematic lighting, high fidelity details, smooth motion, professional video quality, lifelike textures, authentic atmosphere"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

height = 512
width = 512
num_frames = 81

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    video_paths = get_video_paths("/mnt/data/datasets/extracted_tars/kbm/fps")[:50]
    print(f"Found {len(video_paths)} videos.")
    
    # Create organized output directories
    output_base = "./outputs"
    depth_dir = os.path.join(output_base, "depth")
    input_dir = os.path.join(output_base, "input")
    output_dir = os.path.join(output_base, "output")
    depth_npy_dir = os.path.join(output_base, "depth_npy")
    
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depth_npy_dir, exist_ok=True)
    
    # Step 1: Initialize depth model once (not per video)
    print("\n" + "="*60)
    print("INITIALIZING DEPTH MODEL")
    print("="*60)
    depth_model = initialize_depth_model(encoder=DEPTH_ENCODER, device=DEVICE)
    
    # Step 2: Pre-compute all depth maps
    print("\n" + "="*60)
    print("COMPUTING DEPTH MAPS FOR ALL VIDEOS")
    print("="*60)
    all_depths = {}
    all_frames = {}
    all_fps = {}
    
    for idx, video_path in enumerate(video_paths):
        video_name = Path(video_path).stem
        print(f"\n[{idx+1}/{len(video_paths)}] Processing depth for: {video_name}")
        
        try:
            # First, load and sample only the frames we need
            sampled_frames, fps = load_and_sample_frames(
                video_path, 
                num_frames=num_frames,
                max_res=1280
            )
            
            # Then estimate depth only for those sampled frames
            depths, fps = estimate_depth(
                depth_model, 
                sampled_frames,
                fps,
                input_size=518, 
                device=DEVICE
            )
            
            all_depths[video_name] = depths
            all_frames[video_name] = sampled_frames
            all_fps[video_name] = fps
            
            print(f"  ✓ Depth shape: {depths.shape}, FPS: {fps}")
            
            # Save depth maps as numpy array
            np.save(os.path.join(depth_npy_dir, f"{video_name}.npy"), depths)
            
        except Exception as e:
            print(f"  ✗ Error computing depth for {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Clean up depth model to free memory
    del depth_model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"DEPTH COMPUTATION COMPLETE - {len(all_depths)}/{len(video_paths)} successful")
    print("="*60)
    
    # Step 3: Prepare for multi-GPU style transfer
    print("\n" + "="*60)
    print("PREPARING MULTI-GPU STYLE TRANSFER")
    print("="*60)
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")
    
    if num_gpus == 0:
        print("ERROR: No GPUs available!")
        exit(1)
    
    # Filter videos that have successful depth computation
    successful_videos = [(video_path, Path(video_path).stem) 
                         for video_path in video_paths 
                         if Path(video_path).stem in all_depths]
    
    print(f"Processing {len(successful_videos)} videos with successful depth computation")
    
    # Split videos across GPUs
    videos_per_gpu = [[] for _ in range(num_gpus)]
    for idx, video_info in enumerate(successful_videos):
        gpu_id = idx % num_gpus
        videos_per_gpu[gpu_id].append(video_info)
    
    # Print distribution
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {len(videos_per_gpu[gpu_id])} videos")
    
    # Prepare output directories dict
    output_dirs = {
        'output': output_dir,
        'input': input_dir,
        'depth': depth_dir
    }
    
    # Step 4: Run style transfer in parallel across all GPUs
    print("\n" + "="*60)
    print("RUNNING MULTI-GPU STYLE TRANSFER")
    print("="*60)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create processes for each GPU
    processes = []
    for gpu_id in range(num_gpus):
        if len(videos_per_gpu[gpu_id]) > 0:
            p = mp.Process(
                target=process_video_on_gpu,
                args=(
                    gpu_id,
                    videos_per_gpu[gpu_id],
                    all_depths,
                    all_frames,
                    all_fps,
                    output_dirs,
                    prompt,
                    negative_prompt,
                    height,
                    width,
                    num_frames
                )
            )
            p.start()
            processes.append(p)
            print(f"Started process on GPU {gpu_id}")
    
    # Wait for all processes to complete
    print("\nWaiting for all GPUs to complete...")
    for p in processes:
        p.join()
    
    print("\n" + "="*60)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*60)