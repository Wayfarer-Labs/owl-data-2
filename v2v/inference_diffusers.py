import cv2
import imageio
import os
import sys
import torch
import torch.multiprocessing as mp
import PIL.Image
import numpy as np
from pathlib import Path
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Add Video-Depth-Anything to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Video-Depth-Anything'))
from video_depth_anything.video_depth import VideoDepthAnything

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

DEPTH_ENCODER = 'vitl'

def initialize_depth_model(encoder='vitl', device='cuda'):
    """Initialize Video-Depth-Anything model."""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    checkpoint_path = f'Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth'
    
    model = VideoDepthAnything(**model_configs[encoder], metric=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    
    return model

def load_and_sample_frames(video_path, num_frames, max_res=1280, from_middle=True):
    """Load video and sample frames from the middle section."""
    try:
        from decord import VideoReader, cpu
        DECORD_AVAILABLE = True
    except:
        DECORD_AVAILABLE = False
    
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vid)
        fps = vid.get_avg_fps()
        
        if total_frames > num_frames:
            if from_middle:
                start_idx = (total_frames - num_frames) // 2
                end_idx = start_idx + num_frames
                indices = list(range(start_idx, end_idx))
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        else:
            indices = list(range(total_frames))
        
        original_height, original_width = vid[0].shape[:2]
        height, width = original_height, original_width
        
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale))
            width = int(round(original_width * scale))
            height = height if height % 2 == 0 else height + 1
            width = width if width % 2 == 0 else width + 1
            
            vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        
        sampled_frames = vid.get_batch(indices).asnumpy()
        
    else:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames > num_frames:
            if from_middle:
                start_idx = (total_frames - num_frames) // 2
                end_idx = start_idx + num_frames
                indices = np.arange(start_idx, end_idx)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
        
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height, width = original_height, original_width
        
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale))
            width = int(round(original_width * scale))
        
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
    """Estimate depth maps for pre-loaded frames."""
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
    """Resample depth maps to match target number of frames."""
    if len(depth_maps) == num_frames:
        return depth_maps
    
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
    """Convert numpy frames array to PIL Images and resample to target dimensions."""
    if len(frames_array) != num_frames:
        indices = np.linspace(0, len(frames_array)-1, num_frames, dtype=int)
        frames_array = frames_array[indices]
    
    pil_frames = []
    for frame in frames_array:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        pil_frames.append(PIL.Image.fromarray(frame))
    
    return pil_frames

def save_pil_to_video(frames, output_path, fps=16):
    """Save PIL frames to video file."""
    imageio.mimwrite(output_path, [np.array(frame) for frame in frames], fps=fps)

def depth_to_rgb(depth_maps):
    """Convert depth maps to RGB images for visualization."""
    depth_frames = []
    for depth in depth_maps:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_rgb = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
        depth_frames.append(PIL.Image.fromarray(depth_rgb))
    return depth_frames

def depth_to_grayscale_pil(depth_maps):
    """Convert raw depth maps to grayscale PIL images for model conditioning."""
    depth_frames = []
    for depth in depth_maps:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_gray = PIL.Image.fromarray(depth_uint8, mode='L')
        depth_rgb = depth_gray.convert('RGB')
        depth_frames.append(depth_rgb)
    
    return depth_frames

def initialize_vace_pipeline(device='cuda'):
    """Initialize VACE pipeline for style transfer."""
    model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )

    flow_shift = 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to(device)
    
    return pipe

def process_video_on_gpu(gpu_id, video_info_list, all_depths, all_frames, all_fps, 
                         output_dirs, prompt, negative_prompt, height, width, num_frames):
    """Worker function to process videos on a specific GPU."""
    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        
        pipe = initialize_vace_pipeline(device=device)
        
        for video_path, video_name in video_info_list:
            try:
                depths = all_depths[video_name]
                frames = all_frames[video_name]
                fps = all_fps[video_name]
                
                depth_resampled = resample_depth_maps(depths, num_frames)
                video_frames = frames_to_pil(frames, height, width, num_frames)
                depth_condition_frames = depth_to_grayscale_pil(depth_resampled)
                depth_condition_frames = [frame.resize((width, height)) for frame in depth_condition_frames]
                depth_rgb_frames = depth_to_rgb(depth_resampled)
                
                torch.compiler.cudagraph_mark_step_begin()
                
                output = pipe(
                    video=depth_condition_frames,
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    guidance_scale=5.0,
                    generator=torch.Generator(device=device).manual_seed(42),
                ).frames[0]
                
                export_to_video(output, os.path.join(output_dirs['output'], f"{video_name}.mp4"), fps=16)
                save_pil_to_video(video_frames, os.path.join(output_dirs['input'], f"{video_name}.mp4"), fps=16)
                save_pil_to_video(depth_rgb_frames, os.path.join(output_dirs['depth'], f"{video_name}.mp4"), fps=16)
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {video_name}: {e}")
                continue
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {e}")

# Constants
# prompt = "First-person perspective, realistic photorealistic footage, natural camera movement, real-world environment, cinematic lighting, high fidelity details, smooth motion, professional video quality, lifelike textures, authentic atmosphere"
prompt = "First-person POV, GoPro footage style, handheld camera, natural shaky cam, real-world location, " \
        "authentic lighting conditions, practical camera work, documentary style, raw unedited feel, bodycam perspective, " \
        "action sports footage, parkour POV, extreme sports camera, chest mount perspective, helmet cam view, urban exploration, real life recording, amateur videography aesthetic, found footage style, dashcam quality, vlog camera movement, smartphone video quality, security camera realism, CCTV footage, livestream quality, unfiltered reality, natural color grading, real-world imperfections, authentic motion blur, actual camera physics, practical effects only, zero CGI, documentary filmmaking, cinema verite, observational camera, candid footage, street photography video, real location shoot, natural environment, outdoor lighting, overcast sky lighting, golden " \
        "hour natural light, practical shadows, real-world reflections, authentic depth of field, camera operator visible, lens flare from real sun, dust particles in air, natural wind movement, realistic weather conditions, genuine human reactions, unscripted moments, real-time recording, continuous shot, long take, no cuts, diegetic sound only, environmental audio, real background noise"


negative_prompt = "Gameplay, unrealistic, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

height = 512
width = 512
num_frames = 81

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    video_paths = get_video_paths("/mnt/data/datasets/extracted_tars/kbm/fps")[:50]
    
    # Create output directories
    output_base = "./outputs"
    depth_dir = os.path.join(output_base, "depth")
    input_dir = os.path.join(output_base, "input")
    output_dir = os.path.join(output_base, "output")
    depth_npy_dir = os.path.join(output_base, "depth_npy")
    
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depth_npy_dir, exist_ok=True)
    
    # Initialize depth model
    depth_model = initialize_depth_model(encoder=DEPTH_ENCODER, device=DEVICE)
    
    # Pre-compute depth maps
    all_depths = {}
    all_frames = {}
    all_fps = {}
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        
        try:
            sampled_frames, fps = load_and_sample_frames(video_path, num_frames=num_frames, max_res=1280)
            depths, fps = estimate_depth(depth_model, sampled_frames, fps, input_size=512, device=DEVICE)
            
            all_depths[video_name] = depths
            all_frames[video_name] = sampled_frames
            all_fps[video_name] = fps
            
            np.save(os.path.join(depth_npy_dir, f"{video_name}.npy"), depths)
            
        except Exception as e:
            print(f"Error computing depth for {video_name}: {e}")
            continue
    
    # Clean up depth model
    del depth_model
    torch.cuda.empty_cache()
    
    # Prepare multi-GPU processing
    num_gpus = torch.cuda.device_count()
    
    successful_videos = [(video_path, Path(video_path).stem) 
                         for video_path in video_paths 
                         if Path(video_path).stem in all_depths]
    
    videos_per_gpu = [[] for _ in range(num_gpus)]
    for idx, video_info in enumerate(successful_videos):
        gpu_id = idx % num_gpus
        videos_per_gpu[gpu_id].append(video_info)
    
    output_dirs = {
        'output': output_dir,
        'input': input_dir,
        'depth': depth_dir
    }
    
    # Run multi-GPU processing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    processes = []
    for gpu_id in range(num_gpus):
        if len(videos_per_gpu[gpu_id]) > 0:
            p = mp.Process(
                target=process_video_on_gpu,
                args=(gpu_id, videos_per_gpu[gpu_id], all_depths, all_frames, all_fps,
                      output_dirs, prompt, negative_prompt, height, width, num_frames)
            )
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
