import torch
import PIL.Image
import numpy as np
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from inference_diffusers import initialize_depth_model, estimate_depth, depth_to_grayscale_pil

torch.backends.cuda.matmul.allow_tf32 = True
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


def prepare_video_and_mask(first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask

def resize_frames(frames: list[PIL.Image.Image], height: int, width: int):
    resized_frames = []
    for frame in frames:
        resized_frame = frame.resize((width, height))
        resized_frames.append(resized_frame)
    return resized_frames

def get_depth(frames: list[PIL.Image.Image], height: int, width: int):
    DEPTH_ENCODER = 'vitl'
    depth_model = initialize_depth_model(encoder=DEPTH_ENCODER, device='cuda')
    depth_maps = estimate_depth(depth_model, frames, fps=60, device='cuda')[0]
    print(f"Estimated depth maps shape: {depth_maps[0].shape}")
    depth_condition_frames = depth_to_grayscale_pil(depth_maps)
    depth_condition_frames = [frame.resize((width, height)) for frame in depth_condition_frames]
                
    del depth_model
    torch.cuda.empty_cache()
    return depth_condition_frames

def main():
    model_id = "Wan-AI/Wan2.1-VACE-14B-diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cuda")

    prompt = "First-person POV, GoPro footage style, natural shaky cam, real-world location, authentic lighting conditions, practical camera work, documentary style, raw unedited feel, bodycam perspective, action sports footage, parkour POV, extreme sports camera, helmet cam view, urban exploration, real life recording, amateur videography aesthetic, found footage style, dashcam quality, vlog camera movement, smartphone video quality, livestream quality, unfiltered reality, natural color grading, real-world imperfections, authentic motion blur, practical effects only, zero CGI, documentary filmmaking, cinema verite, candid footage, street photography video, real location shoot, natural environment, outdoor lighting, overcast sky lighting, golden hour natural light, practical shadows, real-world reflections, authentic depth of field, camera operator visible, lens flare from real sun, dust particles in air, natural wind movement, realistic weather conditions, genuine human reactions, unscripted moments, real-time recording, continuous shot"
    # prompt = "realistic and cinema like, perfect 8K, high quality, film"

    # height = 512
    # width = 512
    height = 36*12
    width = 36*16
    num_frames = 81
    STEPS = 40
    VIDEO_PATH = "/mnt/data/waypoint_1/owl_control/kbm/fps/00ca712fc72c40cb/2025-09-11 10-28-05.mp4"
    TOTAL_FRAMES = 1610
    REF_IMAGE = None
    video = load_and_sample_frames(
        video_path=VIDEO_PATH,
        num_frames=TOTAL_FRAMES,
        max_res=1280,
        from_middle=True,
    )[0]
    depth = get_depth(video, height, width)
    video = [PIL.Image.fromarray(frame) for frame in video]
    video = resize_frames(video, height, width)
    export_to_video(video, "input_video.mp4", fps=60)
    export_to_video(depth, "input_depth.mp4", fps=60)
    print(f"Video length: {len(video)} frames, video type: {type(video[0])}")
    print(f"Frame resolution: {video[0].size[0]}x{video[0].size[1]}")

    n = 10
    len_per_segment = (TOTAL_FRAMES) // n
    final_output = []
    print(f"Processing video in {n} segments of {len_per_segment} frames each")
    for i in range(5):

        # video_segment = video[i * len_per_segment: (i + 1) * len_per_segment]
        video_segment = depth[i * len_per_segment: (i + 1) * len_per_segment]
        print(f"Video segment type: {type(video_segment)}, {len(video_segment)} frames")
        # print(f"Video segment shape: {(video_segment[0]).shape}")
        print(f"Processing segment {i+1}/{n} with {len(video_segment)} frames")
        if REF_IMAGE is None:
            output = pipe(
                video=video_segment,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=len_per_segment,
                num_inference_steps=STEPS,
                guidance_scale=5.0,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
        else:
            output = pipe(
                video=video_segment,
                reference_images=REF_IMAGE,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=len_per_segment,
                num_inference_steps=STEPS,
                guidance_scale=5.0,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
        REF_IMAGE = [
            PIL.Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8))
            for frame in output[-1:]
        ]
        final_output.append(output)
    all_frames = np.concatenate(final_output, axis=0)
    export_to_video(all_frames, "long_gen.mp4", fps=60)


if __name__ == "__main__":
    main()