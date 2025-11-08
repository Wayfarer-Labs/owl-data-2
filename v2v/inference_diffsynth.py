import torch
from diffsynth import save_video,VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

def save_first_frame(video_path, filename):
    video = VideoData(video_path)
    first_frame = video[:, :, 0]
    first_frame_pil = Image.fromarray((first_frame.cpu().numpy() * 255).astype('uint8'))
    first_frame_pil.save(filename)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/control_video.mp4", "data/examples/wan/reference_image_girl.png"]
# )
PROMPT = "First-person POV, GoPro footage style, handheld camera, natural shaky cam, real-world location, authentic lighting conditions, practical camera work, documentary style, raw unedited feel, bodycam perspective, action sports footage, parkour POV, extreme sports camera, chest mount perspective, helmet cam view, urban exploration, real life recording, amateur videography aesthetic, found footage style, dashcam quality, vlog camera movement, smartphone video quality, security camera realism, CCTV footage, livestream quality, unfiltered reality, natural color grading, real-world imperfections, authentic motion blur, actual camera physics, practical effects only, zero CGI, documentary filmmaking, cinema verite, observational camera, candid footage, street photography video, real location shoot, natural environment, outdoor lighting, overcast sky lighting, golden hour natural light, practical shadows, real-world reflections, authentic depth of field, camera operator visible, lens flare from real sun, dust particles in air, natural wind movement, realistic weather conditions, genuine human reactions, unscripted moments, real-time recording, continuous shot, long take, no cuts, diegetic sound only, environmental audio, real background noise"
NEGATIVE_PROMPT = "Gameplay, unrealistic, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
HEIGHT=512
WIDTH=512
VIDEO_PATH = "/home/sky/anmol/owl-data-2/v2v/outputs/depth/2025-09-11 10-28-05.mp4"
IMAGE_PATH = "scripts/ref.png"
control_video = VideoData(VIDEO_PATH, height=HEIGHT, width=WIDTH)
reference_image = Image.open(IMAGE_PATH).resize((HEIGHT, WIDTH))
# control_video = VideoData(VIDEO_PATH)
video = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    control_video=control_video, 
    reference_image=reference_image,
    height=HEIGHT, width=WIDTH, #num_frames=49,
    num_inference_steps=20,
    seed=1, #tiled=True
)
save_video(video, "FUN_CONTROL_video.mp4", fps=15, quality=5)
