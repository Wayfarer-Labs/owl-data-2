import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys
from pathlib import Path
sys.path.append('Video-Depth-Anything')
torch.backends.cuda.enable_flash_sdp(True)
import pdb
from video_depth_anything.video_depth import VideoDepthAnything
from metric_depth.video_depth_anything.video_depth import VideoDepthAnything as MetricVideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

class VideoDepthPipeline:
    def __init__(self, depth_encoder:str='vitl', save_video_depth:bool=False, epsilon:float=1e-8):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_video = save_video_depth

        # self.video_depth_anything = VideoDepthAnything(**self.model_configs[depth_encoder])
        self.video_depth_anything = MetricVideoDepthAnything(**self.model_configs[depth_encoder])

        # Get path to Video-Depth-Anything checkpoints directory dynamically
        project_root = os.path.normpath(os.path.join(os.path.abspath(__file__),'..','..','..'))
        checkpoint_path = os.path.join(project_root, 'Video-Depth-Anything' , 'checkpoints' , f'video_depth_anything_{depth_encoder}.pth')
        
        self.video_depth_anything.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
        self.video_depth_anything = self.video_depth_anything.to(self.device).eval()

        #use to invert the depth map
        self.epsilon = epsilon
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high") 
    
    def convert_tensor_to_frames(self, frames_tensor: torch.Tensor) -> np.ndarray:
        """
        [N,C,H,W] torch.uint8 (CPU) -> [N,H,W,C] np.uint8 without extra copies.
        """
        assert frames_tensor.device.type == "cpu", "Keep frames on CPU for VDA to avoid GPU->CPU sync."
        if frames_tensor.dtype != torch.uint8:
            # Normalize on CPU only if needed
            if frames_tensor.dtype.is_floating_point:
                frames_tensor = (frames_tensor.clamp(0,1) * 255).to(torch.uint8)
            else:
                frames_tensor = frames_tensor.to(torch.uint8)

        # NCHW -> NHWC (Torch->NumPy shares memory; make contiguous to be safe)
        frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous().numpy()
        return frames_np  # (N,H,W,C) uint8
    
    def batch_to_numpy_uint8(self, frames_cuda: torch.Tensor) -> np.ndarray:
        '''
        (N,C,H,W) on CUDA -> (N,H,W,C) NumPy uint8 via one async D2H copy.
        '''
        assert frames_cuda.is_cuda and frames_cuda.ndim == 4 and frames_cuda.shape[1] == 3
        # Make it small on GPU first (uint8 NHWC)
        x = frames_cuda
        if x.dtype.is_floating_point:
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
        else:
             x = x.to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,H,W,C) uint8 on GPU
        # Single pinned CPU buffer + single async copy
        cpu_buf = torch.empty_like(x, device="cpu", pin_memory=True)
        cpu_buf.copy_(x, non_blocking=True)
        torch.cuda.current_stream().synchronize()  # one sync for the whole batch
        return cpu_buf.numpy()  # zero-copy NumPy view

    def process_video(self, frames: torch.Tensor, video_dir: str, video_name:str, target_width: int, target_height:int, video_fps: int):
        # Convert tensor to format expected by infer_video_depth
        if frames.device.type == "cpu":
            frames_np = self.convert_tensor_to_frames(frames) #get (N,H,W,C) uint8
        else:
            frames_np = self.batch_to_numpy_uint8(frames) #get (N,H,W,C) uint8

        with torch.inference_mode():
            depths, fps = self.video_depth_anything.infer_video_depth(
                frames = frames_np, 
                target_fps = video_fps, 
                target_width=target_width, 
                target_height=target_height, 
                device=self.device, 
                fp32=True
            )
            depths = self.replace_inf(depths)
            depths = self.invert_and_rescale_depth(depths)
        
        if self.save_video:
            save_path = os.path.join(video_dir, f'{video_name}_depth.mp4')
            save_video(depths, save_path, fps=fps, is_depths=True, grayscale=True)
        
        #Convert depths back to tensors
        return torch.from_numpy(depths), fps
    
    def invert_and_rescale_depth(self, depths:np.array):
        max_depth = depths.max()
        min_depth = depths.min()
        depths = (depths-min_depth)/(max_depth-min_depth)
        depths = 1-depths
        return depths

    def replace_inf(self, tensor:np.array):
        finite = np.isfinite(tensor)
        if not finite.any():
            return tensor  # or set to zeros
        max_finite = tensor[finite].max()
        tensor[~finite] = max_finite
        return tensor

    def __call__(self, frames: torch.Tensor, video_dir: str, video_name:str, target_width: int, target_height:int, video_fps: int):
        depths, fps = self.process_video(frames=frames, video_dir=video_dir, video_name=video_name, target_width=target_width, target_height=target_height, video_fps=video_fps)
        return depths, fps
