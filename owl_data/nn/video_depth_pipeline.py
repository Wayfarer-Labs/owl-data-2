import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys

torch.backends.cuda.enable_flash_sdp(True)

sys.path.append('Video-Depth-Anything')
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

class VideoDepthPipeline:
    def __init__(self, encoder:str='vitl', save_depth_video:bool=False):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_video = save_depth_video

        self.video_depth_anything = VideoDepthAnything(**self.model_configs[encoder])
        self.video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{encoder}.pth', map_location='cpu'), strict=True)
        self.video_depth_anything = self.video_depth_anything.to(self.device).eval()
    
    def convert_tensor_to_frames(self, frames_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch.Tensor from tensors_from_mp4s.py format to numpy array format expected by VideoDepthAnything infer_video_depth
        
        Input format: [N, C, H, W] - torch.uint8 [0,255]
        Output format: [N, H, W, C] - numpy.uint8 [0,255]
        """
        # Ensure tensor is on CPU and convert to numpy
        frames_np = frames_tensor.cpu().numpy()
        
        # Convert from [N, C, H, W] to [N, H, W, C]
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        
        # Ensure uint8 format [0,255]
        frames_np = frames_np.astype(np.uint8)
        
        return frames_np
    
    def process_video(self, frames: torch.Tensor, video_dir: str, video_name:str, target_width: int, target_height:int, video_fps: int):
        # Convert tensor to format expected by infer_video_depth
        frames_np = self.convert_tensor_to_frames(frames)
        depths, fps = self.video_depth_anything.infer_video_depth(frames_np, video_fps, target_width=target_width, target_height=target_height, device=self.device, fp32=True)
        
        #Convert depths back to tensors
        depths = torch.from_numpy(depths)

        if self.save_video:
            save_path = os.path.join(video_dir, f'{video_name}_depth.mp4')
            save_video(depths, save_path, fps=fps, is_depths=True, grayscale=True)
            
        return depths, fps
    
    def __call__(self, frames: torch.Tensor, video_dir: str, video_name:str, target_width: int, target_height:int, video_fps: int):
        depths, fps = self.process_video(frames=frames, video_dir=video_dir, video_name=video_name, target_width=target_width, target_height=target_height, video_fps=video_fps)
        return depths, fps
