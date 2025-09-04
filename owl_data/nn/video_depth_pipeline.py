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
        self.video_depth_anything.eval()
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
            torch.cuda.empty_cache()
            
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
        max_finite = tensor[np.isfinite(tensor)].max()
        tensor = np.where(np.isinf(tensor), max_finite, tensor)
        return tensor

    def __call__(self, frames: torch.Tensor, video_dir: str, video_name:str, target_width: int, target_height:int, video_fps: int):
        depths, fps = self.process_video(frames=frames, video_dir=video_dir, video_name=video_name, target_width=target_width, target_height=target_height, video_fps=video_fps)
        return depths, fps
