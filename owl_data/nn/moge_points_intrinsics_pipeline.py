import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys
from typing import Optional, Tuple, List
import imageio
import matplotlib.cm as cm
sys.path.append('MoGe')
from MoGe.moge.model.v2 import MoGeModel
import pdb

class MoGePointsIntrinsicsPipeline:
    def __init__(self, intrinsics_only:bool=False, save_video_depth:bool=False, epsilon:float=1e-8):
        self.device = torch.device("cuda")
        self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        self.intrinsics_only = intrinsics_only
        self.save_video = save_video_depth
        self.epsilon = epsilon
    
    def process_frames(self, frames: torch.Tensor):
        """
        Convert frames from decode_video() format to MoGe expected format.
        
        Args:
            frames: torch.Tensor of shape (N, C, H, W) with uint8 values [0, 255]
        
        Returns:
            MoGe model outputs
        """
        # Convert from uint8 [0, 255] to float32 [0, 1]
        frames = frames.float() / 255.0
        
        # Move to device
        frames = frames.to(self.device)
        
        # Process each frame
        outputs = []
        for i in range(frames.shape[0]):
            frame = frames[i]  # Shape: (C, H, W)
            # MoGe expects input in range [0, 1] - which we now have
            # The model will internally apply ImageNet normalization:
            # (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            output = self.moge_model.infer(frame)
            #normalize scale of depth/points map to remove inf
            output['depth'] = self.replace_inf(output['depth'])
            output['points'] = self.replace_inf(output['points'])
            outputs.append(output)
        
        return outputs
    
    def _save_video(self, frame_output, output_video_path, fps=10, is_depths=False, grayscale=False):
        
        depth_frames = []
        for output in frame_output:
            if 'depth' in output and output['depth'] is not None:
                # MoGe depth is a tensor of shape (H, W) or (1, H, W)
                depth = self.replace_inf(output['depth'])
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                    if depth.ndim == 3 and depth.shape[0] == 1:
                        depth = depth.squeeze(0)  # Remove batch dimension if present
                depth_frames.append(depth)
            else:
                print(f"[WARNING] Frame {len(depth_frames)} missing depth data, skipping")
        
        if not depth_frames:
            print("[ERROR] No valid depth frames found")
            return
        
        # Stack all depth frames
        depth_frames = np.array(depth_frames)

        writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
        if is_depths:
            colormap = np.array(cm.get_cmap("inferno").colors)
            d_min, d_max = depth_frames.min(), depth_frames.max()
            for i in range(depth_frames.shape[0]):
                depth = depth_frames[i]
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
                writer.append_data(depth_vis)
        else:
            for i in range(depth_frames.shape[0]):
                writer.append_data(frames[i])
        writer.close()

    def _save_depth_video(self, frame_output: list, save_path: str):
        """
        Compile depth frames from MoGe output into a continuous depth video.
        
        Args:
            frame_output (list): List of MoGe output dictionaries, each containing 'depth' key
            save_path (str): Path where the depth video will be saved
        """
        if not frame_output:
            print("[WARNING] No frames to save for depth video")
            return
        
        print(f"[INFO] Saving depth video with {len(frame_output)} frames to {save_path}")
        
        # Extract depth maps from MoGe outputs
        depth_frames = []
        for output in frame_output:
            if 'depth' in output and output['depth'] is not None:
                # MoGe depth is a tensor of shape (H, W) or (1, H, W)
                depth = output['depth']
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                    if depth.ndim == 3 and depth.shape[0] == 1:
                        depth = depth.squeeze(0)  # Remove batch dimension if present
                
                depth_frames.append(depth)
            else:
                print(f"[WARNING] Frame {len(depth_frames)} missing depth data, skipping")
        
        if not depth_frames:
            print("[ERROR] No valid depth frames found")
            return
        
        # Stack all depth frames
        depth_frames = np.array(depth_frames, dtype=np.uint8)  # Shape: (N, H, W)
        # Apply colormap for better visualization
        colormap = cm.get_cmap('grey')  # You can change to 'viridis', 'plasma', 'magma', etc.
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Method 1: Use imageio (recommended for better quality and compatibility)
        try:
            h, w = depth_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1','H264' if available
            vw = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
            for f in depth_frames:                  # f: uint8 BGR, HxWx3, contiguous
                vw.write(f)
            vw.release()
            print(f"[SUCCESS] Depth video saved to {save_path}")
        except Exception as e:
            print(f'[ERROR] could not process depth video to {save_path}')
        
    def replace_inf(self, tensor:torch.Tensor):
        max_finite = tensor[torch.isfinite(tensor)].max()
        tensor = torch.where(torch.isinf(tensor), max_finite, tensor)
        return tensor

    def __call__(self, frames: torch.Tensor, video_dir: str, video_name: str, video_fps:int) -> dict:
        output = self.process_frames(frames)
        
        if self.save_video:
            save_path = os.path.join(video_dir, f'{video_name}_depth.mp4')
            self._save_video(
                frame_output=output, 
                output_video_path=save_path,
                fps=video_fps,
                is_depths=True,
                grayscale=True
            )

        if self.intrinsics_only:
            return {'intrinsics': torch.stack([out.get('intrinsics') for out in output],0).cpu()}
        else:
            # Aggregate outputs from all frames
            return {
                'points': torch.stack([out.get('points', None) for out in output],0).cpu(), 
                'depth': torch.stack([out.get('depth', None) for out in output], 0).cpu(), 
                'intrinsics': torch.stack([out.get('intrinsics', None) for out in output],0).cpu()
            }
