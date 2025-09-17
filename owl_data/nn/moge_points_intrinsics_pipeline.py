import cv2
import os
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
import numpy as np
from torch.nn import functional as F
import sys
from typing import Optional, Tuple, List
import imageio
import matplotlib
sys.path.append('MoGe')
from MoGe.moge.model.v2 import MoGeModel
import pdb

class MoGePointsIntrinsicsPipeline:
    def __init__(self, intrinsics_only:bool=False, save_video_depth:bool=False, epsilon:float=1e-8, batch_size:int=8):
        self.device = torch.device("cuda")
        self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        self.moge_model.eval()
        self.intrinsics_only = intrinsics_only
        self.save_video = save_video_depth
        self.epsilon = epsilon
        self.batch_size = batch_size
    
    def process_frames(self, frames: torch.Tensor):
        """
        Convert frames from decode_video() format to MoGe expected format.
        
        Args:
            frames: torch.Tensor of shape (N, C, H, W) with uint8 values [0, 255]
        
        Returns:
            MoGe model outputs
        """
        if frames.device.type!= "cpu":
            frames = frames.cpu()
        if frames.dtype.is_floating_point:
            frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        else:
            frames = frames.to(torch.uint8)
        frames = frames.pin_memory()
        
        # Process each frame
        outputs: List[dict] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0,frames.shape[0],self.batch_size):
                frame_batch = frames[i: min(i+self.batch_size,frames.shape[0])]  # Shape: (C, H, W)
                # Move to GPU (async if coming from pinned CPU)
                frame_batch = frame_batch.to(self.device, non_blocking=True)

                # Convert to float on GPU and normalize
                # Keep channels_last for faster convs
                frame_batch = (frame_batch.to(memory_format=torch.channels_last)
                            .to(torch.float32)
                            .div_(255.0))
                
                for j in range(0, frame_batch.shape[0]):
                    # The model will internally apply ImageNet normalization:
                    # (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    output = self.moge_model.infer(frame_batch[j])
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
                depth = self.replace_inf(output['depth'].clone())
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
        # Compute global min/max once
        d_min = np.nanmin([np.nanmin(d) for d in depth_frames])
        d_max = np.nanmax([np.nanmax(d) for d in depth_frames])
        denom = max(d_max - d_min, self.epsilon)
        writer = imageio.get_writer(
            output_video_path, fps=fps, macro_block_size=1,
            codec='libx264', ffmpeg_params=['-crf', '18']
        )

        try:
            if is_depths:
                colormap = np.array(matplotlib.colormaps.get_cmap("gray"))
                depth_norm = (np.clip((depth_frames - d_min) / denom, 0, 1) * 255).astype(np.uint8)
                for i in range(depth_norm.shape[0]):
                    writer.append_data(
                        cv2.cvtColor(
                            cv2.applyColorMap(depth_norm[i], cv2.COLORMAP_INFERNO), 
                            cv2.COLOR_BGR2RGB
                        )
                    )
            else:
                for rgb_frame in depth_frames:
                    writer.append_data(rgb_frame)
        finally:
            writer.close()
               
    def replace_inf(self, tensor:torch.Tensor):
        finite = torch.isfinite(tensor)
        if not finite.any():
            return tensor  # or set to zeros
        max_finite = tensor[finite].max()
        tensor[~finite] = max_finite
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
            return {'intrinsics': torch.stack([out.get('intrinsics') for out in output],0).to('cpu', non_blocking=True)}
        else:
            # Aggregate outputs from all frames
            return {
                'points': torch.stack([out.get('points', None) for out in output],0).to('cpu', non_blocking=True), 
                'depth': torch.stack([out.get('depth', None) for out in output], 0).to('cpu', non_blocking=True), 
                'intrinsics': torch.stack([out.get('intrinsics', None) for out in output],0).to('cpu', non_blocking=True)
            }
