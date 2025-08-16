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

class MoGePointsIntrinsicsPipeline:
    def __init__(self, intrinsics_only:bool=False, save_depth_video:bool=False):
        self.device = torch.device("cuda")
        self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        self.intrinsics_only = intrinsics_only
        self.save_depth_video = save_depth_video
    
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
            outputs.append(output)
        
        return outputs
    
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
        depth_frames = np.array(depth_frames)  # Shape: (N, H, W)
        
        # Handle infinite values (masked areas) by setting them to a reasonable value
        finite_mask = np.isfinite(depth_frames)
        if not np.all(finite_mask):
            # Get the max finite depth value for replacement
            max_finite_depth = np.max(depth_frames[finite_mask]) if np.any(finite_mask) else 10.0
            depth_frames[~finite_mask] = max_finite_depth
            print(f"[INFO] Replaced infinite depth values with {max_finite_depth}")
        
        # Apply colormap for better visualization
        colormap = cm.get_cmap('inferno')  # You can change to 'viridis', 'plasma', 'magma', etc.
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Method 1: Use imageio (recommended for better quality and compatibility)
        try:
            # Convert normalized depth to RGB using colormap
            colored_frames = []
            for i in range(depth_frames.shape[0]):
                depth_frame = depth_frames[i]
                # Apply colormap and convert to RGB (0-255)
                colored = (colormap(depth_frame)[:, :, :3] * 255).astype(np.uint8)
                colored_frames.append(colored)
            
            # Save using imageio
            writer = imageio.get_writer(
                save_path, 
                fps=30,  # You might want to make this configurable
                macro_block_size=1,
                codec='libx264',
                ffmpeg_params=['-crf', '18']  # High quality encoding
            )
            
            for frame in colored_frames:
                writer.append_data(frame)
            
            writer.close()
            print(f"[SUCCESS] Depth video saved to {save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save with imageio: {e}")
            print("[INFO] Trying OpenCV fallback...")
            
            # Method 2: OpenCV fallback
            try:
                if not colored_frames:
                    # If imageio failed before creating colored_frames
                    colored_frames = []
                    for i in range(depth_frames.shape[0]):
                        depth_frame = depth_frames[i]
                        colored = (colormap(depth_frame)[:, :, :3] * 255).astype(np.uint8)
                        # Convert RGB to BGR for OpenCV
                        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
                        colored_frames.append(colored_bgr)
                
                # Get frame dimensions
                height, width = colored_frames[0].shape[:2]
                
                # Try different codecs/extensions
                fourcc_options = [
                    ('mp4v', '.mp4'),
                    ('MJPG', '.avi'),
                    ('XVID', '.avi'),
                ]
                
                success = False
                for fourcc_str, ext in fourcc_options:
                    try:
                        # Change extension if needed
                        base_path = os.path.splitext(save_path)[0]
                        test_path = base_path + ext
                        
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                        out = cv2.VideoWriter(test_path, fourcc, 30.0, (width, height))
                        
                        if not out.isOpened():
                            continue
                            
                        for frame in colored_frames:
                            out.write(frame)
                        
                        out.release()
                        print(f"[SUCCESS] Depth video saved to {test_path} using {fourcc_str}")
                        success = True
                        break
                        
                    except Exception as codec_error:
                        print(f"[WARNING] Failed with {fourcc_str}: {codec_error}")
                        continue
                
                if not success:
                    print("[ERROR] All OpenCV codecs failed")
                    
            except Exception as cv_error:
                print(f"[ERROR] OpenCV fallback failed: {cv_error}")

    def __call__(self, frames: torch.Tensor, video_dir: str, video_name: str) -> dict:
        output = self.process_frames(frames)
        
        if self.save_depth_video:
            save_path = os.path.join(video_dir, f'{video_name}_depth.mp4')
            self._save_depth_video(frame_output=output, save_path=save_path)

        if self.intrinsics_only:
            return {'intrinsics': [out.get('intrinsics') for out in output]}
        else:
            # Aggregate outputs from all frames
            return {
                'points': [out.get('points', None) for out in output], 
                'depth': [out.get('depth', None) for out in output], 
                'intrinsics': [out.get('intrinsics') for out in output]
            }
