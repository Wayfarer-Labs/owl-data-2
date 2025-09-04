import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys
import ssl
sys.path.append('mmpose')
from typing import Optional, Tuple, List, Dict
import pdb
from tqdm import tqdm

# Fix SSL certificate verification error
ssl._create_default_https_context = ssl._create_unverified_context

torch.backends.cuda.enable_flash_sdp(True)
from mmpose.apis import MMPoseInferencer

class PoseKeypointPipeline:
    def __init__(self, keypoint_threshold: float = 0.3):
        """
        Initialize the pose keypoint pipeline with human and animal pose models.
        
        Args:
            keypoint_threshold: Minimum confidence threshold for keypoints
        """
        self.human_pose_model = MMPoseInferencer(
            pose2d='rtmpose-m_8xb256-420e_coco-256x192'
        )
        # Use the animal alias which corresponds to rtmpose-m_8xb64-210e_ap10k-256x256
        self.animal_pose_model = MMPoseInferencer(
            pose2d='animal'
        )
        self.keypoint_threshold = keypoint_threshold
    
    def _frame_generator(self, frames: torch.Tensor):
        """
        Generator that yields frames from the input tensor.
        
        Args:
            frames: Input video tensor of shape [N, C, H, W] or [N, H, W, C]
        """
        n_frames = frames.shape[0]
        
        for i in range(n_frames):
            # Handle different input formats
            if frames.dim() == 4:
                if frames.shape[1] == 3:  # [N, C, H, W]
                    frame = frames[i].permute(1, 2, 0).numpy()
                else:  # [N, H, W, C]
                    frame = frames[i].numpy()
            else:
                raise ValueError(f"Unsupported frame tensor shape: {frames.shape}")
                
            # Ensure frame is in uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
            yield frame
    
    def _extract_keypoints_from_predictions(self, predictions: Dict, height:int, width:int) -> List[np.ndarray]:
        """
        Extract keypoints from model predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            List of keypoint arrays with shape [K, 2] containing only x,y coordinates
        """
        all_keypoints = []
        
        for pred_instance in predictions['predictions'][0]:
            keypoints = np.array(pred_instance.get('keypoints', []))
            #clip keypoints within bounds of image tensor
            keypoints[:,0] = np.clip(keypoints[:,0], 0, width-1)
            keypoints[:,1] = np.clip(keypoints[:,1], 0, height-1)
            
            if np.mean(pred_instance.get('keypoint_scores',[])) >= self.keypoint_threshold:
                
                # Only keep keypoint coordinates - remove confidence scores if present
                all_keypoints.append(torch.Tensor(keypoints).float())
                    
        return all_keypoints
    
    def process_video(self, frames: torch.Tensor) -> tuple[torch.nested.nested_tensor, torch.Tensor]:
        """
        Process video frames and extract keypoints from both human and animal models.
        
        Args:
            frames: Input video tensor of shape [N, C, H, W] or [N, H, W, C]
            
        Returns:
            tuple: (nested_tensor, flattened_points)
                - nested_tensor: Nested tensor containing keypoints from all instances
                  Each element has shape [K, 2] where K varies per instance
                - flattened_points: Tensor of shape [total_points, 2] with all keypoints
                  flattened from all objects and frames
        """
        if frames.numel() == 0:
            return torch.nested.nested_tensor([]), torch.empty(0, 2)
            
        for frame_idx, frame in tqdm(enumerate(self._frame_generator(frames))):
            try:
                # Process with human pose model
                human_predictions = next(self.human_pose_model(frame, show=False))
                human_keypoints = self._extract_keypoints_from_predictions(
                                    human_predictions,
                                    height = frame.shape[0],
                                    width = frame.shape[1]
                )
                
                # Process with animal pose model  
                animal_predictions = next(self.animal_pose_model(frame, show=False))
                animal_keypoints = self._extract_keypoints_from_predictions(
                                    animal_predictions,
                                    height = frame.shape[0],
                                    width = frame.shape[1]
                )
                
                # Combine keypoints from both models
                frame_keypoints = human_keypoints + animal_keypoints
                
                # Create nested tensor to handle ragged dimensions
                if frame_keypoints:
                    nested_tensor = torch.nested.nested_tensor(frame_keypoints)
                    flattened_points = torch.cat([t for t in nested_tensor.unbind()], dim=0)
                    yield nested_tensor, flattened_points
                else:
                    yield torch.nested.nested_tensor([]), torch.empty(0, 2)
                     
            except Exception as e:
                print(f"Warning: Failed to process frame {frame_idx}: {e}")
                continue
        

    def __call__(self, frames: torch.Tensor) -> tuple[torch.nested.nested_tensor, torch.Tensor]:
        """
        Callable interface for the pipeline.
        
        Args:
            frames: Input video tensor
            
        Returns:
            tuple: (nested_tensor, flattened_points)
                - nested_tensor: Nested tensor containing keypoints from all instances
                - flattened_points: Tensor of shape [total_points, 2] with all keypoints
                  flattened from all objects and frames
        """
        nested_frame_data = []
        flattened_frame_data = []
        
        for i, (frame_nested, frame_flattened) in enumerate(self.process_video(frames=frames)):
            nested_frame_data.append(frame_nested)
            flattened_frame_data.append(frame_flattened)
        
        return nested_frame_data, torch.nested.nested_tensor(flattened_frame_data)
