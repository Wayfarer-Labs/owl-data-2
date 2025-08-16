import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys
from typing import Optional, Tuple, List

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
            pose2d='rtmpose-m_8xb256-420e_coco-256x192', 
            det_model='rtmdet_tiny_8xb32-300e_coco'
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
                    frame = frames[i].permute(1, 2, 0).cpu().numpy()
                else:  # [N, H, W, C]
                    frame = frames[i].cpu().numpy()
            else:
                raise ValueError(f"Unsupported frame tensor shape: {frames.shape}")
                
            # Ensure frame is in uint8 format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
            yield frame
    
    def _extract_keypoints_from_predictions(self, predictions: List) -> List[np.ndarray]:
        """
        Extract keypoints from model predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            List of keypoint arrays with shape [K, 2] containing only x,y coordinates
        """
        all_keypoints = []
        
        for pred in predictions:
            preds = pred['predictions'][0]
            instances = preds.get('pred_instances', preds.get('instances', []))
            
            for inst in instances:
                if np.mean(inst['keypoint_scores']) >= self.keypoint_threshold:
                    
                    # Only keep keypoint coordinates - remove confidence scores if present
                    all_keypoints.append(torch.from_numpy(inst['keypoints']).float())
                    
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
            
        for frame_idx, frame in enumerate(self._frame_generator(frames)):
            try:
                # Process with human pose model
                human_predictions = list(self.human_pose_model(frame, show=False, vis_out_dir=None))
                human_keypoints = self._extract_keypoints_from_predictions(human_predictions)
                
                # Process with animal pose model  
                animal_predictions = list(self.animal_pose_model(frame, show=False, vis_out_dir=None))
                animal_keypoints = self._extract_keypoints_from_predictions(animal_predictions)
                
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
        for frame_nested, frame_flattened in self.process_video(frames=frames):
            return frame_nested, frame_flattened
