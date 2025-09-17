import cv2
import os
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
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
    def __init__(self, keypoint_threshold:float = 0.3, batch_size:int=8):
        """
        Initialize the pose keypoint pipeline with human and animal pose models.
        
        Args:
            keypoint_threshold: Minimum confidence threshold for keypoints
        """
        self.device = torch.device("cuda")
        self.human_pose_model = MMPoseInferencer(
            pose2d='rtmpose-m_8xb256-420e_coco-256x192',
            device=self.device
        )
        # Use the animal alias which corresponds to rtmpose-m_8xb64-210e_ap10k-256x256
        self.animal_pose_model = MMPoseInferencer(
            pose2d='animal',
            device=self.device
        )
        self.keypoint_threshold = keypoint_threshold
        self.batch_size = int(batch_size)
    
    def _preprocess_frames(self, frames: torch.Tensor):
        """
        Generator that yields frames from the input tensor.
        
        Args:
            frames: Input video tensor of shape [N, C, H, W] or [N, H, W, C]
        """
        assert frames.device.type == 'cpu', "Keep pose inputs on CPU to avoid GPU->CPU sync"
        assert frames.ndim == 4, "Expected batched 4D tensor for keypoint detection"

        if frames.dtype != torch.uint8:
            frames = (frames.clamp(0,1).mul(255)).to(torch.uint8) if frames.dtype.is_floating_point else frames.to(torch.uint8)
        #convert NCHW image to NHWC image
        if frames.shape[1] == 3:
            frames = frames.permute(0,2,3,1).contiguous()
        elif frames.shape[-1]==3:
            frames = frames.contiguous()
        else:
            raise Exception(f"Channel dimension is not 3: {frame.shape}")

        #convert to numpy on CPU
        frames = frames.numpy() 
        frames = [frames[i] for i in range(frames.shape[0])]
        return frames
    
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
    
    def process_video(self, frames: torch.Tensor) -> tuple[List, torch.Tensor]:
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
        frames = self._preprocess_frames(frames)
        if len(frames) == 0:
            return [], torch.empty(0, 2)

        H,W = frames[0].shape[0:2]
        try:
            # Process with human pose model
            human_predictions = self.human_pose_model(frames, show=False, batch_size=self.batch_size)
            # Process with animal pose model  
            animal_predictions = self.animal_pose_model(frames, show=False, batch_size=self.batch_size)

            #aggregate points from both human and animal predictions
            flattened_points = []
            per_frame_keypoints = []
            for (human_pred, animal_pred) in tqdm(zip(human_predictions, animal_predictions), total=len(frames)):

                human_keypoints = self._extract_keypoints_from_predictions(
                                    human_pred,
                                    height = H,
                                    width = W
                )
                animal_keypoints = self._extract_keypoints_from_predictions(
                                    animal_pred,
                                    height = H,
                                    width = W
                )
                combined_kp = human_keypoints + animal_keypoints
                per_frame_keypoints.append(combined_kp)
                flattened_points.append(np.concatenate(combined_kp, axis=0))
            
            #merge keypoints for all frames into 1 tensor output
            flattened_points = np.concatenate(flattened_points, axis=0) if flattened_points else np.empty((0,2), dtype=np.float32)
            return per_frame_keypoints, torch.from_numpy(flattened_points)

        except Exception as e:
            raise Exception(f"Warning: Failed to process frames: {e}")
            
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
        per_frame_keypoints, flat_keypoints = self.process_video(frames=frames)
        return per_frame_keypoints, flat_keypoints
