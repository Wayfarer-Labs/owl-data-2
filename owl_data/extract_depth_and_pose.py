from .nn.video_depth_pipeline import VideoDepthPipeline
from .nn.mmpose_keypoint_pipeline import PoseKeypointPipeline
from .nn.sam_keypoint_pipeline import SegmentationPipeline
from .nn.moge_points_intrinsics_pipeline import MoGePointsIntrinsicsPipeline
import torch
import os
import numpy as np

class DepthPosePipeline:

    def __init__(self, depth_encoder:str='vitl', segmentation_model:str = 'vit_h', ckpt:str, save_video_depth:bool=False, keypoint_threshold:float=0.3, segmentation_threshold:float=0.5):
        self.video_depth_pipeline = VideoDepthPipeline(
                                    depth_encoder = depth_encoder,
                                    save_video_depth = save_video_depth
                                )
        self.pose_keypoint_pipeline = PoseKeypointPipeline(
                                    keypoint_threshold = keypoint_threshold
                                )
        
        self.segmentation_pipeline = SegmentationPipeline(
                                    model_name = segmentation_model,
                                    ckpt = ckpt,
                                    score_threshold = segmentation_threshold
                                )
        
        self.moge_pipeline = MoGePointsIntrinsicsPipeline()

    def _aggregate_keypoints(self, seg_keypoint, agg_keypoint):
        """
        Aggregate keypoints from segmentation and pose estimation.
        
        Args:
            seg_keypoint: Keypoints from segmentation pipeline
            agg_keypoint: Keypoints from pose estimation pipeline
            
        Returns:
            List of (x, y) coordinate tuples
        """
        all_keypoints = []
        
        # Process segmentation keypoints
        if seg_keypoint is not None:
            for keypoint_set in seg_keypoint:
                if isinstance(keypoint_set, list):
                    for kp in keypoint_set:
                        if kp is not None and len(kp) >= 2:
                            # Ensure we have valid x, y coordinates
                            x, y = int(kp[0]), int(kp[1])
                            all_keypoints.append((x, y))
        
        # Process aggregated keypoints (pose estimation)
        if agg_keypoint is not None:
            if isinstance(agg_keypoint, (list, tuple)):
                for kp in agg_keypoint:
                    if kp is not None and len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        all_keypoints.append((x, y))
            elif isinstance(agg_keypoint, (torch.Tensor, np.ndarray)):
                # Handle tensor/array format
                if agg_keypoint.ndim == 2 and agg_keypoint.shape[1] >= 2:
                    for i in range(agg_keypoint.shape[0]):
                        x, y = int(agg_keypoint[i, 0]), int(agg_keypoint[i, 1])
                        all_keypoints.append((x, y))
        
        return all_keypoints

    def _create_coord_depth_map_moge(self, keypoints, coord_map, depth_map):
        """
        Create a 4-channel tensor with 3D coordinates and depth values for keypoints.
        
        Args:
            keypoints: List of (x, y) coordinate tuples
            coord_map: 3D coordinate map tensor of shape (H, W, 3)
            depth_map: Depth map tensor of shape (H, W)
            
        Returns:
            4-channel tensor where:
            - Channel 0: X coordinates of keypoints
            - Channel 1: Y coordinates of keypoints  
            - Channel 2: Z coordinates of keypoints
            - Channel 3: Depth values at keypoint locations
        """
        if not keypoints:
            # Return empty 4-channel tensor if no keypoints
            return torch.zeros((4, 0), dtype=torch.float32)
        
        # Convert inputs to tensors if needed
        if isinstance(coord_map, np.ndarray):
            coord_map = torch.from_numpy(coord_map)
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        
        # Ensure coord_map is in the right format (H, W, 3)
        if coord_map.dim() == 4 and coord_map.shape[0] == 1:
            coord_map = coord_map.squeeze(0)  # Remove batch dimension
        if coord_map.dim() == 3 and coord_map.shape[0] == 3:
            coord_map = coord_map.permute(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)
        
        # Ensure depth_map is 2D (H, W)
        if depth_map.dim() == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map.squeeze(0)  # Remove batch dimension
        if depth_map.dim() == 3:
            depth_map = depth_map.squeeze()
        
        height, width = depth_map.shape
        
        # Filter keypoints to be within image bounds
        valid_keypoints = []
        for x, y in keypoints:
            if 0 <= x < width and 0 <= y < height:
                valid_keypoints.append((x, y))
        
        if not valid_keypoints:
            return torch.zeros((4, 0), dtype=torch.float32)
        
        # Extract coordinates and depth values for valid keypoints
        num_keypoints = len(valid_keypoints)
        coord_depth_tensor = torch.zeros((4, num_keypoints), dtype=torch.float32)
        
        for i, (x, y) in enumerate(valid_keypoints):
            # Extract 3D coordinates (X, Y, Z) from coord_map
            if coord_map.shape[-1] >= 3:
                coord_depth_tensor[0, i] = coord_map[y, x, 0]  # X coordinate
                coord_depth_tensor[1, i] = coord_map[y, x, 1]  # Y coordinate
                coord_depth_tensor[2, i] = coord_map[y, x, 2]  # Z coordinate
            
            # Extract depth value from depth_map
            coord_depth_tensor[3, i] = depth_map[y, x]  # Depth value
        
        return coord_depth_tensor

    def _create_coord_depth_map(self, keypoints, depth_map, estimated_intrinsics):
        """
        Create a 4-channel tensor with 3D coordinates and depth values for keypoints.
        
        Args:
            keypoints: List of (x, y) coordinate tuples or tensor of shape (k, 2)
            depth_map: Depth map tensor of shape (height, width, 1) or (height, width)
            estimated_intrinsics: Camera intrinsics tensor of shape (3, 3)
            
        Returns:
            4-channel tensor of shape (height, width, 4) where:
            - Channel 0: X coordinates (0 everywhere except at keypoint locations)
            - Channel 1: Y coordinates (0 everywhere except at keypoint locations)
            - Channel 2: Z coordinates (0 everywhere except at keypoint locations)
            - Channel 3: Depth values (0 everywhere except at keypoint locations)
        """
        # Convert inputs to tensors if needed
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if isinstance(estimated_intrinsics, np.ndarray):
            estimated_intrinsics = torch.from_numpy(estimated_intrinsics)
        
        # Ensure depth_map is 2D (height, width)
        if depth_map.dim() == 3:
            if depth_map.shape[-1] == 1:
                depth_map = depth_map.squeeze(-1)  # Remove last dimension if it's 1
            elif depth_map.shape[0] == 1:
                depth_map = depth_map.squeeze(0)   # Remove first dimension if it's 1
        
        height, width = depth_map.shape
        
        # Initialize the 4-channel output tensor (height, width, 4)
        coord_depth_tensor = torch.zeros((height, width, 4), dtype=torch.float32)
        
        # Extract camera intrinsic parameters
        fx = estimated_intrinsics[0, 0]  # Focal length in x
        fy = estimated_intrinsics[1, 1]  # Focal length in y
        cx = estimated_intrinsics[0, 2]  # Principal point x
        cy = estimated_intrinsics[1, 2]  # Principal point y
        
        # Convert keypoints to tensor if it's a list
        if isinstance(keypoints, list):
            if not keypoints:
                return coord_depth_tensor
            keypoints = torch.tensor(keypoints, dtype=torch.float32)
        elif isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).float()
        
        # Ensure keypoints is 2D tensor of shape (k, 2)
        if keypoints.dim() == 1 and len(keypoints) == 2:
            keypoints = keypoints.unsqueeze(0)  # Single keypoint: (2,) -> (1, 2)
        
        if keypoints.numel() == 0:
            return coord_depth_tensor
        
        
        # Convert to integer pixel coordinates
        pixel_x = valid_keypoints[:, 0].long()
        pixel_y = valid_keypoints[:, 1].long()
        
        # Extract depth values at keypoint locations
        depth = depth_map[pixel_y, pixel_x]
        
        # Convert 2D pixel coordinates to 3D world coordinates using camera intrinsics
        # Formula: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth
        world_x = (pixel_x.float() - cx) * depth / fx
        world_y = (pixel_y.float() - cy) * depth / fy
        world_z = depth
        
        # Fill the coordinate depth map at keypoint locations
        coord_depth_tensor[pixel_y, pixel_x, 0] = world_x  # X coordinates
        coord_depth_tensor[pixel_y, pixel_x, 1] = world_y  # Y coordinates
        coord_depth_tensor[pixel_y, pixel_x, 2] = world_z  # Z coordinates
        coord_depth_tensor[pixel_y, pixel_x, 3] = depth   # Depth values
        
        return coord_depth_tensor

    def run_pipeline_moge(self, frame_filename:str, video_dir: str, video_name:str, input_size: int, video_fps: int):
        frames = torch.load(frame_filename)
        frame_idx = frame_filename.split('_')[0]

        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, aggregated_keypoints = self.pose_keypoint_pipeline(frames=frames)
        #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
        masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames)

        #Step 3: run MoGe to get depth maps and coordinates directly from the images
        moge_output = self.moge_pipeline(
                                        frames=frames,
                                        video_dir=video_dir,
                                        video_name=video_name,
                                        video_fps = video_fps
                    )
        
        full_depth_path = os.path.join(video_dir, 'full_depth_splits')
        full_seg_path = os.path.join(video_dir, 'full_seg_splits')
        keypoint_depth_path = os.path.join(video_dir, 'keypoint_depth_splits')
        keypoint_coord_path = os.path.join(video_dir, 'keypoint_coord_splits')

        # Create directories if they don't exist
        os.makedirs(full_depth_path, exist_ok=True)
        os.makedirs(full_seg_path, exist_ok=True)
        os.makedirs(keypoint_depth_path, exist_ok=True)
        os.makedirs(keypoint_coord_path, exist_ok=True)

        #Step 4: save the depth map for each frame in frames as 1D pt file
        torch.save(torch.stack(moge_output['depth']), os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt'))
        #Step 7: save consistently colored segmentation for each frame in frames as RGB pt file
        torch.save(torch.stack(masks), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))
        
        all_keypoint_depths = []
        all_coord_depth_maps = []
        for i, (depth_map, coord_map, seg_keypoint, agg_keypoint) in enumerate(zip(
                                                                            moge_output['depth'], 
                                                                            moge_output['points'], 
                                                                            segmentation_keypoints, 
                                                                            aggregated_keypoints
                                                                            )):
            #Step 2.5: merge keypoints from segmentations and human/animals
            all_keypoints = self._aggregate_keypoints(seg_keypoint, agg_keypoint)
            
            #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
            #Step 6: save the 3D coordinate for KEYPOINTS in each frame as RGB pt file
            coord_depth_map = self._create_coord_depth_map_moge(all_keypoints, coord_map, depth_map)
            
            keypoint_depths = coord_depth_map[3, :] if coord_depth_map.shape[1] > 0 else torch.tensor([])
            all_keypoint_depths.append(keypoint_depths)
            all_coord_depth_maps.append(coord_depth_map)

        # Save keypoint depth values (4th channel of coord_depth_map)
        torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
        # Save the full 4-channel coord_depth_map
        torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))

    def run_pipeline(self, frame_filename:str, video_dir: str, video_name:str, input_size: int, video_fps: int):
        frames = torch.load(frame_filename)
        frame_idx = frame_filename.split('_')[0]
       
        #Step 0: extract the video FOV using the first frame with MoGe
        estimated_intrinsicts = self.moge_pipeline(
                                        frames=frames[0],
                                        video_dir=video_dir,
                                        video_name=video_name,
                                        video_fps = video_fps
                                )['intrinsics']
        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, aggregated_keypoints = self.pose_keypoint_pipeline(frames=frames)
        #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
        masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames)
        #Step 3: run VideoDepthAnything to get consistent depth maps
        depth_tensors, ___ = self.video_depth_pipeline(
                                frames=frames,
                                video_dir=video_dir,
                                video_name=video_name,
                                input_size=input_size,
                                video_fps = video_fps
                            )

        full_depth_path = os.path.join(video_dir, 'full_depth_splits')
        full_seg_path = os.path.join(video_dir, 'full_seg_splits')
        keypoint_depth_path = os.path.join(video_dir, 'keypoint_depth_splits')
        keypoint_coord_path = os.path.join(video_dir, 'keypoint_coord_splits')

        # Create directories if they don't exist
        os.makedirs(full_depth_path, exist_ok=True)
        os.makedirs(full_seg_path, exist_ok=True)
        os.makedirs(keypoint_depth_path, exist_ok=True)
        os.makedirs(keypoint_coord_path, exist_ok=True)


        #Step 4: save the depth map for each frame in frames as 1D pt file
        torch.save(depth_tensors, os.path.join(video_dir, os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt')))
        #Step 7: save consistently colored segmentation for each frame in frames as RGB pt file
        torch.save(torch.stack(masks), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))

        all_keypoint_depths = []
        all_coord_depth_maps = []
        #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
        #Step 6: COMPUTE and save the 3D coordinate for KEYPOINTS in each frame as RGB pt file => using FOV + (cx,cy) formula
        for i, (depth_map, seg_keypoint, agg_keypoint) in enumerate(zip(depth_tensors
                                                                            segmentation_keypoints, 
                                                                            aggregated_keypoints
                                                                            )):
            #Step 2.5: merge keypoints from segmentations and human/animals
            all_keypoints = self._aggregate_keypoints(seg_keypoint, agg_keypoint)
            
            #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
            #Step 6: save the 3D coordinate for KEYPOINTS in each frame as RGB pt file
            coord_depth_map = self._create_coord_depth_map(all_keypoints, depth_map, estimated_intrinsics)
            keypoint_depths = coord_depth_map[3, :] if coord_depth_map.shape[1] > 0 else torch.tensor([])
            all_keypoint_depths.append(keypoint_depths)
            all_coord_depth_maps.append(coord_depth_map)
        
        # Save keypoint depth values (4th channel of coord_depth_map)
        torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
        # Save the full 4-channel coord_depth_map
        torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))
        