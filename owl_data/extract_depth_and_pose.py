import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__),'../..')))
from typing import Optional, Tuple
from owl_data.nn.video_depth_pipeline import VideoDepthPipeline
from owl_data.nn.moge_points_intrinsics_pipeline import MoGePointsIntrinsicsPipeline
from owl_data.nn.mmpose_keypoint_pipeline import PoseKeypointPipeline
from owl_data.nn.multi_gpu_sam_pipeline import MultiGPUSegmentationWrapper
from owl_data.nn.sam_keypoint_pipeline import SegmentationPipeline
import torch
import cv2
import numpy as np 
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt



class DepthPosePipeline:

    def __init__(self, 
                depth_encoder:str='vitl', 
                segmentation_model:str = 'vit_h', 
                save_video_depth:bool=False, 
                keypoint_threshold:float=0.3, 
                segmentation_threshold:float=0.5,
                flow_static_threshold:float=0.7,
                photo_static_threshold:float=0.08,
                grad_min_thresh:float=0.05,
                huber_delta:float=0.05,
                iters:int=6,
                ema_alpha:float=0.8,
                radius:int=15,
                eps_frac:float=1e-3,
                var_window:int=6,
                tau:float=0.03,
                edge_thresh:float=0.07,
                beta:float=0.7
                ):
        self.video_depth_pipeline = VideoDepthPipeline(
                                    depth_encoder = depth_encoder,
                                    save_video_depth = save_video_depth
                                )
        self.pose_keypoint_pipeline = PoseKeypointPipeline(
                                    keypoint_threshold = keypoint_threshold
                                )
        
        self.segmentation_pipeline = SegmentationPipeline(
                                    model_name = segmentation_model,
                                    score_threshold = segmentation_threshold
                                )
        
        self.moge_pipeline = MoGePointsIntrinsicsPipeline(save_video_depth=save_video_depth)


    def normalize_coord_map(self, coords_tensor: torch.Tensor):
        # Suppose coords is your (height, width, 3) coordinates tensor
        coords = coords_tensor.detach().cpu().numpy().astype(np.float32)

        # Normalize each channel (x,y,z) → (r,g,b) into [0,1]
        coords_min = coords.min(axis=(0,1), keepdims=True)
        coords_max = coords.max(axis=(0,1), keepdims=True)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        return coords_norm

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
            # Handle torch tensor with shape (num_instances, num_keypoints, 2)
            if isinstance(seg_keypoint, torch.Tensor) and seg_keypoint.dim() == 3 and seg_keypoint.shape[-1] == 2:
                flattened_keypoints = seg_keypoint.view(-1, 2)
            elif isinstance(seg_keypoint, np.ndarray) and seg_keypoint.ndim == 3 and seg_keypoint.shape[-1] == 2:
                flattened_keypoints = seg_keypoint.reshape(-1, 2)
            else:
                raise ValueError("seg_keypoint must be (num_instances, num_keypoints, 2) tensor/array")

            # Convert to list of (x, y) tuples
            all_keypoints.extend([(int(x), int(y)) for x, y in flattened_keypoints])
        
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
            return torch.zeros((depth_map.shape[0],depth_map.shape[1],4), dtype=torch.float32)
        
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
            return torch.zeros((0, 4), dtype=torch.float32)
        
        # Extract coordinates and depth values for valid keypoints
        num_keypoints = len(valid_keypoints)
        coord_depth_tensor = torch.zeros((height, width, 4), dtype=torch.float32)
        
        for i, (x, y) in enumerate(valid_keypoints):
            # Extract 3D coordinates (X, Y, Z) from coord_map
            coord_depth_tensor[y, x, 0] = coord_map[y, x, 0]  # X coordinate
            coord_depth_tensor[y, x, 1] = coord_map[y, x, 1]  # Y coordinate
            coord_depth_tensor[y, x, 2] = coord_map[y, x, 2]  # Z coordinate
            
            # Extract depth value from depth_map
            coord_depth_tensor[y, x, 3] = depth_map[y, x]  # Depth value
        
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
        fx = estimated_intrinsics[0, 0]*width  # Focal length in x
        fy = estimated_intrinsics[1, 1]*height  # Focal length in y
        cx = estimated_intrinsics[0, 2]*width  # Principal point x
        cy = estimated_intrinsics[1, 2]*height  # Principal point y
        
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
        pixel_x = keypoints[:, 0].long()
        pixel_y = keypoints[:, 1].long()
        
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

    def _create_full_coord_depth_map(
        self,
        depth_map,
        estimated_intrinsics
    ):
        """
        Create a (H, W, 3) tensor where channels are [X, Y, Z] for *all* pixels.

        Args:
            depth_map:
                - torch.Tensor or np.ndarray shaped (H, W) or (H, W, 1).
                - Values are depths Z (e.g., meters). Z<=0 or non-finite are masked to 0.
            estimated_intrinsics:
                - 3x3 intrinsics matrix (torch or numpy).
                - If `assume_normalized_intrinsics=True`, fx,fy,cx,cy are assumed
                normalized to width/height and are scaled by (W,H) respectively.
                If False, they are used as-is (pixel units).
            assume_normalized_intrinsics (bool): see above.

        Returns:
            torch.FloatTensor of shape (H, W, 3): [X, Y, Z].
            - X = (u - cx) * Z / fx
            - Y = (v - cy) * Z / fy
            - Z = depth
        """
        # ---- Convert inputs to tensors ----
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if isinstance(estimated_intrinsics, np.ndarray):
            estimated_intrinsics = torch.from_numpy(estimated_intrinsics)

        # Ensure depth is (H, W)
        if depth_map.dim() == 3:
            if depth_map.shape[-1] == 1:
                depth_map = depth_map.squeeze(-1)
            elif depth_map.shape[0] == 1:
                depth_map = depth_map.squeeze(0)
        assert depth_map.dim() == 2, f"depth_map must be (H, W) or (H, W, 1); got {tuple(depth_map.shape)}"

        H, W = depth_map.shape
        device = depth_map.device
        dtype = depth_map.dtype if depth_map.dtype.is_floating_point else torch.float32

        K = estimated_intrinsics.to(device=device, dtype=torch.float32)

        # ---- Extract intrinsics (handle normalized vs absolute) ----
        fx = K[0, 0] * W
        fy = K[1, 1] * H
        cx = K[0, 2] * W
        cy = K[1, 2] * H

        # Avoid division by zero if fx or fy are zero
        eps = 1e-8
        fx = torch.as_tensor(fx, device=device, dtype=torch.float32).clamp_min(eps)
        fy = torch.as_tensor(fy, device=device, dtype=torch.float32).clamp_min(eps)
        cx = torch.as_tensor(cx, device=device, dtype=torch.float32)
        cy = torch.as_tensor(cy, device=device, dtype=torch.float32)

        # ---- Build pixel coordinate grids (u = x, v = y) ----
        # Create coordinate grids for each pixel
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")  # vv:(H,W), uu:(H,W)

        # ---- Depth and mask ----
        Z = depth_map.to(device=device, dtype=torch.float32)
        valid = torch.isfinite(Z) & (Z > 0)

        # ---- Back-project to 3D ----
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        # Zero-out invalids
        X = torch.where(valid, X, torch.zeros((), device=device, dtype=X.dtype))
        Y = torch.where(valid, Y, torch.zeros((), device=device, dtype=Y.dtype))
        Z = torch.where(valid, Z, torch.zeros((), device=device, dtype=Z.dtype))

        # ---- Stack [X, Y, Z] ----
        full_coord_map = torch.stack([X, Y, Z], dim=-1).to(dtype=torch.float32)  # (H, W, 3)
        return full_coord_map

    def run_pipeline_moge(self, frame_filename:str, video_dir: str, video_name:str, video_fps:int):
        frames = torch.load(frame_filename)[:20]
        frame_idx = frame_filename.split('/')[-1].split('_')[0]

        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, aggregated_keypoints = self.pose_keypoint_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
        
        #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
        masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after segmentation
        
        #Step 3: run MoGe to get depth maps and coordinates directly from the images
        moge_output = self.moge_pipeline(
                                        frames=frames,
                                        video_dir=video_dir,
                                        video_name=video_name,
                                        video_fps=video_fps
                                        )
        torch.cuda.empty_cache()  # Clear GPU memory after MoGe processing
        
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
        torch.save(torch.stack(moge_output['depth']).unsqueeze(-1), os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt'))
        #Step 7: save consistently colored segmentation for each frame in frames as RGB pt file
        torch.save(torch.Tensor(masks), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))
        
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
            keypoint_coord_map = self._create_coord_depth_map_moge(all_keypoints, coord_map, depth_map)
            keypoint_depths = keypoint_coord_map[:, :, 3] if keypoint_coord_map.shape[0] > 0 else torch.tensor([])
            all_keypoint_depths.append(keypoint_depths.cpu())
            all_coord_depth_maps.append(keypoint_coord_map[:, :, :3].cpu())

        # Save keypoint depth values (4th channel of coord_depth_map)
        torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
        # Save the full 4-channel coord_depth_map
        torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))

    def run_pipeline(self, frame_filename:str, video_dir: str, video_name:str, video_fps: int):
        frames = torch.load(frame_filename)[:20]
        frame_idx = frame_filename.split('/')[-1].split('_')[0]

        #Step 0: extract the video FOV using the first frame with MoGe
        self.moge_pipeline.save_video = False
        estimated_intrinsics = self.moge_pipeline(
                                        frames=frames[0].unsqueeze(0),
                                        video_dir=video_dir,
                                        video_name=video_name,
                                        video_fps=video_fps
                                        )['intrinsics'][0].cpu()
        
        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, pose_keypoints = self.pose_keypoint_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
        
        #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
        masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after segmentation
        
        #Step 3: run VideoDepthAnything to get consistent depth maps
        depth_tensors, ___ = self.video_depth_pipeline(
                                frames=frames,
                                video_dir=video_dir,
                                video_name=video_name,
                                target_width = frames[0].shape[2],
                                target_height = frames[0].shape[1],
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
        torch.save(torch.Tensor(masks), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))

        all_keypoint_depths = []
        all_coord_depth_maps = []
        #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
        #Step 6: COMPUTE and save the 3D coordinate for KEYPOINTS in each frame as RGB pt file => using FOV + (cx,cy) formula
        for i, (depth_map, seg_keypoint, pos_keypoint) in enumerate(zip(depth_tensors,
                                                                        segmentation_keypoints, 
                                                                        pose_keypoints
                                                                    )):
            #Step 2.5: merge keypoints from segmentations and human/animals
            all_keypoints = self._aggregate_keypoints(seg_keypoint, pos_keypoint)
            
            #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
            #Step 6: save the 3D coordinate for KEYPOINTS in each frame as RGB pt file
            keypoint_coord_map = self._create_coord_depth_map(all_keypoints, depth_map, estimated_intrinsics)
            
            full_coord_map = self._create_full_coord_depth_map(depth_map, estimated_intrinsics)
            keypoint_depths = keypoint_coord_map[:,:,3] if keypoint_coord_map.shape[1] > 0 else torch.tensor([])
            
            # Move to CPU to save GPU memory
            all_keypoint_depths.append(keypoint_depths.cpu())
            all_coord_depth_maps.append(keypoint_coord_map[:,:,:3].cpu())
            
            # Clear GPU memory after each iteration
            torch.cuda.empty_cache()
        
        # Save keypoint depth values (4th channel of coord_depth_map)
        torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
        # Save the full 4-channel coord_depth_map
        torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))

    def run_pipeline_hybrid(self, frame_filename:str, video_dir:str, video_name:str, video_fps:int):
        frames = torch.load(frame_filename)[:20]
        frame_idx = frame_filename.split('/')[-1].split('_')[0]

        #Step 0: extract the video FOV using the first frame with MoGe
        self.moge_pipeline.save_video = False
        moge_output = self.moge_pipeline(
                                        frames=frames,
                                        video_dir=video_dir,
                                        video_name=video_name,
                                        video_fps=video_fps
                                        )
        estimated_intrinsics = moge_output['intrinsics'][0].cpu()
        
        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, pose_keypoints = self.pose_keypoint_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
        
        #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
        masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames)
        torch.cuda.empty_cache()  # Clear GPU memory after segmentation
        
        #Step 3: run VideoDepthAnything to get consistent depth maps
        depth_tensors, ___ = self.video_depth_pipeline(
                                frames=frames,
                                video_dir=video_dir,
                                video_name=video_name,
                                target_width = frames[0].shape[2],
                                target_height = frames[0].shape[1],
                                video_fps = video_fps
                            )

        #Step 3.5: merge depth map from VDA and MoGE
        pdb.set_trace()
        flows_fwd, flows_bwd = self.build_flows_opencv(frames)
        merged_depth_maps = self.merge_depth_maps(depthmap_vda=depthmap_vda, 
                                        depthmap_moge=depthmap_moge,
                                        frames=frames,
                                        flows_fwd=flows_fwd,
                                        flows_bwd=flows_bwd)

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
        torch.save(merged_depth_maps, os.path.join(video_dir, os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt')))
        #Step 7: save consistently colored segmentation for each frame in frames as RGB pt file
        torch.save(torch.Tensor(masks), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))

        all_keypoint_depths = []
        all_coord_depth_maps = []
        #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
        #Step 6: COMPUTE and save the 3D coordinate for KEYPOINTS in each frame as RGB pt file => using FOV + (cx,cy) formula
        for i, (depthmap, coord_map, seg_keypoint, pos_keypoint) in enumerate(zip(merged_depth_maps,
                                                                        moge_output['points'],
                                                                        segmentation_keypoints, 
                                                                        pose_keypoints
                                                                    )):
            #Step 2.5: merge keypoints from segmentations and human/animals
            all_keypoints = self._aggregate_keypoints(seg_keypoint, pos_keypoint)

            
            #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
            #Step 6: save the 3D coordinate for KEYPOINTS in each frame as RGB pt file
            keypoint_coord_map_estim = self._create_coord_depth_map(all_keypoints, depth_map, estimated_intrinsics)
            keypoint_coord_map_moge = self._create_coord_depth_map_moge(all_keypoints, coord_map, depth_map)

            keypoint_coord_map_merged = None

            full_coord_map = self._create_full_coord_depth_map(depth_map, estimated_intrinsics)
            keypoint_depths = keypoint_coord_map_merged[:,:,3] if keypoint_coord_map_merged.shape[1] > 0 else torch.tensor([])
            
            # Move to CPU to save GPU memory
            all_keypoint_depths.append(keypoint_depths.cpu())
            all_coord_depth_maps.append(keypoint_coord_map_merged[:,:,:3].cpu())
            
            # Clear GPU memory after each iteration
            torch.cuda.empty_cache()
        
        # Save keypoint depth values (4th channel of coord_depth_map)
        torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
        # Save the full 4-channel coord_depth_map
        torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))

    def merge_depth_maps(self, 
        depthmap_vda: torch.Tensor,          # (T,H,W) relative VDA
        depthmap_moge: torch.Tensor,       # (T,H,W) metric MoGE
        frames: Optional[torch.Tensor] = None,     # (T,H,W,3) RGB in [0..1] or [0..255]
        flows_fwd: Optional[torch.Tensor] = None,  # (T-1,H,W,2) t-1->t
        flows_bwd: Optional[torch.Tensor] = None,  # (T-1,H,W,2) t->t-1
        flow_static_thresh: float = None,           # px
        photo_static_thresh: float = None,         # 0..1
        grad_min_thresh: float = None,
        huber_delta:float=None,
        iters:int=None,
        ema_alpha:float=None,
        radius:int=None,
        eps_frac:float=None,
        var_window:int=None,
        tau:float=None,
        edge_thresh:float=None,
        beta:float=None):

        T,H,W = depthmap_vda.shape
        assert depthmap_moge.shape == (T,H,W)

        flow_static_thresh = self.flow_static_thresh if not flow_static_thresh else flow_static_thresh
        photo_static_thresh = self.photo_static_thresh if not photo_static_thresh else photo_static_thresh
        grad_min_thresh = self.grad_min_thresh if not grad_min_thresh else grad_min_thresh
        huber_delta = self.huber_delta if not huber_delta else huber_delta
        iters = self.iters if not iters else iters
        ema_alpha = self.ema_alpha if not ema_alpha else ema_alpha

        # 1) VALID MASK (static & reliable) for s,b fit
        valid_mask = None
        if frames is not None and flows_fwd is not None:
            gray = _to_gray(frames)  # (T,H,W)
            vm = torch.ones((T,H,W), device=depthmap_vda.device, dtype=torch.bool)
            for t in range(1, T):
                # flow magnitude gate
                mag = torch.linalg.norm(flows_fwd[t-1], dim=-1)  # (H,W)
                m_static = (mag <= flow_static_thresh)

                # photometric gate (if backward flow available)
                if flows_bwd is not None:
                    prev_warped_to_cur = _warp_prev_to_cur(gray[t-1], flows_bwd[t-1])  # (H,W)
                    diff = (gray[t] - prev_warped_to_cur).abs()
                    # normalize diff to [0,1] per-frame
                    mx = max(diff.max().item(), self.epsilon)
                    diff = diff / mx
                    m_photo = (diff <= photo_static_thresh)
                else:
                    # fallback: no motion compensation, just allow photometric check to be disabled
                    m_photo = torch.ones_like(m_static, dtype=torch.bool)

                # texture gate (avoid super-flat regions & reflections)
                grad_mag = self._sobel_mag(gray[t]).detach()  # (H,W) in [0,1]
                m_texture = (grad_mag >= grad_min_thresh)

                vm[t] = m_static & m_photo & m_texture

            valid_mask = vm

        # 2) GUIDANCE for guided low-pass
        guidance = frames if frames is not None else None

        # --- run the 3 steps ---
        vda_metric, s, b = self.align_scale_shift(
            depthmap_vda, depthmap_moge, valid_mask=valid_mask,
            huber_delta=huber_delta, iters=iters, ema_alpha=ema_alpha, clip=(0.1, 100.0)
        )

        fused = self.per_frame_fuse(
            vda_metric, depthmap_moge,
            guidance=guidance, radius=radius, eps_frac=eps_frac,
            var_window=var_window, tau=tau, edge_thresh=edge_thresh, clip=(0.1, 100.0)
        )

        stable = self.temporal_stabilize(
            fused, flows_fwd=flows_fwd, flows_bwd=flows_bwd, beta=beta
        )

        return stable, fused, vda_metric, (s, b)

    def run_video(self, frame_path:str, video_dir:str, video_name:str, video_fps:int):
        for frame_file in tqdm(os.listdir(frame_path)):
            self.run_pipeline(
                frame_filename=os.path.join(frame_path, frame_file),
                video_dir= video_dir,
                video_name = video_name,
                video_fps = video_fps
            )
    
    def run_video_moge(self, frame_path:str, video_dir:str, video_name:str, video_fps:int):
        for frame_file in tqdm(os.listdir(frame_path)):
            self.run_pipeline_moge(
                frame_filename=os.path.join(frame_path, frame_file),
                video_dir= video_dir,
                video_name = video_name,
                video_fps = video_fps
            )

    def run_video_hybrid(self, frame_path:str, video_dir:str, video_name:str, video_fps:int):
        for frame_file in tqdm(os.listdir(frame_path)):
            self.run_pipeline_hybrid(
                frame_filename=os.path.join(frame_path, frame_file),
                video_dir= video_dir,
                video_name = video_name,
                video_fps = video_fps
            )
    
    def align_scale_shift(
        self,
        vda_depth: torch.Tensor,         # (T,H,W) in [0,1] relative
        moge_depth: torch.Tensor,      # (T,H,W) metric (m)
        valid_mask: Optional[torch.Tensor] = None,  # (T,H,W) bool; static/reliable pixels (optional)
        huber_delta: float = 0.05,
        iters: int = 6,
        ema_alpha: float = 0.8,
        clip: Tuple[float,float] = (0.1, 100.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust IRLS per-frame fit of s,b in  Dv*s + b ≈ Dm. Then EWMA-smooth (s,b) over time.
        Returns:
        vda_metric: (T,H,W)
        s_all: (T,)
        b_all: (T,)
        """
        assert vda_depth.shape == moge_depth.shape
        T, H, W = vda_depth.shape
        device, dtype = vda_depth.device, vda_depth.dtype

        s_list = []
        b_list = []
        s_prev = None
        b_prev = None

        for t in range(T):
            x = vda_depth[t].reshape(-1)
            y = moge_depth[t].reshape(-1)
            if valid_mask is not None:
                m = valid_mask[t].reshape(-1).bool()
            else:
                m = torch.isfinite(x) & torch.isfinite(y)

            # IRLS (Huber)
            s = torch.tensor(1.0, device=device, dtype=dtype)
            b = torch.tensor(0.0, device=device, dtype=dtype)

            xm = x[m]; ym = y[m]
            if xm.numel() >= 3:
                ones = torch.ones_like(xm)
                for _ in range(iters):
                    r = s * xm + b - ym
                    abs_r = r.abs()
                    w = torch.ones_like(r)
                    mask = abs_r > huber_delta
                    w[mask] = (huber_delta / abs_r[mask])

                    wx = w * xm
                    A11 = (wx * xm).sum()
                    A12 = wx.sum()
                    A22 = w.sum()
                    rhs1 = (wx * ym).sum()
                    rhs2 = (w * ym).sum()

                    det = A11 * A22 - A12 * A12 + 1e-12
                    s_new = ( A22 * rhs1 - A12 * rhs2) / det
                    b_new = (-A12 * rhs1 + A11 * rhs2) / det
                    if torch.isfinite(s_new) and torch.isfinite(b_new):
                        s, b = s_new, b_new

            # temporal EWMA smoothing
            if s_prev is None:
                s_smooth, b_smooth = s, b
            else:
                s_smooth = ema_alpha * s_prev + (1 - ema_alpha) * s
                b_smooth = ema_alpha * b_prev + (1 - ema_alpha) * b
            s_prev, b_prev = s_smooth, b_smooth
            s_list.append(s_smooth)
            b_list.append(b_smooth)

        s_all = torch.stack(s_list)  # (T,)
        b_all = torch.stack(b_list)  # (T,)

        vda_metric = (s_all.view(T,1,1) * vda_depth + b_all.view(T,1,1))
        vda_metric = self._safe_clip(vda_metric, *clip)
        return vda_metric, s_all, b_all

    def per_frame_fuse(
        self,
        vda_metric: torch.Tensor,      # (T,H,W) after step1
        moge_depth: torch.Tensor,      # (T,H,W)
        radius: int = 15,
        eps_frac: float = 1e-3,
        var_window: int = 5,
        tau: float = 0.025,            # m^2 for variance->weight
        edge_thresh: float = 0.08,
        clip: Tuple[float,float] = (0.1, 100.0),
    ) -> torch.Tensor:
        """
        Edge-aware base from VDA + high-frequency residual from MoGE, weighted by
        non-learned temporal stability (exp(-var/tau)). No temporal EMA here.
        Returns fused per-frame depth (T,H,W).
        """
        assert vda_metric.shape == moge_depth.shape
        T, H, W = vda_metric.shape
        vda_metric = self._safe_clip(vda_metric, *clip)
        moge_depth = self._safe_clip(moge_depth, *clip)

        # Guidance
        g = (vda_metric - vda_metric.view(T,-1).mean(dim=1, keepdim=True).view(T,1,1))
        g = g / (g.view(T,-1).std(dim=1, keepdim=True).view(T,1,1) + 1e-6)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
       

        # Guided bases
        eps_v = _auto_eps_per_frame(vda_metric, eps_frac)  # (T,)
        eps_m = _auto_eps_per_frame(moge_depth, eps_frac)  # (T,)
        Bv = self._guided_filter_gray(vda_metric, g, radius, eps_v)  # (T,H,W)
        Bm = self._guided_filter_gray(moge_depth, g, radius, eps_m)

        # High-frequency detail from MoGE
        Hm = moge_depth - Bm  # (T,H,W)

        # Temporal variance of MoGE (short window) -> weight W in [0,1]
        var = torch.zeros_like(moge_depth)
        half = var_window // 2
        for t in range(T):
            a = max(0, t - half)
            b = min(T, t + half + 1)
            seg = moge_depth[a:b]  # (len,H,W)
            var[t] = seg.var(dim=0, unbiased=False)
        W = torch.exp(-var / max(tau, 1e-8))


        fused = self._safe_clip(Bv + W * Hm, *clip)
        return fused

    def _sobel_mag(self, g: torch.Tensor) -> torch.Tensor:
        # g: (T,H,W) gray
        T,H,W = g.shape
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=g.device, dtype=g.dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=g.device, dtype=g.dtype).view(1,1,3,3)
        gx = F.conv2d(g.view(T,1,H,W), kx, padding=1).view(T,H,W)
        gy = F.conv2d(g.view(T,1,H,W), ky, padding=1).view(T,H,W)
        mag = torch.sqrt(gx**2 + gy**2)
        # normalize per-frame
        d = torch.clamp(mag.view(T,-1).max(dim=1).values, min=1e-8).view(T,1,1)
        return mag / d
   
    def temporally_stabilize(
        self,
        fused: torch.Tensor,                 # (T,H,W) from step2
        flows_fwd: Optional[torch.Tensor] = None,  # (T-1,H,W,2) forward flow (t-1->t)  [optional]
        flows_bwd: Optional[torch.Tensor] = None,  # (T-1,H,W,2) backward flow (t->t-1) [optional]
        beta: float = 0.75
    ) -> torch.Tensor:
        """
        Motion-aware EMA across time.
        If backward flow is provided, uses backward sampling for warping.
        If only forward flow is provided, does a plain EMA (no warp).
        Returns stabilized depth (T,H,W).
        """
        T, H, W = fused.shape
        out = torch.empty_like(fused)
        out[0] = fused[0]

        for t in range(1, T):
            if flows_bwd is not None:
                D_prev_warped = self._warp_prev_to_cur(out[t-1], flows_bwd[t-1])  # (H,W)
                # optional occlusion via fwd-bwd consistency if both flows provided
                if flows_fwd is not None:
                    # Warp forward flow into current frame with backward; compare sum ≈ 0
                    fwd_warp = self._warp_prev_to_cur(flows_fwd[t-1].permute(2,0,1), flows_bwd[t-1])
                    fwd_warp = fwd_warp.permute(1,2,0)  # (H,W,2)
                    err = torch.linalg.norm(fwd_warp + flows_bwd[t-1], dim=-1)  # (H,W)
                    non_occ = (err < 1.5).float()
                else:
                    non_occ = torch.ones((H,W), device=fused.device, dtype=fused.dtype)

                out[t] = non_occ * (beta * D_prev_warped + (1 - beta) * fused[t]) + (1 - non_occ) * fused[t]
            else:
                # No warping available -> simple EMA
                out[t] = beta * out[t-1] + (1 - beta) * fused[t]

        return out
    
    def _guided_filter_gray(self, p: torch.Tensor, I: torch.Tensor, r: int, eps: torch.Tensor) -> torch.Tensor:
        """
        Classic guided filter (He et al.) per-frame with gray guidance.
        p: (T,H,W) to smooth; I: (T,H,W) guidance in [0,1];
        eps: (T,) per-frame regularizer.
        Returns (T,H,W).
        """
        T, H, W = p.shape
        mean_I = _box_filter_2d(I, r)
        mean_p = _box_filter_2d(p, r)
        corr_I = _box_filter_2d(I * I, r)
        corr_Ip = _box_filter_2d(I * p, r)

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        # broadcast eps(T,) -> (T,H,W)
        a = cov_Ip / (var_I + eps.view(T,1,1))
        b = mean_p - a * mean_I

        mean_a = _box_filter_2d(a, r)
        mean_b = _box_filter_2d(b, r)
        q = mean_a * I + mean_b
        return q
    
    def _to_cv_gray(self, img_hwc_float):
        im = img_hwc_float.detach().cpu().numpy()
        if im.max() <= 1.0: 
            im = (im * 255.0).astype(np.uint8)
        else:               
            im = im.astype(np.uint8)
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    def cv_flow_tvl1(self, img1_hwc, img2_hwc):
        g1 = self._to_cv_gray(img1_hwc)
        g2 = self._to_cv_gray(img2_hwc)
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(g1, g2, None)  # HxWx2, float32, (dx,dy) in pixels
        return torch.from_numpy(flow).to(img1_hwc.device)

    def build_flows_opencv(self, frames):  # frames: (T,H,W,3) float [0..1] or [0..255]
        print('STARTING OPT FLOW')
        pdb.set_trace()
        if frames.shape[1]==3: #reshape (B,C,H,W) to (B,H,W,C)
            frames = frames.permute(0,2,3,1)
        T,H,W,_ = frames.shape
        fwd, bwd = [], []
        for t in range(1, T):
            f_fwd = self.cv_flow_tvl1(frames[t-1],frames[t])
            f_bwd = self.cv_flow_tvl1(frames[t],frames[t-1])
            fwd.append(f_fwd); bwd.append(f_bwd)
        return torch.stack(fwd,0), torch.stack(bwd,0)

    def _auto_eps_per_frame(self, depth: torch.Tensor, eps_frac: float) -> torch.Tensor:
        """
        eps per frame = eps_frac * (range_2-98%)^2
        depth: (T,H,W) -> (T,)
        """
        T, H, W = depth.shape
        q00 = torch.nanquantile(depth.view(T, -1), 0.00, dim=1)
        q100 = torch.nanquantile(depth.view(T, -1), 1.00, dim=1)
        rng = torch.clamp(q100 - q00, min=self.epsilon)
        return (eps_frac * (rng ** 2)).to(depth.dtype).to(depth.device)  # (T,)
    
    def _safe_clip(self, D: torch.Tensor, dmin: float, dmax: float) -> torch.Tensor:
        D = torch.nan_to_num(D, nan=0.0, posinf=dmax, neginf=dmin)
        return D.clamp(dmin, dmax)
    
    def _warp_prev_to_cur(self, D_prev: torch.Tensor, flow_t_to_prev: torch.Tensor) -> torch.Tensor:
        """
        D_prev: (H,W), flow_t_to_prev: (H,W,2) backward flow F_{t->t-1}.
        returns warped (H,W).
        """
        H, W = D_prev.shape
        grid = _grid_from_backward_flow(flow_t_to_prev)
        warped = F.grid_sample(D_prev.view(1,1,H,W), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return warped.view(H, W)

if __name__ == '__main__':
    pdb.set_trace()
    depth_pose_pipeline = DepthPosePipeline(
        depth_encoder='vitl', 
        segmentation_model='vit_l',
        segmentation_threshold = 0.0,
        save_video_depth = True,
        keypoint_threshold = 0.0
    )

    BASE_PATH = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../temp_files/splits'))
    VIDEO_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../temp_files'))
    VIDEO_NAME = 'to_send.mp4'
    VIDEO_FPS = 50 #TODO: set fps for depth video

    # depth_pose_pipeline.run_video_moge(
    #     frame_path = BASE_PATH,
    #     video_dir = VIDEO_DIR,
    #     video_name = VIDEO_NAME,
    #     video_fps = VIDEO_FPS
    # )

    # depth_pose_pipeline.run_video(
    #     frame_path = BASE_PATH,
    #     video_dir = VIDEO_DIR,
    #     video_name = VIDEO_NAME,
    #     video_fps = VIDEO_FPS
    # )

    depth_pose_pipeline.run_video_hybrid(
        frame_path = BASE_PATH,
        video_dir = VIDEO_DIR,
        video_name = VIDEO_NAME,
        video_fps = VIDEO_FPS
    )