import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__),'../..')))
from typing import Optional, Tuple
from owl_data.nn.video_depth_pipeline import VideoDepthPipeline
from owl_data.nn.moge_points_intrinsics_pipeline import MoGePointsIntrinsicsPipeline
from owl_data.nn.mmpose_keypoint_pipeline import PoseKeypointPipeline
from owl_data.nn.sam_keypoint_pipeline import SegmentationPipeline
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import cv2
import matplotlib
import numpy as np 
from tqdm import tqdm
import pdb
from time import time
import imageio

class DepthPosePipeline:

    def __init__(self, 
                depth_encoder:str='vitl', 
                segmentation_model:str = 'vit_l', 
                save_video_depth:bool=False, 
                keypoint_threshold:float=0.3, 
                segmentation_threshold:float=0.5,
                flow_static_threshold:float=0.7,
                photo_static_threshold:float=0.06,
                grad_min_threshold:float=0.04,
                huber_delta:float=0.05,
                iters:int=6,
                ema_alpha:float=0.90,
                radius:int=15,
                eps_frac:float=2e-3,
                var_window:int=1,
                tau:float=0.03,
                edge_threshold:float=0.1,
                beta:float=0.6,
                use_compile:bool=False,
                batch_size:int=50,
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
        self.save_video_depth = save_video_depth

        self.raft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(self.raft_device).eval()
        self.raft_preproc = Raft_Large_Weights.DEFAULT.transforms()

        #parameters from config
        self.keypoint_threshold=0.3 if not keypoint_threshold else keypoint_threshold
        self.segmentation_threshold=0.5 if not segmentation_threshold else segmentation_threshold
        self.flow_static_threshold=0.9 if not flow_static_threshold else flow_static_threshold
        self.photo_static_threshold=0.07 if not photo_static_threshold else photo_static_threshold
        self.grad_min_threshold=0.04 if not grad_min_threshold else grad_min_threshold
        self.huber_delta=0.05 if not huber_delta else huber_delta
        self.iters=6 if not iters else iters 
        self.ema_alpha=0.90 if not ema_alpha else ema_alpha
        self.radius=15 if not radius else radius
        self.epsilon = 1e-8
        self.eps_frac=1e-3 if not eps_frac else eps_frac
        self.var_window=1 if not var_window else var_window
        self.tau=0.03 if not tau else tau
        self.edge_threshold=0.07 if not edge_threshold else edge_threshold
        self.beta=0.7 if not beta else beta
        self.batch_size = 50 if not batch_size else batch_size

        #separately save the merged depth video if needed
        self.save_video_depth = save_video_depth

        # after all attributes are set (models, epsilon, etc.)
        if use_compile:
            try:
                # Stable shapes/dtypes? Try more aggressive mode.
                compile_mode = "max-autotune"   # or "reduce-overhead" if shapes vary a lot
                self._guided_filter_gray = torch.compile(self._guided_filter_gray, mode=compile_mode)
                self._box_filter_2d      = torch.compile(self._box_filter_2d,      mode=compile_mode)
                self.smooth_residual_three_tap_vec = torch.compile(self.smooth_residual_three_tap_vec, mode=compile_mode)
            except Exception as e:
                print(f"[compile] fell back to eager: {e}")


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

        if not keypoints: 
            return coord_depth_tensor
        
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
        pixel_x = keypoints[:, 0].long().clamp_(0, width-1)
        pixel_y = keypoints[:, 1].long().clamp_(0, height-1)
        
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
        eps = self.epsilon
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
        frames = torch.load(frame_filename)
        frame_idx = frame_filename.split('/')[-1].split('_')[0]
        
        full_depth_path = os.path.join(video_dir, 'full_depth_splits')
        full_seg_path = os.path.join(video_dir, 'full_seg_splits')
        keypoint_depth_path = os.path.join(video_dir, 'keypoint_depth_splits')
        keypoint_coord_path = os.path.join(video_dir, 'keypoint_coord_splits')

        # Create directories if they don't exist
        os.makedirs(full_depth_path, exist_ok=True)
        os.makedirs(full_seg_path, exist_ok=True)
        os.makedirs(keypoint_depth_path, exist_ok=True)
        os.makedirs(keypoint_coord_path, exist_ok=True)

        #accumulate results over ALL batches on cpu
        all_keypoint_depths = []
        all_coord_depth_maps = []
        all_merged_depths = []
        all_segmentation_maps = []

        for k in range(0, frames.shape[0], self.batch_size):
            frame_batch = frames[k:k+self.batch_size]
            #Step 1: extract the RTM pose pixels for humans and animals in the frames
            separate_entity_keypoints, aggregated_keypoints = self.pose_keypoint_pipeline(frames=frame_batch)
            torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
            
            #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
            masks, segmentation_keypoints = self.segmentation_pipeline(frames=frame_batch)
            torch.cuda.empty_cache()  # Clear GPU memory after segmentation
            
            #Step 3: run MoGe to get depth maps and coordinates directly from the images
            moge_output = self.moge_pipeline(
                                            frames=frame_batch,
                                            video_dir=video_dir,
                                            video_name=video_name,
                                            video_fps=video_fps
                                            )
            torch.cuda.empty_cache()  # Clear GPU memory after MoGe processing
            
            #accumulate the merged depth map and masks on cpu
            all_segmentation_maps.append(masks)

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

                # Move to CPU to save GPU memory
                all_keypoint_depths.append(keypoint_depths.cpu())
                all_coord_depth_maps.append(keypoint_coord_map[:, :, :3].cpu())
                all_merged_depths.append(depth_map.cpu())

                # Clear GPU memory after each iteration
                torch.cuda.empty_cache()

            #after all frame batches processed: stack keypoint depths, depth maps, coord maps, segmentation maps 
            #save merged depth map for each frame in frames as 1D pt file
            torch.save(torch.cat(all_merged_depths, dim=0), os.path.join(video_dir, os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt')))
            #save colored segmentation for each frame in frames as RGB pt file
            torch.save(torch.cat(all_segmentation_maps, dim=0), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))
            #save keypoint depth values (4th channel of coord_depth_map)
            torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
            #save the full 4-channel coord_depth_map
            torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))
            

    def run_pipeline_hybrid(self, frame_filename:str, video_dir:str, video_name:str, video_fps:int):
        frames = torch.load(frame_filename)
        frame_idx = frame_filename.split('/')[-1].split('_')[0]

        full_depth_path = os.path.join(video_dir, 'full_depth_splits')
        full_seg_path = os.path.join(video_dir, 'full_seg_splits')
        keypoint_depth_path = os.path.join(video_dir, 'keypoint_depth_splits')
        keypoint_coord_path = os.path.join(video_dir, 'keypoint_coord_splits')

        # Create directories if they don't exist
        os.makedirs(full_depth_path, exist_ok=True)
        os.makedirs(full_seg_path, exist_ok=True)
        os.makedirs(keypoint_depth_path, exist_ok=True)
        os.makedirs(keypoint_coord_path, exist_ok=True)

        #Step 3: run VideoDepthAnything to get consistent depth maps
        self.video_depth_pipeline.save_video = False
        depth_tensors, ___ = self.video_depth_pipeline(
                                frames=frames,
                                video_dir=video_dir,
                                video_name=video_name,
                                target_width = frames[0].shape[2],
                                target_height = frames[0].shape[1],
                                video_fps = video_fps
                            )
        torch.cuda.empty_cache()  # Clear GPU memory after VDA inference

        #accumulate results over ALL batches on cpu
        all_keypoint_depths = []
        all_coord_depth_maps = []
        all_merged_depths = []
        all_segmentation_maps = []

        for k in range(0, frames.shape[0], self.batch_size):
            frames_cpu = frames[k:k+self.batch_size]
            frames_cuda = frames_cpu.to(self.raft_device, non_blocking=True)\
                                .to(memory_format=torch.channels_last)\
                                .to(torch.float32).div(255.0)

            #Step 0: extract the video FOV using the first frame with MoGe
            self.moge_pipeline.save_video = True
            moge_output = self.moge_pipeline(
                                            frames=frames_cuda,
                                            video_dir=video_dir,
                                            video_name=video_name,
                                            video_fps=video_fps
                                            )
            torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
            estimated_intrinsics = moge_output['intrinsics'][0]

            #Step 1: extract the RTM pose pixels for humans and animals in the frames
            separate_entity_keypoints, pose_keypoints = self.pose_keypoint_pipeline(frames=frames_cpu)
            torch.cuda.empty_cache()  # Clear GPU memory after pose estimation
            #Step 2: extract the visual segmentations using SAM + compute keypoints for segment blobs
            masks, segmentation_keypoints = self.segmentation_pipeline(frames=frames_cpu)
            torch.cuda.empty_cache()  # Clear GPU memory after segmentation
            
            #Step 3.5: merge depth map from VDA and MoGE
            flows_fwd, flows_bwd = self.build_optical_flow_raft(frames_cuda)
            merged_depth_maps = self.merge_depth_maps(depthmap_vda=depth_tensors[k:k+self.batch_size], 
                                            depthmap_moge=moge_output['depth'],
                                            frames=frames_cpu,
                                            flows_fwd=flows_fwd,
                                            flows_bwd=flows_bwd)
            
            #save the merged depth maps if needed
            if self.save_video_depth:
                self._save_video(depth_frames=merged_depth_maps,
                                video_name=video_name,
                                video_dir=video_dir, 
                                fps=video_fps
                )
            #accumulate the merged depth map on cpu
            all_merged_depths.append(merged_depth_maps.cpu())
            all_segmentation_maps.append(masks)

            #Step 5: save the depth map for KEYPOINTS in each frame as 1D pt file
            #Step 6: COMPUTE and save the 3D coordinate for KEYPOINTS in each frame as RGB pt file => using FOV + (cx,cy) formula
            for i, (depth_map, seg_keypoint, pos_keypoint) in enumerate(zip(merged_depth_maps,
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
            
            #after all frame batches processed: stack keypoint depths, depth maps, coord maps, segmentation maps 
            #save merged depth map for each frame in frames as 1D pt file
            torch.save(torch.cat(all_merged_depths, dim=0), os.path.join(video_dir, os.path.join(full_depth_path, f'{frame_idx}_fulldepth.pt')))
            #save colored segmentation for each frame in frames as RGB pt file
            torch.save(torch.cat(all_segmentation_maps, dim=0), os.path.join(full_seg_path, f'{frame_idx}_segmap.pt'))
            #save keypoint depth values (4th channel of coord_depth_map)
            torch.save(torch.stack(all_keypoint_depths), os.path.join(keypoint_depth_path, f'{frame_idx}_kpdepth.pt'))
            #save the full 4-channel coord_depth_map
            torch.save(torch.stack(all_coord_depth_maps), os.path.join(keypoint_coord_path, f'{frame_idx}_coorddepth.pt'))

    def _save_video(self, depth_frames:torch.Tensor, video_dir:str, video_name:str, fps:int, is_depths:bool=True):
        # Stack all depth frames
        depth_frames = depth_frames.numpy()
        #create save path for video depth 
        output_video_path = os.path.join(video_dir, f'{video_name}_depth.mp4')
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
    def _to_gray(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert frames to grayscale using torchvision.
        Accepts (T,H,W,3) RGB in [0..1] or [0..255], or (T,H,W) already gray.
        Returns (T,H,W) float in [0,1].
        """
        # Already grayscale: (T,H,W)
        if img.ndim == 3:
            g = img.float()
            if g.max() > 1.5:
                g = g / 255.0
            return g.contiguous()

        # RGB: (T,3,H,W)
        if img.ndim == 4: 
            if not img.shape[1] == 3:
                #(T,H,W,3) -> (T,3,H,W)
                img = img.permute(0,3,1,2)
            x = img.float()
            if x.max() > 1.5:
                x = x / 255.0
            # torchvision expects BCHW
            g_bchw = TF.rgb_to_grayscale(x, num_output_channels=1)  # (T,1,H,W)
            g = g_bchw[:, 0]                         # (T,H,W)
            return g.contiguous()
        raise ValueError("Expected (T,H,W) or (T,H,W,3) tensor.")

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
        edge_threshold:float=None,
        beta:float=None):

        T,H,W = depthmap_vda.shape
        assert depthmap_moge.shape == (T,H,W)
        #access the config/params from class __init__ or merge function
        flow_static_thresh = self.flow_static_threshold if not flow_static_thresh else flow_static_thresh
        photo_static_thresh = self.photo_static_threshold if not photo_static_thresh else photo_static_thresh
        grad_min_thresh = self.grad_min_threshold if not grad_min_thresh else grad_min_thresh
        huber_delta = self.huber_delta if not huber_delta else huber_delta
        iters = self.iters if not iters else iters
        ema_alpha = self.ema_alpha if not ema_alpha else ema_alpha

        #move depth map tensors to CUDA and move back after merging depth maps
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else depthmap_vda.device
        if flows_fwd is not None:
            flows_fwd = flows_fwd.to(device, non_blocking=True)
        if flows_bwd is not None:
            flows_bwd = flows_bwd.to(device, non_blocking=True)

        # 1) VALID MASK (static & reliable) for s,b fit
        valid_mask = None
        if frames is not None and flows_fwd is not None:
            gray = self._to_gray(frames).to(device) # (T, H, W), float in [0,1]
            vm = torch.ones((T, H, W), device=device, dtype=torch.bool)

            # ---- Flow-magnitude gate for frames 1..T-1 (uses forward flow t-1->t) ----
            # flows_fwd: (T-1, H, W, 2)  -> per-pixel magnitude
            mag = torch.linalg.norm(flows_fwd, dim=-1)   # (T-1, H, W)
            m_static = (mag <= flow_static_thresh)       # (T-1, H, W)

            # ---- Texture gate for frames 1..T-1 (Sobel on current frame) ----
            # self._sobel_mag handles (T,H,W) and returns per-frame normalized grads
            grad_mag_all = self._sobel_mag(gray)         # (T, H, W) in [0,1]
            m_texture = (grad_mag_all[1:] >= grad_min_thresh)   # (T-1, H, W)

            # ---- Photometric gate (batched) using backward flow t->t-1 ----
            if flows_bwd is not None:
                B = flows_bwd.shape[0]                  # B = T-1
                # Build normalized sampling grid for all frames at once
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=flows_bwd.device, dtype=flows_bwd.dtype),
                    torch.arange(W, device=flows_bwd.device, dtype=flows_bwd.dtype),
                    indexing="ij"
                )
                xx = xx.expand(B, H, W)
                yy = yy.expand(B, H, W)
                grid_x = (xx + flows_bwd[..., 0]) / max(W - 1, 1) * 2 - 1
                grid_y = (yy + flows_bwd[..., 1]) / max(H - 1, 1) * 2 - 1
                grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)
                del grid_x, grid_y, xx, yy #free space on GPU/cuda

                prev = gray[:-1].unsqueeze(1)  # (B, 1, H, W)
                prev_warped = F.grid_sample(prev, grid, mode="bilinear",
                                            padding_mode="border", align_corners=True)[:, 0]  # (B,H,W)
                diff = (gray[1:] - prev_warped).abs()  # (B, H, W)
                del prev, gray

                # per-frame normalization to [0,1]
                mx = diff.view(B, -1).amax(dim=1, keepdim=True).clamp_min(self.epsilon).view(B, 1, 1)
                diff_norm = diff / mx
                m_photo = (diff_norm <= photo_static_thresh)  # (B, H, W)
            else:
                m_photo = torch.ones_like(m_static, dtype=torch.bool)

            # ---- Combine gates (add & m_photo if you enable the commented block) ----
            vm[1:] = m_static & m_texture & m_photo
            valid_mask = vm.cpu()
            del m_static, m_texture, m_photo

        # 2) GUIDANCE for guided low-pass
        guidance = frames if frames is not None else None

        # --- run the 2 steps ---
        vda_metric, s, b = self.align_scale_shift(
            depthmap_vda, depthmap_moge, valid_mask=valid_mask,
            huber_delta=huber_delta, iters=iters, ema_alpha=ema_alpha, clip=(0.1, 100.0)
        )
        del s, b

        #move tensors to CUDA for heavy-lifting in frame fusion
        vda_metric  = vda_metric.to(device, non_blocking=True)
        depthmap_moge = depthmap_moge.to(device, non_blocking=True)
        fused = self.per_frame_fuse(
            vda_metric, moge_depth=depthmap_moge, flows_fwd=flows_fwd, flows_bwd=flows_bwd, clip=(0.1, 100.0)
        )
        #move result back to CPU
        return fused.to("cpu", non_blocking=True)

    def run_video_base(self, frame_path:str, video_dir:str, video_name:str, video_fps:int):
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
        moge_depth: torch.Tensor,        # (T,H,W) metric (m)
        valid_mask: Optional[torch.Tensor] = None,  # (T,H,W) bool; static/reliable pixels (optional)
        huber_delta: float = 0.05,
        iters: int = 6,
        ema_alpha: float = 0.8,
        clip: Tuple[float,float] = (0.1, 100.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust IRLS per-frame fit of s,b in Dv*s + b ≈ Dm (vectorized across frames).
        Then EWMA-smooth (s,b) over time (vectorized, no per-frame loop).
        Returns:
        vda_metric: (T,H,W)
        s_all: (T,)
        b_all: (T,)
        """
        assert vda_depth.shape == moge_depth.shape
        T, H, W = vda_depth.shape
        dtype = vda_depth.dtype
        device = torch.device("cuda") if torch.cuda.is_available() else depthmap_vda.device

        #create and move depth estimates to contiguous arrays on same 'cuda' device
        vda_depth  = vda_depth.to(device, dtype=torch.float32, non_blocking=True).contiguous()
        moge_depth = moge_depth.to(device, dtype=torch.float32, non_blocking=True).contiguous()
        if valid_mask is not None and valid_mask.device != device:
            valid_mask = valid_mask.to(device, dtype=torch.bool, non_blocking=True).contiguous()

        # Flatten spatial dims: (T, N)
        N = H * W
        x = vda_depth.view(T, N)
        y = moge_depth.view(T, N)

        # Valid pixels per frame
        if valid_mask is not None:
            m_bool = valid_mask.view(T, N).bool()
        else:
            m_bool = torch.isfinite(x) & torch.isfinite(y)
        m = m_bool.float()                        # (T,N)
        has_data = (m_bool.sum(dim=1) >= 3)       # (T,)

        # Initialize s,b per frame (as in original code)
        s = torch.ones(T, 1, device=device, dtype=dtype)   # (T,1)
        b = torch.zeros(T, 1, device=device, dtype=dtype)  # (T,1)

        # IRLS over all frames in parallel
        for _ in range(iters):
            r = s * x + b - y                   # (T,N)
            abs_r = r.abs()
            w = torch.ones_like(r)
            mask_large = abs_r > huber_delta
            # Huber weights
            w[mask_large] = (huber_delta / (abs_r[mask_large] + 1e-12))
            # Zero out invalid pixels
            w = w * m

            # Weighted normal equation terms (per frame)
            wx  = w * x
            sum_wx2 = (wx * x).sum(dim=1)        # (T,)
            sum_wx  =  wx.sum(dim=1)             # (T,)
            sum_w   =  w.sum(dim=1)              # (T,)
            sum_wxy = (wx * y).sum(dim=1)        # (T,)
            sum_wy  = (w  * y).sum(dim=1)        # (T,)

            det = sum_wx2 * sum_w - sum_wx * sum_wx + 1e-12
            s_new = ( sum_w * sum_wxy - sum_wx * sum_wy) / det
            b_new = (-sum_wx * sum_wxy + sum_wx2 * sum_wy) / det

            # Only update frames that have enough data; keep defaults otherwise
            s = torch.where(has_data.view(T,1), s_new.view(T,1), s)
            b = torch.where(has_data.view(T,1), b_new.view(T,1), b)

        s_raw = s.view(T)   # (T,)
        b_raw = b.view(T)   # (T,)

        # -------- Vectorized EWMA over time (no per-frame loop) --------
        # s_smooth[t] = ema_alpha * s_smooth[t-1] + (1-ema_alpha) * s_raw[t], s_smooth[0] = s_raw[0]
        # Closed form: s_smooth[t] = (1-a) * a^t * cumsum(s_raw / a^k) + a^(t+1) * s_raw[0]
        a = float(ema_alpha)
        one_minus_a = 1.0 - a
        idx = torch.arange(T, device=device, dtype=dtype)

        a_pow_t   = (a ** idx)                    # (T,)
        a_pow_tp1 = (a ** (idx + 1.0))            # (T,)
        # Avoid division by zero if a == 0 (degenerate EMA -> s_smooth = s_raw)
        if a == 0.0:
            s_smooth = s_raw
            b_smooth = b_raw
        else:
            z_s = s_raw / (a ** idx)              # (T,)
            z_b = b_raw / (a ** idx)
            csum_s = torch.cumsum(z_s, dim=0)     # (T,)
            csum_b = torch.cumsum(z_b, dim=0)
            s_smooth = one_minus_a * a_pow_t * csum_s + a_pow_tp1 * s_raw[0]
            b_smooth = one_minus_a * a_pow_t * csum_b + a_pow_tp1 * b_raw[0]

        s_all = s_smooth.to(dtype)
        b_all = b_smooth.to(dtype)

        # Apply to depth and clip
        vda_metric = (s_all.view(T,1,1) * vda_depth + b_all.view(T,1,1))
        vda_metric = self._safe_clip(vda_metric, *clip)

        return vda_metric.to("cpu", non_blocking=True), s_all.to("cpu", non_blocking=True), b_all.to("cpu", non_blocking=True)
    
    def per_frame_fuse(
        self,
        vda_metric: torch.Tensor,      # (T,H,W) after step1
        moge_depth: torch.Tensor,      # (T,H,W)
        flows_fwd: torch.Tensor, 
        flows_bwd: torch.Tensor,
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
        eps_v = self._auto_eps_per_frame(vda_metric, self.eps_frac)  # (T,)
        eps_m = self._auto_eps_per_frame(moge_depth, self.eps_frac)  # (T,)
        Bv = self._guided_filter_gray(vda_metric, g, self.radius, eps_v)  # (T,H,W)
        Bm = self._guided_filter_gray(moge_depth, g, self.radius, eps_m)

        # High-frequency detail from MoGE
        Hm = moge_depth - Bm  # (T,H,W)
        # Hm = self._median_residual(Hm)
        
        # High frequency detail from VDA
        # Hv = vda_metric - Bv

        # Temporal variance of MoGE (short window) -> weight W in [0,1]
        # var = torch.zeros_like(moge_depth)
        # window_size = self.var_window // 2
        # for t in range(T):
        #     i = max(0, t - window_size)
        #     j = min(T, t + window_size + 1)
        #     seg = moge_depth[i:j]  # (len,H,W)
        #     var[t] = seg.var(dim=0, unbiased=False)
        var = moge_depth.unsqueeze(1).var(dim=1, unbiased=False)
        W = torch.exp(-var / max(self.tau, self.epsilon))
        Hm = self.smooth_residual_three_tap_vec(Hm=W*Hm, flows_fwd=flows_fwd, flows_bwd=flows_bwd)
        fused = self._safe_clip(Bv + Hm,*clip)
        return fused
    
    def _grid_from_flow(self, flows: torch.Tensor) -> torch.Tensor:
        # flows_bwd: (B,H,W,2) t->t-1
        B,H,W,_ = flows.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=flows.device, dtype=flows.dtype),
            torch.arange(W, device=flows.device, dtype=flows.dtype),
            indexing="ij"
        )
        xx = xx.expand(B, H, W); yy = yy.expand(B, H, W)
        gx = (xx + flows[..., 0]) / max(W-1, 1) * 2 - 1
        gy = (yy + flows[..., 1]) / max(H-1, 1) * 2 - 1
        return torch.stack([gx, gy], dim=-1).contiguous()  # (B,H,W,2)
    
    @torch.no_grad()
    def smooth_residual_three_tap_vec(self, Hm: torch.Tensor,
                                    flows_fwd: torch.Tensor,  # (T-1,H,W,2) t->t+1
                                    flows_bwd: torch.Tensor,  # (T-1,H,W,2) t->t-1
                                    w_prev: float = 0.25,
                                    w_cur:  float = 0.50,
                                    w_next: float = 0.25) -> torch.Tensor:
        """
        out[t] = w_prev*warp(H_{t-1}, F_bwd[t-1]) + w_cur*H_t + w_next*warp(H_{t+1}, F_fwd[t])
        Zero-phase-ish (less lag), fully vectorized.
        """
        T,H,W = Hm.shape
        out = Hm.clone()
        if T == 1: return out

        # prev->t for t=1..T-1
        grid_b = self._grid_from_flow(flows_bwd)                # (T-1,H,W,2)
        prev = Hm[:-1].unsqueeze(1).contiguous()               # (T-1,1,H,W)
        prev_warp = F.grid_sample(prev, grid_b, mode="bilinear",
                                padding_mode="border", align_corners=True)[:,0]  # (T-1,H,W)

        # next->t for t=0..T-2
        grid_f = self._grid_from_flow(flows_fwd)                # (T-1,H,W,2)
        nxt  = Hm[1:].unsqueeze(1).contiguous()                # (T-1,1,H,W)
        next_warp = F.grid_sample(nxt, grid_f, mode="bilinear",
                                padding_mode="border", align_corners=True)[:,0]  # (T-1,H,W)

        # assemble aligned tensors shaped (T,H,W)
        prev_aln = torch.zeros_like(Hm); prev_aln[1:] = prev_warp
        next_aln = torch.zeros_like(Hm); next_aln[:-1] = next_warp

        # simple fixed weights:
        w_sum = (w_prev + w_cur + w_next)
        w_prev_map = (w_prev / w_sum)
        w_cur_map  = (w_cur  / w_sum)
        w_next_map = (w_next / w_sum)

        out = (w_prev_map * prev_aln) + (w_cur_map * Hm) + (w_next_map * next_aln)

        # boundary handling: keep ends closer to valid neighbors
        out[0]  = (w_cur_map + w_next_map) * Hm[0]  + w_next_map * next_aln[0]
        out[-1] = (w_prev_map) * prev_aln[-1]         + (w_cur_map + w_prev_map) * Hm[-1]
        return out

    def _median_residual(self, Hm: torch.Tensor) -> torch.Tensor:
        """3-frame temporal median, vectorized (pad at ends by replication). x: (T,H,W)."""
        T,H,W = Hm.shape
        Hm_pad = torch.cat([Hm[[0]], Hm, Hm[[-1]]], dim=0)         # (T+2,H,W)
        trip = torch.stack([Hm_pad[:-2], Hm_pad[1:-1], Hm_pad[2:]], dim=0)  # (3,T,H,W)
        return trip.median(dim=0).values 

    def _sobel_mag(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute per-frame Sobel gradient magnitude, normalized to [0,1].
        Accepts (H,W) single frame or (T,H,W) batch; returns same rank/shape.
        """
        # Sobel kernels (on same device/dtype as g)
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                        device=g.device, dtype=g.dtype).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                        device=g.device, dtype=g.dtype).view(1,1,3,3)

        T, H, W = g.shape
        g = g.view(T,1,H,W)
        gx = F.conv2d(g, kx, padding=1)
        gy = F.conv2d(g, ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy).view(T, H, W)
        # per-frame normalization
        d = torch.clamp(mag.view(T, -1).amax(dim=1), min=self.epsilon).view(T,1,1)
        return mag / d
   
    
    def _box_filter_2d(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """
        Fast box filter via integral image (summed-area table).
        x: (T,H,W) -> returns (T,H,W) mean over (2r+1)x(2r+1) with zero-padding semantics.
        """
        assert x.ndim == 3, "Expected x of shape (T,H,W)"
        T, H, W = x.shape
        k = 2 * r + 1

        # integral image with a 1-pixel zero pad (top/left)
        x_pad = F.pad(x, (1, 0, 1, 0))           # (T,H+1,W+1)
        ii = x_pad.cumsum(dim=1).cumsum(dim=2)   # (T,H+1,W+1)

        # window bounds (clipped), then +1 for integral indexing on the "max" side
        ys = torch.arange(H, device=x.device)
        xs = torch.arange(W, device=x.device)
        y0 = (ys - r).clamp_(0, H-1)
        y1 = (ys + r).clamp_(0, H-1) + 1
        x0 = (xs - r).clamp_(0, W-1)
        x1 = (xs + r).clamp_(0, W-1) + 1

        # use (H,1) and (1,W) index shapes to avoid creating a singleton batch dim
        y0i = y0.view(H, 1); y1i = y1.view(H, 1)
        x0i = x0.view(1, W); x1i = x1.view(1, W)

        # 4-corner sums -> (T,H,W) directly
        A = ii[:, y1i, x1i]   # (T,H,W)
        B = ii[:, y0i, x1i]   # (T,H,W)
        C = ii[:, y1i, x0i]   # (T,H,W)
        D = ii[:, y0i, x0i]   # (T,H,W)
        box_sum = A - B - C + D
        return box_sum / float(k * k)

    def _guided_filter_gray(self, p: torch.Tensor, I: torch.Tensor, r: int, eps: torch.Tensor) -> torch.Tensor:
        """
        Classic guided filter (He et al.) per-frame with gray guidance.
        p: (T,H,W) to smooth; I: (T,H,W) guidance in [0,1];
        eps: (T,) per-frame regularizer.
        Returns (T,H,W).
        """
        T, H, W = p.shape
        mean_I = self._box_filter_2d(I, r)
        mean_p = self._box_filter_2d(p, r)
        corr_I = self._box_filter_2d(I * I, r)
        corr_Ip = self._box_filter_2d(I * p, r)

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        # broadcast eps(T,) -> (T,H,W)
        a = cov_Ip / (var_I + eps.view(T,1,1))
        b = mean_p - a * mean_I

        mean_a = self._box_filter_2d(a, r)
        mean_b = self._box_filter_2d(b, r)
        q = mean_a * I + mean_b
        return q
    
    @torch.no_grad()
    def build_optical_flow_raft(self, frames:torch.Tensor):
        """
        frames: (T,H,W,3) or (T,3,H,W), float in [0..1] or [0..255]
        returns:
        flows_fwd: (T-1,H,W,2)   t-1 -> t
        flows_bwd: (T-1,H,W,2)   t   -> t-1
        """
        # Normalize to [0,1], keep device
        frames_device = frames.device
        if frames.shape[1] != 3 and frames.shape[-1] == 3:
            #(T,H,W,3) -> (T,3,H,W)
            frames = frames.permute(0,3,1,2).contiguous()
        if frames.device != self.raft_device:
            #align frame and RAFT model devices
            frames = frames.to(self.raft_device)
        if frames.max() > 1.5:
            frames = frames / 255.0

        # To BCHW
        T, C, H, W = frames.shape
        if T < 2:
            raise ValueError("Need at least 2 frames for optical flow")

        # Prepare batch pairs for all consecutive frames
        img_pre = frames[:-1].contiguous().to(self.raft_device)  # (T-1,3,H,W)
        img_post = frames[1:].contiguous().to(self.raft_device)   # (T-1,3,H,W)

        # # One forward pass to get BOTH directions: concat pairs along batch
        # # First half: forward (t-1 -> t); second half: backward (t -> t-1)
        # img_fwd = torch.cat([img_pre, img_post], dim=0).to(self.raft_device)  # (2*(T-1),3,H',W')
        # img_bwd = torch.cat([img_post, img_pre], dim=0).to(self.raft_device)

        # # Run RAFT once (returns list of flows at multiple iterations; get last most refined flow estimates)
        # flows_output = self.raft_model(img_fwd, img_bwd)[-1] # list of (B,2,H',W')
        # # Move back to original device
        # flows_output = flows_output.to(frames_device)
        # flow_fwd, flow_bwd = flows_output[:img_pre.shape[0]], flows_output[img_pre.shape[0]:]  # first B flows are fwd, last B are bwd

        #First compute forward flows + move back to original device
        flow_fwd = self.raft_model(img_pre, img_post)[-1] # list of (B,2,H',W')
        flow_fwd = flow_fwd.to(frames_device)
        #Next compute backward flows + move back to original device
        flow_bwd = self.raft_model(img_post, img_pre)[-1] # list of (B,2,H',W')
        flow_bwd = flow_bwd.to(frames_device)

        #Remove img_pre and img_post from device
        img_pre = img_pre.cpu()
        img_post = img_post.cpu()

        # Permute to (B,H,W,2) to match your downstream API & TV-L1 outputs
        flows_fwd = flow_fwd.permute(0, 2, 3, 1).contiguous().to(torch.float32)  # (T-1,H,W,2)
        flows_bwd = flow_bwd.permute(0, 2, 3, 1).contiguous().to(torch.float32)
        return flows_fwd, flows_bwd

    def _auto_eps_per_frame(self, depth: torch.Tensor, eps_frac: float) -> torch.Tensor:
        """
        eps per frame = eps_frac * (range_0-100%)^2
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
    
    def run_video(self, frame_path:str, video_dir:str, video_name:str, video_fps:int):

        if self.mode == 'moge':
            self.run_video_moge(
                frame_path = frame_path,
                video_dir = video_dir,
                video_name = video_name,
                video_fps = video_fps
            )
        elif self.mode == 'merged':
            self.run_video_hybrid(
                frame_path = frame_path,
                video_dir = video_dir,
                video_name = video_name,
                video_fps = video_fps
            )
        else:
            raise Exception(f'ERROR: mode {self.mode} is invalid')

if __name__ == '__main__':
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