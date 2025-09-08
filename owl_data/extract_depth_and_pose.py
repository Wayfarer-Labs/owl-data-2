import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.abspath(__file__),'../..')))

from owl_data.nn.video_depth_pipeline import VideoDepthPipeline
from owl_data.nn.moge_points_intrinsics_pipeline import MoGePointsIntrinsicsPipeline
from owl_data.nn.mmpose_keypoint_pipeline import PoseKeypointPipeline
from owl_data.nn.multi_gpu_sam_pipeline import MultiGPUSegmentationWrapper
from owl_data.nn.sam_keypoint_pipeline import SegmentationPipeline
import torch
import numpy as np 
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt



class DepthPosePipeline:

    def __init__(self, depth_encoder:str='vitl', segmentation_model:str = 'vit_h', save_video_depth:bool=False, keypoint_threshold:float=0.3, segmentation_threshold:float=0.5):
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
        
        self.moge_pipeline = MoGePointsIntrinsicsPipeline()

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

    def run_pipeline_moge(self, frame_filename:str, video_dir: str, video_name:str):
        frames = torch.load(frame_filename)[:10]
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
                                        video_name=video_name
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
        frames = torch.load(frame_filename)[:10]
        frame_idx = frame_filename.split('/')[-1].split('_')[0]
       
        #Step 0: extract the video FOV using the first frame with MoGe
        estimated_intrinsics = self.moge_pipeline(
                                        frames=frames[0].unsqueeze(0),
                                        video_dir=video_dir,
                                        video_name=video_name
                                        )['intrinsics'][0].cpu()
        #Step 1: extract the RTM pose pixels for humans and animals in the frames
        separate_entity_keypoints, aggregated_keypoints = self.pose_keypoint_pipeline(frames=frames)
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
        for i, (depth_map, seg_keypoint, agg_keypoint) in enumerate(zip(depth_tensors,
                                                                        segmentation_keypoints, 
                                                                        aggregated_keypoints
                                                                    )):
            #Step 2.5: merge keypoints from segmentations and human/animals
            all_keypoints = self._aggregate_keypoints(seg_keypoint, agg_keypoint)
            
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
            )

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

    depth_pose_pipeline.run_video(
        frame_path = BASE_PATH,
        video_dir = VIDEO_DIR,
        video_name = VIDEO_NAME,
        video_fps = VIDEO_FPS
    )