import os
import torch
import numpy as np
import sys
sys.path.append('segment-anything')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
torch.backends.cuda.enable_flash_sdp(True)
from tqdm import tqdm
import colorsys
import hashlib
import pdb


class SegmentationPipeline:
    def __init__(self, 
                 model_name: str = 'vit_l',  # Changed default from vit_h to vit_l for better speed
                 score_threshold: float = 0.0,
                 points_per_side: int = 24, # Fewer points = faster processing
                 points_per_batch: int = 120,  # More points per batch = better GPU utilization
                 pred_iou_thresh: float = 0.9, # Higher threshold = fewer low-quality masks
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 100, #remove small masks for efficiency
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        sam_model_checkpoints = {
            'vit_h': 'sam_vit_h.pth',        # ViT-Huge: ~632M params (slowest, highest quality)
            'vit_l': 'sam_vit_l.pth', # ViT-Large: ~308M params (balanced)
            'vit_b': 'sam_vit_b_01ec64.pth'  # ViT-Base: ~91M params (fastest, lower quality)
        }
        base_ckpt_path = os.path.normpath(os.path.join(
                                            os.path.abspath(__file__), 
                                            '..',
                                            '..',
                                            '..',
                                            'segment-anything',
                                            'ckpts')
                                        )
        model_ckpt = os.path.join(base_ckpt_path, sam_model_checkpoints[model_name])

        # Load model and move to device
        sam_model = sam_model_registry[model_name](checkpoint=model_ckpt)
        sam_model.to(device=device)
        
        # Optimized SAM configuration for speed
        self.sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side, 
            points_per_batch=points_per_batch, 
            pred_iou_thresh=pred_iou_thresh,  
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=0,  # No crop layers for speed
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask"  # Most efficient output format
        )
        self.score_threshold = score_threshold
        self.device = device
    
    def process_video(self, frames: torch.Tensor):
        """
        Process video frames to generate segmentation masks.
        
        Args:
            frames (torch.Tensor): Input frames tensor of shape (N, C, H, W) or (C, H, W)
        
        Returns:
            list: List of masks for each frame, where each mask contains segmentation info
        """
        # Handle single frame case
        if frames.dim() == 3:  # (C, H, W)
            frames = frames.unsqueeze(0)  # Add batch dimension
        
        all_masks = []
        all_segmentation_keypoints = []
        
        for i in tqdm(range(frames.shape[0])):
            frame = frames[i]  # (C, H, W)
            
            # Convert from torch tensor to numpy array
            # Assuming input is in format (C, H, W) with values in [0, 1] or [0, 255]
            if frame.dtype == torch.float32 or frame.dtype == torch.float64:
                # Convert from [0, 1] to [0, 255] if needed
                if frame.max() <= 1.0:
                    frame = frame * 255
                frame = frame.byte()
            
            # Convert to numpy and change from (C, H, W) to (H, W, C)
            frame_np = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            original_shape = frame_np.shape[:2]  # (H, W)
            
            # Generate masks using SAM on the numpy frame
            masks = self.sam_mask_generator.generate(frame_np)
            
            # Extract necessary information from masks
            processed_masks = self._extract_masks(masks)
            processed_masks = self._filter_masks_by_score(processed_masks)
            all_masks.append(self._consolidate_masks(processed_masks['masks']))
            all_segmentation_keypoints.append(processed_masks['all_keypoints'])
        
        return np.stack(all_masks,axis=0), all_segmentation_keypoints
    
    def _colors_from_ids(self, ids, s=0.65, v=0.95):
        # Deterministic colors from IDs (int or str)
        rgbs = []
        for i, x in enumerate(ids):
            hv = int.from_bytes(hashlib.md5(str(x).encode()).digest()[:8], "big") / 2**64
            h = (hv + i * (137.508/360.0)) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append((int(255*r), int(255*g), int(255*b)))
        return np.array(rgbs, dtype=np.uint8)

    def _distinct_colors(self, N, s=0.65, v=0.95):
        # Distinct per-frame colors if no IDs provided
        hues = ((np.arange(N) * 137.508) % 360) / 360.0
        rgbs = [tuple(int(255*c) for c in colorsys.hsv_to_rgb(h, s, v)) for h in hues]
        return np.array(rgbs, dtype=np.uint8)

    def _consolidate_masks(self, masks, ids=None):
        """
        masks: (N,H,W) bool/0-1 ==> where N is number of masks
        ids:   optional list/array of length N to keep colors stable across frames
        returns: (H,W,3) uint8 RGB where SMALLER objects are drawn ON TOP
        """
        masks = (masks > 0)
        if masks.ndim == 2:
            masks = masks[None, ...]
        N, H, W = masks.shape

        # pick colors
        colors = self._colors_from_ids(ids) if ids is not None else self._distinct_colors(N)

        # compute areas and paint largest->smallest so small overwrite (on top)
        areas = masks.reshape(N, -1).sum(axis=1)
        order = np.argsort(areas)[::-1]  # largest first, smallest last

        out = np.zeros((H, W, 3), dtype=np.uint8)
        for i in order:
            m = masks[i]
            if m.any():
                out[m] = colors[i]
        return out
    
    def find_closest_mask_pixel(self, x_coords, y_coords, target_y, target_x):
        distances = (y_coords - target_y)**2 + (x_coords - target_x)**2
        closest_idx = np.argmin(distances)
        return [x_coords[closest_idx], y_coords[closest_idx]]  # Return as [x, y]
    
    def extract_mask_keypoints(self, segmentation):
        """
        Extract keypoints from a segmentation mask.
        
        Args:
            segmentation: Either a binary mask (numpy array) or RLE format dict
        
        Returns:
            dict: Dictionary containing keypoint coordinates for different positions
        """
        # Convert segmentation to binary mask if needed
        if isinstance(segmentation, dict):
            # RLE format - convert to binary mask
            from segment_anything.utils.amg import rle_to_mask
            mask = rle_to_mask(segmentation)
        else:
            # Already a binary mask
            mask = segmentation
        
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Find all mask pixels
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) == 0:
            # Empty mask
            return {
                'top_left': None, 'top_middle': None, 'top_right': None,
                'left_middle': None, 'center': None, 'right_middle': None,
                'bottom_left': None, 'bottom_middle': None, 'bottom_right': None,
                'intermediate_points': []
            }
        
        # Get bounding box of the mask
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        
       
        # Extract corner and edge keypoints
        keypoints = {}
        
        # Corners
        keypoints['top_left'] = self.find_closest_mask_pixel(x_coords, y_coords, min_y, min_x)
        keypoints['top_right'] = self.find_closest_mask_pixel(x_coords, y_coords, min_y, max_x)
        keypoints['bottom_left'] = self.find_closest_mask_pixel(x_coords, y_coords, max_y, min_x)
        keypoints['bottom_right'] = self.find_closest_mask_pixel(x_coords, y_coords, max_y, max_x)
        
        # Edge midpoints
        mid_y = (min_y + max_y) // 2
        mid_x = (min_x + max_x) // 2
        
        keypoints['top_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, min_y, mid_x)
        keypoints['bottom_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, max_y, mid_x)
        keypoints['left_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, mid_y, min_x)
        keypoints['right_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, mid_y, max_x)
        
        # Center point
        keypoints['center'] = self.find_closest_mask_pixel(x_coords, y_coords, mid_y, mid_x)
        
        
        # Points between corners and edges
        quarter_y_top = (min_y + mid_y) // 2
        quarter_y_bottom = (mid_y + max_y) // 2
        quarter_x_left = (min_x + mid_x) // 2
        quarter_x_right = (mid_x + max_x) // 2
        
        # Top edge intermediate points
        keypoints['top_left_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, min_y, quarter_x_left)  # top-left-middle
        keypoints['top_right_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, min_y, quarter_x_right) # top-right-middle
        
        # Bottom edge intermediate points
        keypoints['bottom_left_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, max_y, quarter_x_left)  # bottom-left-middle
        keypoints['bottom_right_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, max_y, quarter_x_right)  # bottom-right-middle
        
        # Left edge intermediate points
        keypoints['left_top_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_top, min_x)  # left-top-middle
        keypoints['left_bottom_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_bottom, min_x)  # left-bottom-middle
        
        # Right edge intermediate points
        keypoints['right_top_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_top, max_x)  # right-top-middle
        keypoints['right_bottom_middle'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_bottom, max_x)  # right-bottom-middle
        
        # Diagonal intermediate points
        keypoints['top_left_quadrant'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_top, quarter_x_left)  # top-left quadrant
        keypoints['top_right_quadrant'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_top, quarter_x_right)  # top-right quadrant
        keypoints['bottom_left_quadrant'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_bottom, quarter_x_left)  # bottom-left quadrant
        keypoints['bottom_right_quadrant'] = self.find_closest_mask_pixel(x_coords, y_coords, quarter_y_bottom, quarter_x_right)  # bottom-right quadrant
        
        return keypoints
    
    def _extract_masks(self, sam_output):
        """
        Extract and post-process masks from SAM output.
        
        Args:
            sam_output (list): List of dictionaries from SAM generate() method
        
        Returns:
            dict: Dictionary containing processed mask information
        """
        if not sam_output:
            return {
                'masks': np.array(),
                'point_coords': np.array(),
                'all_keypoints': np.array(),
                'boxes': np.array(),
                'areas': np.array(),
                'scores': np.array(),
                'stability_scores': np.array()
            }
        
        masks = []
        point_coords = []
        all_keypoints = []
        boxes = []
        areas = []
        scores = []
        stability_scores = []
        
        for mask_data in sam_output:
            # Extract segmentation mask
            segmentation = mask_data['segmentation']
            masks.append(segmentation)
            
            # Extract keypoints from the segmentation mask
            mask_keypoints = self.extract_mask_keypoints(segmentation)
            
            # Extract the original point coordinates input to the model
            reference_keypoints = mask_data['point_coords'] if isinstance(mask_data['point_coords'], list) else mask_data['point_coords'].tolist()
            
            # Combine original point coordinates with extracted keypoints
            all_keypoints.append(reference_keypoints + list(mask_keypoints.values()))
            
            # Add intermediate points
            point_coords.append(reference_keypoints)
            
            # Extract bounding box (XYWH format)
            boxes.append(mask_data['bbox'])
            
            # Extract area
            areas.append(mask_data['area'])
            
            # Extract predicted IoU score
            scores.append(mask_data['predicted_iou'])
            
            # Extract stability score
            stability_scores.append(mask_data['stability_score'])
        
        return {
            'masks': np.array(masks),
            'point_coords': np.array(point_coords),
            'all_keypoints': np.array(all_keypoints),
            'boxes': np.array(boxes),
            'areas':  np.array(areas),
            'scores':  np.array(scores),
            'stability_scores':  np.array(stability_scores),
            'num_masks': len(masks)
        }
    
    def _filter_masks_by_score(self, processed_masks):
        """
        Filter masks by prediction score.
        
        Args:
            processed_masks (dict): Output from _extract_masks
            min_score (float): Minimum score threshold
        
        Returns:
            dict: Filtered mask data
        """
        if len(processed_masks['masks'])==0:
            return processed_masks
        
        filtered_data = {
            'masks': [],
            'point_coords': [],
            'all_keypoints': [],
            'boxes': [],
            'areas': [],
            'scores': [],
            'stability_scores': []
        }
        
        for i, score in enumerate(processed_masks['scores']):
            if score >= self.score_threshold:
                filtered_data['masks'].append(processed_masks['masks'][i])
                filtered_data['point_coords'].append(processed_masks['point_coords'][i])
                filtered_data['all_keypoints'].append(processed_masks['all_keypoints'][i])
                filtered_data['boxes'].append(processed_masks['boxes'][i])
                filtered_data['areas'].append(processed_masks['areas'][i])
                filtered_data['scores'].append(processed_masks['scores'][i])
                filtered_data['stability_scores'].append(processed_masks['stability_scores'][i])
        
        filtered_data['num_masks'] = len(filtered_data['masks'])

        for (k,v) in filtered_data.items():
            filtered_data[k] = np.array(v)
        
        return filtered_data
    
    def __call__(self, frames: torch.Tensor):
        """
        Main callable method for the pipeline.
        
        Args:
            frames (torch.Tensor): Input frames tensor
        
        Returns:
            list: Processed masks for each frame
        """
        masks, seg_points = self.process_video(frames=frames)
        return masks, seg_points
        