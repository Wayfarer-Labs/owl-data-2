import os
import torch
import numpy as np
import sys
sys.path.append('segment-anything')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
torch.backends.cuda.enable_flash_sdp(True)



class SegmentationPipeline:
    def __init__(self, model_name: str = 'vit_h', score_threshold:float=0.0):
        sam_model_checkpoints = {
            'vit_h': 'sam_vit_h.pth'
        }
        base_ckpt_path = os.path.join(os.path.abspath(__file__), '..','..','segment-anything','ckpts')
        model_ckpt = os.path.join(base_ckpt_path, sam_model_checkpoints[model_name])

        self.sam_mask_generator = SamAutomaticMaskGenerator(
            sam_model_registry[model_name](checkpoint=model_ckpt)
        )
        self.score_threshold = score_threshold
    
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
        
        for i in range(frames.shape[0]):
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
            
            # Generate masks using SAM
            masks = self.sam_mask_generator.generate(frame_np)
            
            # Extract necessary information from masks
            processed_masks = self._extract_masks(masks)
            processed_masks = self._filter_masks_by_score(processed_masks)
            all_masks.append(processed_masks['masks'])
            all_segmentation_keypoints.append(processed_masks['all_keypoints'])
        
        return all_masks, all_segmentation_keypoints
    
    def find_closest_mask_pixel(self, target_y, target_x):
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
        keypoints['top_left'] = self.find_closest_mask_pixel(min_y, min_x)
        keypoints['top_right'] = self.find_closest_mask_pixel(min_y, max_x)
        keypoints['bottom_left'] = self.find_closest_mask_pixel(max_y, min_x)
        keypoints['bottom_right'] = self.find_closest_mask_pixel(max_y, max_x)
        
        # Edge midpoints
        mid_y = (min_y + max_y) // 2
        mid_x = (min_x + max_x) // 2
        
        keypoints['top_middle'] = self.find_closest_mask_pixel(min_y, mid_x)
        keypoints['bottom_middle'] = self.find_closest_mask_pixel(max_y, mid_x)
        keypoints['left_middle'] = self.find_closest_mask_pixel(mid_y, min_x)
        keypoints['right_middle'] = self.find_closest_mask_pixel(mid_y, max_x)
        
        # Center point
        keypoints['center'] = self.find_closest_mask_pixel(mid_y, mid_x)
        
        
        # Points between corners and edges
        quarter_y_top = (min_y + mid_y) // 2
        quarter_y_bottom = (mid_y + max_y) // 2
        quarter_x_left = (min_x + mid_x) // 2
        quarter_x_right = (mid_x + max_x) // 2
        
        # Top edge intermediate points
        keypoints['top_left_middle'] = find_closest_mask_pixel(min_y, quarter_x_left)  # top-left-middle
        keypoints['top_right_middle'] = find_closest_mask_pixel(min_y, quarter_x_right) # top-right-middle
        
        # Bottom edge intermediate points
        keypoints['bottom_left_middle'] = find_closest_mask_pixel(max_y, quarter_x_left)  # bottom-left-middle
        keypoints['bottom_right_middle'] = find_closest_mask_pixel(max_y, quarter_x_right)  # bottom-right-middle
        
        # Left edge intermediate points
        keypoints['left_top_middle'] = find_closest_mask_pixel(quarter_y_top, min_x)  # left-top-middle
        keypoints['left_bottom_middle'] = find_closest_mask_pixel(quarter_y_bottom, min_x)  # left-bottom-middle
        
        # Right edge intermediate points
        keypoints['right_top_middle'] = find_closest_mask_pixel(quarter_y_top, max_x)  # right-top-middle
        keypoints['right_bottom_middle'] = find_closest_mask_pixel(quarter_y_bottom, max_x)  # right-bottom-middle
        
        # Diagonal intermediate points
        keypoints['top_left_quadrant'] = find_closest_mask_pixel(quarter_y_top, quarter_x_left)  # top-left quadrant
        keypoints['top_right_quadrant'] = find_closest_mask_pixel(quarter_y_top, quarter_x_right)  # top-right quadrant
        keypoints['bottom_left_quadrant'] = find_closest_mask_pixel(quarter_y_bottom, quarter_x_left)  # bottom-left quadrant
        keypoints['bottom_right_quadrant'] = find_closest_mask_pixel(quarter_y_bottom, quarter_x_right)  # bottom-right quadrant
        
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
                'masks': [],
                'point_coords': [],
                'extracted_keypoints': [],
                'boxes': [],
                'areas': [],
                'scores': [],
                'stability_scores': []
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
            if isinstance(segmentation, dict):
                # RLE format - convert to binary mask if needed
                # For now, we'll keep the RLE format
                masks.append(segmentation)
            else:
                # Binary mask format
                masks.append(segmentation)
            
            # Extract keypoints from the segmentation mask
            mask_keypoints = self.extract_mask_keypoints(segmentation)
            
            # Extract the original point coordinates input to the model
            reference_keypoints = mask_data['point_coords'] if isinstance(original_point_coords, list) else mask_data['point_coords'].tolist()
            
            # Combine original point coordinates with extracted keypoints
            all_keypoints.append(reference_keypoints + mask_keypoints)
            
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
            'masks': masks,
            'point_coords': point_coords,
            'all_keypoints': all_keypoints,
            'boxes': boxes,
            'areas': areas,
            'scores': scores,
            'stability_scores': stability_scores,
            'num_masks': len(masks)
        }
    
    def filter_masks_by_score(self, processed_masks):
        """
        Filter masks by prediction score.
        
        Args:
            processed_masks (dict): Output from _extract_masks
            min_score (float): Minimum score threshold
        
        Returns:
            dict: Filtered mask data
        """
        if not processed_masks['masks']:
            return processed_masks
        
        filtered_data = {
            'masks': [],
            'point_coords': [],
            'extracted_keypoints': [],
            'boxes': [],
            'areas': [],
            'scores': [],
            'stability_scores': []
        }
        
        for i, score in enumerate(processed_masks['scores']):
            if score >= self.score_threshold:
                filtered_data['masks'].append(processed_masks['masks'][i])
                filtered_data['point_coords'].append(processed_masks['point_coords'][i])
                filtered_data['extracted_keypoints'].append(processed_masks['extracted_keypoints'][i])
                filtered_data['boxes'].append(processed_masks['boxes'][i])
                filtered_data['areas'].append(processed_masks['areas'][i])
                filtered_data['scores'].append(processed_masks['scores'][i])
                filtered_data['stability_scores'].append(processed_masks['stability_scores'][i])
        
        filtered_data['num_masks'] = len(filtered_data['masks'])
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
        