import os
import torch
import torch.multiprocessing as mp
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import time
import queue
import threading

sys.path.append('segment-anything')
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class MultiGPUSegmentationPipeline:
    """
    Multi-GPU SAM segmentation pipeline for 3-4x speedup.
    Distributes frames across multiple GPUs for parallel processing.
    """
    
    def __init__(self, 
                 model_name: str = 'vit_l',
                 score_threshold: float = 0.0,
                 points_per_side: int = 24, # Fewer points = faster processing
                 points_per_batch: int = 120,  # More points per batch = better GPU utilization
                 pred_iou_thresh: float = 0.9, # Higher threshold = fewer low-quality masks
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 100, #remove small masks for efficiency
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_workers: int = None):
        """
        Initialize multi-GPU SAM pipeline.
        
        Args:
            model_name: SAM model variant ('vit_h', 'vit_l', 'vit_b')
            score_threshold: Minimum score threshold for masks
            performance_preset: Performance preset ('fast', 'balanced', 'quality')
            max_workers: Maximum number of GPU workers (None = auto-detect)
        """
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.performance_preset = performance_preset
        
        # Auto-detect GPU count or use specified max_workers
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.max_workers = min(max_workers or self.num_gpus, self.num_gpus)
        
        # Initialize single-GPU pipelines for each device
        self.gpu_pipelines = {}
        self._initialize_gpu_pipelines(points_per_side,
                                points_per_batch,
                                pred_iou_thresh,
                                stability_score_thresh,
                                min_mask_area)
    
    def _initialize_gpu_pipelines(self, 
                                points_per_side,
                                points_per_batch,
                                pred_iou_thresh,
                                stability_score_thresh,
                                min_mask_area):
        """Initialize SAM pipeline on each available GPU."""
        sam_model_checkpoints = {
            'vit_h': 'sam_vit_h.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }
        
        base_ckpt_path = os.path.normpath(os.path.join(
            os.path.abspath(__file__), 
            '..',
            '..',
            '..',
            'segment-anything',
            'ckpts'
        ))
        model_ckpt = os.path.join(base_ckpt_path, sam_model_checkpoints[self.model_name])
        
        for gpu_id in range(self.max_workers):
            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            
            # Load model on specific GPU
            sam_model = sam_model_registry[self.model_name](checkpoint=model_ckpt)
            sam_model.to(device=device)
            
            # Create mask generator for this GPU
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=0,
                min_mask_region_area=min_mask_area,
                output_mode="binary_mask"
            )
            
            self.gpu_pipelines[gpu_id] = {
                'mask_generator': mask_generator,
                'device': device,
                'model': sam_model
            }
            
            print(f"  - Initialized GPU {gpu_id} ({device})")
    
    def _process_frame_on_gpu(self, gpu_id: int, frame_data: Tuple[int, torch.Tensor]) -> Tuple[int, Any, Any]:
        """
        Process a single frame on a specific GPU.
        
        Args:
            gpu_id: GPU device ID
            frame_data: Tuple of (frame_index, frame_tensor)
        
        Returns:
            Tuple of (frame_index, masks, keypoints)
        """
        frame_idx, frame = frame_data
        
        try:
            # Convert frame to numpy format
            if frame.dtype == torch.float32 or frame.dtype == torch.float64:
                if frame.max() <= 1.0:
                    frame = frame * 255
                frame = frame.byte()
            
            frame_np = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # Get the pipeline for this GPU
            pipeline = self.gpu_pipelines[gpu_id]
            mask_generator = pipeline['mask_generator']
            
            # Generate masks
            masks = mask_generator.generate(frame_np)
            
            # Process masks (simplified version of the original processing)
            processed_masks = self._extract_masks(masks)
            processed_masks = self._filter_masks_by_score(processed_masks)
            
            return frame_idx, processed_masks['masks'], processed_masks['all_keypoints']
            
        except Exception as e:
            print(f"Error processing frame {frame_idx} on GPU {gpu_id}: {e}")
            return frame_idx, [], []
    
    def _extract_masks(self, sam_output):
        """Extract and post-process masks from SAM output (simplified version)."""
        if not sam_output:
            return {
                'masks': [],
                'all_keypoints': [],
                'scores': []
            }
        
        masks = []
        all_keypoints = []
        scores = []
        
        for mask_data in sam_output:
            masks.append(mask_data['segmentation'])
            
            # Extract basic keypoints from mask
            if isinstance(mask_data['segmentation'], dict):
                from segment_anything.utils.amg import rle_to_mask
                binary_mask = rle_to_mask(mask_data['segmentation'])
            else:
                binary_mask = mask_data['segmentation']
            
            # Get mask center as a simple keypoint
            y_coords, x_coords = np.where(binary_mask)
            if len(y_coords) > 0:
                center_y, center_x = y_coords.mean(), x_coords.mean()
                all_keypoints.append([[center_x, center_y]])
            else:
                all_keypoints.append([])
            
            scores.append(mask_data['predicted_iou'])
        
        return {
            'masks': masks,
            'all_keypoints': all_keypoints,
            'scores': scores
        }
    
    def _filter_masks_by_score(self, processed_masks):
        """Filter masks by score threshold."""
        if not processed_masks['masks']:
            return processed_masks
        
        filtered_masks = []
        filtered_keypoints = []
        filtered_scores = []
        
        for i, score in enumerate(processed_masks['scores']):
            if score >= self.score_threshold:
                filtered_masks.append(processed_masks['masks'][i])
                filtered_keypoints.append(processed_masks['all_keypoints'][i])
                filtered_scores.append(score)
        
        return {
            'masks': filtered_masks,
            'all_keypoints': filtered_keypoints,
            'scores': filtered_scores
        }
    
    def process_frames_parallel(self, frames: torch.Tensor) -> Tuple[List, List]:
        """
        Process frames in parallel across multiple GPUs with optimized batching.
        
        Args:
            frames: Tensor of shape (N, C, H, W) or (C, H, W)
        
        Returns:
            Tuple of (all_masks, all_keypoints)
        """
        # Handle single frame case
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        
        num_frames = frames.shape[0]
        print(f"Processing {num_frames} frames across {self.max_workers} GPUs...")
        
        # Calculate optimal batch size per GPU for better throughput
        frames_per_gpu = max(1, num_frames // self.max_workers)
        batch_size_per_gpu = min(2, frames_per_gpu)  # Process 2 frames per batch per GPU
        
        # Prepare batched frame data
        all_masks = [None] * num_frames
        all_keypoints = [None] * num_frames
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_indices = {}
            
            # Submit batched tasks to GPUs
            for start_idx in range(0, num_frames, self.max_workers * batch_size_per_gpu):
                for gpu_offset in range(self.max_workers):
                    gpu_id = gpu_offset
                    batch_start = start_idx + gpu_offset * batch_size_per_gpu
                    batch_end = min(batch_start + batch_size_per_gpu, num_frames)
                    
                    if batch_start < num_frames:
                        # Create batch for this GPU
                        batch_indices = list(range(batch_start, batch_end))
                        batch_frames = [(i, frames[i]) for i in batch_indices]
                        
                        future = executor.submit(self._process_frame_batch_on_gpu, gpu_id, batch_frames)
                        future_to_indices[future] = batch_indices
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_indices):
                batch_indices = future_to_indices[future]
                batch_results = future.result()
                
                # Store results in correct positions
                for i, (frame_idx, masks, keypoints) in enumerate(batch_results):
                    all_masks[frame_idx] = masks
                    all_keypoints[frame_idx] = keypoints
                    completed += 1
                
                if completed % max(1, num_frames // 5) == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (num_frames - completed) / rate if rate > 0 else 0
                    print(f"  Processed {completed}/{num_frames} frames ({rate:.1f} fps, ETA: {eta:.1f}s)")
        
        total_time = time.time() - start_time
        fps = num_frames / total_time if total_time > 0 else 0
        print(f"Multi-GPU processing completed in {total_time:.1f}s ({fps:.1f} fps)")
        
        return all_masks, all_keypoints
    
    def _process_frame_batch_on_gpu(self, gpu_id: int, batch_frames: List[Tuple[int, torch.Tensor]]) -> List[Tuple[int, Any, Any]]:
        """
        Process a batch of frames on a specific GPU for better efficiency.
        
        Args:
            gpu_id: GPU device ID
            batch_frames: List of (frame_index, frame_tensor) tuples
        
        Returns:
            List of (frame_index, masks, keypoints) tuples
        """
        results = []
        
        try:
            # Get the pipeline for this GPU
            pipeline = self.gpu_pipelines[gpu_id]
            mask_generator = pipeline['mask_generator']
            
            # Process each frame in the batch
            for frame_idx, frame in batch_frames:
                # Convert frame to numpy format
                if frame.dtype == torch.float32 or frame.dtype == torch.float64:
                    if frame.max() <= 1.0:
                        frame = frame * 255
                    frame = frame.byte()
                
                frame_np = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # Generate masks
                masks = mask_generator.generate(frame_np)
                
                # Process masks
                processed_masks = self._extract_masks(masks)
                processed_masks = self._filter_masks_by_score(processed_masks)
                
                results.append((frame_idx, processed_masks['masks'], processed_masks['all_keypoints']))
                
        except Exception as e:
            print(f"Error processing batch on GPU {gpu_id}: {e}")
            # Return empty results for failed frames
            for frame_idx, _ in batch_frames:
                results.append((frame_idx, [], []))
        
        return results
    
    def __call__(self, frames: torch.Tensor):
        """Main callable method for the pipeline."""
        return self.process_frames_parallel(frames)


class MultiGPUSegmentationWrapper:
    """
    Drop-in replacement wrapper for the original SegmentationPipeline
    that provides multi-GPU acceleration.
    """
    
    def __init__(self, model_name: str = 'vit_l', score_threshold: float = 0.0, **kwargs):
        """Initialize with same interface as original SegmentationPipeline."""
        # Check if we have multiple GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if num_gpus > 1:
            print(f"Using Multi-GPU SAM Pipeline with {num_gpus} GPUs for maximum speed!")
            self.pipeline = MultiGPUSegmentationPipeline(
                model_name=model_name,
                score_threshold=score_threshold,
                performance_preset='balanced'
            )
            self.is_multi_gpu = True
        else:
            print("Single GPU detected, falling back to optimized single-GPU pipeline")
            # Import and use the original optimized pipeline
            from .sam_keypoint_pipeline import SegmentationPipeline
            self.pipeline = SegmentationPipeline(
                model_name=model_name,
                score_threshold=score_threshold,
                performance_preset='balanced'
            )
            self.is_multi_gpu = False
    
    def __call__(self, frames: torch.Tensor):
        """Process frames using the appropriate pipeline."""
        return self.pipeline(frames)
    
    def process_video(self, frames: torch.Tensor):
        """Compatibility method."""
        if self.is_multi_gpu:
            return self.pipeline.process_frames_parallel(frames)
        else:
            return self.pipeline.process_video(frames)
