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
from typing import List, Tuple, Optional


class SegmentationPipeline:
    def __init__(self, 
                 model_name: str = 'vit_l',  # Changed default from vit_h to vit_l for better speed
                 score_threshold: float = 0.0,
                 points_per_side: int = 18, # Fewer points = faster processing
                 points_per_batch: int = 256,  # More points per batch = better GPU utilization
                 pred_iou_thresh: float = 0.9, # Higher threshold = fewer low-quality masks
                 stability_score_thresh: float = 0.95,
                 min_mask_region_area: int = 150, #remove small masks for efficiency
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
        sam_model.eval()
        
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
        torch.backends.cudnn.benchmark = True  # faster speed when input sizes vary; lets cuDNN autotune
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
        if frames.dim == 3:  # (C, H, W)
            frames = frames.unsqueeze(0)  # Add batch dimension
        #assign frames to cpu device
        if frames.device.type != 'cpu':
            frames = frames.cpu()
        #convert NCHW -> NHWC formatting
        if frames.shape[1] == 3: 
            frames = frames.permute(0, 2, 3, 1)
        if frames.dtype != torch.uint8:
            frames = (frames.clamp(0,1).mul(255)).to(torch.uint8) if frames.dtype.is_floating_point else frames.to(torch.uint8)
        frames = frames.contiguous()
        frames = frames.numpy()
        
        all_masks = []
        all_keypoints = []
        with torch.inference_mode():
            for i in tqdm(range(frames.shape[0])):
                frame = frames[i]  # (C, H, W)
                masks = self.sam_mask_generator.generate(frame)
                
                # Extract necessary information from masks
                processed_masks = self.extract_masks(masks, 
                                                    score_th=self.score_threshold
                                                   )
                
                # processed_masks = self._filter_masks_by_score(processed_masks)
                all_masks.append(self._consolidate_masks(processed_masks['masks']))
                all_keypoints.append(processed_masks['keypoints_xy'])
        return torch.from_numpy(np.stack(all_masks,axis=0)), all_keypoints
    
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
    
    def _stack_masks_from_sam(self,
                          sam_output: List[dict], *,
                          score_th: Optional[float] = None,
                          allow_rle: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        masks: (N,H,W) bool
        boxes_xywh: (N,4) int
        areas: (N,) int
        scores: (N,) float32
        """
        if not sam_output:
            return (np.zeros((0,1,1), bool),
                    np.zeros((0,4), np.int32),
                    np.zeros((0,),  np.int32),
                    np.zeros((0,),  np.float32))

        if allow_rle:
            try:
                from segment_anything.utils.amg import rle_to_mask
            except Exception:
                rle_to_mask = None

        ms = []
        sc = []
        # Filter first (cheap) before stacking
        for d in sam_output:
            if score_th is not None and d.get("predicted_iou", 0.0) < score_th:
                continue
            seg = d["segmentation"]
            if isinstance(seg, dict):
                if not allow_rle:
                    # Skip or raise if you don't want RLE here
                    continue
                if rle_to_mask is None:
                    raise RuntimeError("RLE mask received but rle_to_mask not available.")
                seg = rle_to_mask(seg)
            ms.append(seg.astype(bool, copy=False))
            sc.append(float(d.get("predicted_iou", 0.0)))

        if not ms:
            return (np.zeros((0,1,1), bool),
                    np.zeros((0,4), np.int32),
                    np.zeros((0,),  np.int32),
                    np.zeros((0,),  np.float32))

        masks = np.stack(ms, axis=0)               # (N,H,W) bool
        scores = np.asarray(sc, dtype=np.float32)  # (N,)

        N, H, W = masks.shape

        # Areas (vectorized)
        areas = masks.sum(axis=(1,2), dtype=np.int32)

        # Bounding boxes (vectorized via projections)
        rows_any = masks.any(axis=2)           # (N,H) True where any pixel in row
        cols_any = masks.any(axis=1)           # (N,W)

        # Handle empty masks gracefully (return -1s)
        has_row = rows_any.any(axis=1)
        has_col = cols_any.any(axis=1)

        y_min = np.where(has_row, rows_any.argmax(axis=1), -1)
        y_max = np.where(
            has_row,
            H - 1 - rows_any[:, ::-1].argmax(axis=1),
            -1
        )
        x_min = np.where(has_col, cols_any.argmax(axis=1), -1)
        x_max = np.where(
            has_col,
            W - 1 - cols_any[:, ::-1].argmax(axis=1),
            -1
        )

        w = np.maximum(0, x_max - x_min + 1).astype(np.int32)
        h = np.maximum(0, y_max - y_min + 1).astype(np.int32)
        boxes_xywh = np.stack([x_min, y_min, w, h], axis=1).astype(np.int32)

        return masks, boxes_xywh, areas, scores

    def extract_mask_keypoints(self, masks: np.ndarray):
        """
        Batched keypoint extraction directly from masks (no boxes, no SciPy).
        masks: (B,H,W) bool/0-1
        Returns:
        pts_xy: (B,K,2) int32, keypoints snapped to nearest True pixel
        names:  list of K names
        """
        assert masks.ndim == 3
        B, H, W = masks.shape
        masks = masks.astype(bool, copy=False)

        names = [
            "top_left","top_middle","top_right",
            "left_middle","center","right_middle",
            "bottom_left","bottom_middle","bottom_right",
            "top_left_middle","top_right_middle",
            "bottom_left_middle","bottom_right_middle",
            "left_top_middle","left_bottom_middle",
            "right_top_middle","right_bottom_middle",
            "top_left_quadrant","top_right_quadrant",
            "bottom_left_quadrant","bottom_right_quadrant",
        ]
        K = len(names)
        pts_xy = np.full((B, K, 2), -1, dtype=np.int32)

        # Precompute a coordinate grid for quick indexing
        ys_grid, xs_grid = np.mgrid[0:H, 0:W]

        for i in range(B):
            m = masks[i]
            if not m.any():
                continue

            y_coords = ys_grid[m]
            x_coords = xs_grid[m]

            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            ym  = (min_y + max_y) // 2
            xm  = (min_x + max_x) // 2
            qy_t = (min_y + ym) // 2
            qy_b = (ym + max_y) // 2
            qx_l = (min_x + xm) // 2
            qx_r = (xm + max_x) // 2

            # targets as (K,2) in (y,x) order
            targets = np.array([
                [min_y, min_x], [min_y, xm],   [min_y, max_x],
                [ym,    min_x], [ym,    xm],   [ym,    max_x],
                [max_y, min_x], [max_y, xm],   [max_y, max_x],
                [min_y, qx_l],  [min_y, qx_r],
                [max_y, qx_l],  [max_y, qx_r],
                [qy_t,  min_x], [qy_b,  min_x],
                [qy_t,  max_x], [qy_b,  max_x],
                [qy_t,  qx_l],  [qy_t,  qx_r],
                [qy_b,  qx_l],  [qy_b,  qx_r],
            ], dtype=np.int32)

            # Vectorized nearest search for all K targets
            # dist: (K, P_i)
            dy = y_coords[None, :] - targets[:, 0, None]
            dx = x_coords[None, :] - targets[:, 1, None]
            dist = dy*dy + dx*dx
            nn_idx = dist.argmin(axis=1)  # (K,)

            pts_xy[i, :, 0] = x_coords[nn_idx]
            pts_xy[i, :, 1] = y_coords[nn_idx]

        return pts_xy, names


    def extract_masks(
        self,
        sam_output: List[dict],
        score_th: Optional[float] = None,
        allow_rle: bool = False
    ) -> dict[str, object]:
        """
        Vectorized mask extraction from SAM output. ~10–100x less Python overhead than per-mask loops.
        """
        masks, boxes_xywh, areas, scores = self._stack_masks_from_sam(
            sam_output, score_th=score_th, allow_rle=allow_rle
        )
        N = masks.shape[0]
        out = {
            "masks": masks,                            # (N,H,W) bool
            "boxes": boxes_xywh,                       # (N,4)  int XYWH
            "areas": areas,                            # (N,)   int
            "scores": scores,                          # (N,)   float32
            "num_masks": int(N),
            # Keep SAM’s original points as a list (variable-length)
            "point_coords": [d.get("point_coords", []) for d in sam_output][:N],
            # Stability score if present; otherwise zeros
            "stability_scores": np.asarray(
                [d.get("stability_score", 0.0) for d in sam_output][:N], dtype=np.float32
            ),
        }
        pts_xy, kp_names = self.extract_mask_keypoints(masks)  # (N,K,2)
        out["keypoints_xy"] = pts_xy           # (N,K,2) int
        out["keypoint_names"] = kp_names       # list[str]

        return out
    
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
        