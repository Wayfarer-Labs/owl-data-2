"""
Quality check functions for game data pipeline.
"""

import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from owl_data.waypoint_1.game_data.owl_types import ExtractedData
from owl_data.waypoint_1.game_data.constants import MENU_THRESHOLD
from owl_data.waypoint_1.game_data.quality_checks.vlm import _encode_frame, _classify_single_frame



def check_darkness(data: ExtractedData, threshold: float = 15.0) -> bool:
    """Checks if the video is excessively dark."""
    # Use the frames with a long stride to get a diverse sample
    frames = data.sampled_frames["stride-30_chw"]
    avg_brightness = frames.mean()
    logging.info(f"'{data.s3_key}' brightness check: {avg_brightness:.2f}")
    return bool(avg_brightness < threshold)

def check_dpi_scaling(data: ExtractedData, border_ratio: float = 0.1) -> bool:
    """Checks for large black borders indicating a DPI scaling issue."""
    # Just need one representative frame
    frame = data.sampled_frames["stride-3_chw"][0] # C, H, W
    h, w = frame.shape[1], frame.shape[2]
    
    # Check top and bottom borders
    top_border = frame[:, :int(h * border_ratio), :]
    bottom_border = frame[:, int(h * (1 - border_ratio)):, :]

    is_letterboxed = top_border.mean() < 10.0 and bottom_border.mean() < 10.0
    # A more robust check would also look at left/right borders.
    return bool(is_letterboxed)


def check_for_menus(data) -> dict:
    """
    Classifies frames as 'menu' or 'gameplay' and checks for consecutive menus.

    Args:
        data: An ExtractedData object containing the `sampled_frames` dictionary.

    Returns:
        A dictionary containing detailed menu classification metadata:
        {
            'is_consecutive_menu': bool,
            'stride-30_chw_menus': [bool, ...],
            'stride-60_chw_menus': [bool, ...]
        }
    """
    # 1. Combine frames, remembering the split point
    frames_30s = data.sampled_frames.get("stride-30_chw", np.array([]))
    frames_60s = data.sampled_frames.get("stride-60_chw", np.array([]))
    num_frames_30s = frames_30s.shape[0]
    
    if frames_30s.size == 0 and frames_60s.size == 0:
        logging.warning(f"No frames found for specified strides in '{data.s3_key}'.")
        return {
            'is_consecutive_menu': False,
            'stride-30_chw_menus': [],
            'stride-60_chw_menus': []
        }
        
    all_frames = np.concatenate([frames_30s, frames_60s], axis=0)
    num_frames = all_frames.shape[0]

    logging.info(f"Checking {num_frames} frames for menus in '{data.s3_key}'...")
    
    # 2. Classify all frames in parallel
    results = [""] * num_frames # Pre-allocate list to preserve order
    with ThreadPoolExecutor(max_workers=5) as executor:
        encoded_frames = [_encode_frame(frame) for frame in all_frames]
        future_to_index = {executor.submit(_classify_single_frame, frame): i for i, frame in enumerate(encoded_frames)}
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logging.error(f"Frame {index} generated an exception: {e}")
                results[index] = "error"

    # 3. Convert results to boolean lists (True if 'menu')
    is_menu_list = [res == 'menu' for res in results]

    # 4. Check for consecutive menu frames from the full boolean list
    is_consecutive_menu = any(is_menu_list[i] and is_menu_list[i+1] for i in range(len(is_menu_list) - 1))

    # 5. Split the boolean list back into stride-specific lists
    menu_bools_30s = is_menu_list[:num_frames_30s]
    menu_bools_60s = is_menu_list[num_frames_30s:]
    
    logging.info(f"VLM results for '{data.s3_key}': Consecutive menus: {is_consecutive_menu}.")
    
    return {
        'is_consecutive_menu': is_consecutive_menu,
        'stride-30_chw_menus': menu_bools_30s,
        'stride-60_chw_menus': menu_bools_60s,
    }


def _run_all_quality_checks(data: ExtractedData) -> dict:
    """
    Runs all quality checks on the extracted data and returns the results as a dictionary.
    
    Args:
        data: ExtractedData object containing the video metadata and sampled frames
        
    Returns:
        Dictionary containing all quality check results with standardized keys
    """
    # Run all quality checks
    is_video_mostly_dark = check_darkness(data)
    is_dpi_scale_issue = check_dpi_scaling(data)
    video_menu_percent = check_for_menus(data)
    is_video_mostly_menu = video_menu_percent > MENU_THRESHOLD
    
    return {
        'is_video_mostly_dark': is_video_mostly_dark,
        'is_dpi_scale_issue': is_dpi_scale_issue,
        'video_menu_percent': video_menu_percent,
        'is_video_mostly_menu': is_video_mostly_menu,
    } 