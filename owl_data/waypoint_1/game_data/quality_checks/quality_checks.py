"""
Quality check functions for game data pipeline.
"""

import logging
from owl_data.waypoint_1.game_data.owl_types import ExtractedData
from owl_data.waypoint_1.game_data.constants import MENU_THRESHOLD


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

def check_for_menus(data: ExtractedData) -> float:
    """Estimates the percentage of time spent in a menu (placeholder)."""
    # This would eventually be a call to a VLM or a more complex model.
    # For now, it's a placeholder.
    return MENU_THRESHOLD # Placeholder: 10% of the time is in menus


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