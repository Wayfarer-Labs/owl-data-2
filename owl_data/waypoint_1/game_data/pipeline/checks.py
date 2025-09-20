import logging

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData

MENU_THRESHOLD = 0.1


def check_darkness(data: ExtractedData, threshold: float = 15.0) -> bool:
    """Checks if the video is excessively dark."""
    # Use the frames with a long stride to get a diverse sample
    frames = data.sampled_frames["stride-30_chw"]
    avg_brightness = frames.mean()
    logging.info(f"'{data.s3_key}' brightness check: {avg_brightness:.2f}")
    return avg_brightness < threshold

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
    return is_letterboxed

def check_for_menus(data: ExtractedData) -> float:
    """Estimates the percentage of time spent in a menu (placeholder)."""
    # This would eventually be a call to a VLM or a more complex model.
    # For now, it's a placeholder.
    return MENU_THRESHOLD # Placeholder: 10% of the time is in menus

