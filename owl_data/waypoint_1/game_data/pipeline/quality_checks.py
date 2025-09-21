"""
Quality check functions for game data pipeline.
"""

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData


def _run_all_quality_checks(data: ExtractedData) -> dict:
    """
    Runs all quality checks on the extracted data and returns the results as a dictionary.
    
    Args:
        data: ExtractedData object containing the video metadata and sampled frames
        
    Returns:
        Dictionary containing all quality check results with standardized keys
    """
    from owl_data.waypoint_1.game_data.pipeline.checks import check_darkness, check_dpi_scaling, check_for_menus, MENU_THRESHOLD
    
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