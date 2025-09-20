import json
import traceback
import pandas as pd

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.checks import MENU_THRESHOLD

def _create_manifest_record(s3_key: str, extracted_data: ExtractedData, quality_flags: dict, error: Exception = None) -> dict:
    """
    Creates a standardized record for the manifest, ensuring all columns are present.

    This function acts as a central builder for your output rows. It populates
    fields based on successful processing or fills in error details, always
    returning a dictionary with a consistent schema.

    Args:
        s3_key: The S3 key of the processed TAR.
        extracted_data: The successful result from extract_and_sample.
        quality_flags: The successful result from the quality checks.
        error: The exception object if processing failed.

    Returns:
        A dictionary that conforms to the MANIFEST_COLUMNS schema.
    """
    # 1. Start with a template containing default values for all possible columns
    record = {
        's3_key': s3_key,
        'video_id': None,
        'video_metadata': None,
        'session_metadata': None,
        'sampled_frames': None, # Storing large arrays in Parquet is not ideal, consider storing paths instead
        'is_video_mostly_dark': None,
        'is_dpi_scale_issue': None,
        'video_menu_percent': None,
        'menu_flag_threshold': MENU_THRESHOLD,
        'is_video_mostly_menu': None,
        'error': None,
        'error_traceback': None,
        'processed_time': pd.Timestamp.now()
    }

    # 2. If processing was successful, populate the data fields
    if error is None:
        # NOTE: This assumes `extract_and_sample` now returns a single object again,
        # since you enforced the "1 video per TAR" rule.
        # If it still returns a list, you'd call this function inside a loop.
        record.update({
            'video_id': extracted_data.video_id,
            'video_metadata': json.dumps(extracted_data.video_metadata),
            'session_metadata': json.dumps(extracted_data.session_metadata),
            **quality_flags
        })
    # 3. If an error occurred, populate the error fields
    elif error:
        record['error'] = str(error)
        record['error_traceback'] = traceback.format_exc()

    return record