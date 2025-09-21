import json
import traceback
import pandas as pd
from functools import cache
import logging
import os
import threading

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.checks import MENU_THRESHOLD

@cache
def get_commit_hash() -> str:
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

def create_manifest_record(
    s3_key: str, 
    extracted_data: ExtractedData = None, 
    quality_flags: dict = None,
    error: Exception = None
) -> dict:
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
        'processed_time': pd.Timestamp.now(),
        'commit_hash': get_commit_hash()
    }

    # 2. If processing was successful, populate the data fields
    if error is None and extracted_data and quality_flags:
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


class ParquetBatchWriter:
    """A helper class to manage the batching and writing of records to a Parquet file."""
    def __init__(self, output_path: str, file_lock: threading.Lock, batch_size: int = 50):
        self.output_path = output_path
        self.lock = file_lock
        self.batch_size = batch_size
        self.batch = []

    def add(self, record: dict):
        """Adds a record to the batch and writes the batch to disk if it's full."""
        self.batch.append(record)
        if len(self.batch) >= self.batch_size:
            self._write_batch()

    def flush(self):
        """Writes any remaining records in the batch to disk."""
        if self.batch:
            self._write_batch()

    def _write_batch(self):
        """The actual file-writing logic."""
        logging.info(f"Writing batch of {len(self.batch)} records to {self.output_path}...")
        with self.lock:
            try:
                df = pd.DataFrame(self.batch)
                if not os.path.exists(self.output_path):
                    df.to_parquet(self.output_path, engine='fastparquet', index=False)
                else:
                    df.to_parquet(self.output_path, engine='fastparquet', append=True)
                self.batch = [] # Clear the batch after a successful write
            except Exception as e:
                logging.critical(f"FATAL: Could not write batch to {self.output_path}: {e}")
                # Optional: Decide how to handle failed writes, e.g., save to a recovery file.
