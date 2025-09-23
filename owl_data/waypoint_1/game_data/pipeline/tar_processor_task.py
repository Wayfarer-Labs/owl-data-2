import logging
import threading
import queue

from owl_data.waypoint_1.game_data.utils.tar_utils import extract_and_sample
from owl_data.waypoint_1.game_data.utils.manifest_utils import create_manifest_record, ParquetBatchWriter
from owl_data.waypoint_1.game_data.quality_checks.quality_checks import _run_all_quality_checks


def _process_single_tar(s3_key: str, tar_bytes: bytes) -> dict:
    """
    Handles the complete processing for one TAR file.

    This function contains the core business logic: extraction, quality checks,
    and the creation of a standardized record for success or failure.

    Returns:
        A single dictionary representing the manifest record.
    """
    try:
        # --- Success Path ---
        extracted_data = extract_and_sample(tar_bytes, s3_key)
        if not extracted_data:
             raise ValueError("extract_and_sample did not return valid data.")

        quality_flags = _run_all_quality_checks(extracted_data)
        
        # Create and return the success record.
        return create_manifest_record(
            s3_key=s3_key, 
            extracted_data=extracted_data, 
            quality_flags=quality_flags
        )

    except Exception as e:
        # --- Failure Path ---
        logging.error(f"Failed to process {s3_key}: {e}", exc_info=True)
        # Create and return the failure record.
        return create_manifest_record(s3_key=s3_key, error=e)


def processor_task(
    buffer_queue: queue.Queue,
    output_path: str,
    file_lock: threading.Lock
):
    """
    Consumer: Orchestrates fetching data, processing it, and batch-writing results.
    """
    batch_writer = ParquetBatchWriter(output_path, file_lock, batch_size=50)

    while True:
        s3_key, tar_bytes = buffer_queue.get()

        if s3_key is None: # Shutdown signal
            batch_writer.flush() # Write any remaining records
            logging.info("Shutdown signal received. Processor terminating.")
            buffer_queue.task_done()
            break
        
        # --- The entire processing logic is now a single, clean function call ---
        final_record = _process_single_tar(s3_key, tar_bytes)
        
        # --- The I/O logic is now a single, clean method call ---
        batch_writer.add(final_record)

        # Signal that this item from the queue is finished
        buffer_queue.task_done()
