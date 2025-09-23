"""
Pipeline 2: Manifest Generation Pipeline

This pipeline downloads .pt files from the manifest bucket (game-data-manifest),
loads the ExtractedData objects, runs quality checks, and generates a parquet
manifest file.

This allows for fast iteration when experimenting with different quality check
methods without re-downloading and re-extracting the original TAR files.
"""

import os
import queue
import logging
import threading
import traceback
from typing import List

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

from owl_data.waypoint_1.game_data.utils.s3_utils import download_extracted_data_from_s3, list_pt_files_in_bucket
from owl_data.waypoint_1.game_data.utils.manifest_utils import create_manifest_record, ParquetBatchWriter
from owl_data.waypoint_1.game_data.quality_checks.quality_checks import _run_all_quality_checks

S3Client = type(boto3.client('s3'))

load_dotenv()


def manifest_downloader_task(
    manifest_bucket: str,
    master_queue: queue.Queue,
    buffer_queue: queue.Queue,
    s3_client: S3Client,
    num_processors: int
):
    """
    Producer: Downloads .pt files from manifest bucket and places ExtractedData in buffer queue.
    """
    while True:
        pt_s3_key = master_queue.get()

        if pt_s3_key is None:
            logging.info("Shutdown signal received. Manifest downloader terminating.")
            # Propagate shutdown signal to all processor threads
            for _ in range(num_processors):
                buffer_queue.put((None, None))
            master_queue.task_done()
            break

        try:
            logging.info(f"Downloading .pt file: {pt_s3_key}")
            extracted_data = download_extracted_data_from_s3(
                s3_client=s3_client,
                manifest_bucket=manifest_bucket,
                pt_s3_key=pt_s3_key
            )
            
            buffer_queue.put((pt_s3_key, extracted_data))
            logging.info(f"'{pt_s3_key}' placed on buffer. Buffer size: {buffer_queue.qsize()}")

        except FileNotFoundError:
            logging.warning(f"PT file not found: {pt_s3_key}")
        except ClientError as e:
            logging.error(f"S3 client error for {pt_s3_key}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error downloading {pt_s3_key}: {e}")
        finally:
            master_queue.task_done()


def manifest_processor_task(
    buffer_queue: queue.Queue,
    output_path: str,
    file_lock: threading.Lock
):
    """
    Consumer: Runs quality checks on ExtractedData and writes results to parquet manifest.
    """
    batch_writer = ParquetBatchWriter(output_path, file_lock, batch_size=50)

    while True:
        pt_s3_key, extracted_data = buffer_queue.get()

        if pt_s3_key is None:  # Shutdown signal
            batch_writer.flush()  # Write any remaining records
            logging.info("Shutdown signal received. Manifest processor terminating.")
            buffer_queue.task_done()
            break

        try:
            # Run quality checks on the extracted data
            logging.info(f"Running quality checks on {extracted_data.s3_key}")
            quality_flags = _run_all_quality_checks(extracted_data)
            
            # Create manifest record
            manifest_record = create_manifest_record(
                s3_key=extracted_data.s3_key,
                extracted_data=extracted_data,
                quality_flags=quality_flags
            )
            
            # Add to batch writer
            batch_writer.add(manifest_record)
            
            logging.info(f"Successfully processed quality checks for {extracted_data.s3_key}")

        except Exception as e:
            # Create error record for failed quality checks
            logging.error(f"Failed to process quality checks for {pt_s3_key}: {e}", exc_info=True)
            
            # Try to create an error record if we have extracted_data
            if extracted_data:
                error_record = create_manifest_record(
                    s3_key=extracted_data.s3_key,
                    error=e
                )
                batch_writer.add(error_record)
            
        finally:
            # Signal that this item from the queue is finished
            buffer_queue.task_done()


def run_manifest_pipeline(
    manifest_bucket: str,
    output_path: str,
    pt_s3_keys: List[str] = None,
    prefix: str = ""
):
    """
    Runs the manifest generation pipeline to create parquet manifest from .pt files.
    
    Args:
        manifest_bucket: Name of the manifest bucket containing .pt files
        output_path: Path where the parquet manifest will be saved
        pt_s3_keys: Optional list of specific .pt S3 keys to process. If None, processes all .pt files.
        prefix: Optional prefix to filter .pt files (only used if pt_s3_keys is None)
    """
    # --- 1. Configuration and Initialization ---
    NUM_PROCESSORS = os.cpu_count() or 4
    BUFFER_QUEUE_SIZE = NUM_PROCESSORS * 2
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )

    # --- 2. Get list of .pt files to process ---
    if pt_s3_keys is None:
        logging.info(f"Listing .pt files in bucket {manifest_bucket} with prefix '{prefix}'...")
        tasks_to_process = list_pt_files_in_bucket(
            s3_client=s3_client,
            manifest_bucket=manifest_bucket,
            prefix=prefix
        )
    else:
        tasks_to_process = pt_s3_keys
        logging.info(f"Processing {len(tasks_to_process)} specified .pt files")

    if not tasks_to_process:
        logging.info("No .pt files to process.")
        return

    logging.info(f"Found {len(tasks_to_process)} .pt files to process")

    # --- 3. Set up queues and threads ---
    master_queue: queue.Queue = queue.Queue()
    buffer_queue: queue.Queue = queue.Queue(maxsize=BUFFER_QUEUE_SIZE)
    file_lock = threading.Lock()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Fill the master queue with tasks
    for task in tasks_to_process:
        master_queue.put(task)

    # Add shutdown signal
    master_queue.put(None)

    logging.info(f"Starting manifest pipeline with {NUM_PROCESSORS} processors...")

    # Start downloader thread
    threads: List[threading.Thread] = []
    downloader = threading.Thread(
        target=manifest_downloader_task,
        args=(manifest_bucket, master_queue, buffer_queue, s3_client, NUM_PROCESSORS),
        name="ManifestDownloader"
    )
    threads.append(downloader)
    downloader.start()

    # Start processor threads
    for i in range(NUM_PROCESSORS):
        processor = threading.Thread(
            target=manifest_processor_task,
            args=(buffer_queue, output_path, file_lock),
            name=f"ManifestProcessor-{i+1}"
        )
        threads.append(processor)
        processor.start()

    # --- 4. Wait for completion ---
    logging.info("All threads started. Waiting for queues to be processed...")
    master_queue.join()
    logging.info("Master queue is empty. Downloader has finished its work.")
    
    buffer_queue.join()
    logging.info("Buffer queue is empty. All manifest processing is complete.")

    # --- 5. Cleanly join threads ---
    for thread in threads:
        thread.join()

    logging.info("Manifest pipeline finished successfully.")


if __name__ == '__main__':
    # Example usage
    MANIFEST_BUCKET = "game-data-manifest"
    OUTPUT_PARQUET_PATH = "/mnt/data/sami/manifests/gamedata_quality_manifest_from_pt.parquet"
    
    # You can specify specific .pt files or process all files with a prefix
    # Option 1: Process all .pt files
    # pt_files_to_process = None
    
    # Option 2: Process specific .pt files
    # pt_files_to_process = [
    #     "example/video1.pt",
    #     "example/video2.pt",
    # ]
    
    # Option 3: Process all .pt files with a specific prefix
    # prefix_filter = "2024/"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_manifest_pipeline(
        manifest_bucket=MANIFEST_BUCKET,
        output_path=OUTPUT_PARQUET_PATH,
        pt_s3_keys=None,  # Process all .pt files
        prefix=""  # No prefix filter
    ) 