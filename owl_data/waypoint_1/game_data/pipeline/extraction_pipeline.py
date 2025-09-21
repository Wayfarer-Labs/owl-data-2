"""
Pipeline 1: TAR Extraction Pipeline

This pipeline downloads TAR files from the source bucket (game-data), 
extracts the video data and metadata, and uploads the ExtractedData 
objects as .pt files to the manifest bucket (game-data-manifest).

This separates the expensive extraction process from the quality checks,
allowing for faster iteration when experimenting with different quality
check methods.
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

from owl_data.waypoint_1.game_data.pipeline.tar_utils import extract_and_sample
from owl_data.waypoint_1.game_data.pipeline.s3_utils import upload_extracted_data_to_s3, get_missing_pt_files
from owl_data.waypoint_1.game_data.pipeline.downloader import S3Client

load_dotenv()
from owl_data.waypoint_1.game_data.pipeline.constants import MAX_FILE_SIZE_BYTES


def extraction_downloader_task(
    bucket_name: str,
    master_queue: queue.Queue,
    buffer_queue: queue.Queue,
    s3_client: S3Client,
    num_processors: int
):
    """
    Producer: Downloads TAR files from source bucket and places them in buffer queue.
    Similar to the original downloader but for the extraction pipeline.
    """
    while True:
        s3_key = master_queue.get()

        if s3_key is None:
            logging.info("Shutdown signal received. Extraction downloader terminating.")
            # Propagate shutdown signal to all processor threads
            for _ in range(num_processors):
                buffer_queue.put((None, None))
            master_queue.task_done()
            break

        try:
            meta = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            size = meta['ContentLength']

            if size > MAX_FILE_SIZE_BYTES or size == 0:
                logging.warning(f"SKIPPING {s3_key} - Invalid size: {size / 1e6:.2f} MB.")
                master_queue.task_done()
                continue

            logging.info(f"Downloading {s3_key} ({size / 1e6:.2f} MB)...")
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            tar_bytes = response['Body'].read()

            buffer_queue.put((s3_key, tar_bytes))
            logging.info(f"'{s3_key}' placed on buffer. Buffer size: {buffer_queue.qsize()}")

        except ClientError as e:
            logging.error(f"S3 client error for {s3_key}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error downloading {s3_key}: {e}")
        finally:
            master_queue.task_done()


def extraction_processor_task(
    buffer_queue: queue.Queue,
    manifest_bucket: str,
    s3_client: S3Client
):
    """
    Consumer: Extracts data from TAR files and uploads .pt files to manifest bucket.
    """
    while True:
        s3_key, tar_bytes = buffer_queue.get()

        if s3_key is None:  # Shutdown signal
            logging.info("Shutdown signal received. Extraction processor terminating.")
            buffer_queue.task_done()
            break

        try:
            # Extract data from TAR file
            logging.info(f"Extracting data from {s3_key}")
            extracted_data = extract_and_sample(tar_bytes, s3_key)
            
            if not extracted_data:
                raise ValueError("extract_and_sample did not return valid data.")

            # Upload extracted data as .pt file to manifest bucket
            logging.info(f"Uploading extracted data for {s3_key} to manifest bucket")
            pt_s3_key = upload_extracted_data_to_s3(
                extracted_data=extracted_data,
                s3_client=s3_client,
                manifest_bucket=manifest_bucket,
                original_s3_key=s3_key
            )
            
            logging.info(f"Successfully processed {s3_key} -> {pt_s3_key}")

        except Exception as e:
            logging.error(f"Failed to process {s3_key}: {e}", exc_info=True)
        finally:
            # Signal that this item from the queue is finished
            buffer_queue.task_done()


def run_extraction_pipeline(
    source_bucket: str,
    manifest_bucket: str,
    master_task_list: List[str],
    skip_existing: bool = True
):
    """
    Runs the extraction pipeline to convert TAR files to .pt files.
    
    Args:
        source_bucket: Name of the source bucket containing TAR files (e.g., "game-data")
        manifest_bucket: Name of the manifest bucket for .pt files (e.g., "game-data-manifest")
        master_task_list: List of TAR S3 keys to process
        skip_existing: If True, skip TAR files that already have corresponding .pt files
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

    # --- 2. Filter out already processed files if requested ---
    if skip_existing:
        logging.info("Checking for existing .pt files to skip...")
        tasks_to_process = get_missing_pt_files(
            s3_client=s3_client,
            source_bucket=source_bucket,
            manifest_bucket=manifest_bucket,
            tar_s3_keys=master_task_list
        )
        logging.info(f"Processing {len(tasks_to_process)} out of {len(master_task_list)} TAR files")
    else:
        tasks_to_process = master_task_list
        logging.info(f"Processing all {len(tasks_to_process)} TAR files")

    if not tasks_to_process:
        logging.info("No TAR files to process. All .pt files already exist.")
        return

    # --- 3. Set up queues and threads ---
    master_queue: queue.Queue = queue.Queue()
    buffer_queue: queue.Queue = queue.Queue(maxsize=BUFFER_QUEUE_SIZE)

    # Fill the master queue with tasks
    for task in tasks_to_process:
        master_queue.put(task)

    # Add shutdown signal
    master_queue.put(None)

    logging.info(f"Starting extraction pipeline with {NUM_PROCESSORS} processors...")

    # Start downloader thread
    threads: List[threading.Thread] = []
    downloader = threading.Thread(
        target=extraction_downloader_task,
        args=(source_bucket, master_queue, buffer_queue, s3_client, NUM_PROCESSORS),
        name="ExtractionDownloader"
    )
    threads.append(downloader)
    downloader.start()

    # Start processor threads
    for i in range(NUM_PROCESSORS):
        processor = threading.Thread(
            target=extraction_processor_task,
            args=(buffer_queue, manifest_bucket, s3_client),
            name=f"ExtractionProcessor-{i+1}"
        )
        threads.append(processor)
        processor.start()

    # --- 4. Wait for completion ---
    logging.info("All threads started. Waiting for queues to be processed...")
    master_queue.join()
    logging.info("Master queue is empty. Downloader has finished its work.")
    
    buffer_queue.join()
    logging.info("Buffer queue is empty. All extraction processing is complete.")

    # --- 5. Cleanly join threads ---
    for thread in threads:
        thread.join()

    logging.info("Extraction pipeline finished successfully.")


if __name__ == '__main__':
    # Example usage
    SOURCE_BUCKET = "game-data"
    MANIFEST_BUCKET = "game-data-manifest"
    
    # Load task list
    task_list_file = "task_list.txt"
    if os.path.exists(task_list_file):
        with open(task_list_file, 'r') as f:
            tasks = [line.strip() for line in f if line.strip()]
    else:
        # Fallback to a small test list
        tasks = [
            "example/video1.tar",
            "example/video2.tar",
        ]
        logging.warning(f"Task list file {task_list_file} not found. Using test tasks.")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_extraction_pipeline(
        source_bucket=SOURCE_BUCKET,
        manifest_bucket=MANIFEST_BUCKET,
        master_task_list=tasks,
        skip_existing=False
    ) 