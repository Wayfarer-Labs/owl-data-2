import os
import queue
import threading
import logging

from dotenv import load_dotenv
import boto3
import pandas as pd
from botocore.exceptions import ClientError

from owl_data.waypoint_1.game_data.pipeline.checks import (
    check_darkness,
    check_dpi_scaling,
    check_for_menus
)
from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.tar_utils import extract_and_sample


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()


# Constants for the pipeline
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
MENU_THRESHOLD = 0.1
# Using a placeholder type for boto3 client for cleaner annotations
S3Client = type(boto3.client('s3'))

def _run_all_quality_checks(data: ExtractedData) -> dict:
    """Runs all modular checks and returns a dictionary of flags."""
    menu_percent = check_for_menus(data)
    flags = {
        "is_video_mostly_dark": check_darkness(data),
        "is_dpi_scale_issue": check_dpi_scaling(data),
        "video_menu_percent": menu_percent,
        "menu_flag_threshold": MENU_THRESHOLD,
        "is_video_mostly_menu": menu_percent > MENU_THRESHOLD,
    }
    return flags

def downloader_task(
    bucket_name: str,
    master_queue: queue.Queue,
    buffer_queue: queue.Queue,
    s3_client: S3Client,
    num_processors: int
):
    """
    Producer: Fetches tasks from master_queue, downloads from S3, and
    places the data into the buffer_queue for the processors.
    """
    while True:
        s3_key = master_queue.get()

        if s3_key is None:
            logging.info("Shutdown signal received. Downloader terminating.")
            # Propagate the shutdown signal to all processor threads
            for _ in range(num_processors):
                buffer_queue.put((None, None))
            master_queue.task_done()
            break

        try:
            meta = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            size = meta['ContentLength']

            if size > MAX_FILE_SIZE_BYTES or size == 0:
                logging.warning(f"SKIPPING {s3_key} - Invalid size: {size / 1e6:.2f} MB.")
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

def processor_task(
    buffer_queue: queue.Queue,
    output_path: str,
    file_lock: threading.Lock
):
    """
    Consumer: Fetches data from buffer_queue, processes it, and writes
    results to the output Parquet file.
    """
    while True:
        s3_key, tar_bytes = buffer_queue.get()

        if s3_key is None:
            logging.info("Shutdown signal received. Processor terminating.")
            buffer_queue.task_done()
            break

        record = {"tar_url": s3_key, "processed_time": pd.Timestamp.now()}
        
        try:
            logging.info(f"Processing '{s3_key}'...")
            extracted_data: ExtractedData = extract_and_sample(tar_bytes, s3_key)
            quality_flags = _run_all_quality_checks(extracted_data)
            record.update(quality_flags)

            logging.info(f"Finished processing '{s3_key}'.")

        except Exception as e:
            import traceback
            logging.error(f"Failed to process {s3_key}: {e}", exc_info=True)
            record["error"] = str(e)
            record["error_traceback"] = traceback.format_exc()

        finally:
            with file_lock:
                try:
                    df = pd.DataFrame([record])
                    # You may want to write to separate files and merge later for performance
                    df.to_parquet(output_path, engine='pyarrow', append=True)
                except Exception as e:
                    logging.critical(f"FATAL: Could not write results for '{s3_key}' to {output_path}: {e}")
            buffer_queue.task_done()


def main_orchestrator(bucket: str, master_task_list: list[str], output_path: str):
    """Initializes and runs the entire multi-threaded data processing pipeline.

    This function orchestrates a producer-consumer workflow to efficiently process
    a large number of TAR files from S3. It sets up a single "downloader" thread
    that fetches files and places them into a bounded in-memory buffer. A pool of
    "processor" threads then consumes from this buffer, running data extraction
    and quality checks in parallel.

    The pipeline is designed for graceful shutdown and ensures that all tasks are
    completed before the program exits.

    Args:
        bucket (str): The name of the S3 bucket where the source TAR files are located.
        master_task_list (list[str]): A list of S3 object keys. Each key is a string
            representing the full path to a TAR file within the specified bucket
            (e.g., ['tars/video1.tar', 'tars/video2.tar']). This list serves
            as the complete set of tasks to be processed.
        output_path (str): The file path where the results will be saved. The output
            is a Parquet file, and results from each processed TAR will be
            appended to it. If the file does not exist, it will be created.
    """

    # --- 1. Configuration and Initialization ---
    NUM_PROCESSORS = os.cpu_count() or 4
    # Buffer size is a trade-off: larger uses more RAM but prevents stalls
    BUFFER_QUEUE_SIZE = NUM_PROCESSORS * 2
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    
    file_lock = threading.Lock()
    # queue containing tars to process
    master_queue = queue.Queue()
    # queue that holds in extracted data 
    buffer_queue = queue.Queue(maxsize=BUFFER_QUEUE_SIZE)
    logging.info(f"Starting pipeline with {NUM_PROCESSORS} processor threads.")
    
    # --- 2. Populate Master Task Queue ---
    logging.info(f"Populating master queue with {len(master_task_list)} tasks...")
    for task in master_task_list:
        master_queue.put(task)
    master_queue.put(None) # Sentinel for the downloader
    
    # --- 3. Start Worker Threads ---
    threads: list[threading.Thread] = []
    downloader = threading.Thread(
        target=downloader_task,
        args=(bucket, master_queue, buffer_queue, s3_client, NUM_PROCESSORS),
        name="Downloader"
    )
    threads.append(downloader)
    downloader.start()

    for i in range(NUM_PROCESSORS):
        processor = threading.Thread(
            target=processor_task,
            args=(buffer_queue, output_path, file_lock),
            name=f"Processor-{i+1}"
        )
        threads.append(processor)
        processor.start()

    # --- 4. Wait for Completion ---
    logging.info("All threads started. Waiting for queues to be processed...")
    # This blocks until the downloader has called task_done() for all items
    master_queue.join()
    logging.info("Master queue is empty. Downloader has finished its work.")
    
    # This blocks until processors have called task_done() for all items
    buffer_queue.join()
    logging.info("Buffer queue is empty. All processing is complete.")

    # --- 5. Cleanly Join Threads ---
    for thread in threads:
        thread.join()

    logging.info("Pipeline finished successfully.")


if __name__ == '__main__':
    # This is an example of how you would run the orchestrator
    BUCKET_NAME = "game-data"
    OUTPUT_PARQUET_PATH = "/mnt/data/sami/results/quality_manifest.parquet"
    
    # In a real run, you would list objects from S3 here.
    # For this example, we'll use a small, dummy list.
    # s3 = boto3.client(...)
    # response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="some/folder/")
    # all_tar_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.tar')]
    
    # dummy_task_list = [
    #     "tars/vid1.tar",
    #     "tars/vid2.tar",
    #     "tars/large_file.tar", # Will be skipped by size check
    #     "tars/non_existent.tar", # Will be skipped by S3 error
    #     "tars/vid3.tar",
    # ]
    task_list = "task_list.txt"
    tasks = [line.strip() for line in open(task_list, 'r').readlines() if line.strip()]
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)
    
    main_orchestrator(
        bucket=BUCKET_NAME,
        master_task_list=tasks,
        output_path=OUTPUT_PARQUET_PATH
    )
