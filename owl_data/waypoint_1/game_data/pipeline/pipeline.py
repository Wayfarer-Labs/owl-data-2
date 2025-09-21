import os
import queue
import logging
import threading
import traceback

from dotenv import load_dotenv
import boto3
import pandas as pd
from botocore.exceptions import ClientError

from owl_data.waypoint_1.game_data.pipeline.downloader import downloader_task
from owl_data.waypoint_1.game_data.pipeline.processor import processor_task


load_dotenv()
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


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
    OUTPUT_PARQUET_PATH = "/mnt/data/sami/manifests/gamedata_quality_manifest.parquet"
    
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
