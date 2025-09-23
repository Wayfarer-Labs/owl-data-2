import os
import sys
import queue
import threading
import logging
import boto3
import tempfile
import pandas as pd
from dotenv import load_dotenv

# --- Important: Import the tasks from your main pipeline script ---
# This assumes your main script is named 'pipeline.py' and is in the same directory.
from owl_data.waypoint_1.game_data.pipeline.pipeline import downloader_task, processor_task

# Load environment variables for S3 access
load_dotenv()

# Set up logging for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def test_downloader(s3_key_to_test: str):
    """
    Tests the downloader_task in isolation.

    It creates the necessary queues, adds a single S3 key, and runs the
    downloader. It then verifies that the downloader fetched the data and
    placed it on the buffer queue correctly.
    """
    logging.info("--- Running Downloader Test ---")
    
    # 1. Setup
    bucket_name = "game-data"
    master_queue = queue.Queue()
    buffer_queue = queue.Queue(maxsize=5)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )

    # 2. Add a single task and the shutdown signal
    master_queue.put(s3_key_to_test)
    master_queue.put(None) # Sentinel for the downloader

    # 3. Run the downloader in a thread
    # We pass num_processors=1 because we only need to shut down one (imaginary) processor
    downloader_thread = threading.Thread(
        target=downloader_task,
        args=(bucket_name, master_queue, buffer_queue, s3_client, 1),
        name="TestDownloader"
    )
    downloader_thread.start()
    
    # 4. Wait for the downloader to finish
    master_queue.join()
    downloader_thread.join()
    logging.info("Downloader thread has finished.")

    # 5. Verify the result
    try:
        s3_key, tar_bytes = buffer_queue.get_nowait()
        if s3_key == s3_key_to_test and isinstance(tar_bytes, bytes) and len(tar_bytes) > 0:
            logging.info(f"✅ SUCCESS: Downloader fetched '{s3_key}' ({len(tar_bytes)/1e6:.2f} MB) correctly.")
        else:
            logging.error("❌ FAILURE: Data in buffer queue is incorrect.")
            
        # Check that the shutdown signal was propagated for the processor
        shutdown_signal = buffer_queue.get_nowait()
        if shutdown_signal == (None, None):
            logging.info("✅ SUCCESS: Downloader propagated the shutdown signal.")
        else:
            logging.error("❌ FAILURE: Shutdown signal was not propagated correctly.")

    except queue.Empty:
        logging.error("❌ FAILURE: Buffer queue is empty. Downloader did not place data on it.")
    
    logging.info("--- Downloader Test Complete ---")


def test_processor(s3_key_to_test: str):
    """
    Tests the processor_task in isolation.

    It first downloads a single TAR file to use as test data. It then
    runs the processor on this data and checks if it produces an output file.
    """
    logging.info("--- Running Processor Test ---")

    # 1. Setup: First, we need data to process. Let's download one file.
    logging.info(f"Setting up test by downloading '{s3_key_to_test}'...")
    s3_client = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    try:
        response = s3_client.get_object(Bucket="game-data", Key=s3_key_to_test)
        tar_bytes = response['Body'].read()
    except Exception as e:
        logging.error(f"❌ FAILURE: Could not download test data '{s3_key_to_test}'. Error: {e}")
        return

    # 2. Setup queues and a temporary output file
    buffer_queue = queue.Queue()
    file_lock = threading.Lock()
    
    # Use a temporary file for the output so we don't clutter the directory
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        output_path = tmp.name
        logging.info(f"Processor will write results to temporary file: {output_path}")

        # 3. Add the test data and shutdown signal to the buffer
        buffer_queue.put((s3_key_to_test, tar_bytes))
        buffer_queue.put((None, None)) # Sentinel for the processor

        # 4. Run the processor in a thread
        processor_thread = threading.Thread(
            target=processor_task,
            args=(buffer_queue, output_path, file_lock),
            name="TestProcessor"
        )
        processor_thread.start()

        # 5. Wait for the processor to finish
        buffer_queue.join()
        processor_thread.join()
        logging.info("Processor thread has finished.")

        # 6. Verify the result
        try:
            results_df = pd.read_parquet(output_path)
            if not results_df.empty and results_df.iloc[0]['tar_url'] == s3_key_to_test:
                logging.info(f"✅ SUCCESS: Processor wrote a valid record to the output file.")
                logging.info("Output record:\n" + str(results_df.iloc[0]))
            else:
                logging.error("❌ FAILURE: Output file is incorrect or malformed.")
        except Exception as e:
            logging.error(f"❌ FAILURE: Could not read or verify the output Parquet file. Error: {e}")

    logging.info("--- Processor Test Complete ---")


if __name__ == '__main__':
    num_tars = 10
    with open('task_list.txt', 'r') as f:
        task_list = f.readlines()

    for i in range(num_tars):
        test_downloader(task_list[i].strip())
        test_processor(task_list[i].strip())
