import logging
import queue
import boto3
from botocore.exceptions import ClientError

S3Client = type(boto3.client('s3'))

from owl_data.waypoint_1.game_data.pipeline.constants import MAX_FILE_SIZE_BYTES


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
