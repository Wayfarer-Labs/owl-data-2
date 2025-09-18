import os
import sys
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BUCKET_NAME = "game-data"
TASK_LIST_DIR = '/mnt/data/sami/cache/game_data_manifests'
TASK_LIST_PATH = os.path.join(TASK_LIST_DIR, 'task_list.txt')

# Set up basic logging to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_task_list(bucket_name: str, output_path: str):
    """
    Scans an S3 bucket for all objects ending in '.tar' and writes their
    keys to a text file.

    This function handles pagination to ensure all objects in the bucket are found.

    Args:
        bucket_name (str): The name of the S3 bucket to scan.
        output_path (str): The local file path to write the list of S3 keys to.
                           Each key will be on a new line.
    """
    logging.info(f"Connecting to S3 to scan bucket '{bucket_name}'...")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    
    all_tar_keys = []
    
    try:
        # Use a paginator to automatically handle fetching all objects
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        logging.info("Fetching object list... This may take a while for large buckets.")
        page_count = 0
        for page in pages:
            page_count += 1
            if 'Contents' in page:
                # Filter for .tar files in the current page of results
                tar_keys_on_page = [
                    obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.tar')
                ]
                all_tar_keys.extend(tar_keys_on_page)
                logging.info(f"Scanned page {page_count}, found {len(tar_keys_on_page)} TAR files. Total found: {len(all_tar_keys)}")
            else:
                logging.info(f"Scanned page {page_count}, no objects found.")


    except ClientError as e:
        logging.error(f"A boto3 client error occurred: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

    if not all_tar_keys:
        logging.warning("No .tar files were found in the bucket.")
        return

    logging.info(f"Found a total of {len(all_tar_keys)} TAR files.")

    # --- Write the list to the output file ---
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Writing task list to: {output_path}")
        with open(output_path, 'w') as f:
            for key in all_tar_keys:
                f.write(key + '\n')
        
        logging.info("Task list created successfully.")

    except IOError as e:
        logging.error(f"Failed to write to file {output_path}: {e}")


if __name__ == '__main__':
    # Run the function
    create_task_list(bucket_name=BUCKET_NAME, output_path=TASK_LIST_PATH)
