import os
import sys
import logging
import argparse
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Load environment variables from .env file
load_dotenv()

BUCKET_NAME = "game-data"
TASK_LIST_DIR = './'
TASK_LIST_PATH = os.path.join(TASK_LIST_DIR, 'task_list.txt')

# Set up basic logging to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_user_datetime(value: str) -> datetime:
    """
    Parse a datetime string.
    Supports:
      - YYYY-MM-DD
      - YYYY-MM-DDTHH:MM[:SS][.ffffff][Z|±HH:MM]
    Returns timezone-aware UTC datetime.
    """
    v = value.strip()
    if v.endswith('Z'):
        v = v[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        # try plain date
        dt = datetime.strptime(v, '%Y-%m-%d')
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def format_for_filename(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime('%Y%m%d_%H%M%S')

def create_task_list(bucket_name: str, output_path: str, start_dt: datetime = None, end_dt: datetime = None):
    """
    Scans an S3 bucket for all objects ending in '.tar' and writes their keys to a text file.
    Optionally filters by LastModified within [start_dt, end_dt] (inclusive). Datetimes are treated as UTC.
    """
    if start_dt or end_dt:
        logging.info(f"Filtering by LastModified: start={start_dt} end={end_dt}")
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
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        logging.info("Fetching object list... This may take a while for large buckets.")
        page_count = 0
        for page in pages:
            page_count += 1
            if 'Contents' in page:
                tar_keys_on_page = []
                for obj in page['Contents']:
                    key = obj['Key']
                    if not key.endswith('.tar'):
                        continue
                    lm = obj.get('LastModified')
                    if start_dt and lm < start_dt:
                        continue
                    if end_dt and lm > end_dt:
                        continue
                    tar_keys_on_page.append(key)
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
        logging.warning("No .tar files were found in the bucket with the given filters." if (start_dt or end_dt) else "No .tar files were found in the bucket.")
        return

    logging.info(f"Found a total of {len(all_tar_keys)} TAR files.")

    try:
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
    parser = argparse.ArgumentParser(description="Generate a task list of .tar keys from an S3 bucket, optionally filtered by LastModified datetime interval.")
    parser.add_argument('--start', type=str, default=None, help="Start datetime (inclusive). ISO 8601 (e.g., 2025-09-01 or 2025-09-01T12:00:00Z).")
    parser.add_argument('--end', type=str, default=None, help="End datetime (inclusive). ISO 8601 (e.g., 2025-09-15 or 2025-09-15T23:59:59Z). If only a date is provided, the end is treated as end-of-day UTC.")

    args = parser.parse_args()

    start_dt = parse_user_datetime(args.start) if args.start else None
    end_dt = parse_user_datetime(args.end) if args.end else None

    # If end was provided as date-only, bump to end-of-day UTC
    if args.end and len(args.end.strip()) == 10:
        end_dt = end_dt + timedelta(days=1) - timedelta(microseconds=1)

    if start_dt and end_dt and start_dt > end_dt:
        logging.error("Start datetime must be <= end datetime.")
        sys.exit(1)

    output_path = TASK_LIST_PATH
    if start_dt or end_dt:
        start_label = format_for_filename(start_dt) if start_dt else 'min'
        end_label = format_for_filename(end_dt) if end_dt else 'max'
        filename = f"task_list_{start_label}_to_{end_label}.txt"
        output_path = os.path.join(TASK_LIST_DIR, filename)

    create_task_list(bucket_name=BUCKET_NAME, output_path=output_path, start_dt=start_dt, end_dt=end_dt)