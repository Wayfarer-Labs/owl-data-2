import os
import sys
import logging
import boto3
import tarfile
import io
import uuid
from collections import defaultdict
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- 1. Function to Peer into TAR Structure ---
def peer_into_tar(tar_bytes: bytes) -> dict:
    """
    Inspects the structure of an in-memory TAR file without full extraction.

    Args:
        tar_bytes: The raw bytes of the TAR file.

    Returns:
        A dictionary summarizing the contents, including a count of MP4 files.
    """
    structure = {
        "mp4_count": 0,
        "json_count": 0,
        "csv_count": 0,
        "member_names": []
    }
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    structure["member_names"].append(member.name)
                    if member.name.lower().endswith('.mp4'):
                        structure["mp4_count"] += 1
                    elif member.name.lower().endswith('.json'):
                        structure["json_count"] += 1
                    elif member.name.lower().endswith('.csv'):
                        structure["csv_count"] += 1
    except tarfile.TarError as e:
        logging.error(f"Could not read TAR file structure: {e}")
    return structure

# --- 2. Function to Check if Splitting is Needed ---
def is_multi_video_tar(tar_structure: dict) -> bool:
    """
    Determines if a TAR file represents more than one video.

    Args:
        tar_structure: The structure dictionary from peer_into_tar.

    Returns:
        True if the TAR contains more than one MP4 file, False otherwise.
    """
    return tar_structure.get("mp4_count", 0) > 1

# --- 3. Function to Split a Multi-Video TAR ---
def split_multi_video_tar(tar_bytes: bytes) -> dict[str, io.BytesIO]:
    """
    Splits a single multi-video TAR into multiple in-memory single-video TARs.

    Args:
        tar_bytes: The raw bytes of the multi-video TAR file.

    Returns:
        A dictionary where keys are new, unique TAR names (e.g., 'uuid.tar')
        and values are BytesIO objects containing the new single-video TARs.
    """
    new_tars = {}
    file_groups = defaultdict(list)

    # First, group all members by their base name (e.g., '000000')
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as original_tar:
        for member in original_tar.getmembers():
            if member.isfile():
                base_name = os.path.splitext(os.path.basename(member.name))[0]
                file_groups[base_name].append(member)

        # Now, create a new TAR for each group that contains an MP4
        for base_name, members in file_groups.items():
            if any(m.name.endswith('.mp4') for m in members):
                new_tar_name = f"{uuid.uuid4().hex[:16]}.tar"
                new_tar_bytes_io = io.BytesIO()

                with tarfile.open(fileobj=new_tar_bytes_io, mode='w') as new_tar:
                    for member in members:
                        # Add each file from the group to the new TAR
                        file_content = original_tar.extractfile(member).read()
                        tarinfo = tarfile.TarInfo(name=os.path.basename(member.name))
                        tarinfo.size = len(file_content)
                        new_tar.addfile(tarinfo, io.BytesIO(file_content))
                
                new_tar_bytes_io.seek(0) # Rewind buffer to the beginning for reading
                new_tars[new_tar_name] = new_tar_bytes_io
                logging.info(f"Created new in-memory TAR '{new_tar_name}' for group '{base_name}'.")

    return new_tars

# --- 4. Main Orchestrator to Process and Split Tars ---
def process_and_split_tars(bucket_name: str, s3_keys_to_split: list[str], dry_run: bool = False):
    """
    Orchestrates the full workflow: download, split, upload new, and delete old.

    Args:
        bucket_name: The S3 bucket name.
        s3_keys_to_split: A list of S3 keys that need to be split.
        dry_run: If True, will only print actions without performing them.
    """
    s3_client = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )

    for s3_key in s3_keys_to_split:
        logging.info(f"--- Processing '{s3_key}' ---")
        try:
            # Download the multi-video TAR
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            tar_bytes = response['Body'].read()

            # Split into new in-memory TARs
            new_tars_map = split_multi_video_tar(tar_bytes)

            if not new_tars_map:
                logging.warning(f"Splitting '{s3_key}' resulted in no new TARs. Skipping.")
                continue

            # Upload new TARs and delete the old one
            for new_name, new_bytes_io in new_tars_map.items():
                new_key = os.path.join(os.path.dirname(s3_key), new_name)
                logging.info(f"Uploading new TAR to '{new_key}'.")
                if not dry_run:
                    s3_client.put_object(Bucket=bucket_name, Key=new_key, Body=new_bytes_io.read())
                else:
                    logging.info(f"[DRY RUN] Would upload {len(new_bytes_io.getvalue())} bytes to '{new_key}'.")

            logging.info(f"Deleting original TAR '{s3_key}'.")
            if not dry_run:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            else:
                logging.info(f"[DRY RUN] Would delete '{s3_key}'.")

        except ClientError as e:
            logging.error(f"An S3 error occurred while processing '{s3_key}': {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing '{s3_key}': {e}")

# --- 5. Function to Find All Multi-Video Tars ---
def find_multi_video_tars(bucket_name: str, all_s3_keys: list[str], output_file: str):
    """
    Scans a list of S3 keys, identifies all multi-video TARs, and saves the list.

    Args:
        bucket_name: The S3 bucket name.
        all_s3_keys: A list of all TAR S3 keys to check.
        output_file: Path to a text file to save the list of multi-video TAR keys.
    """
    s3_client = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    
    multi_video_keys = []
    total_keys = len(all_s3_keys)
    for i, s3_key in enumerate(all_s3_keys):
        logging.info(f"Checking [{i+1}/{total_keys}]: '{s3_key}'...")
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            tar_bytes = response['Body'].read()
            structure = peer_into_tar(tar_bytes)
            if is_multi_video_tar(structure):
                logging.warning(f"Found multi-video TAR: '{s3_key}' ({structure['mp4_count']} videos).")
                multi_video_keys.append(s3_key)
        except ClientError as e:
            logging.error(f"Could not check '{s3_key}': {e}")

    logging.info(f"Found {len(multi_video_keys)} multi-video TARs.")
    with open(output_file, 'w') as f:
        for key in multi_video_keys:
            f.write(key + '\n')
    logging.info(f"List of multi-video TARs saved to '{output_file}'.")


if __name__ == '__main__':
    # --- Example Usage ---
    # This is an example of how you would use the functions above.
    
    # Prerequisite: You need a 'task_list.txt' from the create_task_list.py script.
    MASTER_TASK_LIST = 'task_list.txt'
    MULTI_VIDEO_LIST_OUTPUT = 'multi_video_tars.txt'
    BUCKET = "game-data"

    if not os.path.exists(MASTER_TASK_LIST):
        logging.error(f"Master task list '{MASTER_TASK_LIST}' not found.")
        logging.error("Please run the 'create_task_list.py' script first.")
        sys.exit(1)

    with open(MASTER_TASK_LIST, 'r') as f:
        all_keys = [line.strip() for line in f if line.strip()]

    # --- STEP 1: Find all the TARs that need splitting ---
    # Uncomment the following line to run the discovery process.
    find_multi_video_tars(BUCKET, all_keys, MULTI_VIDEO_LIST_OUTPUT)

    # --- STEP 2: Process the list of bad TARs ---
    if os.path.exists(MULTI_VIDEO_LIST_OUTPUT):
        with open(MULTI_VIDEO_LIST_OUTPUT, 'r') as f:
            keys_to_split = [line.strip() for line in f if line.strip()]
        
        logging.info(f"Found {len(keys_to_split)} TARs to split from '{MULTI_VIDEO_LIST_OUTPUT}'.")
        
        # Set dry_run=True to see what would happen without making changes.
        # Set dry_run=False to execute the split, upload, and delete operations.
        process_and_split_tars(BUCKET, keys_to_split, dry_run=True)
    else:
        logging.info(f"'{MULTI_VIDEO_LIST_OUTPUT}' not found.")
        logging.info("Run the 'find_multi_video_tars' function first to generate it.")
