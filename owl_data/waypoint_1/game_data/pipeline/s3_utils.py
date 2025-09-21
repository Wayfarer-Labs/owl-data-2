import io
import logging
import tempfile
from typing import Optional
import boto3
from botocore.exceptions import ClientError

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.pt_utils import (
    save_extracted_data_to_pt, 
    load_extracted_data_from_pt,
    convert_s3_key_to_pt_key
)


def upload_extracted_data_to_s3(
    extracted_data: ExtractedData,
    s3_client,
    manifest_bucket: str,
    original_s3_key: str
) -> str:
    """
    Uploads an ExtractedData object to S3 as a .pt file.
    
    Args:
        extracted_data: ExtractedData object to upload
        s3_client: Boto3 S3 client
        manifest_bucket: Name of the manifest bucket
        original_s3_key: Original S3 key from the TAR file
        
    Returns:
        S3 key where the .pt file was uploaded
    """
    pt_s3_key = convert_s3_key_to_pt_key(original_s3_key)
    
    try:
        # Create a temporary file to save the .pt data
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            save_extracted_data_to_pt(extracted_data, temp_file.name)
            temp_file.flush()
            
            # Upload the temporary file to S3
            with open(temp_file.name, 'rb') as pt_file:
                s3_client.put_object(
                    Bucket=manifest_bucket,
                    Key=pt_s3_key,
                    Body=pt_file.read(),
                    ContentType='application/octet-stream'
                )
                
        logging.info(f"Uploaded ExtractedData to s3://{manifest_bucket}/{pt_s3_key}")
        return pt_s3_key
        
    except Exception as e:
        logging.error(f"Failed to upload ExtractedData to S3: {e}")
        raise
    finally:
        # Clean up temporary file
        try:
            import os
            os.unlink(temp_file.name)
        except:
            pass


def download_extracted_data_from_s3(
    s3_client,
    manifest_bucket: str,
    pt_s3_key: str
) -> ExtractedData:
    """
    Downloads and loads an ExtractedData object from S3.
    
    Args:
        s3_client: Boto3 S3 client
        manifest_bucket: Name of the manifest bucket
        pt_s3_key: S3 key of the .pt file
        
    Returns:
        ExtractedData object loaded from S3
    """
    try:
        # Download the .pt file from S3
        response = s3_client.get_object(Bucket=manifest_bucket, Key=pt_s3_key)
        pt_bytes = response['Body'].read()
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            temp_file.write(pt_bytes)
            temp_file.flush()
            
            extracted_data = load_extracted_data_from_pt(temp_file.name)
            
        logging.info(f"Downloaded ExtractedData from s3://{manifest_bucket}/{pt_s3_key}")
        return extracted_data
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logging.warning(f"PT file not found: s3://{manifest_bucket}/{pt_s3_key}")
            raise FileNotFoundError(f"PT file not found: s3://{manifest_bucket}/{pt_s3_key}")
        else:
            logging.error(f"S3 client error downloading {pt_s3_key}: {e}")
            raise
    except Exception as e:
        logging.error(f"Failed to download ExtractedData from S3: {e}")
        raise
    finally:
        # Clean up temporary file
        try:
            import os
            os.unlink(temp_file.name)
        except:
            pass


def check_pt_file_exists(
    s3_client,
    manifest_bucket: str,
    pt_s3_key: str
) -> bool:
    """
    Checks if a .pt file exists in the manifest bucket.
    
    Args:
        s3_client: Boto3 S3 client
        manifest_bucket: Name of the manifest bucket
        pt_s3_key: S3 key of the .pt file
        
    Returns:
        True if the file exists, False otherwise
    """
    try:
        s3_client.head_object(Bucket=manifest_bucket, Key=pt_s3_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Re-raise other errors (permissions, etc.)
            raise


def list_pt_files_in_bucket(
    s3_client,
    manifest_bucket: str,
    prefix: str = ""
) -> list[str]:
    """
    Lists all .pt files in the manifest bucket with optional prefix filter.
    
    Args:
        s3_client: Boto3 S3 client
        manifest_bucket: Name of the manifest bucket
        prefix: Optional prefix to filter results
        
    Returns:
        List of S3 keys for .pt files
    """
    try:
        pt_files = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=manifest_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.pt'):
                        pt_files.append(key)
        
        logging.info(f"Found {len(pt_files)} .pt files in s3://{manifest_bucket}/{prefix}")
        return pt_files
        
    except Exception as e:
        logging.error(f"Failed to list .pt files in S3: {e}")
        raise


def get_missing_pt_files(
    s3_client,
    source_bucket: str,
    manifest_bucket: str,
    tar_s3_keys: list[str]
) -> list[str]:
    """
    Returns a list of TAR S3 keys that don't have corresponding .pt files in the manifest bucket.
    
    Args:
        s3_client: Boto3 S3 client
        source_bucket: Name of the source bucket containing TAR files
        manifest_bucket: Name of the manifest bucket containing .pt files
        tar_s3_keys: List of TAR S3 keys to check
        
    Returns:
        List of TAR S3 keys that need to be processed (missing .pt files)
    """
    missing_tars = []
    
    for tar_key in tar_s3_keys:
        pt_key = convert_s3_key_to_pt_key(tar_key)
        
        if not check_pt_file_exists(s3_client, manifest_bucket, pt_key):
            missing_tars.append(tar_key)
    
    logging.info(f"Found {len(missing_tars)} TAR files missing corresponding .pt files")
    return missing_tars 