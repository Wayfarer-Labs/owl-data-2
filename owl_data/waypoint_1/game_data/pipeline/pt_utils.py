import torch
import json
import logging
from typing import Optional
from pathlib import Path

from owl_data.waypoint_1.game_data.pipeline.owl_types import ExtractedData


def serialize_extracted_data(data: ExtractedData) -> dict:
    """
    Serializes an ExtractedData object into a dictionary suitable for saving with torch.save.
    
    Args:
        data: ExtractedData object to serialize
        
    Returns:
        Dictionary containing all the data from ExtractedData object
    """
    return {
        's3_key': data.s3_key,
        'video_id': data.video_id,
        'video_metadata': data.video_metadata,
        'session_metadata': data.session_metadata,
        'sampled_frames': data.sampled_frames,
        # Add metadata about serialization
        '_serialization_version': '1.0',
        '_data_type': 'ExtractedData'
    }


def deserialize_extracted_data(serialized_dict: dict) -> ExtractedData:
    """
    Deserializes a dictionary back into an ExtractedData object.
    
    Args:
        serialized_dict: Dictionary containing serialized ExtractedData
        
    Returns:
        ExtractedData object reconstructed from the dictionary
    """
    # Validate serialization format
    if serialized_dict.get('_data_type') != 'ExtractedData':
        raise ValueError(f"Invalid data type: {serialized_dict.get('_data_type')}")
    
    return ExtractedData(
        s3_key=serialized_dict['s3_key'],
        video_id=serialized_dict['video_id'],
        video_metadata=serialized_dict['video_metadata'],
        session_metadata=serialized_dict['session_metadata'],
        sampled_frames=serialized_dict['sampled_frames']
    )


def save_extracted_data_to_pt(data: ExtractedData, pt_path: str) -> None:
    """
    Saves an ExtractedData object to a .pt file.
    
    Args:
        data: ExtractedData object to save
        pt_path: Path where to save the .pt file
    """
    serialized_data = serialize_extracted_data(data)
    
    # Ensure directory exists
    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(serialized_data, pt_path)
        logging.info(f"Saved ExtractedData to {pt_path}")
    except Exception as e:
        logging.error(f"Failed to save ExtractedData to {pt_path}: {e}")
        raise


def load_extracted_data_from_pt(pt_path: str) -> ExtractedData:
    """
    Loads an ExtractedData object from a .pt file.
    
    Args:
        pt_path: Path to the .pt file to load
        
    Returns:
        ExtractedData object loaded from the file
    """
    try:
        serialized_data = torch.load(pt_path, map_location='cpu', weights_only=False)
        return deserialize_extracted_data(serialized_data)
    except Exception as e:
        logging.error(f"Failed to load ExtractedData from {pt_path}: {e}")
        raise


def convert_s3_key_to_pt_key(s3_key: str, bucket_suffix: str = "-manifest") -> str:
    """
    Converts a TAR S3 key to a corresponding .pt S3 key for the manifest bucket.
    
    Args:
        s3_key: Original S3 key (e.g., "path/to/file.tar")
        bucket_suffix: Suffix to add to bucket name (default: "-manifest")
        
    Returns:
        Converted S3 key (e.g., "path/to/file.pt")
    """ 
    if s3_key.endswith('.tar'):
        return s3_key[:-4] + '.pt'
    else:
        return s3_key + '.pt'


def convert_pt_key_to_s3_key(pt_key: str) -> str:
    """
    Converts a .pt S3 key back to the original TAR S3 key.
    
    Args:
        pt_key: PT S3 key (e.g., "path/to/file.pt")
        
    Returns:
        Original S3 key (e.g., "path/to/file.tar")
    """
    if pt_key.endswith('.pt'):
        return pt_key[:-3] + '.tar'
    else:
        return pt_key + '.tar' 