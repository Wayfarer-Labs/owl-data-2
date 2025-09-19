import os
import tarfile
import json
import gc
import av
import io
from collections import defaultdict
import logging

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.mp4_utils import sample_frames_from_bytes

def _process_single_video_tar(tar: tarfile.TarFile, s3_key: str) -> list[ExtractedData]:
    """Processes a TAR file containing one video with descriptively named files."""
    results = []
    files = {}
    for member in tar.getmembers():
        if member.isfile():
            if member.name.endswith('.mp4'):
                files['mp4'] = tar.extractfile(member).read()
                files['video_name'] = member.name
            elif member.name == 'metadata.json':
                files['json'] = tar.extractfile(member).read()
            # Add other specific files like inputs.csv if needed

    if 'mp4' in files and 'json' in files:
        try:
            video_bytes = files['mp4']
            video_id = os.path.splitext(os.path.basename(files['video_name']))[0]
            session_metadata = json.loads(files['json'])
            video_metadata = av.probe(io.BytesIO(video_bytes))
            
            strides_spec = {3: 15, 30: 15, 60: 5}
            sampled_frames = sample_frames_from_bytes(video_bytes, strides_spec)
            
            del video_bytes
            gc.collect()

            results.append(ExtractedData(
                s3_key=s3_key,
                video_id=video_id,
                video_metadata=video_metadata,
                session_metadata=session_metadata,
                sampled_frames=sampled_frames
            ))
        except Exception as e:
            logging.error(f"Failed to process single-video TAR '{s3_key}'. Error: {e}")
    else:
        logging.warning(f"Skipping single-video TAR '{s3_key}': missing mp4 or metadata.json.")
        
    return results


def _process_multi_video_tar(tar: tarfile.TarFile, s3_key: str) -> list[ExtractedData]:
    """Processes a TAR file containing multiple videos grouped by a numerical base name."""
    results = []
    file_groups = defaultdict(dict)
    
    for member in tar.getmembers():
        if member.isfile():
            base_name, extension = os.path.splitext(os.path.basename(member.name))
            extension = extension.lower().strip('.')
            if extension in ['mp4', 'json', 'csv']:
                file_groups[base_name][extension] = tar.extractfile(member).read()

    for video_id, files in file_groups.items():
        if 'mp4' in files and 'json' in files:
            try:
                video_bytes = files['mp4']
                session_metadata = json.loads(files['json'])
                video_metadata = av.probe(io.BytesIO(video_bytes))
                
                strides_spec = {3: 15, 30: 15, 60: 5}
                sampled_frames = sample_frames_from_bytes(video_bytes, strides_spec)
                
                del video_bytes
                gc.collect()

                results.append(ExtractedData(
                    s3_key=f"{s3_key}/{video_id}",
                    video_id=video_id,
                    video_metadata=video_metadata,
                    session_metadata=session_metadata,
                    sampled_frames=sampled_frames
                ))
            except Exception as e:
                logging.error(f"Failed to process group '{video_id}' in TAR '{s3_key}'. Error: {e}")
        else:
            logging.warning(f"Skipping group '{video_id}' in TAR '{s3_key}': missing mp4 or json file.")
            
    return results

def extract_and_sample(tar_bytes: bytes, s3_key: str) -> list[ExtractedData]:
    """
    Detects the TAR format and extracts data accordingly.

    This function acts as a dispatcher. It checks for the presence of
    'metadata.json' to decide whether to process the TAR as a single-video
    or multi-video archive.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as tar:
        member_names = [m.name for m in tar.getmembers()]
        
        # --- Detection Logic ---
        if 'metadata.json' in member_names:
            logging.info(f"Detected 'Single-Video' format for TAR '{s3_key}'.")
            return _process_single_video_tar(tar, s3_key)
        else:
            logging.info(f"Detected 'Multi-Video' format for TAR '{s3_key}'.")
            return _process_multi_video_tar(tar, s3_key)


def main():
    import sys
    video_path = "/home/sky/owl-data-2/owl_data/waypoint_1/game_data/tars_by_size/205_500MB/4e3e5cbc95e64420/000000.mp4"

    try:
        logging.info(f"Reading video file from: {video_path}")
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
    except FileNotFoundError:
        logging.error(f"Error: The file was not found at '{video_path}'")
        logging.error("Please update the 'video_path' variable in this script.")
        sys.exit(1)

    # --- Define the desired strides and frame counts here ---
    # Key = stride in seconds, Value = number of frames to sample
    strides_to_sample = {
        3: 15,  # Sample 15 frames, 3 seconds apart
        30: 15, # Sample 15 frames, 30 seconds apart
        60: 5   # Sample 5 frames, 60 seconds apart
    }

    # Run the frame sampling function with the new specification
    sampled_frames = sample_frames_from_bytes(video_bytes, strides_to_sample)

    # Print the results to verify
    print("\n--- Frame Sampling Results ---")
    for name, array in sampled_frames.items():
        if array.size > 0:
            # Shape is (num_frames, channels, height, width)
            print(f"'{name}': Found {array.shape[0]} frames with shape {array.shape}")
        else:
            print(f"'{name}': No frames were sampled (video might be too short for this stride).")
    print("----------------------------")


if __name__ == "__main__":
    main()
