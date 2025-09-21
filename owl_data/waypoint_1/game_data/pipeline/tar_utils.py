import os, gc, io, json, tarfile, logging, ffmpeg, boto3, traceback
from collections import defaultdict

from owl_data.waypoint_1.game_data.pipeline.owl_types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.mp4_utils import sample_frames_from_bytes

def _process_single_video_tar(tar: tarfile.TarFile, s3_key: str) -> ExtractedData:
    """Processes a TAR file containing one video with descriptively named files."""
    files = {}
    for member in tar.getmembers():
        if member.isfile():
            if member.name.endswith('.mp4'):
                files['mp4'] = tar.extractfile(member).read()
                files['video_name'] = member.name
            elif member.name.endswith('.json'):
                files['json'] = tar.extractfile(member).read()
            elif member.name.endswith('.csv'):
                files['csv'] = tar.extractfile(member).read()

    if not ('mp4' in files and 'json' in files):
        raise Exception(f"Skipping single-video TAR '{s3_key}': missing mp4={'mp4' in files} or metadata.json={'json' in files}.")
    
    try:
        video_bytes = files['mp4']
        # -- save a tmp mp4
        tmp_dir = "/tmp"
        tmp_mp4_path = os.path.join(tmp_dir, s3_key, os.path.basename(files['video_name']))
        os.makedirs(os.path.dirname(tmp_mp4_path), exist_ok=True)
        with open(tmp_mp4_path, 'wb') as f:
            f.write(video_bytes)

        video_id = os.path.splitext(os.path.basename(files['video_name']))[0]
        session_metadata = json.loads(files['json'])
        video_metadata = ffmpeg.probe(tmp_mp4_path)
        
        strides_spec = {3: 15, 30: 15, 60: 5}
        sampled_frames = sample_frames_from_bytes(video_bytes, strides_spec)
        
        del video_bytes
        gc.collect()

        return ExtractedData(
            s3_key=s3_key,
            video_id=video_id,
            video_metadata=video_metadata,
            session_metadata=session_metadata,
            sampled_frames=sampled_frames,
        )
    except Exception as e:
        logging.error(f"Failed to process single-video TAR '{s3_key}'. Error: {e} with traceback: {traceback.format_exc()}")
        raise e


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
                video_metadata = ffmpeg.probe(io.BytesIO(video_bytes))
                
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
                logging.error(f"Failed to process group '{video_id}' in TAR '{s3_key}'. Error: {e} with traceback: {traceback.format_exc()}")
        else:
            logging.warning(f"Skipping group '{video_id}' in TAR '{s3_key}': missing mp4 or json file.")
            
    return results

def extract_and_sample(tar_bytes: bytes, s3_key: str) -> ExtractedData:
    """
    Detects the TAR format and extracts data accordingly.

    This function acts as a dispatcher. It checks for the presence of
    'metadata.json' to decide whether to process the TAR as a single-video
    or multi-video archive.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as tar:
        member_names = [m.name for m in tar.getmembers()]
        
        # --- Detection Logic ---
        if any(member_name.endswith('.json') for member_name in member_names):
            logging.info(f"Detected 'Single-Video' format for TAR '{s3_key}'.")
            return _process_single_video_tar(tar, s3_key)
        else:
            logging.error(f"Detected 'Multi-Video' format for TAR '{s3_key}' with members: {member_names} and {len(member_names)} members.")
            raise Exception(f"TAR does not conform to single-video format. {member_names}")

if __name__ == "__main__":
    import boto3
    from dotenv import load_dotenv
    load_dotenv()

    task_list = "task_list.txt"
    num_samples = 10
    with open(task_list, 'r') as f:
        s3_keys = [line.strip() for line in f if line.strip()]
    s3_keys = s3_keys[:num_samples]

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )

    for s3_key in s3_keys:
        logging.info(f"Processing TAR '{s3_key}'")
        response = s3_client.get_object(Bucket='game-data', Key=s3_key)
        tar_bytes = response['Body'].read()
        result = extract_and_sample(tar_bytes, s3_key)