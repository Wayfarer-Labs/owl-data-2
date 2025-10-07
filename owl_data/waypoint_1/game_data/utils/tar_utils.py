import os, io, json, tarfile, logging, ffmpeg, boto3, traceback, tempfile, pathlib

from owl_data.waypoint_1.game_data.owl_types import ExtractedData
from owl_data.waypoint_1.game_data.utils.mp4_utils import downsample_video_from_path


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
        session_metadata = json.loads(files['json'])

        with tempfile.TemporaryDirectory(prefix="owl_tar_extract_") as tmpdir:
            # Extract full TAR contents to temp directory
            tar.extractall(path=tmpdir)

            # Resolve mp4 path inside the extracted dir
            mp4_rel = files.get('video_name')
            mp4_path = pathlib.Path(tmpdir) / mp4_rel if mp4_rel else None
            if not mp4_path or not mp4_path.exists():
                # Fallback: find the first .mp4 in extracted tree
                for root, _, fnames in os.walk(tmpdir):
                    for fn in fnames:
                        if fn.lower().endswith(".mp4"):
                            mp4_path = pathlib.Path(root) / fn
                            break
                    if mp4_path and mp4_path.exists():
                        break
            if not mp4_path or not mp4_path.exists():
                raise FileNotFoundError(f"Extracted mp4 not found in TAR '{s3_key}'")

            # Probe original video via file path
            video_metadata = ffmpeg.probe(str(mp4_path))

            # Downsample from extracted path
            downsampled_paths = downsample_video_from_path(mp4_path)

            # Read downsampled files as bytes and probe from bytes (keeps current behavior)
            downsampled_video_bytes = []
            downsampled_video_metadata = []
            for downsampled_path in downsampled_paths:
                mp4_bytes = io.BytesIO(downsampled_path.read_bytes())
                downsampled_video_bytes.append(mp4_bytes)
                downsampled_video_metadata.append(ffmpeg.probe(downsampled_path))

        # Build output object
        data = ExtractedData(
            s3_key=s3_key,
            in_video_metadata=video_metadata,
            out_video_metadata=downsampled_video_metadata,
            session_metadata=session_metadata,
            downsampled_video_bytes=downsampled_video_bytes,
            controls_csv_str=str(files['csv'])
        )

        # Cleanup downsampled files and their directory
        for downsampled_path in downsampled_paths:
            try:
                downsampled_path.unlink(missing_ok=True)
            except Exception:
                logging.warning(f"Failed to remove {downsampled_path}")
        for p in downsampled_paths:
            try:
                p.parent.rmdir()
            except Exception:
                pass
        logging.info(f"Removed downsampled paths for {s3_key}")

        return data

    except Exception as e:
        logging.error(f"Failed to process single-video TAR '{s3_key}'. Error: {e} with traceback: {traceback.format_exc()}")
        raise e

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
    num_samples = 1
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