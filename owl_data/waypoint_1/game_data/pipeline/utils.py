import io
import gc
import json
import logging
import tarfile
import numpy as np

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData


def _sample_frames_from_bytes(video_bytes: bytes) -> dict[str, np.ndarray]:
    """
    Helper function to decode and sample frames from in-memory video data.
    Replace this with your actual OpenCV or Decord implementation.
    """
    logging.debug("Decoding video and sampling frames...")
    # Placeholder logic
    sampled_data = {
        "stride-3_chw": np.random.rand(15, 3, 360, 640),
        "stride-30_chw": np.random.rand(15, 3, 360, 640),
    }
    logging.debug("Frame sampling complete.")
    return sampled_data


def _extract_and_sample(tar_bytes: bytes, s3_key: str) -> ExtractedData:
    """
    Extracts data from an in-memory TAR, samples video frames,
    and immediately discards the large video file to free RAM.
    """
    video_bytes = None
    metadata_dict = None

    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if not f: continue
            content = f.read()
            if member.name.endswith('.mp4'):
                logging.debug(f"Extracting {member.name} into memory...")
                video_bytes = content
            elif member.name.endswith('.json'):
                metadata_dict = json.loads(content.decode('utf-8'))

    if not video_bytes or not metadata_dict:
        raise ValueError(f"TAR file {s3_key} is missing required .mp4 or .json file.")

    sampled_frames = _sample_frames_from_bytes(video_bytes)

    logging.debug("Deleting in-memory MP4 file to free RAM.")
    del video_bytes
    gc.collect()

    return ExtractedData(
        s3_key=s3_key,
        metadata=metadata_dict,
        sampled_frames=sampled_frames
    )
