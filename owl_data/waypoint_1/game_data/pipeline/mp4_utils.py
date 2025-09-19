import io
import os
import tarfile
import json
import gc
import av
from collections import defaultdict
import logging
import numpy as np
from typing import Optional

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData


def _sample_frames_at_stride(
    container: av.container.Container,
    stride_seconds: int,
    num_frames: int,
    resize_dims: Optional[tuple[int, int]]
) -> np.ndarray:
    """Helper to perform the actual frame sampling for a given stride."""
    frames = []
    stream = container.streams.video[0]
    
    total_duration_seconds = stream.duration * stream.time_base
    
    # Reset stream to the beginning for each stride sampling pass
    container.seek(0)

    for i in range(num_frames):
        target_time_seconds = (i + 1) * stride_seconds
        
        if target_time_seconds > total_duration_seconds:
            logging.warning(
                f"Cannot sample at {target_time_seconds}s for stride {stride_seconds}s; video duration is only {total_duration_seconds:.2f}s."
            )
            break
            
        try:
            container.seek(int(target_time_seconds * 1_000_000), backward=True, any_frame=False, stream=stream)
            
            for frame in container.decode(video=0):
                if frame.pts * stream.time_base >= target_time_seconds:
                    frame_np = frame.to_ndarray(format='rgb24')
                    
                    if resize_dims:
                        resized_frame = frame.reformat(width=resize_dims[0], height=resize_dims[1], format='rgb24')
                        frame_np = resized_frame.to_ndarray(format='rgb24')
                        
                    frame_chw = frame_np.transpose(2, 0, 1)
                    frames.append(frame_chw)
                    break 
        except (av.AVError, StopIteration) as e:
            logging.error(f"Error while seeking/decoding at {target_time_seconds}s for stride {stride_seconds}s: {e}")
            break
            
    if not frames:
        return np.array([])
        
    return np.stack(frames)


def sample_frames_from_bytes(
    video_bytes: bytes,
    strides_spec: dict[int, int]
) -> dict[str, np.ndarray]:
    """
    Decodes an in-memory video and samples frames based on a given specification.

    This function resizes frames to a maximum height of 360p while maintaining
    aspect ratio. It returns a dictionary of NumPy arrays in CHW format.

    Args:
        video_bytes: The raw bytes of the video file.
        strides_spec: A dictionary specifying the sampling.
            - key (int): The stride in seconds between samples.
            - value (int): The total number of frames to sample for this stride.
            Example: {3: 15, 30: 15} will sample 15 frames at a 3s stride
                     and 15 frames at a 30s stride.

    Returns:
        A dictionary mapping stride information to a NumPy array of frames.
        e.g., {"stride-3_chw": array, "stride-30_chw": array}
    """
    sampled_data = {}
    
    try:
        # Use a BytesIO object to allow seeking, which is required for sampling multiple strides
        video_stream = io.BytesIO(video_bytes)
        with av.open(video_stream) as container:
            stream = container.streams.video[0]
            
            resize_dims = None
            if stream.height > 360:
                scale = 360 / stream.height
                new_width = int(stream.width * scale)
                resize_dims = (new_width, 360)

            # --- Loop through the user-defined strides ---
            for stride_seconds, num_frames in strides_spec.items():
                logging.info(f"Sampling {num_frames} frames with {stride_seconds}-second stride...")
                
                frames = _sample_frames_at_stride(
                    container=container,
                    stride_seconds=stride_seconds,
                    num_frames=num_frames,
                    resize_dims=resize_dims
                )
                
                key = f"stride-{stride_seconds}_chw"
                sampled_data[key] = frames

    except Exception as e:
        logging.error(f"Failed to open or process video bytes with PyAV: {e}")
        # On failure, populate keys with empty arrays
        for stride_seconds in strides_spec:
            key = f"stride-{stride_seconds}_chw"
            sampled_data[key] = np.array([])

    return sampled_data
