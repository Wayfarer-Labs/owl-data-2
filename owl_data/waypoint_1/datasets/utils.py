from pathlib import Path
import torch
import numpy as np
import av
import os
from typing import Generator
from multimethod import multimethod, overload
import torchvision.transforms as T
import cv2

DIMENSION_ORDER = 'NCHW'
STRIDE_SEC = 5


CHUNK_FRAME_NUM = 512
COMMA_2K19_ZIP_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/comma2k19')
COMMA_2K19_ZIP_OUT_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/processed')
COMMA_2K19_NORMALIZED_360_DIR = Path('/mnt/data/waypoint_1/normalized360/comma2k19')

MKIF_DIR = Path('/mnt/data/waypoint_1/datasets/MKIF/videos')
MKIF_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/mkif')

EGOEXPLORE_DIR = Path('/mnt/data/waypoint_1/datasets/egoexplore/videos')
EGOEXPLORE_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/egoexplore/videos')

EPIC_KITCHENS_100_DIR = Path('/mnt/data/waypoint_1/datasets/epic_kitchens_100/2g1n6qdydwa9u22shpxqzp0t8m')
EPIC_KITCHENS_100_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/epic_kitchens_100')

KINETICS_700_DIR = Path('/mnt/data/waypoint_1/datasets/kinetics700/Kinetics-700/')
KINETICS_700_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/kinetics700/Kinetics-700')


def _to_16_9_crop_fn(frame_chw: torch.Tensor) -> callable:
    """
    Creates a transform that crops to 16:9 aspect ratio.
    For standard resolutions (360p, 720p, 1080p, etc.), preserves original resolution.
    For weird aspect ratios, resizes then crops to 360p (640x360).
    """
    original_width = frame_chw.shape[2]
    original_height = frame_chw.shape[1]
    
    target_aspect = 16 / 9
    original_aspect = original_width / original_height
    
    # Define standard resolutions we want to preserve
    standard_resolutions = [
        (640, 360),   # 360p
        (854, 480),   # 480p
        (1280, 720),  # 720p
        (1920, 1080), # 1080p
        (2560, 1440), # 1440p
        (3840, 2160), # 4K
    ]
    
    # Check if this is close to a standard resolution (within 10% tolerance)
    is_standard_resolution = False
    for std_width, std_height in standard_resolutions:
        width_ratio = min(original_width, std_width) / max(original_width, std_width)
        height_ratio = min(original_height, std_height) / max(original_height, std_height)
        if width_ratio > 0.9 and height_ratio > 0.9:
            is_standard_resolution = True
            break
    
    if is_standard_resolution:
        # Standard resolution: only crop to 16:9, preserve resolution
        if original_aspect > target_aspect:
            # Original is wider - crop width to achieve 16:9
            crop_width = int(original_height * target_aspect)
            crop_height = original_height
        else:
            # Original is taller - crop height to achieve 16:9
            crop_width = original_width
            crop_height = int(original_width / target_aspect)
        
        transform = T.Compose([
            T.CenterCrop(size=(crop_height, crop_width))
        ])
    else:
        # Weird aspect ratio: resize then crop to 360p
        # First resize to make one dimension fit 360p, then crop to 16:9
        if original_aspect > target_aspect:
            # Wider than 16:9 - fit height to 360, then crop width
            resize_height = 360
            resize_width = int(360 * original_aspect)
        else:
            # Taller than 16:9 - fit width to 640, then crop height  
            resize_width = 640
            resize_height = int(640 / original_aspect)
        
        transform = T.Compose([
            T.Resize(size=(resize_height, resize_width), antialias=True),
            T.CenterCrop(size=(360, 640))
        ])
    
    return transform


is_mp4 = lambda path: isinstance(path, Path) and path.suffix == '.mp4'
is_hevc = lambda path: isinstance(path, Path) and path.suffix == '.hevc'
is_webm = lambda path: isinstance(path, Path) and path.suffix == '.webm'


@overload
def process_video_seek(
    path: is_mp4 | is_webm,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM
) -> Generator[torch.Tensor, None, None]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]

        # decoder threading helps a lot on CPU
        cc = stream.codec_context
        try:
            cc.thread_type = "FRAME"   # or "SLICE" depending on codec
            cc.thread_count = os.cpu_count()
        except Exception:
            pass

        # duration in seconds
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        else: duration_s = float(container.duration) / av.time_base  # fallback

        transform = None
        buf = []
        t = 0.0

        while t < duration_s:
            # Fast seek to nearest keyframe before t
            container.seek(int(t * av.time_base), stream=stream, any_frame=False, backward=True)

            # Decode forward until we reach t (or next frame past it)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                if ts + 1e-6 >= t:  # reached our target
                    torch_frame = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                    transform = transform or _to_16_9_crop_fn(torch_frame)
                    buf.append(transform(torch_frame))
                    print(f'appended frame at ts {ts}')
                    if len(buf) == chunk_size:
                        yield torch.stack(buf); buf = []
                    break  # go seek to next timestamp

            t += stride_sec

        if buf:
            yield torch.stack(buf)


@overload
def process_video_seek(
    path: is_hevc,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM
) -> Generator[torch.Tensor, None, None]:
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        print(f"Cannot open video file: {path}")
        return
    
    transform_fn = _to_16_9_crop_fn(torch.zeros(
        3,
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    )
    
    frames_buffer = []
    frame_count = 0
    stride_frames = int(stride_sec * cap.get(cv2.CAP_PROP_FPS))
    frame_idx = -1
    while True:
        ret, frame = cap.read() ; frame_idx += 1
        if not ret: break
        if frame_idx % stride_frames != 0:
            continue

        # convert BGR to RGB and to tensor [H, W, C] -> [C, H, W]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().to(torch.uint8)
        # apply transforms
        frame_processed = transform_fn(frame_tensor)
        frames_buffer.append(frame_processed)
        frame_count += 1

        if len(frames_buffer) == chunk_size:
            yield torch.stack(frames_buffer)
            frames_buffer = []
    
    if frames_buffer:
        yield torch.stack(frames_buffer)
    
    cap.release()
    print(f"Processed {frame_count} frames -> output shape: [t, 3, 360, 640]")

