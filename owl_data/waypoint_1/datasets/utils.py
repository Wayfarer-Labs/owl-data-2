from pathlib import Path
import torch
import numpy as np
import av
import os
from typing import Generator
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



def _to_360p_fn(frame_chw: torch.Tensor) -> callable:
    # Define transforms: center crop to 16:9 then resize to 640x360
    original_width = frame_chw.shape[2]
    original_height = frame_chw.shape[1]
    
    target_aspect = 640 / 360  # 16:9
    original_aspect = original_width / original_height
    
    if original_aspect > target_aspect:
        # Original is wider - crop width
        crop_width = int(original_height * target_aspect)
        crop_height = original_height
    else:
        # Original is taller - crop height  
        crop_width = original_width
        crop_height = int(original_width / target_aspect)
    
    # Update transform with calculated crop size
    transform = T.Compose([
        T.CenterCrop(size=(crop_height, crop_width)),
        T.Resize(size=(360, 640), antialias=True)
    ])
    return transform

is_mp4 = lambda path: isinstance(path, Path) and path.suffix == '.mp4'
is_hevc = lambda path: isinstance(path, Path) and path.suffix == '.hevc'

def process_video_seek(
    path: is_mp4,
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
                    transform = transform or _to_360p_fn(torch_frame)
                    buf.append(transform(torch_frame))
                    print(f'appended frame at ts {ts}')
                    if len(buf) == chunk_size:
                        yield torch.stack(buf); buf = []
                    break  # go seek to next timestamp

            t += stride_sec

        if buf:
            yield torch.stack(buf)


def process_video_seek(
    path: is_hevc,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM
) -> Generator[torch.Tensor, None, None]:
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        print(f"Cannot open video file: {path}")
        return
    
    transform_fn = _to_360p_fn(torch.zeros(
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

