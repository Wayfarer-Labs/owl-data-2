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

def process_video_seek(
    path: Path,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM,
) -> Generator[torch.Tensor, None, None]:
    match path.suffix.lower():
        case '.mp4' | '.webm': yield from process_video_seek_mp4_or_webm(path, stride_sec, chunk_size)
        case '.hevc': yield from process_video_seek_hevc(path, stride_sec, chunk_size)
        case _: raise NotImplementedError(f"Unsupported extension: {path.suffix}")


def peer_chunks(
    path: Path,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM,
) -> list[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return list(range(0, int(stream.duration * stream.time_base), stride_sec * stream.time_base))


def process_video_seek_mp4_or_webm(
    path: Path,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM,
) -> Generator[torch.Tensor, None, None]:
    """
    Seek-decoding sampler:
      - Seeks in *stream time_base* ticks (correct for stream=...).
      - Clamps to the actual last frame timestamp (prevents tail loops).
      - Breaks cleanly when no frame >= target time is found after a seek.
      - Dedupes repeats near EOF and emits fixed-size chunks.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]

        # Threading helps a lot on CPU decodes
        try:
            cc = stream.codec_context
            cc.thread_type = "FRAME"  # or "SLICE" depending on codec
            cc.thread_count = max(1, os.cpu_count() or 1)
        except Exception:
            pass

        # --- Helpers for timebases ---
        tick = float(stream.time_base)                   # seconds per tick, e.g., 1/90000
        def sec_to_pts(t: float) -> int:                 # seconds -> stream ticks
            return int(round(t / tick))

        # --- Establish duration_s (metadata) ---
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = float(container.duration) / av.time_base  # av.time_base = 1_000_000
        else:
            duration_s = float("inf")  # will be clamped below

        # --- Find actual last frame timestamp (ts_last) once; clamp loop end ---
        # Seek to "very end", decode forward to get last decodable frame
        try:
            container.seek(2**63 - 1, stream=stream, any_frame=True, backward=True)
        except OverflowError:
            # Fallback for platforms with smaller int
            container.seek(9223372036854775807, stream=stream, any_frame=True, backward=True)

        ts_last = None
        for f in container.decode(stream):
            if f.pts is not None:
                ts_last = float(f.pts * tick)
        if ts_last is None:
            # Fall back to metadata if no frame decoded at tail
            ts_last = duration_s if duration_s != float("inf") else 0.0

        # Clamp the nominal duration to actual last frame
        duration_s = min(duration_s, ts_last)

        # Tolerance: a few ticks is safer than a fixed microsecond epsilon
        eps = 3 * tick

        # --- Main stride/seek loop ---
        transform = None
        buf: list[torch.Tensor] = []
        last_ts_emitted: float | None = None

        t = 0.0
        while t <= duration_s + eps:
            # Seek to nearest keyframe at or before target in stream ticks
            target_pts = sec_to_pts(t)
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)

            hit = False
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = float(frame.pts * tick)

                # If we've reached/passed the requested target (with tolerance), we can take this frame
                if ts + eps >= t:
                    # Dedupe if a repeated last frame is being surfaced near EOF or due to GOP structure
                    if last_ts_emitted is None or ts > last_ts_emitted + 1e-12:
                        torch_frame = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)
                        if transform is None:
                            transform = _to_16_9_crop_fn(torch_frame)  # your provided function
                        buf.append(transform(torch_frame))
                        print(f"appended frame at ts: {ts}, t: {t}, eps: {eps}")
                        last_ts_emitted = ts

                        if len(buf) == chunk_size:
                            yield torch.stack(buf)
                            buf = []
                    hit = True
                    break  # move on to next stride target

            if not hit:
                # No frame >= t after seek+decode -> we've reached/passed EOF
                break

            t += stride_sec

        if buf:
            yield torch.stack(buf)


def process_video_seek_hevc(
    path: Path,
    stride_sec: float = 5.0,
    chunk_size: int = CHUNK_FRAME_NUM
) -> Generator[torch.Tensor, None, None]:
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        print(f"Cannot open video file: {path}")
        return
    
    transform_fn = _to_16_9_crop_fn(torch.zeros(
        3,
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ))
    
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

