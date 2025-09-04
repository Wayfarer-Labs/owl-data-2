import os
import tqdm
import zipfile
import subprocess
from pathlib import Path
import cv2

import torchvision.transforms as T
import torch
import torchcodec
from typing import Generator

from owl_data.waypoint_1.datasets.globals import (
    COMMA_2K19_ZIP_DIR,
    COMMA_2K19_ZIP_OUT_DIR,
    COMMA_2K19_NORMALIZED_360_DIR,
    CHUNK_FRAME_NUM,
    STRIDE_SEC
)

_unzip = lambda src, dst: subprocess.run(['sudo', 'unzip', str(src.absolute()), '-d', str(dst.absolute())])


def unzip_chunks(
    zip_dir: Path = COMMA_2K19_ZIP_DIR,
    output_dir: Path = COMMA_2K19_ZIP_OUT_DIR,
    exclude: list[str] = []
):
    for zip in tqdm.tqdm(zip_dir.glob('Chunk_*.zip'), desc='Unzipping chunks'):
        if zip.name in exclude: continue
        _unzip(zip, output_dir)


def _to_360p_fn(cap: cv2.VideoCapture) -> callable:
    # Define transforms: center crop to 16:9 then resize to 640x360
    transform = T.Compose([
        T.CenterCrop(size=(874, 1164)),  # This will be calculated dynamically
        T.Resize(size=(360, 640), antialias=True)
    ])
    
    # Get original dimensions and calculate crop size for 16:9 aspect ratio
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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


def hevc_to_chunks(hevc_path: Path, stride_sec: int = STRIDE_SEC) -> Generator[torch.Tensor, None, None]:
    """
    Resize to standard 360p (640x360) using center crop + resize
    """
    cap = cv2.VideoCapture(str(hevc_path))
    
    if not cap.isOpened():
        print(f"Cannot open video file: {hevc_path}")
        return
    
    transform_fn = _to_360p_fn(cap)
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

        if len(frames_buffer) == CHUNK_FRAME_NUM:
            yield torch.stack(frames_buffer)
            frames_buffer = []
    
    if frames_buffer:
        yield torch.stack(frames_buffer)
    
    cap.release()
    print(f"Processed {frame_count} frames -> output shape: [t, 3, 360, 640]")


def process_dongle_dir(dongle_dir: Path, out_dir: Path) -> None:
    # Takes a dongle dir such as 'b0c9d2329ad1606b|2018-07-31--20-50-28' and processes
    #  the hevc from subdir into .pt files in the out_dir:
    chunk_name = dongle_dir.parent.name
    for subdir in tqdm.tqdm(
        dongle_dir.glob('*'),
        desc=f'Processing dongle data from {dongle_dir.name}'
    ):
        for i, hevc_chunk in tqdm.tqdm(
            enumerate(hevc_to_chunks(subdir / 'video.hevc', stride_sec = STRIDE_SEC)),
            desc=f'Processing {subdir.name}'
        ):
            # copy the directory structure to the output dir
            normalized_dir = out_dir / chunk_name / dongle_dir.name / subdir.name
            normalized_dir.mkdir(parents=True, exist_ok=True)
            torch.save(hevc_chunk, normalized_dir / f'{i:08d}_rgb.pt')


if __name__ == '__main__':
    for dongle_dir in COMMA_2K19_ZIP_OUT_DIR.glob('Chunk_*/*'):
        process_dongle_dir(dongle_dir, COMMA_2K19_NORMALIZED_360_DIR)
