from pathlib import Path
from typing import Generator

import torch
import av
from owl_data.waypoint_1.datasets.globals import STRIDE_SEC, MKIF_DIR, MKIF_OUT_DIR, CHUNK_FRAME_NUM
import random
import torchvision.transforms as T
import multiprocessing

def _to_360p_fn(frame_chw: torch.Tensor) -> callable:
    # Define transforms: center crop to 16:9 then resize to 640x360
    transform = T.Compose([
        T.CenterCrop(size=(874, 1164)),  # This will be calculated dynamically
        T.Resize(size=(360, 640), antialias=True)
    ])
    
    # Get original dimensions and calculate crop size for 16:9 aspect ratio
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


def process_webm(webm_path: Path, stride_sec: int = 5, chunk_size: int = CHUNK_FRAME_NUM) -> Generator[torch.Tensor, None, None]:
    container = av.open(str(webm_path))
    stride_num_frames = int(stride_sec * container.streams.video[0].average_rate)
    stream = container.streams.video[0]
    frame_buffer = []
    transform = None
    for i, frame in enumerate(container.decode(stream)):
        if i % stride_num_frames != 0: continue
        torch_frame = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2,0,1) # hwc to chw
        
        if transform is None:
            transform = _to_360p_fn(torch_frame)

        frame_buffer.append(transform(torch_frame))
        
        if len(frame_buffer) == chunk_size:
            print(f'Processed chunk for {webm_path} at {i}')
            yield torch.stack(frame_buffer)
            frame_buffer = []

    if frame_buffer:
        yield torch.stack(frame_buffer)

def process_mp4(mp4_path: Path, stride_sec: int = 5, chunk_size: int = CHUNK_FRAME_NUM) -> Generator[torch.Tensor, None, None]:
    container = av.open(str(mp4_path))
    stride_num_frames = int(stride_sec * container.streams.video[0].average_rate)
    stream = container.streams.video[0]
    frame_buffer = []
    transform = None
    for i, frame in enumerate(container.decode(stream)):
        if i % stride_num_frames != 0: continue
        torch_frame = torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2,0,1) # hwc to chw
        
        if transform is None:
            transform = _to_360p_fn(torch_frame)

        frame_buffer.append(transform(torch_frame))
        
        if len(frame_buffer) == chunk_size:
            print(f'Processed chunk for {mp4_path} at {i}')
            yield torch.stack(frame_buffer)
            frame_buffer = []

    if frame_buffer:
        yield torch.stack(frame_buffer)

def process_single_file(path: Path, stride_sec: int, out_dir: Path) -> None:
    process_fn = process_mp4 if path.suffix == '.mp4' else process_webm
    filedir = path.stem
    outdir = out_dir / filedir
    outdir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(process_fn(path, stride_sec, chunk_size=CHUNK_FRAME_NUM)):
        print(f'Saved chunk for {path} at {i}')
        torch.save(chunk, outdir / f'{i:08d}_rgb.pt')
    print(f'Processed {path}')


def process_mkif(mkif_path: Path, stride_sec: int = 5, out_dir: Path = MKIF_OUT_DIR) -> None:
    paths = list(mkif_path.glob('*.mp4')) + list(mkif_path.glob('*.webm'))
    random.shuffle(paths)
    
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(process_single_file, [(path, stride_sec, out_dir) for path in paths])

if __name__ == '__main__':
    process_mkif(MKIF_DIR, out_dir=MKIF_OUT_DIR)