from pathlib import Path
from typing import Generator
import torch
import torchvision.transforms as T
import av
from owl_data.waypoint_1.datasets.globals import CHUNK_FRAME_NUM, STRIDE_SEC, EPIC_KITCHENS_100_OUT_DIR
import os
import multiprocessing

top_path = Path('/mnt/data/waypoint_1/datasets/epic_kitchens_100/2g1n6qdydwa9u22shpxqzp0t8m')


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



def process_mp4_seek(mp4_path: Path, stride_sec: float = 5.0, chunk_size: int = CHUNK_FRAME_NUM
                     ) -> Generator[torch.Tensor, None, None]:
    with av.open(str(mp4_path)) as container:
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
        else:
            duration_s = float(container.duration) / av.time_base  # fallback

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
                    if transform is None:
                        transform = _to_360p_fn(torch_frame)
                    buf.append(transform(torch_frame))
                    print(f'appended frame at ts {ts}')
                    if len(buf) == chunk_size:
                        yield torch.stack(buf); buf = []
                    break  # go seek to next timestamp

            t += stride_sec

        if buf:
            yield torch.stack(buf)


def mp4_paths() -> Generator[Path, None, None]:
    for path in top_path.glob('P*/videos/*.MP4'):
        yield path
        
def process_single_file(path: Path) -> None:
    dst_path = EPIC_KITCHENS_100_OUT_DIR / path.parent.name / path.stem
    dst_path.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(process_mp4_seek(path)):
        torch.save(chunk, dst_path / f'{i:08d}_rgb.pt')
        print(f'Saved chunk for {path} at {i} to {dst_path / f"{i:08d}_rgb.pt"}')


if __name__ == '__main__':
    paths = list(mp4_paths())
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(process_single_file, paths)
