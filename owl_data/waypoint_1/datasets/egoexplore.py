from pathlib import Path
from typing import Generator

import torch
import av
from owl_data.waypoint_1.datasets.utils import STRIDE_SEC, EGOEXPLORE_DIR, EGOEXPLORE_OUT_DIR, CHUNK_FRAME_NUM
import random
import torchvision.transforms as T
import multiprocessing
import os



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

import av
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Generator

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


def process_single_file(path: Path, stride_sec: int, out_dir: Path) -> None:
    process_fn = process_mp4_seek
    filedir = path.stem
    outdir = out_dir / filedir
    outdir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(process_fn(path, stride_sec, chunk_size=CHUNK_FRAME_NUM)):
        torch.save(chunk, outdir / f'{i:08d}_rgb.pt')
        print(f'Saved chunk for {path} at {i} to { outdir / f"{i:08d}_rgb.pt"}')
    print(f'Processed {path}')


def process_egoexplore(egoexplore_path: Path, stride_sec: int = 5, out_dir: Path = EGOEXPLORE_OUT_DIR) -> None:
    paths = list(egoexplore_path.glob('*.mp4?*'))
    random.shuffle(paths)
    # for path in paths:
    #     process_single_file(path, stride_sec, out_dir)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_single_file, [(path, stride_sec, out_dir) for path in paths])

if __name__ == '__main__':
    process_egoexplore(EGOEXPLORE_DIR, out_dir=EGOEXPLORE_OUT_DIR)