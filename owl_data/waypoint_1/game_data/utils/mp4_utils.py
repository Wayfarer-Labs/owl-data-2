import io, os
import av
import logging
import numpy as np
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches



def downsample_video_from_path(
    in_video_path: pathlib.Path,
    intervals_seconds: list[float] = [60.] * 10,
    downsampled_fps: int = 10,
    new_height: int = 240,
    crf: int = 18,
) -> list[pathlib.Path]:
    """
    Decodes an in-memory video and downsamples it based on a given specification.
    """
    import subprocess, ffmpeg
    out_path = pathlib.Path('/tmp') / in_video_path.stem
    out_path.mkdir(parents=True, exist_ok=True)
    
    start_time, out_path_chunks = 0., [],
    try: 
        video_duration_seconds = int(float(ffmpeg.probe(in_video_path)['streams'][0]['duration']))
    except:
        logging.error(f'Error getting video duration for {in_video_path}')
        video_duration_seconds = 600

    try:
        for duration in intervals_seconds:
                out_path_chunk = out_path / f"{out_path.stem}_{start_time:.0f}_{start_time + duration:.0f}.mp4"
                
                if start_time + duration > video_duration_seconds:
                    duration = video_duration_seconds - start_time

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", f"{start_time:.3f}",            # input-seek (fast)
                    "-i", str(in_video_path),
                    "-t", f"{duration:.3f}",               # duration, not absolute end time
                    "-vf", f"fps={downsampled_fps},scale=-2:{new_height}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", str(crf),
                    "-threads", "1",
                    "-an",
                    str(out_path_chunk)
                ]

                inst = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                start_time += duration
                out_path_chunks.append(out_path_chunk)
    finally:
        for out_path_chunk in out_path_chunks:
            if not out_path_chunk.exists():
                logging.error(f"Error processing chunk out_path_chunk: {out_path_chunk}")
            else:
                logging.debug(f"Processed chunk: {out_path_chunk}")

        return out_path_chunks



def save_contact_sheet_for_video(video_path: pathlib.Path, out_dir: pathlib.Path, num_frames: int = 5) -> pathlib.Path:
    """
    Decode frames from a video and save a 1xN contact sheet PNG with N frames.
    Frames are sampled approximately uniformly across the clip if duration/fps are available,
    otherwise the first N frames are used.
    """
    import ffmpeg  # uses ffmpeg-python; falls back gracefully if probe fails
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe for duration/fps to estimate frame indices
    duration, fps = 0.0, 0.0
    try:
        info = ffmpeg.probe(str(video_path))
        stream = next(s for s in info.get("streams", []) if s.get("codec_type") == "video")
        duration = float(
            (stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
        )
        avg_rate = stream.get("avg_frame_rate", "0/0")
        num, den = avg_rate.split("/")
        fps = (float(num) / float(den)) if float(den) != 0 else 0.0
    except Exception:
        duration, fps = 0.0, 0.0

    # Choose target frame indices
    target_indices = None
    if duration > 0 and fps > 0 and np.isfinite(duration):
        total_est = max(1, int(round(duration * fps)))
        # Spread indices across (0, total_est)
        positions = np.linspace(1, max(1, total_est - 1), num_frames, dtype=int)
        target_indices = set(int(i) for i in positions)

    # Decode and collect frames
    images = []
    container = None
    try:
        container = av.open(str(video_path))
        for i, frame in enumerate(container.decode(video=0)):
            if target_indices is None:
                if len(images) < num_frames:
                    images.append(frame.to_ndarray(format="rgb24"))
                if len(images) >= num_frames:
                    break
            else:
                if i in target_indices:
                    images.append(frame.to_ndarray(format="rgb24"))
                    if len(images) >= num_frames:
                        break
    finally:
        try:
            if container is not None:
                container.close()
        except Exception:
            pass

    # Pad if fewer than requested frames were found
    if not images:
        images = [np.zeros((240, 320, 3), dtype=np.uint8)] * num_frames
    elif len(images) < num_frames:
        images += [images[-1]] * (num_frames - len(images))

    # Create contact sheet
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 3, 3))
    if num_frames == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx])
        ax.axis("off")
        ax.set_title(f"{video_path.stem} [{idx+1}/{num_frames}]", fontsize=8)
    fig.tight_layout()

    out_file = out_dir / f"{pathlib.Path(video_path).stem}_contact.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def main():
    import sys
    import os
    from datetime import datetime
    import pathlib

    video_path = "/home/sky/owl-data-2/test_tar/2025-09-30 16-04-03.mp4"
    out_vis_dir = pathlib.Path("/home/sky/owl-data-2/test_tar")

    downsampled_paths: list[pathlib.Path] = downsample_video_from_path(
        in_video_path=pathlib.Path(video_path),
    )

    print("\n--- Frame Sampling Results ---")
    for p in downsampled_paths:
        print(p)

    # Create 5-frame contact sheets for each chunk
    for chunk_path in downsampled_paths:
        out_file = save_contact_sheet_for_video(pathlib.Path(chunk_path), out_dir=out_vis_dir, num_frames=5)
        print(f"Saved contact sheet: {out_file}")

if __name__ == "__main__":
    main()