import io, os
import av
import logging
import numpy as np
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pathlib, logging, subprocess
import ffmpeg


def downsample_video_from_path(
    in_video_path: pathlib.Path,
    intervals_seconds: list[float] | None = None,
    downsampled_fps: int = 10,
    new_height: int = 240,
    crf: int = 18,
) -> list[pathlib.Path]:
    if intervals_seconds is None:
        intervals_seconds = [60.0] * 10

    out_dir = pathlib.Path("/tmp") / in_video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe duration (fall back to 600s if probe fails)
    try:
        probe = ffmpeg.probe(str(in_video_path))
        # Prefer container duration; fall back to first video stream
        dur_str = probe.get("format", {}).get("duration") or probe["streams"][0]["duration"]
        video_duration_seconds = float(dur_str)
    except Exception:
        logging.error(f"Error getting video duration for {in_video_path}; defaulting to 600s")
        video_duration_seconds = 600.0

    start = 0.0
    out_paths: list[pathlib.Path] = []

    for idx, want in enumerate(intervals_seconds):
        remaining = video_duration_seconds - start
        if remaining <= 0:
            break  # we're done; don't make extra files

        duration = min(want, remaining)  # allow a final partial chunk

        out_path = out_dir / f"{out_dir.stem}_{int(start)}_{int(start + duration)}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",         # input seek
            "-i", str(in_video_path),
            "-t", f"{duration:.3f}",       # duration (not absolute end)
            "-vf", f"fps={downsampled_fps},scale=-2:{new_height}",
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(crf),
            "-threads", "1",
            "-an",
            str(out_path),
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if out_path.exists() and out_path.stat().st_size > 0 and proc.returncode == 0:
            logging.debug(f"Processed chunk: {out_path}")
            out_paths.append(out_path)
        else:
            logging.error(f"Failed chunk (skipping): {out_path}")

        start += duration  # advance by the actual duration we used

        # If we just consumed the tail exactly, stop.
        if start >= video_duration_seconds:
            break

    return out_paths




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