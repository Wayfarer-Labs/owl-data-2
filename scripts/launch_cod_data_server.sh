#!/bin/bash

python3 -m owl_data.video_loader_server \
    --root_dir /mnt/data/shahbuland/video-proc-2/datasets/cod-yt \
    --num_workers 64 \
    --queue_max 200000 \
    --frame_skip 1 \
    --n_frames 2 \
    --known_fps 60 \
    --suffix .mp4
