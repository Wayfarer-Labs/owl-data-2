#!/bin/bash

python3 -m owl_data.video_loader_server \
    --root_dir /mnt/data/datasets/1x_latents \
    --num_workers 64 \
    --queue_max 200000 \
    --frame_skip 1 \
    --n_frames 300 \
    --known_fps 30 \
    --resize_to 512 512 \
    --suffix .mp4
