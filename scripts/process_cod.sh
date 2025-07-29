#!/bin/bash

set -e

INPUT_DIR="/mnt/data/shahbuland/video-proc-2/datasets/cod-yt"
OUTPUT_DIR="/mnt/data/datasets/cod-yt-jpegs"

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the frames_to_jpegs.py script
python3 -m owl_data.frames_to_jpegs \
    --root_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --frame_skip 30 \
    --suffix mp4 \
    --images_per_subdir 1000
