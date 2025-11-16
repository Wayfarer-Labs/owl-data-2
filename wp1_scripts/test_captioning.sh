#!/bin/bash

python -m owl_data.waypoint_1.captions_from_tensors \
    --root_dir /mnt/data/waypoint_1/data/egoexplore_360P \
    --output_dir /mnt/data/waypoint_1/data_pt/egoexplore_360P \
    --batch_size 8