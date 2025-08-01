#!/bin/bash

python -m owl_data.latents_from_mp4s \
    --vae_cfg_path /mnt/data/shahbuland/owl-vaes/configs/1x/no_depth.yml \
    --vae_ckpt_path /mnt/data/shahbuland/owl-vaes/checkpoints/1x_rgb_no_depth/step_200000.pt \
    --vae_batch_size 32 \
    --num_workers 64