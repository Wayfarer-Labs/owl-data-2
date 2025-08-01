"""
Assuming an RGB based encoder from OWL VAEs,
Uses 8 parallel ray jobs with a data sequential data server to encode dataset.
File paths for writing are based on the input video file name
"""

from owl_data.video_loader import VideoServerIterableDataset
from owl_data.nn.owl_image_vae import BatchedEncodingPipe

import ray
import os
import torch

@ray.remote(num_gpus=1)
class EncodingWorker:
    def __init__(
        self,
        rank,
        world_size,
        vae_cfg_path,
        vae_ckpt_path,
        vae_batch_size=32,
        num_workers=64,
    ):
        self.rank = rank
        self.world_size = world_size
        self.loader = VideoServerIterableDataset(num_workers=num_workers, shuffle_buffer = 1)
        self.pipe = BatchedEncodingPipe(vae_cfg_path, vae_ckpt_path, vae_batch_size)

    def __call__(self):
        # Just process one batch for now, as per instructions
        for item in self.loader:
            # Get the latent
            frames = item["frames"]  # [n, h, w, c], uint8
            frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0  # [n, h, w, c]
            frames = frames.permute(0, 3, 1, 2)  # -> [n, c, h, w]
            frames = frames.to(torch.bfloat16)

            latents = self.pipe(frames.cuda()).cpu()
            print("latents shape:", latents.shape)

            # Metadata and subsequently pathing
            # New vid path will be based on original path + splits + chunk_idx
            metadata = item["metadata"]
            chunk_idx = metadata['idx_in_vid']
            vid_dir = metadata['vid_dir']
            output_path = os.path.join(vid_dir,"splits",f"{chunk_idx:06d}_rgblatent.pt")
            torch.save(latents, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_cfg_path", type=str, required=True)
    parser.add_argument("--vae_ckpt_path", type=str, required=True)
    parser.add_argument("--vae_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    # num_gpus is world_size
    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    # Launch one EncodingWorker per GPU, as in depth_from_jpegs.py
    workers = []
    for gpu_id in range(args.num_gpus):
        worker = EncodingWorker.remote(
            gpu_id,
            args.num_gpus,
            args.vae_cfg_path,
            args.vae_ckpt_path,
            args.vae_batch_size,
            args.num_workers,
        )
        workers.append(worker)

    # Launch all workers in parallel
    ray.get([worker.__call__.remote() for worker in workers])