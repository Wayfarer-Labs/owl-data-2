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
import tqdm

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
        import time
        from tqdm import tqdm

        total_frames = 0
        t0 = time.time()
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=0,
                desc="Frames Processed (all workers, est)",
                unit="frames",
                dynamic_ncols=True,
                smoothing=0.1,
            )
            last_update = time.time()
            last_frames = 0

        for item in self.loader:
            frames = item["frames"]  # [n, h, w, c], uint8
            n_frames = frames.shape[0]
            total_frames += n_frames

            # tqdm update only on rank 0
            if self.rank == 0:
                # Estimate total frames processed across all workers
                est_total = total_frames * self.world_size
                now = time.time()
                elapsed = now - t0
                fps = est_total / elapsed if elapsed > 0 else 0
                pbar.total = None  # no fixed total
                pbar.n = est_total
                pbar.set_postfix({"fps": f"{fps:.1f}"})
                pbar.refresh()

            frames = torch.from_numpy(frames).cuda().bfloat16() / 255.0 * 2.0 - 1.0  # [n, h, w, c]
            frames = frames.permute(0, 3, 1, 2)  # -> [n, c, h, w]

            latents = self.pipe(frames).cpu()

            # Metadata and subsequently pathing
            metadata = item["metadata"]
            chunk_idx = metadata['idx_in_vid']
            vid_dir = metadata['vid_dir']
            os.makedirs(os.path.join(vid_dir,"splits"), exist_ok=True)
            output_path = os.path.join(vid_dir,"splits",f"{chunk_idx:06d}_rgblatent.pt")
            torch.save(latents, output_path)

            del frames, latents, item

        if self.rank == 0 and pbar is not None:
            pbar.close()

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