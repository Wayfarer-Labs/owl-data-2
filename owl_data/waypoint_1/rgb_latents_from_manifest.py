import os
import argparse
import ray
from tqdm import tqdm
import torch
import torch.nn.functional as F
from ..nn.owl_image_vae import BatchedEncodingPipe
import pandas as pd
import random
import math
import numpy as np

def find_tensor_paths(csv_path, force_overwrite=False):
    """
    For each row in the manifest, generate all *_rgb.pt and *_rgblatent.pt output paths
    for each chunk (ceil(vid_duration * vid_fps)), assuming all exist.
    Shuffle and return the lists.
    """
    print("Reading manifest...")
    df = pd.read_csv(csv_path, usecols=["vid_dir_path",
                                        "vid_duration",
                                        "vid_fps"])         # read only what you need

    chunks = np.ceil(df["vid_duration"].to_numpy()
                     * df["vid_fps"].to_numpy()).astype(np.uint16)

    dirs   = df["vid_dir_path"].to_numpy()

    return dirs, chunks

    n_tot  = int(chunks.sum())

    # 1. repeat directory names
    dir_rep   = np.repeat(dirs, chunks)

    # 2. 0…n-1 sequence for each video
    chunk_ids = np.concatenate([np.arange(n, dtype=np.uint16) for n in chunks])

    # 3. zero-padded byte strings
    chunk_str = np.char.zfill(chunk_ids.astype("U"), 8)
    
    print("Constructing paths from manifest... (this might take a few minutes)")
    # 4. build final paths (still NumPy, very fast)
    rgb_paths    = dir_rep + "/splits/" + chunk_str + "_rgb.pt"
    latent_paths = dir_rep + "/splits/" + chunk_str + "_rgblatent.pt"
    
    return rgb_paths, latent_paths

def compile_preprocess():
    @torch.compile
    def preprocess(rgb):
        # rgb: [b,3,360,640] uint8
        # Output: [b,4,360,640] float16, normalized to [-1,1]
        rgb = rgb.cuda(non_blocking=True)
        rgb = rgb.to(torch.float32) / 255.0
        # Normalize to [-1,1]
        rgb = rgb * 2.0 - 1.0
        x = rgb.float16()
        return x
    return preprocess

@ray.remote(num_gpus=1)
class LatentWorker:
    def __init__(self, vae_cfg_path, vae_ckpt_path, vae_batch_size, force_overwrite=False):
        self.pipe = BatchedEncodingPipe(
            vae_cfg_path, vae_ckpt_path, batch_size=vae_batch_size
        )
        self.preprocess = compile_preprocess()
        self.force_overwrite = force_overwrite

    def run(self, vid_dir_path, n_chunks):
        # Construct rgb and latent paths given the vid_dir_path and n_chunks
        for i in range(n_chunks):
            rgb_path = vid_dir_path + "/splits/" + str(i).zfill(8) + "_rgb.pt"
            latent_path = vid_dir_path + "/splits/" + str(i).zfill(8) + "_rgblatent.pt"

            self._run(rgb_path, latent_path)

    def _run(self, rgb_path, latent_path):
        if not self.force_overwrite and os.path.exists(latent_path):
            return
        if not os.path.exists(rgb_path):
            return
        rgb = torch.load(rgb_path, map_location='cuda', weights_only=False)  # [b,3,360,640] uint8
        x = self.preprocess(rgb)  # [b,3,360,640] float16, normalized to [-1,1]
        with torch.no_grad():
            latent = self.pipe(x)  # [b,128,8,8]
        torch.save(latent, latent_path, _use_new_zipfile_serialization=False)

def main():
    parser = argparse.ArgumentParser(description="Compute rgblatents for all *_rgb.pt tensors in a directory using multiple GPUs")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--vae_cfg_path", type=str, required=True, help="Path to VAE config file")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to VAE checkpoint file")
    parser.add_argument("--vae_batch_size", type=int, default=500, help="Batch size for VAE encoding (default: 128)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing depth latent tensors")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node (0-indexed)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")
    args = parser.parse_args()

    dirs, chunks = find_tensor_paths(args.manifest_path, args.force_overwrite)

    print(f"Found {len(dirs)} directories in manifest.")

    # Shard work across nodes
    total_files = len(dirs)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    dirs = dirs[start_idx:end_idx]
    chunks = chunks[start_idx:end_idx]

    if len(dirs) == 0:
        print(f"Node {args.node_rank}: No files assigned to this node after sharding.")
        return

    num_gpus = args.num_gpus

    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    # build one actor per GPU
    workers = [
        LatentWorker.remote(
            args.vae_cfg_path, args.vae_ckpt_path, args.vae_batch_size, args.force_overwrite
        )
        for _ in range(num_gpus)
    ]

    result_refs = []
    for idx, (dir_path, chunk) in enumerate(zip(dirs, chunks)):
        worker = workers[idx % num_gpus]          # simple round-robin
        ref = worker.run.remote(dir_path, chunk)
        result_refs.append(ref)

    with tqdm(total=len(result_refs), desc=f"Extracting rgblatents (node {args.node_rank})", unit="video") as pbar:
        while result_refs:
            done, result_refs = ray.wait(result_refs, num_returns=1, timeout=1)
            pbar.update(len(done))
        # Ensure all exceptions are raised
        ray.get(done)

    print(f"Node {args.node_rank}: rgblatent extraction complete!")

if __name__ == "__main__":
    main()
