import ray
import os
from ..video_reader import VideoReader
import torch
from tqdm import tqdm
import argparse
import math
import json
from dotenv import load_dotenv
import threading
import queue
load_dotenv()

from ..nn.owl_image_vae import BatchedEncodingPipe

def sanitize_filename(filename: str) -> str:
    """Replace leading hyphens and other problematic characters with underscores."""
    # Replace leading hyphens with underscores
    while filename.startswith('-'):
        filename = '_' + filename[1:]
    return filename

def get_vid_info_path_with_output_dirs(root_dir, output_dir, file_type='mp4'):
    """
    Get a list of tuples of (path_to_vid_info, path_to_corresponding_output_dir)
    """
    vid_info_paths_with_output_dirs = []
    # Handle both single path and list of paths
    root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
    
    from tqdm import tqdm

    pbar = tqdm(desc="Collecting video paths", unit="video", total=None)
    for dir in root_dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(file_type):
                    full_path = os.path.join(root, file)

                    # first get raw fp sanitized
                    fp = os.path.basename(full_path)
                    fp = os.path.splitext(fp)[0]
                    fp = sanitize_filename(fp)

                    # full path starts with root_dir, so for output, swap it
                    output_path = os.path.dirname(full_path.replace(root_dir, output_dir))
                    output_path = os.path.join(output_path, fp)

                    vid_info_path = os.path.join(output_path, "vid_info.json")

                    vid_info_paths_with_output_dirs.append((vid_info_path, output_path))
                    pbar.update(1)
    pbar.close()
    return vid_info_paths_with_output_dirs

def check_probe_and_completion(output_dir, chunk_size, force_overwrite=False) -> str:
    """
    For given output directory, check if probe was done, and if all splits are present.
    Will return False if we shouldn't write anything, True if we should.
    """
    vid_info_path = os.path.join(output_dir, "vid_info.json")
    splits_dir = os.path.join(output_dir, "splits")

    if not os.path.exists(output_dir):
        return False
    if not os.path.exists(vid_info_path):
        return False
    if not os.path.exists(splits_dir):
        return True # No splits, good to go
    
    with open(vid_info_path, "r") as f:
        vid_info = json.load(f)
    expected_splits = vid_info["duration"] * vid_info["fps"] / chunk_size
    if expected_splits % 1 < 0.05:
        expected_splits = math.floor(expected_splits)
    else:
        expected_splits = math.ceil(expected_splits)
        
    num_splits = len([f for f in os.listdir(splits_dir) if f.endswith('_rgb.pt')])
    return num_splits != expected_splits # If not enough splits, write everything

def get_chunks(vid_info_path, chunk_size = 2000):
    vid_info = json.load(open(vid_info_path, "r"))
    return math.ceil(vid_info["duration"] * vid_info["fps"] / chunk_size)

@ray.remote(num_gpus=1)
class EncoderWorker:
    def __init__(self, vae_cfg_path, vae_ckpt_path, vae_batch_size, force_overwrite=False):
        self.pipe = BatchedEncodingPipe(
            vae_cfg_path, vae_ckpt_path, batch_size=vae_batch_size
        )
        self.force_overwrite = force_overwrite

    def run(self, pair, chunk_size):
        vid_info_path, output_path = pair

        # Get n_chunks from vid_info
        if not os.path.exists(vid_info_path):
            print("Warning: vid_info.json not found for ", vid_info_path)
            return

        n_chunks = get_chunks(vid_info_path, chunk_size)
        splits_dir = os.path.join(output_path, "splits")

        def rgb_path(i):
            return os.path.join(splits_dir, f"{i:08d}_rgb.pt")
        
        def latent_path(i):
            return os.path.join(splits_dir, f"{i:08d}_rgblatent.pt")

        if not self.force_overwrite and os.path.exists(latent_path(n_chunks-1)):
            return

        for i in range(n_chunks):
            try:
                rgb = torch.load(rgb_path(i), map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Error loading tensor {rgb_path(i)}: {e}")
                continue

            if os.path.exists(latent_path(i)) and not self.force_overwrite:
                continue

            with torch.no_grad():
                latent = self.pipe(rgb)
            torch.save(latent, latent_path(i), _use_new_zipfile_serialization=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rgblatents for all *_rgb.pt tensors in a directory using multiple GPUs")
    parser.add_argument('--root_dir', type=str, required=True, help="Input directory containing mp4 files (used for dir structuring)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory containing *_rgb.pt tensors")
    parser.add_argument("--vae_cfg_path", type=str, required=True, help="Path to VAE config file")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to VAE checkpoint file")
    parser.add_argument("--vae_batch_size", type=int, default=10, help="Batch size for VAE encoding (default: 128)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing rgb latent tensors")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node (0-indexed)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Number of frames per chunk (default: 2000)")
    parser.add_argument("--file_type", type=str, default='mp4', help="File type to search for")
    args = parser.parse_args()

    path_pairs = get_vid_info_path_with_output_dirs(args.root_dir, args.output_dir, args.file_type)

    # Shard work across nodes
    total_files = len(path_pairs)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    path_pairs = path_pairs[start_idx:end_idx]

    if not path_pairs:
        print(f"Node {args.node_rank}: No files assigned to this node after sharding.")
        exit()

    num_gpus = args.num_gpus

    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    # build one actor per GPU
    workers = []
    for _ in range(num_gpus):
        worker = EncoderWorker.remote(args.vae_cfg_path, args.vae_ckpt_path, args.vae_batch_size, args.force_overwrite)
        workers.append(worker)

    futures = []
    for idx, pair in enumerate(path_pairs):
        worker = workers[idx % num_gpus]
        future = worker.run.remote(pair, args.chunk_size)
        futures.append(future)

    with tqdm(total=len(futures), desc=f"Encoding rgblatents (node {args.node_rank})") as pbar:
        while futures:
            done, futures = ray.wait(futures)
            ray.get(done)
            pbar.update(len(done))

    print(f"Node {args.node_rank}: rgblatent encoding complete!")
