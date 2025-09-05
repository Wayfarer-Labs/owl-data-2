import os
import argparse
import ray
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .nn.owl_image_vae import BatchedEncodingPipe

def find_tensor_paths(root_dir, force_overwrite=False):
    """
    Find all *_rgb.pt and *_depth.pt files and corresponding *_depthlatent.pt output paths.
    If force_overwrite is False, skip those where *_depthlatent.pt already exists.
    """
    input_rgb_paths = []
    input_depth_paths = []
    output_latent_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_rgb.pt"):
                rgb_path = os.path.join(root, file)
                depth_path = os.path.join(root, file.replace("_rgb.pt", "_depth.pt"))
                latent_path = os.path.join(root, file.replace("_rgb.pt", "_depthlatent.pt"))
                if not os.path.exists(depth_path):
                    continue
                if not force_overwrite and os.path.exists(latent_path):
                    continue
                input_rgb_paths.append(rgb_path)
                input_depth_paths.append(depth_path)
                output_latent_paths.append(latent_path)
    return input_rgb_paths, input_depth_paths, output_latent_paths

def compile_preprocess():
    @torch.compile
    def preprocess(rgb, depth):
        # rgb: [b,3,360,640] uint8, depth: [b,1,360,640] uint8
        # Output: [b,4,360,640] bfloat16, normalized to [-1,1]
        rgb = rgb.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        rgb = rgb.to(torch.float32) / 255.0
        depth = depth.to(torch.float32) / 255.0
        # Normalize to [-1,1]
        rgb = rgb * 2.0 - 1.0
        depth = depth * 2.0 - 1.0
        x = torch.cat([rgb, depth], dim=1).bfloat16()
        return x
    return preprocess

@ray.remote(num_gpus=1)
class DepthLatentWorker:
    def __init__(self, vae_cfg_path, vae_ckpt_path, vae_batch_size):
        self.pipe = BatchedEncodingPipe(
            vae_cfg_path, vae_ckpt_path, batch_size=vae_batch_size
        )
        self.preprocess = compile_preprocess()

    def run(self, rgb_path, depth_path, latent_path):
        rgb = torch.load(rgb_path, map_location='cuda')  # [b,3,360,640] uint8
        depth = torch.load(depth_path, map_location='cuda')  # [b,1,360,640] uint8
        x = self.preprocess(rgb, depth)  # [b,4,360,640] bfloat16, [-1,1]
        with torch.no_grad():
            latent = self.pipe(x)  # [b,128,8,8]
        torch.save(latent, latent_path)

def split_list(lst, n):
    # Splits lst into n nearly equal parts
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    parser = argparse.ArgumentParser(description="Compute depth latents for all *_rgb.pt and *_depth.pt tensors in a directory using multiple GPUs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_rgb.pt and *_depth.pt tensors")
    parser.add_argument("--vae_cfg_path", type=str, required=True, help="Path to VAE config file")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to VAE checkpoint file")
    parser.add_argument("--vae_batch_size", type=int, default=128, help="Batch size for VAE encoding (default: 128)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing depth latent tensors")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node (0-indexed)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")
    args = parser.parse_args()

    rgb_paths, depth_paths, latent_paths = find_tensor_paths(args.input_dir, args.force_overwrite)
    if len(rgb_paths) == 0:
        print(f"No *_rgb.pt files found in {args.input_dir}")
        return

    # Shard work across nodes
    total_files = len(rgb_paths)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    rgb_paths = rgb_paths[start_idx:end_idx]
    depth_paths = depth_paths[start_idx:end_idx]
    latent_paths = latent_paths[start_idx:end_idx]

    if not rgb_paths:
        print(f"Node {args.node_rank}: No files assigned to this node after sharding.")
        return

    num_gpus = args.num_gpus

    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    # build one actor per GPU
    workers = [
        DepthLatentWorker.remote(
            args.vae_cfg_path, args.vae_ckpt_path, args.vae_batch_size
        )
        for _ in range(num_gpus)
    ]

    result_refs = []
    for idx, (rgb_path, depth_path, latent_path) in enumerate(zip(rgb_paths, depth_paths, latent_paths)):
        worker = workers[idx % num_gpus]          # simple round-robin
        ref = worker.run.remote(rgb_path, depth_path, latent_path)
        result_refs.append(ref)

    with tqdm(total=len(result_refs), desc=f"Extracting depth latents (node {args.node_rank})", unit="video") as pbar:
        while result_refs:
            done, result_refs = ray.wait(result_refs, num_returns=1, timeout=1)
            pbar.update(len(done))
        # Ensure all exceptions are raised
        ray.get(done)

    print(f"Node {args.node_rank}: Depth latent extraction complete!")

if __name__ == "__main__":
    main()
