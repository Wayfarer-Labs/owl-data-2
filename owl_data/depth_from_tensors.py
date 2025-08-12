import os
import argparse
import ray
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .nn.depth_pipeline import BatchedDepthPipeline

def find_tensor_paths(root_dir, force_overwrite=False):
    """
    Find all *_rgb.pt files and corresponding *_depth.pt output paths.
    If force_overwrite is False, skip those where *_depth.pt already exists.
    """
    input_paths = []
    output_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_rgb.pt"):
                in_path = os.path.join(root, file)
                out_path = os.path.join(root, file.replace("_rgb.pt", "_depth.pt"))
                if not force_overwrite and os.path.exists(out_path):
                    continue
                input_paths.append(in_path)
                output_paths.append(out_path)
    return input_paths, output_paths

@ray.remote(num_gpus=1)
def process_tensor_on_gpu(input_path, output_path):
    video = torch.load(input_path, map_location='cpu')  # [N,C,H,W]
    model = BatchedDepthPipeline()
    with torch.no_grad():
        depth = model(video)  # [N,1,H,W] or similar
    torch.save(depth, output_path)

def split_list(lst, n):
    # Splits lst into n nearly equal parts
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    parser = argparse.ArgumentParser(description="Compute depth maps for all *_rgb.pt tensors in a directory using multiple GPUs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_rgb.pt tensors")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--force_overwrite", action="store_true", help="Overwrite existing depth tensors")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node (0-indexed)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")
    args = parser.parse_args()

    input_paths, output_paths = find_tensor_paths(args.input_dir, args.force_overwrite)
    if len(input_paths) == 0:
        print(f"No *_rgb.pt files found in {args.input_dir}")
        return

    # Shard work across nodes
    total_files = len(input_paths)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    input_paths = input_paths[start_idx:end_idx]
    output_paths = output_paths[start_idx:end_idx]

    if not input_paths:
        print(f"Node {args.node_rank}: No files assigned to this node after sharding.")
        return

    num_gpus = args.num_gpus
    splits = split_list(list(zip(input_paths, output_paths)), num_gpus)

    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    result_refs = []
    for gpu_id, split in enumerate(splits):
        if len(split) == 0:
            continue
        for in_path, out_path in split:
            ref = process_tensor_on_gpu.remote(in_path, out_path)
            result_refs.append(ref)

    with tqdm(total=len(result_refs), desc=f"Extracting depth (node {args.node_rank})", unit="video") as pbar:
        while result_refs:
            done, result_refs = ray.wait(result_refs, num_returns=1, timeout=1)
            pbar.update(len(done))
        # Ensure all exceptions are raised
        ray.get(done)

    print(f"Node {args.node_rank}: Depth extraction complete!")

if __name__ == "__main__":
    main()
