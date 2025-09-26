import os
from tqdm import tqdm
import argparse
import math
import json
from dotenv import load_dotenv
import threading
import queue
import torch
load_dotenv()

from ..nn.captioner_parallel import VLMCaptionerParallel

"""
This script assumes rgb.pt tensor files already exist,
and that vid_info.json files already exist.
Creates captions as .jsonl files

I.e. for some chunk XXXXXXXX_rgb.pt,
would create XXXXXXXX_captions.jsonl

This parallel version uses 8 independent vLLM servers for true parallel processing.
"""

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

def get_chunks(vid_info_path, chunk_size = 2000):
    vid_info = json.load(open(vid_info_path, "r"))
    return math.ceil(vid_info["duration"] * vid_info["fps"] / chunk_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing tensors')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for captions')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Number of frames per chunk')
    parser.add_argument('--force_overwrite', action='store_true', help='Force overwrite existing captions')
    parser.add_argument('--port', type=int, default=8000, help='Base port for VLLM servers (will use port to port+7)')
    parser.add_argument('--file_type', type=str, default='.mp4', help='File type to search for')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for parallel processing')
    parser.add_argument('--window_length', type=float, default=5.0, help='Window length')
    parser.add_argument('--kernel', type=int, default=5, help='Kernel size')
    parser.add_argument('--stride', type=float, default=5.0, help='Stride')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help='Model name')
    parser.add_argument('--num_servers', type=int, default=8, help='Number of parallel vLLM servers')
    args = parser.parse_args()

    path_pairs = get_vid_info_path_with_output_dirs(args.root_dir, args.output_dir, args.file_type)

    total_files = len(path_pairs)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    path_pairs = path_pairs[start_idx:end_idx]

    print(f"Initializing parallel captioner with {args.num_servers} servers on ports {args.port}-{args.port+args.num_servers-1}")
    captioner = VLMCaptionerParallel(
        port=args.port,
        window_length=args.window_length,
        kernel=args.kernel,
        stride=args.stride,
        model_name=args.model_name,
        num_servers=args.num_servers
    )

    # Process videos in larger batches for better parallelization
    video_batch_size = 4  # Process multiple videos at once
    video_batches = []
    current_batch = []
    current_batch_data = []

    print(f"Collecting videos for batch processing (node {args.node_rank})...")
    for i in tqdm(range(0, len(path_pairs)), desc=f"Preparing batches (node {args.node_rank})", unit="video"):
        vid_info_path, output_path = path_pairs[i]
        n_chunks = get_chunks(vid_info_path, args.chunk_size)
        splits_dir = os.path.join(output_path, "splits")

        # If all caption files exist we can just skip
        if not args.force_overwrite and all(os.path.exists(os.path.join(splits_dir, f"{j:08d}_captions.jsonl")) for j in range(n_chunks)):
            continue

        rgb_fp_list = [os.path.join(splits_dir, f"{j:08d}_rgb.pt") for j in range(n_chunks)]
        rgb_fp_list = [fp for fp in rgb_fp_list if os.path.exists(fp)]

        if rgb_fp_list:
            current_batch.append((vid_info_path, output_path, rgb_fp_list))
            if len(current_batch) >= video_batch_size:
                video_batches.append(current_batch)
                current_batch = []

    if current_batch:
        video_batches.append(current_batch)

    print(f"Processing {len(video_batches)} batches of videos...")

    for batch_idx, video_batch in enumerate(tqdm(video_batches, desc=f"Processing batches (node {args.node_rank})", unit="batch")):
        # Load all video chunks for this batch
        all_video_chunks = []
        all_fps_list = []
        video_info = []

        for vid_info_path, output_path, rgb_fp_list in video_batch:
            video_chunks = [
                torch.load(fp, map_location='cpu', weights_only=False, mmap=True) for fp in rgb_fp_list
            ]
            fps = json.load(open(vid_info_path, "r"))["fps"]
            fps_list = [fps] * len(video_chunks)

            all_video_chunks.extend(video_chunks)
            all_fps_list.extend(fps_list)
            video_info.append((output_path, len(video_chunks)))

        # Process all chunks from all videos in parallel
        print(f"Batch {batch_idx+1}/{len(video_batches)}: Processing {len(all_video_chunks)} chunks across {len(video_batch)} videos...")
        all_captions = captioner(all_video_chunks, all_fps_list)

        # Save results
        caption_idx = 0
        for output_path, num_chunks in video_info:
            video_captions = all_captions[caption_idx:caption_idx + num_chunks]
            caption_idx += num_chunks

            for j in range(len(video_captions)):
                with open(os.path.join(output_path, "splits", f"{j:08d}_captions.jsonl"), "w") as f:
                    for caption in video_captions[j]:
                        json.dump(caption, f)
                        f.write("\n")

    print(f"Node {args.node_rank} completed processing!")