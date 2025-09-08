import ray
import os
from .video_reader import VideoReader
import torch
from tqdm import tqdm
import argparse

def get_video_paths(root_dir):
    video_paths = []
    # Handle both single path and list of paths
    root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
    
    for dir in root_dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.join(root, file)
                    video_paths.append(full_path)
    return video_paths

def to_tensor(frames, output_size):
    # frames is list of np arrays that are all [h,w,c] uint8 [0,255]
    # Convert to tensor and stack along batch dim
    frames = [torch.from_numpy(f) for f in frames]
    frames = torch.stack(frames, dim=0)  # [N,H,W,C]
    
    # Move channels first and resize
    frames = frames.permute(0,3,1,2)  # [N,C,H,W]
    frames = torch.nn.functional.interpolate(frames, size=output_size, mode='bilinear', align_corners=False)
    
    # Convert to uint8
    frames = frames.to(torch.uint8).cpu()
    
    return frames

@ray.remote
def decode_video_fps_aware(path, chunk_size, output_size, target_fps=15):
    vr = VideoReader(path)
    
    # Calculate sampling stride based on target FPS
    original_fps = vr.get_avg_fps() if hasattr(vr, 'get_avg_fps') else 30
    stride = max(round(original_fps / target_fps), 1)
    
    frames = []
    n_frames = 0
    split_ind = 0
    
    # Sample frames with stride instead of processing all
    for i in tqdm(range(0, len(vr), stride)):
        frame = vr[i]
        frames.append(frame.asnumpy())
        
        n_frames += 1
        if n_frames >= chunk_size:
            chunk = to_tensor(frames, output_size)
            torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))
            frames = []
            n_frames = 0
            split_ind += 1
            del chunk
    
    # Save any remaining frames
    if frames:
        chunk = to_tensor(frames, output_size)
        torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))

@ray.remote
def decode_video(path, chunk_size, output_size, stride=1):
    split_dir = os.path.dirname(path)
    split_dir = os.path.join(split_dir, "splits")
    os.makedirs(split_dir, exist_ok = True)

    vr = VideoReader(path, stride)
    frames = []
    n_frames = 0
    split_ind = 0

    # Use VideoReader from video_reader.py, which is an iterable
    for frame in vr:
        frames.append(frame)  # frame is already a numpy array, HWC, uint8

        n_frames += 1
        if n_frames >= chunk_size:
            chunk = to_tensor(frames, output_size)
            frames = []
            torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))
            n_frames = 0
            split_ind += 1

            del chunk

    if frames:
        chunk = to_tensor(frames, output_size)
        torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing videos')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Number of frames per chunk')
    parser.add_argument('--output_size', type=int, nargs=2, default=[360, 640], help='Output size as two integers: height width')
    parser.add_argument('--force_overwrite', action='store_true', help='Force overwrite existing rgb tensors')
    parser.add_argument('--num_cpus', type=int, default=80, help='Number of CPUs to use for Ray')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node (0-indexed)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--stride', type=int, default=1, help='Frame skip')
    args = parser.parse_args()

    video_paths = get_video_paths(args.root_dir)

    # Shard work across nodes
    total_files = len(video_paths)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    video_paths = video_paths[start_idx:end_idx]

    if not video_paths:
        print(f"Node {args.node_rank}: No videos assigned to this node after sharding.")
        exit()

    # Initialize ray with specified number of CPUs
    ray.init(num_cpus=args.num_cpus)

    # Filter paths and prepare work
    paths_to_process = []
    for path in video_paths:
        split_dir = os.path.join(os.path.dirname(path), "splits")
        if os.path.exists(split_dir) and not args.force_overwrite:
            print(f"Skipping {path} - already processed")
            continue
        elif os.path.exists(split_dir) and args.force_overwrite:
            for file in os.listdir(split_dir):
                if file.endswith('_rgb.pt'):
                    os.remove(os.path.join(split_dir, file))
        paths_to_process.append(path)

    if not paths_to_process:
        print(f"Node {args.node_rank}: No videos to process")
        exit()

    # Launch parallel processing
    futures = [decode_video.remote(path, args.chunk_size, tuple(args.output_size), args.stride) 
              for path in paths_to_process]

    # Wait for results with progress bar
    with tqdm(total=len(futures), desc=f"Processing videos (node {args.node_rank})") as pbar:
        while futures:
            done, futures = ray.wait(futures)
            completed_paths = ray.get(done)
            for path in completed_paths:
                print(f"Completed processing {path}")
            pbar.update(len(done))

    print(f"Node {args.node_rank}: All videos processed!")