import os
import argparse
import ray
import json
import subprocess
from tqdm import tqdm

def sanitize_filename(filename: str) -> str:
    """Replace leading hyphens and other problematic characters with underscores."""
    while filename.startswith('-'):
        filename = '_' + filename[1:]
    return filename

def get_vid_info(path):
    """
    Use ffprobe to get video resolution, fps, and duration in seconds.
    Returns a dict with keys:
        "resolution": str (e.g. "1920x1080")
        "fps": int (e.g. 60)
        "duration": float (seconds)
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video info for {path}: {result.stderr}")
    lines = result.stdout.strip().splitlines()
    if len(lines) < 4:
        raise RuntimeError(f"Unexpected ffprobe output for {path}: {result.stdout}")
    width = lines[0]
    height = lines[1]
    r_frame_rate = lines[2]
    duration = lines[3]
    try:
        num, denom = r_frame_rate.split('/')
        fps = int(round(float(num) / float(denom)))
    except Exception:
        raise RuntimeError(f"Could not parse frame rate '{r_frame_rate}' for {path}")
    try:
        duration = float(duration)
    except Exception:
        raise RuntimeError(f"Could not parse duration '{duration}' for {path}")
    return {
        "resolution": f"{width}x{height}",
        "fps": fps,
        "duration": duration
    }

def get_all_mp4s(root_dir):
    mp4s = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.mp4'):
                mp4s.append(os.path.join(dirpath, fname))
    return mp4s

def get_output_vid_info_path(input_path, root_dir, output_dir):
    # Get relative path from root_dir
    rel_path = os.path.relpath(input_path, root_dir)
    # Remove .mp4 extension and sanitize filename
    rel_dir, mp4_name = os.path.split(rel_path)
    base_name = os.path.splitext(mp4_name)[0]
    base_name = sanitize_filename(base_name)
    out_dir = os.path.join(output_dir, rel_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "vid_info.json")

@ray.remote
def process_video(input_path, root_dir, output_dir):
    try:
        vid_info = get_vid_info(input_path)
        out_path = get_output_vid_info_path(input_path, root_dir, output_dir)
        with open(out_path, 'w') as f:
            json.dump(vid_info, f, indent=2)
        return {"input": input_path, "output": out_path, "status": "ok"}
    except Exception as e:
        return {"input": input_path, "error": str(e), "status": "error"}

def main():
    parser = argparse.ArgumentParser(description="Parallel ffprobe info extraction for all mp4s in a directory tree.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for vid_info.json files')
    parser.add_argument('--num_cpus', type=int, default=8, help='Number of CPUs to use for Ray')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node (0-indexed)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Total number of nodes')
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus)

    mp4_paths = get_all_mp4s(args.root_dir)
    print(f"Found {len(mp4_paths)} mp4 files in {args.root_dir}")

    # Shard work across nodes
    total_files = len(mp4_paths)
    files_per_node = (total_files + args.num_nodes - 1) // args.num_nodes
    start_idx = args.node_rank * files_per_node
    end_idx = min((args.node_rank + 1) * files_per_node, total_files)
    mp4_paths = mp4_paths[start_idx:end_idx]

    print(f"Node {args.node_rank}: Assigned {len(mp4_paths)} mp4 files to process.")

    if not mp4_paths:
        print(f"Node {args.node_rank}: No files assigned to this node after sharding.")
        return

    futures = [
        process_video.remote(path, args.root_dir, args.output_dir)
        for path in mp4_paths
    ]

    results = []
    with tqdm(total=len(futures), desc=f"Processing videos (node {args.node_rank})") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            for obj_ref in done:
                result = ray.get(obj_ref)
                results.append(result)
                pbar.update(1)

    num_errors = sum(1 for r in results if r.get("status") == "error")
    print(f"Node {args.node_rank}: Completed. {len(results) - num_errors} succeeded, {num_errors} errors.")
    if num_errors > 0:
        print("Some errors:")
        for r in results:
            if r.get("status") == "error":
                print(f"{r['input']}: {r['error']}")

if __name__ == "__main__":
    main()
