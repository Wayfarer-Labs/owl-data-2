import ray
import os
from .video_reader import VideoReader
import torch
from tqdm import tqdm
import argparse
import subprocess
import math
import json

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
    # r_frame_rate is like "60/1" or "30000/1001"
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

def check_video_completion(vid_info_dict, split_dir, chunk_size):
    """
    Check if the video is processed by looking at number of splits generated vs how many
    there should be.
    """
    if not os.path.exists(split_dir):
        return False
    num_splits = len([f for f in os.listdir(split_dir) if f.endswith('_rgb.pt')])
    expected_splits = math.ceil(vid_info_dict["duration"] * vid_info_dict["fps"] / chunk_size)
    return num_splits == expected_splits

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

def sanitize_filename(filename: str) -> str:
    """Replace leading hyphens and other problematic characters with underscores."""
    # Replace leading hyphens with underscores
    while filename.startswith('-'):
        filename = '_' + filename[1:]
    return filename

def is_mp4_in_unique_folder(path: str) -> bool:
    """Check if the MP4 is in a folder with no other contents (except splits/, vid_info.json)."""
    video_dir = os.path.dirname(path)
    video_filename = os.path.basename(path)

    # Get all files/dirs in the video's directory
    contents = os.listdir(video_dir)

    # Filter out the video file itself, splits directory, and vid_info.json
    video_name_without_ext = os.path.splitext(video_filename)[0]
    filtered_contents = [
        item for item in contents
        if item != video_filename
        and item != 'splits'
        and item != 'vid_info.json'
        and item != f'{video_name_without_ext}.json'
    ]

    # If there are other files/directories, the MP4 is not in a unique folder
    return len(filtered_contents) == 0

def get_vid_info_path(path: str, root_dir: str, output_dir: str | None) -> str:
    """Get the path where vid_info.json should be stored for a given video path."""
    # Convert to absolute paths
    path = os.path.abspath(path)
    root_dir = os.path.abspath(root_dir)
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)

    video_dir = os.path.dirname(path)
    video_filename = os.path.basename(path)
    video_name_without_ext = sanitize_filename(os.path.splitext(video_filename)[0])

    if output_dir is None:
        # When no output_dir: check if MP4 is in unique folder
        if is_mp4_in_unique_folder(path):
            return os.path.join(video_dir, 'vid_info.json')
        else:
            # Create vid_info.json in the folder named after the video file
            return os.path.join(video_dir, video_name_without_ext, f'{video_name_without_ext}.json')
    else:
        # When output_dir is provided: create path structure based on absolute paths
        if is_mp4_in_unique_folder(path):
            # MP4 is in unique folder, preserve directory structure
            if video_dir.startswith(root_dir):
                # Video is under root_dir, preserve relative structure
                rel_path = os.path.relpath(video_dir, root_dir)
                # Fix relative path issues - remove leading "./"
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                elif rel_path == '.':
                    rel_path = ''
                return os.path.join(output_dir, rel_path, 'vid_info.json')
            else:
                # Video is outside root_dir, use basename only
                dir_name = sanitize_filename(os.path.basename(video_dir))
                return os.path.join(output_dir, dir_name, 'vid_info.json')
        else:
            # MP4 is not in unique folder, create vid_info.json in the folder for it
            if video_dir.startswith(root_dir):
                # Video is under root_dir, preserve relative structure
                rel_path = os.path.relpath(video_dir, root_dir)
                # Fix relative path issues - remove leading "./"
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                elif rel_path == '.':
                    rel_path = ''
                return os.path.join(output_dir, rel_path, video_name_without_ext, f'{video_name_without_ext}.json')
            else:
                # Video is outside root_dir, use basename only
                dir_name = sanitize_filename(os.path.basename(video_dir))
                return os.path.join(output_dir, dir_name, video_name_without_ext, f'{video_name_without_ext}.json')

def get_split_dir_from_path(path: str, root_dir: str, output_dir: str | None) -> str:
    """Compute the directory where tensor splits will be stored for a given video path.

    If ``output_dir`` is ``None``, we default to creating the ``splits`` directory
    alongside the original ``.mp4``. When ``output_dir`` is provided, we replicate
    the directory structure **relative to ``root_dir``** inside ``output_dir`` and
    place the ``splits`` directory there instead. This effectively mirrors the
    input directory hierarchy while omitting the original ``.mp4`` files.

    Special case: If the MP4 is not in a unique folder (i.e., shares a directory
    with other files), we create a dedicated folder for it named after the video file.
    """
    # Convert to absolute paths
    path = os.path.abspath(path)
    root_dir = os.path.abspath(root_dir)
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)

    video_dir = os.path.dirname(path)
    video_filename = os.path.basename(path)
    video_name_without_ext = sanitize_filename(os.path.splitext(video_filename)[0])

    if output_dir is None:
        # When no output_dir: check if MP4 is in unique folder
        if is_mp4_in_unique_folder(path):
            return os.path.join(video_dir, "splits")
        else:
            # Create a folder named after the video file
            return os.path.join(video_dir, video_name_without_ext, "splits")
    else:
        # When output_dir is provided: create path structure based on absolute paths
        if is_mp4_in_unique_folder(path):
            # MP4 is in unique folder, preserve directory structure
            if video_dir.startswith(root_dir):
                # Video is under root_dir, preserve relative structure
                rel_path = os.path.relpath(video_dir, root_dir)
                # Fix relative path issues - remove leading "./"
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                elif rel_path == '.':
                    rel_path = ''
                return os.path.join(output_dir, rel_path, "splits")
            else:
                # Video is outside root_dir, use basename only
                dir_name = sanitize_filename(os.path.basename(video_dir))
                return os.path.join(output_dir, dir_name, "splits")
        else:
            # MP4 is not in unique folder, create a folder for it
            if video_dir.startswith(root_dir):
                # Video is under root_dir, preserve relative structure
                rel_path = os.path.relpath(video_dir, root_dir)
                # Fix relative path issues - remove leading "./"
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                elif rel_path == '.':
                    rel_path = ''
                return os.path.join(output_dir, rel_path, video_name_without_ext, "splits")
            else:
                # Video is outside root_dir, use basename only
                dir_name = sanitize_filename(os.path.basename(video_dir))
                return os.path.join(output_dir, dir_name, video_name_without_ext, "splits")

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
def decode_video(path, split_dir, vid_info_path, chunk_size, output_size, stride, force_overwrite):
    """Process video, getting vid_info and checking completion as needed."""
    # Step 1: Get or load vid_info
    if os.path.exists(vid_info_path):
        try:
            with open(vid_info_path, 'r') as f:
                vid_info = json.load(f)
        except:
            # If vid_info file is corrupted, regenerate it
            vid_info = get_vid_info(path)
            vid_info_dir = os.path.dirname(vid_info_path)
            os.makedirs(vid_info_dir, exist_ok=True)
            with open(vid_info_path, 'w') as f:
                json.dump(vid_info, f, indent=2)
    else:
        # Generate vid_info
        vid_info = get_vid_info(path)
        vid_info_dir = os.path.dirname(vid_info_path)
        os.makedirs(vid_info_dir, exist_ok=True)
        with open(vid_info_path, 'w') as f:
            json.dump(vid_info, f, indent=2)

    # Ensure output directory exists
    os.makedirs(split_dir, exist_ok=True)

    # Step 2: Check completion if not force_overwrite
    if not force_overwrite and os.path.exists(split_dir):
        try:
            if check_video_completion(vid_info, split_dir, chunk_size):
                return path  # Already completed
        except Exception:
            pass  # Continue with processing if check fails

    # Step 3: Clean existing splits if needed
    if os.path.exists(split_dir):
        for file in os.listdir(split_dir):
            if file.endswith('_rgb.pt'):
                os.remove(os.path.join(split_dir, file))

    

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
    parser.add_argument('--output_dir', type=str, default=None, help='Optional output directory for split tensors')
    args = parser.parse_args()

    # If output_dir is given, make sure it exists
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

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

    # All videos will be processed - workers will handle completion checking
    paths_to_process = video_paths

    if not paths_to_process:
        print(f"Node {args.node_rank}: No videos to process")
        exit()

    # Launch parallel processing - each worker handles vid_info and completion checking
    futures = [decode_video.remote(path,
                                  get_split_dir_from_path(path, args.root_dir, args.output_dir),
                                  get_vid_info_path(path, args.root_dir, args.output_dir),
                                  args.chunk_size,
                                  tuple(args.output_size),
                                  args.stride,
                                  args.force_overwrite)
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