import ray
import os
from .video_reader import VideoReader
import torch
from tqdm import tqdm
import argparse
import subprocess
import math
import json
import wandb
import time

from dotenv import load_dotenv
load_dotenv()

# Log timing metrics every N frames
LOG_EVERY = 100

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
    # Count existing split files without using listdir
    num_splits = 0
    split_index = 0
    while os.path.exists(os.path.join(split_dir, f"{split_index:08d}_rgb.pt")):
        num_splits += 1
        split_index += 1
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


def get_vid_info_path(path: str, root_dir: str, output_dir: str) -> str:
    """Get the path where vid_info.json should be stored for a given video path."""
    # Convert to absolute paths
    path = os.path.abspath(path)
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    video_filename = os.path.basename(path)
    video_name_without_ext = sanitize_filename(os.path.splitext(video_filename)[0])

    # Create vid_info.json in the folder named after the video file
    return os.path.join(output_dir, video_name_without_ext, f'{video_name_without_ext}.json')

def get_split_dir_from_path(path: str, root_dir: str, output_dir: str) -> str:
    """Compute the directory where tensor splits will be stored for a given video path.

    Creates a folder in output_dir named after the video file and places splits there.
    """
    # Convert to absolute paths
    path = os.path.abspath(path)
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    video_filename = os.path.basename(path)
    video_name_without_ext = sanitize_filename(os.path.splitext(video_filename)[0])

    # Create a folder named after the video file
    return os.path.join(output_dir, video_name_without_ext, "splits")

def to_tensor(frames, output_size):
    # frames is list of np arrays that are all [h,w,c] uint8 [0,255]
    # Convert to tensor and stack along batch dim
    frames = torch.stack([torch.from_numpy(f) for f in frames], dim = 0) # nhwc

    # Move channels first and resize
    frames = frames.permute(0,3,1,2)  # [N,C,H,W]
    frames = torch.nn.functional.interpolate(frames, size=output_size, mode='bilinear', align_corners=False)

    return frames

@ray.remote
class VideoWorker:
    def __init__(self, local_rank, is_primary_worker=False):
        self.local_rank = local_rank
        self.is_primary_worker = is_primary_worker

        # Global counters for primary worker across all videos
        if self.is_primary_worker:
            self.global_chunks_written = 0
            self.global_tensor_times = []
            self.global_write_times = []
            self.global_decode_times = []

    def decode_video(self, path, split_dir, vid_info_path, chunk_size, output_size, stride, force_overwrite):
        """Process video, getting vid_info and checking completion as needed."""
        start_time = time.time()

        # Initialize timing lists for tracking performance (per video)
        tensor_times = []
        write_times = []
        decode_times = []

        # Frame counter for periodic logging
        total_frames_processed = 0
        fps_start_time = time.time()  # For FPS calculation

        # Step 1: Get or load vid_info
        if os.path.exists(vid_info_path):
            try:
                with open(vid_info_path, 'r') as f:
                    vid_info = json.load(f)
            except:
                # If vid_info file is corrupted, regenerate it
                try:
                    vid_info = get_vid_info(path)
                    vid_info_dir = os.path.dirname(vid_info_path)
                    os.makedirs(vid_info_dir, exist_ok=True)
                    with open(vid_info_path, 'w') as f:
                        json.dump(vid_info, f, indent=2)
                except Exception as e:
                    # ffprobe failed, return error info for logging
                    return {"path": path, "ffprobe_error": str(e)}
        else:
            # Generate vid_info
            try:
                vid_info = get_vid_info(path)
                vid_info_dir = os.path.dirname(vid_info_path)
                os.makedirs(vid_info_dir, exist_ok=True)
                with open(vid_info_path, 'w') as f:
                    json.dump(vid_info, f, indent=2)
            except Exception as e:
                # ffprobe failed, return error info for logging
                return {"path": path, "ffprobe_error": str(e)}

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
            # Remove existing split files without using listdir
            split_index = 0
            while True:
                split_file = os.path.join(split_dir, f"{split_index:08d}_rgb.pt")
                if os.path.exists(split_file):
                    os.remove(split_file)
                    split_index += 1
                else:
                    break



        vr = VideoReader(path, stride)
        frames = []
        n_frames = 0
        split_ind = 0

        # Use VideoReader from video_reader.py, which is an iterable
        for frame in vr:
            decode_start = time.time()
            frames.append(frame)  # frame is already a numpy array, HWC, uint8
            decode_time = time.time() - decode_start
            decode_times.append(decode_time)
            if self.is_primary_worker:
                self.global_decode_times.append(decode_time)

            n_frames += 1
            total_frames_processed += 1

            # Store timing info for primary worker to return later
            if self.is_primary_worker and total_frames_processed % LOG_EVERY == 0:
                # Use global timing data accumulated across all videos processed by this worker
                avg_decode_time = sum(self.global_decode_times) / len(self.global_decode_times) if self.global_decode_times else 0
                avg_tensor_time = sum(self.global_tensor_times) / len(self.global_tensor_times) if self.global_tensor_times else 0
                avg_write_time = sum(self.global_write_times) / len(self.global_write_times) if self.global_write_times else 0
                elapsed_time = time.time() - fps_start_time
                fps = total_frames_processed / elapsed_time if elapsed_time > 0 else 0

                # Store for later return (will be logged by main process)
                self.timing_metrics = {
                    "chunks_written_so_far": self.global_chunks_written,
                    "avg_decode_time_per_frame_ms": avg_decode_time * 1000,
                    "avg_tensor_conversion_time_ms": avg_tensor_time * 1000,
                    "avg_file_write_time_ms": avg_write_time * 1000,
                    "fps_frames_per_second": fps
                }

            if n_frames >= chunk_size:
                tensor_start = time.time()
                chunk = to_tensor(frames, output_size)
                tensor_time = time.time() - tensor_start
                tensor_times.append(tensor_time)
                if self.is_primary_worker:
                    self.global_tensor_times.append(tensor_time)

                frames = []

                write_start = time.time()
                # Use faster serialization
                torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"), _use_new_zipfile_serialization=False)
                write_time = time.time() - write_start
                write_times.append(write_time)
                if self.is_primary_worker:
                    self.global_write_times.append(write_time)
                    self.global_chunks_written += 1

                n_frames = 0
                split_ind += 1

                del chunk

        if frames:
            tensor_start = time.time()
            chunk = to_tensor(frames, output_size)
            tensor_time = time.time() - tensor_start
            tensor_times.append(tensor_time)
            if self.is_primary_worker:
                self.global_tensor_times.append(tensor_time)

            write_start = time.time()
            # Use faster serialization
            torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"), _use_new_zipfile_serialization=False)
            write_time = time.time() - write_start
            write_times.append(write_time)
            if self.is_primary_worker:
                self.global_write_times.append(write_time)
                self.global_chunks_written += 1

            split_ind += 1
            del chunk

        # Calculate final metrics for this video (return only, no logging)
        total_time = time.time() - start_time
        video_duration_hours = vid_info['duration'] / 3600.0

        # Calculate average timing metrics for return
        avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
        avg_tensor_time = sum(tensor_times) / len(tensor_times) if tensor_times else 0
        avg_write_time = sum(write_times) / len(write_times) if write_times else 0

        result = {
            "path": path,
            "duration_hours": video_duration_hours,
            "chunks_written": split_ind,
            "avg_decode_time_ms": avg_decode_time * 1000,
            "avg_tensor_time_ms": avg_tensor_time * 1000,
            "avg_write_time_ms": avg_write_time * 1000,
            "is_primary_worker": self.is_primary_worker
        }

        # Add timing metrics if this was the primary worker and we have them
        if self.is_primary_worker and hasattr(self, 'timing_metrics'):
            result.update(self.timing_metrics)

        return result

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
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for split tensors')
    parser.add_argument('--wandb_entity', type=str, default='shahbuland', help='Wandb entity')
    parser.add_argument('--wandb_project_name', type=str, default='video_decoding', help='Wandb project name')
    args = parser.parse_args()

    # Make sure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading video paths on node {args.node_rank}...")
    video_paths = get_video_paths(args.root_dir)
    print(f"Loaded {len(video_paths)} video paths on node {args.node_rank}")

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

    # Prepare wandb config for primary worker
    wandb_config = None
    if args.node_rank == 0:
        wandb_config = {
            "project": args.wandb_project_name,
            "entity": args.wandb_entity,
            "name": f"video_decoding_node_{args.node_rank}",
            "config": {
                "chunk_size": args.chunk_size,
                "output_size": args.output_size,
                "stride": args.stride,
                "num_cpus": args.num_cpus,
                "num_nodes": args.num_nodes,
                "total_videos": len(video_paths),
                "node_rank": args.node_rank
            }
        }

    # Create workers with local ranks
    workers = []
    for local_rank in range(args.num_cpus):
        is_primary_worker = (args.node_rank == 0 and local_rank == 0)
        worker = VideoWorker.remote(local_rank, is_primary_worker)
        workers.append(worker)

    # Initialize wandb only on node 0
    if args.node_rank == 0:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"video_decoding_node_{args.node_rank}",
            config={
                "chunk_size": args.chunk_size,
                "output_size": args.output_size,
                "stride": args.stride,
                "num_cpus": args.num_cpus,
                "num_nodes": args.num_nodes,
                "total_videos": len(video_paths),
                "node_rank": args.node_rank
            }
        )

    # All videos will be processed - workers will handle completion checking
    paths_to_process = video_paths

    if not paths_to_process:
        print(f"Node {args.node_rank}: No videos to process")
        exit()

    # Distribute video paths among workers
    futures = []
    for i, path in enumerate(paths_to_process):
        worker_idx = i % len(workers)
        worker = workers[worker_idx]
        future = worker.decode_video.remote(path,
                                           get_split_dir_from_path(path, args.root_dir, args.output_dir),
                                           get_vid_info_path(path, args.root_dir, args.output_dir),
                                           args.chunk_size,
                                           tuple(args.output_size),
                                           args.stride,
                                           args.force_overwrite)
        futures.append(future)

    # Wait for results with progress bar and track cumulative metrics
    total_videos_completed = 0
    total_hours_processed = 0.0
    total_chunks_written = 0
    ffprobe_errors = 0
    total_videos_attempted = len(paths_to_process)

    # Aggregated timing metrics from all workers
    all_decode_times = []
    all_tensor_times = []
    all_write_times = []

    with tqdm(total=len(futures), desc=f"Processing videos (node {args.node_rank})") as pbar:
        while futures:
            done, futures = ray.wait(futures)
            completed_results = ray.get(done)
            for result in completed_results:
                if isinstance(result, dict):
                    path = result["path"]

                    # Check if this was an ffprobe error
                    if "ffprobe_error" in result:
                        ffprobe_errors += 1
                        print(f"ffprobe error for {path}: {result['ffprobe_error']}")
                    else:
                        # Successful processing
                        duration_hours = result["duration_hours"]
                        chunks_written = result["chunks_written"]
                        total_hours_processed += duration_hours
                        total_chunks_written += chunks_written

                        # Collect timing metrics
                        all_decode_times.append(result["avg_decode_time_ms"])
                        all_tensor_times.append(result["avg_tensor_time_ms"])
                        all_write_times.append(result["avg_write_time_ms"])

                        # Log timing metrics from primary worker if available
                        if args.node_rank == 0 and result.get("is_primary_worker") and "chunks_written_so_far" in result:
                            wandb.log({
                                "chunks_written_so_far": result["chunks_written_so_far"],
                                "avg_decode_time_per_frame_ms": result["avg_decode_time_per_frame_ms"],
                                "avg_tensor_conversion_time_ms": result["avg_tensor_conversion_time_ms"],
                                "avg_file_write_time_ms": result["avg_file_write_time_ms"],
                                "fps_frames_per_second": result["fps_frames_per_second"]
                            })

                else:  # Legacy format (just path)
                    path = result

                total_videos_completed += 1

                # Log cumulative progress for node 0
                if args.node_rank == 0:
                    ffprobe_error_rate = ffprobe_errors / total_videos_completed if total_videos_completed > 0 else 0
                    wandb.log({
                        "cumulative_videos_completed": total_videos_completed,
                        "cumulative_hours_processed": total_hours_processed,
                        "cumulative_chunks_written": total_chunks_written,
                        "ffprobe_error_rate": ffprobe_error_rate,
                        "progress_percentage": (total_videos_completed / len(paths_to_process)) * 100
                    })

            pbar.update(len(done))

    print(f"Node {args.node_rank}: All videos processed!")

    # Final summary logging for node 0
    if args.node_rank == 0:
        final_ffprobe_error_rate = ffprobe_errors / total_videos_attempted if total_videos_attempted > 0 else 0

        # Calculate overall average timing metrics
        avg_decode_time = sum(all_decode_times) / len(all_decode_times) if all_decode_times else 0
        avg_tensor_time = sum(all_tensor_times) / len(all_tensor_times) if all_tensor_times else 0
        avg_write_time = sum(all_write_times) / len(all_write_times) if all_write_times else 0

        wandb.log({
            "final_videos_completed": total_videos_completed,
            "final_hours_processed": total_hours_processed,
            "final_chunks_written": total_chunks_written,
            "final_ffprobe_error_rate": final_ffprobe_error_rate,
            "final_avg_decode_time_ms": avg_decode_time,
            "final_avg_tensor_time_ms": avg_tensor_time,
            "final_avg_write_time_ms": avg_write_time,
            "node_processing_complete": 1
        })
        wandb.finish()

        print(f"Summary: Processed {total_videos_completed} videos, {total_hours_processed:.2f} hours of content, wrote {total_chunks_written} chunks")
        print(f"Average timings: decode={avg_decode_time:.2f}ms, tensor={avg_tensor_time:.2f}ms, write={avg_write_time:.2f}ms")