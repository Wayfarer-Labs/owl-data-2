import os
import argparse
import ray
from tqdm import tqdm
from multiprocessing import cpu_count
from local_data.video_reader import extract_frames
import math

def find_all_video_files(root_dir, suffix):
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(f'.{suffix.lower()}'):
                video_files.append(os.path.join(dirpath, fname))
    return video_files

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def extract_frames_with_subdirs(video_path, output_base, frame_skip, images_per_subdir):
    """
    Extract frames from a video and save them in subdirectories, each containing up to images_per_subdir images.
    """
    import av
    from PIL import Image

    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Error: Could not open video file {video_path}: {e}")
        return

    video_stream = container.streams.video[0]
    frame_count = 0
    saved_frame_index = 0

    for frame in container.decode(video_stream):
        if frame_count % (frame_skip + 1) == 0:
            subdir_idx = saved_frame_index // images_per_subdir
            subdir_name = f"{subdir_idx:06d}"
            subdir_path = os.path.join(output_base, subdir_name)
            ensure_dir(subdir_path)
            img_idx = saved_frame_index % images_per_subdir
            filename = f"{img_idx:04d}.jpg"
            filepath = os.path.join(subdir_path, filename)
            frame_array = frame.to_ndarray(format='rgb24')
            pil_image = Image.fromarray(frame_array)
            pil_image.save(filepath, "JPEG")
            saved_frame_index += 1
        frame_count += 1

    container.close()

@ray.remote
def process_videos(rank, video_paths, output_dir, frame_skip, images_per_subdir):
    # All output for this rank goes under output_dir/rank/
    rank_dir = os.path.join(output_dir, f"{rank}")
    ensure_dir(rank_dir)
    frame_counter = 0  # global frame counter for this rank

    for video_path in tqdm(video_paths, desc=f"Rank {rank}", position=rank):
        # For each video, just keep appending frames to the rank's output, using the counting system
        try:
            import av
            from PIL import Image

            container = av.open(video_path)
            video_stream = container.streams.video[0]
            frame_count = 0

            for frame in container.decode(video_stream):
                if frame_count % (frame_skip + 1) == 0:
                    subdir_idx = frame_counter // images_per_subdir
                    subdir_name = f"{subdir_idx:06d}"
                    subdir_path = os.path.join(rank_dir, subdir_name)
                    ensure_dir(subdir_path)
                    img_idx = frame_counter % images_per_subdir
                    filename = f"{img_idx:04d}.jpg"
                    filepath = os.path.join(subdir_path, filename)
                    frame_array = frame.to_ndarray(format='rgb24')
                    pil_image = Image.fromarray(frame_array)
                    pil_image.save(filepath, "JPEG")
                    frame_counter += 1
                frame_count += 1

            container.close()
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

def split_list(lst, n):
    # Splits lst into n nearly equal parts
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    parser = argparse.ArgumentParser(description="Parallel video decoding with Ray")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted jpgs")
    parser.add_argument("--frame_skip", type=int, default=30, help="Number of frames to skip between saved frames")
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPU processes to use (default: all available)")
    parser.add_argument("--suffix", type=str, default="mp4", help="Video file suffix to search for (e.g. mkv, mp4)")
    parser.add_argument("--images_per_subdir", type=int, default=1000, help="Number of images per subdirectory")
    args = parser.parse_args()

    # Find all video files with the given suffix
    video_files = find_all_video_files(args.root_dir, args.suffix)
    if len(video_files) == 0:
        print(f"No .{args.suffix} files found in {args.root_dir}")
        return

    # Determine number of CPUs
    num_cpus = args.num_cpus if args.num_cpus is not None else cpu_count()
    num_cpus = min(num_cpus, len(video_files))  # Don't spawn more workers than files

    # Split video files among workers
    video_splits = split_list(video_files, num_cpus)

    # Start Ray
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # Launch jobs
    result_refs = []
    for rank, video_list in enumerate(video_splits):
        if len(video_list) == 0:
            continue
        ref = process_videos.remote(rank, video_list, args.output_dir, args.frame_skip, args.images_per_subdir)
        result_refs.append(ref)

    # Wait for all jobs to finish
    ray.get(result_refs)
    print("All decoding jobs completed.")

if __name__ == "__main__":
    main()
