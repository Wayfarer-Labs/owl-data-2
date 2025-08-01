import boto3

import os
import argparse
import tarfile
import tempfile
import random
import ray
from multiprocessing import cpu_count
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


def find_jpg_and_depth_pairs(root_dir):
    """
    Walks root_dir and finds all .jpg files (not .depth.jpg), and for each, checks if a corresponding .depth.jpg exists.
    Returns a list of tuples: (rgb_path, depth_path)
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg') and not fname.lower().endswith('.depth.jpg'):
                rgb_path = os.path.join(dirpath, fname)
                base = fname[:-4]  # strip .jpg
                depth_fname = base + '.depth.jpg'
                depth_path = os.path.join(dirpath, depth_fname)
                if os.path.exists(depth_path):
                    pairs.append((rgb_path, depth_path))
    return pairs

def split_list(lst, n):
    # Splits lst into n nearly equal parts
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def upload_file_to_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, s3_key)

@ray.remote
def process_and_upload(rank, pairs, images_per_tar, tars_per_subdir, bucket_name, base_prefix):
    # Shuffle the pairs for this rank
    random.shuffle(pairs)
    # Calculate number of tars
    num_tars = (len(pairs) + images_per_tar - 1) // images_per_tar
    for tar_idx in tqdm(range(num_tars), desc=f"Rank {rank} tarring", position=rank):
        start = tar_idx * images_per_tar
        end = min((tar_idx + 1) * images_per_tar, len(pairs))
        tar_pairs = pairs[start:end]
        # Subdir index for S3
        subdir_idx = tar_idx // tars_per_subdir
        tar_in_subdir_idx = tar_idx % tars_per_subdir
        # S3 key: base_prefix/rank/subdir_idx/tar_in_subdir_idx.tar
        s3_key = f"{base_prefix}/{rank:01d}/{subdir_idx:04d}/{tar_in_subdir_idx:04d}.tar"
        # Create tar in temp file
        with tempfile.NamedTemporaryFile(suffix=".tar") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w") as tar:
                for pair_idx, (rgb_path, depth_path) in enumerate(tar_pairs):
                    # Add rgb
                    arcname_rgb = f"{pair_idx:04d}.jpg"
                    tar.add(rgb_path, arcname=arcname_rgb)
                    # Add depth
                    arcname_depth = f"{pair_idx:04d}.depth.jpg"
                    tar.add(depth_path, arcname=arcname_depth)
            # Upload to S3
            upload_file_to_s3(tmp_tar.name, bucket_name, s3_key)

def main():
    parser = argparse.ArgumentParser(description="Upload shuffled tars of RGB/Depth pairs to S3")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing .jpg and .depth.jpg files")
    parser.add_argument("--images_per_tar", type=int, default=1000, help="Number of image pairs per tar")
    parser.add_argument("--tars_per_subdir", type=int, default=100, help="Number of tars per subdir on S3")
    parser.add_argument("--bucket_name", type=str, default="1x-frames", help="S3 bucket name")
    parser.add_argument("--base_prefix", type=str, default="depth-and-rgb", help="Base prefix in S3 bucket")
    parser.add_argument("--num_cpus", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    # Find all pairs
    print("Finding RGB/Depth pairs...")
    pairs = find_jpg_and_depth_pairs(args.root_dir)
    if len(pairs) == 0:
        print("No RGB/Depth pairs found.")
        return

    # Shuffle all pairs before splitting
    random.shuffle(pairs)

    # Split pairs across workers
    num_cpus = min(args.num_cpus, len(pairs))
    splits = split_list(pairs, num_cpus)

    # Start Ray
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # Launch jobs
    result_refs = []
    for rank, split in enumerate(splits):
        if len(split) == 0:
            continue
        ref = process_and_upload.remote(
            rank, split, args.images_per_tar, args.tars_per_subdir, args.bucket_name, args.base_prefix
        )
        result_refs.append(ref)

    # Wait for all jobs to finish
    ray.get(result_refs)
    print("All uploads completed.")

if __name__ == "__main__":
    main()
