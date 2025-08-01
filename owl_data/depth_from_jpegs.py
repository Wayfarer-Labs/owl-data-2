import os
import argparse
import ray
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor
import itertools

def find_all_jpegs_structured(root_dir, overwrite=False):
    """Ultra-fast JPEG discovery using known directory structure"""
    jpeg_paths = []
    
    # First, discover the top-level directories (0/, 1/, 2/, etc.)
    top_level_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            top_level_dirs.append(item)
    
    top_level_dirs.sort(key=int)  # Sort numerically
    
    for top_dir in top_level_dirs:
        top_path = os.path.join(root_dir, top_dir)
        
        # Get subdirectories (000000/, 000001/, etc.)
        try:
            subdirs = [d for d in os.listdir(top_path) 
                      if os.path.isdir(os.path.join(top_path, d))]
            subdirs.sort()
        except OSError:
            continue
            
        for subdir in subdirs:
            subdir_path = os.path.join(top_path, subdir)
            
            # Generate all possible JPEG paths (0000.jpg to 0999.jpg)
            for i in range(1000):
                jpeg_name = f"{i:04d}.jpg"
                jpeg_path = os.path.join(subdir_path, jpeg_name)
                
                # Only check if file exists (much faster than os.walk)
                if os.path.exists(jpeg_path):
                    if overwrite:
                        jpeg_paths.append(jpeg_path)
                    else:
                        # Check if depth file exists
                        depth_path = os.path.join(subdir_path, f"{i:04d}.depth.jpg")
                        if not os.path.exists(depth_path):
                            jpeg_paths.append(jpeg_path)
    
    return jpeg_paths

def split_list(lst, n):
    # Splits lst into n nearly equal parts
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

@ray.remote(num_gpus=1)
def process_jpegs_on_gpu(jpeg_paths, gpu_id):
    from owl_data.nn.depth_pipeline import DepthPipeline

    depth_pipeline = DepthPipeline()
    for img_path in tqdm(jpeg_paths, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            depth = depth_pipeline(img)
            # Save depth map as .depth.jpg (or .depth.jpeg)
            base, ext = os.path.splitext(img_path)
            depth_path = base + ".depth.jpg"
            cv2.imwrite(depth_path, depth)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compute depth maps for all JPEGs in a directory using 8 GPUs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JPEG images")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth maps")
    args = parser.parse_args()

    jpeg_paths = find_all_jpegs_structured(args.input_dir, args.overwrite)
    if len(jpeg_paths) == 0:
        print(f"No JPEG files found in {args.input_dir}")
        return

    num_gpus = args.num_gpus
    splits = split_list(jpeg_paths, num_gpus)

    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    result_refs = []
    for gpu_id, split in enumerate(splits):
        if len(split) == 0:
            continue
        ref = process_jpegs_on_gpu.remote(split, gpu_id)
        result_refs.append(ref)

    ray.get(result_refs)
    print("All depth map jobs completed.")

if __name__ == "__main__":
    main()
