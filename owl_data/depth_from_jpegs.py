import os
import argparse
import ray
from tqdm import tqdm
import cv2

def find_all_jpegs(root_dir):
    jpeg_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
                jpeg_paths.append(os.path.join(dirpath, fname))
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
    args = parser.parse_args()

    jpeg_paths = find_all_jpegs(args.input_dir)
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
