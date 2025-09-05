import os
import argparse
import torch
import ray
from tqdm import tqdm
import av
import cv2
import numpy as np
from typing import List, Tuple, Dict

def find_all_video_files(root_dir: str, suffix: str = '.mp4') -> List[str]:
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(suffix.lower()):
                video_files.append(os.path.join(dirpath, filename))
    return sorted(video_files)

def extract_frames_from_video(video_path: str, frame_skip: int = 1, max_frames: int = 100, resize_to: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        for frame in container.decode(video_stream):
            if frame_count % (frame_skip + 1) == 0 and saved_count < max_frames:
                # Convert to numpy array (RGB format)
                frame_array = frame.to_ndarray(format='rgb24')
                
                # Resize frame
                frame_resized = cv2.resize(frame_array, resize_to, interpolation=cv2.INTER_LINEAR)
                
                # Normalize to [0, 1]
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                
                frames.append(frame_normalized)
                saved_count += 1
            
            frame_count += 1
            
            if saved_count >= max_frames:
                break
        
        container.close()
        
        if frames:
            # Stack frames into tensor [num_frames, height, width, channels]
            frames_tensor = torch.from_numpy(np.stack(frames))
            return frames_tensor
        else:
            return None
            
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

@ray.remote
def process_video_to_tensor(video_path: str, output_dir: str, frame_skip: int, max_frames: int, resize_to: Tuple[int, int], overwrite: bool) -> Dict:
    try:
        # Extract frames
        frames_tensor = extract_frames_from_video(video_path, frame_skip, max_frames, resize_to)
        
        if frames_tensor is None:
            return {
                'video_path': video_path,
                'status': 'error',
                'reason': 'Failed to extract frames'
            }
        
        # Create output filename with numbered format
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create splits directory if it doesn't exist
        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        # Use numbered filename (e.g., 00000000_caption_rgb.pt)
        # For now, we'll use a simple counter, but this could be enhanced
        existing_files = len([f for f in os.listdir(splits_dir) if f.endswith('_caption_rgb.pt')])
        output_filename = f"{existing_files:08d}_caption_rgb.pt"
        output_path = os.path.join(splits_dir, output_filename)
        
        # Check if file already exists and overwrite is false
        if os.path.exists(output_path) and not overwrite:
            return {
                'video_path': video_path,
                'status': 'skipped',
                'reason': f'Tensor file already exists: {output_filename}'
            }
        
        # Save tensor
        torch.save(frames_tensor, output_path)
        
        return {
            'video_path': video_path,
            'status': 'success',
            'output_path': output_path,
            'tensor_shape': frames_tensor.shape,
            'num_frames': frames_tensor.shape[0]
        }
        
    except Exception as e:
        return {
            'video_path': video_path,
            'status': 'error',
            'reason': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Convert video files to RGB tensor files")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for RGB tensor files")
    parser.add_argument("--frame_skip", type=int, default=30,
                       help="Number of frames to skip between saved frames (default: 30)")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum number of frames per video (default: 100)")
    parser.add_argument("--resize_to", type=int, nargs=2, default=[256, 256],
                       help="Resize frames to (height, width) (default: 256 256)")
    parser.add_argument("--suffix", type=str, default=".mp4",
                       help="Video file suffix to search for (default: .mp4)")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use (default: 1)")
    parser.add_argument("--num_nodes", type=int, default=1,
                       help="Number of nodes being used (default: 1)")
    parser.add_argument("--node_rank", type=int, default=0,
                       help="Node rank among all nodes (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing tensor files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find video files
    print(f"Searching for video files in {args.input_dir}...")
    video_files = find_all_video_files(args.input_dir, args.suffix)
    
    if not video_files:
        print(f"No {args.suffix} files found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Split files across nodes if using multiple nodes
    if args.num_nodes > 1:
        total = len(video_files)
        per_node = total // args.num_nodes
        remainder = total % args.num_nodes
        start = args.node_rank * per_node + min(args.node_rank, remainder)
        end = start + per_node + (1 if args.node_rank < remainder else 0)
        node_files = video_files[start:end]
        print(f"Node {args.node_rank}: processing {len(node_files)} files out of {total} total")
    else:
        node_files = video_files
        print(f"Single node: processing {len(node_files)} files")
    
    # Initialize Ray
    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    
    # Split files across GPUs
    num_gpus = args.num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * len(node_files) // num_gpus
        end = (i + 1) * len(node_files) // num_gpus
        splits.append(node_files[start:end])
    
    # Process videos in parallel
    result_refs = []
    for gpu_id, file_split in enumerate(splits):
        if file_split:
            for video_path in file_split:
                ref = process_video_to_tensor.remote(
                    video_path,
                    args.output_dir,
                    args.frame_skip,
                    args.max_frames,
                    tuple(args.resize_to),
                    args.overwrite
                )
                result_refs.append(ref)
    
    print("Converting videos to RGB tensors...")
    results = ray.get(result_refs)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nConversion complete!")
    print(f"Node {args.node_rank} summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
    if successful > 0:
        print(f"\nGenerated {successful} RGB tensor files in {args.output_dir}")
        print("Sample files:")
        for result in results[:3]:
            if result['status'] == 'success':
                print(f"  {os.path.basename(result['output_path'])} - Shape: {result['tensor_shape']}")
    
    if skipped > 0:
        print(f"\nSkipped {skipped} files (already exist). Use --overwrite to force regeneration.")
    
    if errors > 0:
        print("\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['video_path']}: {result['reason']}")

if __name__ == "__main__":
    main() 