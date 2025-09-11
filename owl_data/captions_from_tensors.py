import os
import argparse
import torch
import ray
from tqdm import tqdm
import glob
from typing import List, Tuple, Dict
import json
import tempfile
from PIL import Image
import numpy as np
import datetime

# Default prompt template for video captioning
# You can customize this prompt to focus on specific aspects of the video
# For example, you could modify it to focus on:
# - Game-specific events and mechanics
# - Character interactions and behaviors  
# - UI elements and interface changes
# - Audio cues and sound events
DEFAULT_PROMPT_TEMPLATE = """
Analyze this video sequence and provide a detailed description of what is happening in the world/environment. 

Focus on:
- Environmental events and world state changes
- Objects, structures, and their interactions
- Atmospheric conditions and lighting
- Movement and dynamics in the scene
- Spatial relationships and positioning

Important guidelines:
- Describe what is happening in the world, NOT what the player/character is doing
- Focus on observable events and environmental changes
- Be specific about locations, objects, and their states
- Note any significant changes between frames
- Avoid subjective interpretations or game-specific terminology

Provide a clear, objective description of the visual content and environmental events shown in this video sequence.
"""

def find_all_tensor_files(root_dir: str) -> List[str]:
    tensor_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_rgb.pt'):
                tensor_files.append(os.path.join(dirpath, filename))
    return sorted(tensor_files)

def load_tensor_file(tensor_path: str) -> torch.Tensor:
    try:
        tensor = torch.load(tensor_path, map_location='cpu', mmap=True)
        return tensor
    except Exception as e:
        print(f"Error loading tensor {tensor_path}: {e}")
        return None

def create_sliding_windows(num_frames: int, kernel_size: int, dilation: int, stride: int) -> List[Tuple[int, int, List[int]]]:
    windows = []
    
    for window_start in range(0, num_frames - (kernel_size - 1) * dilation, stride):
        frame_indices = []
        for i in range(kernel_size):
            frame_idx = window_start + i * dilation
            if frame_idx < num_frames:
                frame_indices.append(frame_idx)
        
        if len(frame_indices) == kernel_size:
            end_frame = frame_indices[-1]
            windows.append((window_start, end_frame, frame_indices))
    
    return windows

def setup_qwen_model():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import os
        
        # Define cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "qwen")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("Loading Qwen2.5-VL model...")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, "Qwen2.5-VL-3B-Instruct"))
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, "Qwen2.5-VL-3B-Instruct"))
        )
        
        return model, processor, process_vision_info
        
    except ImportError:
        print("Missing dependencies. Installing required packages...")
        import subprocess
        subprocess.check_call([
            "pip", "install", 
            "git+https://github.com/huggingface/transformers", 
            "accelerate",
            "qwen-vl-utils[decord]==0.0.8"
        ])
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import os
        
        # Define cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "qwen")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("Loading Qwen2.5-VL model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=cache_dir
        )
        
        return model, processor, process_vision_info
    except Exception as e:
        print(f"Error setting up model: {e}")
        raise

def tensor_to_images(frames: torch.Tensor) -> List[Image.Image]:
    images = []
    
    for i in range(frames.shape[0]):
        frame = frames[i]  # [H, W, C] or [C, H, W]
        
        # Check if we need to permute
        if frame.shape[-1] != 3:  # If last dim is not 3, assume [C, H, W]
            frame = frame.permute(1, 2, 0)  # Convert to [H, W, C]
        
        frame_np = frame.numpy()
        
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        image = Image.fromarray(frame_np)
        images.append(image)
    
    return images

def generate_caption_for_window(frames: torch.Tensor, vlm_model, processor, process_vision_info, prompt_template: str) -> str:
    try:
        images = tensor_to_images(frames)
        
        temp_files = []
        try:
            for i, img in enumerate(images):
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                img.save(temp_file.name, 'JPEG')
                temp_files.append(temp_file.name)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_template},
                    ]
                }
            ]
            
            for temp_file in temp_files:
                messages[0]["content"].insert(-1, {
                    "type": "image", 
                    "image": f"file://{temp_file}"
                })
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            with torch.no_grad():
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            caption = output_text[0].strip()
            return caption
            
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption"

@ray.remote(num_gpus=1)
class CaptionWorker:
    def __init__(self, kernel_size, dilation, stride, prompt_template, verbose: bool = False):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.prompt_template = prompt_template
        self.verbose = verbose
        self.model, self.processor, self.process_vision_info = setup_qwen_model()
    
    def process_tensor_file(self, tensor_path: str, overwrite: bool = False) -> Dict:
        try:
            # Extract the split directory and filename
            tensor_dir = os.path.dirname(tensor_path)
            tensor_filename = os.path.basename(tensor_path)
            
            # Parse the numbered filename (e.g., "00000000_rgb.pt" -> "00000000")
            if '_rgb.pt' in tensor_filename:
                base_number = tensor_filename.replace('_rgb.pt', '')
            elif '_caption_rgb.pt' in tensor_filename:
                base_number = tensor_filename.replace('_caption_rgb.pt', '')
            else:
                return {
                    'tensor_path': tensor_path,
                    'status': 'error',
                    'reason': f'Unexpected tensor filename format: {tensor_filename}'
                }
            
            # Extract just the number part (e.g., "00000000_caption" -> "00000000")
            if '_caption' in base_number:
                base_number = base_number.replace('_caption', '')
            
            # Create caption filename in the same directory
            caption_filename = f"{base_number}_captions.json"
            caption_path = os.path.join(tensor_dir, caption_filename)
            
            if os.path.exists(caption_path) and not overwrite:
                if getattr(self, "verbose", False):
                    print(f"SKIP: {caption_path} (exists)")
                return {
                    'tensor_path': tensor_path,
                    'status': 'skipped',
                    'reason': 'Caption file already exists'
                }
            
            if getattr(self, "verbose", False):
                print(f"PROCESS: {tensor_path} -> {caption_path}")
            tensor = load_tensor_file(tensor_path)
            if tensor is None:
                return {
                    'tensor_path': tensor_path,
                    'status': 'error',
                    'reason': 'Failed to load tensor'
                }
            
            if tensor.dim() == 4:
                num_frames = tensor.shape[0]
            else:
                return {
                    'tensor_path': tensor_path,
                    'status': 'error',
                    'reason': f'Unexpected tensor shape: {tensor.shape}'
                }
            
            windows = create_sliding_windows(num_frames, self.kernel_size, self.dilation, self.stride)
            
            if not windows:
                return {
                    'tensor_path': tensor_path,
                    'status': 'error',
                    'reason': 'No valid windows created'
                }
            
            captions_data = {
                "metadata": {
                    "video_name": base_number,
                    "total_frames": num_frames,
                    "kernel_size": self.kernel_size,
                    "dilation": self.dilation,
                    "stride": self.stride,
                    "prompt_template": self.prompt_template,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": "Qwen2.5-VL-3B"
                },
                "windows": []
            }
            
            for start_frame, end_frame, frame_indices in windows:
                window_frames = tensor[frame_indices]
                window_frames = (window_frames.float() / 255.0) # -> [0,1]
                caption = generate_caption_for_window(window_frames, self.model, self.processor, self.process_vision_info, self.prompt_template)
                window_data = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_indices": frame_indices,
                    "caption": caption
                }
                captions_data["windows"].append(window_data)
            
            with open(caption_path, 'w') as f:
                json.dump(captions_data, f, indent=2)
            if getattr(self, "verbose", False):
                print(f"WROTE: {caption_path} (windows={len(windows)})")
            
            return {
                'tensor_path': tensor_path,
                'status': 'success',
                'num_windows': len(windows),
                'caption_path': caption_path
            }
            
        except Exception as e:
            return {
                'tensor_path': tensor_path,
                'status': 'error',
                'reason': str(e)
            }

def main():
    parser = argparse.ArgumentParser(description="Generate captions from tensor files using Qwen2.5-VL-3B sliding window approach")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing tensor files (*_rgb.pt)")
    parser.add_argument("--kernel_size", type=int, default=5,
                       help="Number of frames in each window (default: 5)")
    parser.add_argument("--dilation", type=int, default=2,
                       help="Gap between consecutive frames in window (default: 2)")
    parser.add_argument("--stride", type=int, default=3,
                       help="Stride for sliding window (default: 3)")
    parser.add_argument("--prompt_template", type=str, 
                       default=DEFAULT_PROMPT_TEMPLATE,
                       help="Template for captioning prompt")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="Number of GPUs to use (default: 8)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose per-file logging (SKIP/PROCESS/WROTE)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing caption files")
    parser.add_argument("--num_nodes", type=int, default=1,
                       help="Number of nodes being used (default: 1)")
    parser.add_argument("--node_rank", type=int, default=0,
                       help="Node rank among all nodes (default: 0)")
    parser.add_argument("--max_files", type=int, default=0,
                       help="Limit the number of tensor files to process (0 means no limit)")
    
    args = parser.parse_args()
    
    print(f"Searching for tensor files in {args.input_dir}...")
    tensor_files = find_all_tensor_files(args.input_dir)
    
    if not tensor_files:
        print(f"No tensor files found in {args.input_dir}")
        return
    
    print(f"Found {len(tensor_files)} tensor files")
    
    if args.num_nodes > 1:
        total = len(tensor_files)
        per_node = total // args.num_nodes
        remainder = total % args.num_nodes
        start = args.node_rank * per_node + min(args.node_rank, remainder)
        end = start + per_node + (1 if args.node_rank < remainder else 0)
        node_files = tensor_files[start:end]
        print(f"Node {args.node_rank}: processing {len(node_files)} files out of {total} total")
    else:
        node_files = tensor_files
        print(f"Single node: processing {len(node_files)} files")
    
    # Optionally limit how many files to process
    if args.max_files > 0:
        node_files = node_files[:args.max_files]
        print(f"Limiting to first {len(node_files)} files due to --max_files={args.max_files}")
    
    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    
    num_gpus = args.num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * len(node_files) // num_gpus
        end = (i + 1) * len(node_files) // num_gpus
        splits.append(node_files[start:end])
    
    workers = []
    for gpu_id in range(num_gpus):
        worker = CaptionWorker.remote(
            args.kernel_size,
            args.dilation,
            args.stride,
            args.prompt_template,
            args.verbose,
        )
        workers.append(worker)
    
    result_refs = []
    for gpu_id, file_split in enumerate(splits):
        if file_split:
            for tensor_path in file_split:
                ref = workers[gpu_id].process_tensor_file.remote(
                    tensor_path,
                    args.overwrite
                )
                result_refs.append(ref)
    
    print("Processing tensor files with Qwen2.5-VL-3B...")
    results = ray.get(result_refs)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nProcessing complete!")
    print(f"Node {args.node_rank} summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
    if errors > 0:
        print("\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['tensor_path']}: {result['reason']}")

if __name__ == "__main__":
    main()
