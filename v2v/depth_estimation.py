"""
Wrapper script to run Video-Depth-Anything and convert output to numpy arrays.
This script reads video path from stdin, runs depth estimation, and returns depth maps.
"""
import sys
import json
import subprocess
import numpy as np
import os
from pathlib import Path
import shutil
import cv2

def run_video_depth_anything(video_path, output_dir, encoder='vitl'):
    """
    Run Video-Depth-Anything depth estimation.
    """
    # Create temporary output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the depth estimation command
    cmd = [
        'python3', 
        'Video-Depth-Anything/run.py',
        '--input_video', video_path,
        '--output_dir', output_dir,
        '--encoder', encoder
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Depth estimation failed: {result.stderr}")
    
    return output_dir

def load_depth_from_output(output_dir, video_path, num_frames=None, height=None, width=None):
    """
    Load depth maps from Video-Depth-Anything output directory.
    The tool typically saves depth maps as images in a subdirectory.
    """
    video_name = Path(video_path).stem
    
    # Find depth output directory (adjust based on actual output structure)
    depth_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(depth_dir):
        # Try alternative naming
        depth_dir = output_dir
    
    # Load all depth images
    depth_files = sorted([f for f in os.listdir(depth_dir) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    
    depth_maps = []
    for depth_file in depth_files:
        depth_path = os.path.join(depth_dir, depth_file)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if dimensions specified
        if height and width:
            depth_img = cv2.resize(depth_img, (width, height))
        
        # Normalize to 0-1 range
        depth_map = depth_img.astype(np.float32) / 255.0
        depth_maps.append(depth_map)
    
    # If num_frames specified, sample uniformly
    if num_frames and len(depth_maps) > num_frames:
        indices = np.linspace(0, len(depth_maps)-1, num_frames, dtype=int)
        depth_maps = [depth_maps[i] for i in indices]
    
    return np.stack(depth_maps)

if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    video_path = input_data['video_path']
    num_frames = input_data.get('num_frames', None)
    height = input_data.get('height', None)
    width = input_data.get('width', None)
    encoder = input_data.get('encoder', 'vitl')
    output_npy_path = input_data.get('output_path', None)
    
    # Create temporary output directory
    temp_output_dir = f"/tmp/depth_output_{Path(video_path).stem}"
    
    try:
        # Run depth estimation
        depth_output_dir = run_video_depth_anything(video_path, temp_output_dir, encoder)
        
        # Load depth maps
        depth_maps = load_depth_from_output(
            depth_output_dir, 
            video_path, 
            num_frames, 
            height, 
            width
        )
        
        # Save as numpy array
        if output_npy_path is None:
            video_name = Path(video_path).stem
            output_npy_path = f"/tmp/depth_{video_name}.npy"
        
        np.save(output_npy_path, depth_maps)
        
        # Clean up temporary files
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        
        # Return result
        result = {
            'success': True,
            'depth_path': output_npy_path,
            'shape': list(depth_maps.shape)
        }
        print(json.dumps(result))
        
    except Exception as e:
        result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(result))
        sys.exit(1)