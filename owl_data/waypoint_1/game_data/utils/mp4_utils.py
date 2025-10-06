import io
import av
import logging
import numpy as np
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches



def downsample_video_from_path(
    in_video_path: pathlib.Path,
    intervals_seconds: list[float] = [60.] * 10,
    downsampled_fps: int = 10,
    new_height: int = 240,
    crf: int = 18,
) -> list[pathlib.Path]:
    """
    Decodes an in-memory video and downsamples it based on a given specification.
    """
    import subprocess, ffmpeg
    out_path = pathlib.Path('/tmp') / in_video_path.stem
    
    start_time, out_path_chunks = 0., [],
    try: 
        video_duration_seconds = float(ffmpeg.probe(in_video_path)['streams'][0]['duration'])
    except:
        video_duration_seconds = 600.

    for duration in intervals_seconds:
        try:
            out_path_chunk = out_path / f"{out_path.stem}_{start_time:.0f}_{start_time + duration:.0f}.mp4"
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", str(in_video_path),
                "-ss", str(start_time),
                "-t", str(start_time + duration),
                "-vf", f"fps={downsampled_fps},scale=-2:{new_height}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(crf),
                "-an",  # Remove audio
                str(out_path_chunk)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            start_time += duration
            out_path_chunks.append(out_path_chunk)
        except:
            logging.info(f"Error processing chunk {start_time:.0f}_{start_time + duration:.0f} of {video_duration_seconds:.0f} seconds, could be out of range?")
            break

    return out_path_chunks


def main():
    import sys
    import os
    from datetime import datetime
    
    video_path = "/home/sky/owl-data-2/pt_visualizations/tars/2025-09-17 14-59-42.mp4"
    output_dir = "/home/sky/owl-data-2/pt_visualizations/manual_visualizations"

    try:
        logging.info(f"Reading video file from: {video_path}")
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
    except FileNotFoundError:
        logging.error(f"Error: The file was not found at '{video_path}'")
        logging.error("Please update the 'video_path' variable in this script.")
        sys.exit(1)

    # --- Define the desired strides and frame counts here ---
    # Key = stride in seconds, Value = number of frames to sample

    # Run the frame sampling function with the new specification
    sampled_frames = downsample_video_from_path(video_path, strides_to_sample)

    # Print the results to verify
    print("\n--- Frame Sampling Results ---")
    for name, array in sampled_frames.items():
        if array.size > 0:
            # Shape is (num_frames, channels, height, width)
            print(f"'{name}': Found {array.shape[0]} frames with shape {array.shape}")
        else:
            print(f"'{name}': No frames were sampled (video might be too short for this stride).")
    print("----------------------------")

    # Create visualizations
    if any(array.size > 0 for array in sampled_frames.values()):
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video filename without extension for naming
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed grid visualization
        detailed_output_path = os.path.join(output_dir, f"{video_name}_detailed_{timestamp}.png")
        _create_detailed_visualization(sampled_frames, video_name, detailed_output_path)
        
        # Create summary visualization
        summary_output_path = os.path.join(output_dir, f"{video_name}_summary_{timestamp}.png")
        _create_summary_visualization(sampled_frames, video_name, summary_output_path)
        
        print(f"\nVisualizations saved to:")
        print(f"  Detailed: {detailed_output_path}")
        print(f"  Summary: {summary_output_path}")
    else:
        print("No frames to visualize.")


def _create_detailed_visualization(sampled_frames, video_name, output_path):
    """Create a detailed grid visualization showing all frames."""
    # Collect all valid frames with metadata
    all_frames = []
    for stride_key, frames in sampled_frames.items():
        if frames.size > 0:
            stride_seconds = int(stride_key.split('-')[1].split('_')[0])
            for i, frame in enumerate(frames):
                timestamp = i * stride_seconds
                all_frames.append({
                    'frame': frame,
                    'stride_seconds': stride_seconds,
                    'timestamp': timestamp,
                    'stride_key': stride_key
                })
    
    if not all_frames:
        return
    
    # Determine grid layout
    num_frames = len(all_frames)
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    import matplotlib.pyplot as plt
    # Create visualization
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    fig.suptitle(f"Detailed Frame Grid: {video_name}", fontsize=16)
    
    # Colors for different strides
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    stride_colors = {}
    
    for i, frame_data in enumerate(all_frames):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        
        frame = frame_data['frame']
        stride_seconds = frame_data['stride_seconds']
        timestamp = frame_data['timestamp']
        
        # Convert CHW to HWC and normalize
        if frame.shape[0] == 3:  # CHW format
            frame_rgb = np.transpose(frame, (1, 2, 0))
        else:
            frame_rgb = frame
        
        frame_rgb = np.clip(frame_rgb / 255.0, 0, 1)
        
        # Get color for this stride
        if stride_seconds not in stride_colors:
            stride_colors[stride_seconds] = colors[len(stride_colors) % len(colors)]
        
        ax.imshow(frame_rgb)
        ax.set_title(f"{stride_seconds}s stride\nt={timestamp}s", fontsize=10)
        ax.axis('off')
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(stride_colors[stride_seconds])
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # Hide unused subplots
    for i in range(num_frames, rows * cols):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    # Add legend
    legend_elements = []
    for stride_seconds, color in stride_colors.items():
        num_frames_for_stride = sum(1 for f in all_frames if f['stride_seconds'] == stride_seconds)
        legend_elements.append(
            patches.Patch(color=color, label=f"{stride_seconds}s stride ({num_frames_for_stride} frames)")
        )
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _create_summary_visualization(sampled_frames, video_name, output_path):
    """Create a summary visualization showing one representative frame per stride."""
    # Get one representative frame from each stride
    stride_frames = []
    for stride_key, frames in sampled_frames.items():
        if frames.size > 0:
            stride_seconds = int(stride_key.split('-')[1].split('_')[0])
            # Take middle frame as representative
            mid_idx = len(frames) // 2
            representative_frame = frames[mid_idx]
            representative_time = mid_idx * stride_seconds
            
            stride_frames.append({
                'stride_seconds': stride_seconds,
                'frame': representative_frame,
                'timestamp': representative_time,
                'total_frames': len(frames)
            })
    
    if not stride_frames:
        return
    
    # Sort by stride
    stride_frames.sort(key=lambda x: x['stride_seconds'])
    
    # Create horizontal layout
    fig, axes = plt.subplots(1, len(stride_frames), figsize=(len(stride_frames) * 3, 4))
    
    if len(stride_frames) == 1:
        axes = [axes]
    
    fig.suptitle(f"Video Summary: {video_name}", fontsize=12)
    
    # Plot representative frames
    for i, (ax, stride_data) in enumerate(zip(axes, stride_frames)):
        frame = stride_data['frame']
        timestamp = stride_data['timestamp']
        stride_seconds = stride_data['stride_seconds']
        total_frames = stride_data['total_frames']
        
        # Convert CHW to HWC and normalize
        if frame.shape[0] == 3:  # CHW format
            frame_rgb = np.transpose(frame, (1, 2, 0))
        else:
            frame_rgb = frame
        
        frame_rgb = np.clip(frame_rgb / 255.0, 0, 1)
        
        ax.imshow(frame_rgb)
        ax.set_title(f"{stride_seconds}s stride\n({total_frames} frames)\nShowing t={timestamp}s", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()