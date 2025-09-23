"""
PT File Visualizer

This script downloads .pt files from the S3 manifest bucket and creates 
visualization grids showing the sampled frames with their timing information.

Each visualization shows:
- All frames from each stride (3s, 30s, 60s)
- Frame timestamps 
- Grid layout organized by stride
- Video metadata and quality information

Usage:
    python pt_from_tar_visualizer.py --pt-keys key1.pt key2.pt --output-dir ./visualizations
    python pt_from_tar_visualizer.py --list-all --limit 10
    python pt_from_tar_visualizer.py --prefix "path/to/" --limit 5
"""

import os
import argparse
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Dict, Tuple
import json
from datetime import datetime

import boto3
from dotenv import load_dotenv

from owl_data.waypoint_1.game_data.utils.s3_utils import (
    download_extracted_data_from_s3, 
    list_pt_files_in_bucket
)
from owl_data.waypoint_1.game_data.owl_types import ExtractedData

load_dotenv()


def parse_stride_info(stride_key: str) -> Tuple[int, str]:
    """
    Parse stride key to extract stride seconds and format.
    
    Args:
        stride_key: Key like "stride-3_chw" or "stride-30_chw"
        
    Returns:
        Tuple of (stride_seconds, format_string)
    """
    # Remove "stride-" prefix and split on "_"
    parts = stride_key.replace("stride-", "").split("_")
    stride_seconds = int(parts[0])
    format_str = parts[1] if len(parts) > 1 else "chw"
    return stride_seconds, format_str


def calculate_frame_timestamps(stride_seconds: int, num_frames: int) -> List[float]:
    """
    Calculate the timestamps for frames based on stride.
    
    Args:
        stride_seconds: Seconds between each frame
        num_frames: Number of frames
        
    Returns:
        List of timestamps in seconds
    """
    return [(i + 1) * stride_seconds for i in range(num_frames)]


def chw_to_hwc(frame_chw: np.ndarray) -> np.ndarray:
    """
    Convert CHW format to HWC format for matplotlib display.
    
    Args:
        frame_chw: Frame in CHW format (channels, height, width)
        
    Returns:
        Frame in HWC format (height, width, channels)
    """
    if len(frame_chw.shape) == 3 and frame_chw.shape[0] == 3:
        return frame_chw.transpose(1, 2, 0)
    return frame_chw


def create_frame_grid_visualization(
    extracted_data: ExtractedData,
    output_path: str,
    max_frames_per_row: int = 8
) -> str:
    """
    Create a grid visualization of all sampled frames from a .pt file.
    
    Args:
        extracted_data: The ExtractedData object containing sampled frames
        output_path: Path where to save the visualization
        max_frames_per_row: Maximum number of frames per row in the grid
        
    Returns:
        Path to the saved visualization
    """
    sampled_frames = extracted_data.sampled_frames
    
    if not sampled_frames:
        logging.warning(f"No sampled frames found for {extracted_data.s3_key}")
        return None
    
    # Calculate total number of frames and grid dimensions
    total_frames = 0
    stride_info = []
    
    for stride_key, frames in sampled_frames.items():
        if frames.size > 0:
            stride_seconds, _ = parse_stride_info(stride_key)
            num_frames = frames.shape[0]
            timestamps = calculate_frame_timestamps(stride_seconds, num_frames)
            stride_info.append({
                'key': stride_key,
                'stride_seconds': stride_seconds,
                'frames': frames,
                'timestamps': timestamps,
                'num_frames': num_frames
            })
            total_frames += num_frames
    
    if total_frames == 0:
        logging.warning(f"No valid frames found for {extracted_data.s3_key}")
        return None
    
    # Sort by stride for consistent ordering
    stride_info.sort(key=lambda x: x['stride_seconds'])
    
    # Calculate grid dimensions
    cols = min(max_frames_per_row, total_frames)
    rows = (total_frames + cols - 1) // cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Set figure title with video info
    video_duration = "Unknown"
    try:
        if extracted_data.video_metadata and 'format' in extracted_data.video_metadata:
            duration_str = extracted_data.video_metadata['format'].get('duration', 'Unknown')
            if duration_str != 'Unknown':
                video_duration = f"{float(duration_str):.1f}s"
    except:
        pass
    
    fig.suptitle(
        f"Video: {extracted_data.video_id}\n"
        f"Duration: {video_duration} | S3 Key: {extracted_data.s3_key}",
        fontsize=10,
        y=0.98
    )
    
    # Plot frames
    frame_idx = 0
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colors for different strides
    
    for stride_idx, stride_data in enumerate(stride_info):
        frames = stride_data['frames']
        timestamps = stride_data['timestamps']
        stride_seconds = stride_data['stride_seconds']
        color = colors[stride_idx % len(colors)]
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            if frame_idx >= len(axes):
                break
                
            ax = axes[frame_idx]
            
            # Convert CHW to HWC and normalize to [0, 1] if needed
            frame_hwc = chw_to_hwc(frame)
            if frame_hwc.dtype == np.uint8:
                frame_hwc = frame_hwc.astype(np.float32) / 255.0
            
            # Display frame
            ax.imshow(frame_hwc)
            ax.set_title(f"{timestamp:.1f}s\n(stride {stride_seconds}s)", fontsize=8)
            ax.axis('off')
            
            # Add colored border to indicate stride
            rect = patches.Rectangle(
                (0, 0), frame_hwc.shape[1], frame_hwc.shape[0],
                linewidth=3, edgecolor=color, facecolor='none',
                transform=ax.transData
            )
            ax.add_patch(rect)
            
            frame_idx += 1
    
    # Hide unused subplots
    for i in range(frame_idx, len(axes)):
        axes[i].axis('off')
    
    # Add stride legend
    legend_elements = []
    for stride_idx, stride_data in enumerate(stride_info):
        color = colors[stride_idx % len(colors)]
        legend_elements.append(
            patches.Patch(color=color, label=f"{stride_data['stride_seconds']}s stride ({stride_data['num_frames']} frames)")
        )
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved visualization to {output_path}")
    return output_path


def create_summary_visualization(
    extracted_data: ExtractedData,
    output_path: str
) -> str:
    """
    Create a summary visualization showing just one representative frame per stride.
    
    Args:
        extracted_data: The ExtractedData object containing sampled frames
        output_path: Path where to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    sampled_frames = extracted_data.sampled_frames
    
    if not sampled_frames:
        logging.warning(f"No sampled frames found for {extracted_data.s3_key}")
        return None
    
    # Get one representative frame from each stride
    stride_frames = []
    for stride_key, frames in sampled_frames.items():
        if frames.size > 0:
            stride_seconds, _ = parse_stride_info(stride_key)
            # Take middle frame as representative
            mid_idx = len(frames) // 2
            representative_frame = frames[mid_idx]
            representative_time = calculate_frame_timestamps(stride_seconds, len(frames))[mid_idx]
            
            stride_frames.append({
                'stride_seconds': stride_seconds,
                'frame': representative_frame,
                'timestamp': representative_time,
                'total_frames': len(frames)
            })
    
    if not stride_frames:
        logging.warning(f"No valid frames found for {extracted_data.s3_key}")
        return None
    
    # Sort by stride
    stride_frames.sort(key=lambda x: x['stride_seconds'])
    
    # Create horizontal layout
    fig, axes = plt.subplots(1, len(stride_frames), figsize=(len(stride_frames) * 3, 4))
    
    if len(stride_frames) == 1:
        axes = [axes]
    
    # Set figure title
    video_duration = "Unknown"
    try:
        if extracted_data.video_metadata and 'format' in extracted_data.video_metadata:
            duration_str = extracted_data.video_metadata['format'].get('duration', 'Unknown')
            if duration_str != 'Unknown':
                video_duration = f"{float(duration_str):.1f}s"
    except:
        pass
    
    fig.suptitle(
        f"Video Summary: {extracted_data.video_id} (Duration: {video_duration})\n"
        f"S3 Key: {extracted_data.s3_key}",
        fontsize=12
    )
    
    # Plot representative frames
    for i, (ax, stride_data) in enumerate(zip(axes, stride_frames)):
        frame = stride_data['frame']
        timestamp = stride_data['timestamp']
        stride_seconds = stride_data['stride_seconds']
        total_frames = stride_data['total_frames']
        
        # Convert CHW to HWC and normalize
        frame_hwc = chw_to_hwc(frame)
        if frame_hwc.dtype == np.uint8:
            frame_hwc = frame_hwc.astype(np.float32) / 255.0
        
        ax.imshow(frame_hwc)
        ax.set_title(
            f"Stride: {stride_seconds}s\n"
            f"Frame at {timestamp:.1f}s\n"
            f"({total_frames} total frames)",
            fontsize=10
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved summary visualization to {output_path}")
    return output_path


def visualize_pt_files(
    manifest_bucket: str,
    pt_s3_keys: List[str],
    output_dir: str,
    create_detailed: bool = True,
    create_summary: bool = True
):
    """
    Download and visualize multiple .pt files from S3.
    
    Args:
        manifest_bucket: Name of the manifest bucket containing .pt files
        pt_s3_keys: List of .pt file S3 keys to visualize
        output_dir: Directory to save visualizations
        create_detailed: Whether to create detailed grid visualizations
        create_summary: Whether to create summary visualizations
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful_visualizations = 0
    failed_visualizations = 0
    
    for i, pt_s3_key in enumerate(pt_s3_keys, 1):
        logging.info(f"Processing [{i}/{len(pt_s3_keys)}]: {pt_s3_key}")
        
        try:
            # Download and load ExtractedData
            extracted_data = download_extracted_data_from_s3(
                s3_client=s3_client,
                manifest_bucket=manifest_bucket,
                pt_s3_key=pt_s3_key
            )
            
            # Create safe filename
            safe_filename = pt_s3_key.replace('/', '_').replace('.pt', '')
            
            # Create detailed visualization
            if create_detailed:
                detailed_output = os.path.join(output_dir, f"{safe_filename}_detailed.png")
                try:
                    create_frame_grid_visualization(extracted_data, detailed_output)
                except Exception as e:
                    logging.error(f"Failed to create detailed visualization for {pt_s3_key}: {e}")
            
            # Create summary visualization
            if create_summary:
                summary_output = os.path.join(output_dir, f"{safe_filename}_summary.png")
                try:
                    create_summary_visualization(extracted_data, summary_output)
                except Exception as e:
                    logging.error(f"Failed to create summary visualization for {pt_s3_key}: {e}")
            
            successful_visualizations += 1
            
        except Exception as e:
            logging.error(f"Failed to process {pt_s3_key}: {e}")
            failed_visualizations += 1
    
    logging.info(f"Visualization complete: {successful_visualizations} successful, {failed_visualizations} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize .pt files from S3 manifest bucket"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pt-keys",
        nargs="+",
        help="Specific .pt file S3 keys to visualize"
    )
    input_group.add_argument(
        "--list-all",
        action="store_true",
        help="Process all .pt files in the bucket"
    )
    input_group.add_argument(
        "--prefix",
        help="Process all .pt files with this prefix"
    )
    
    # Configuration options
    parser.add_argument(
        "--manifest-bucket",
        default="game-data-manifest",
        help="Name of the manifest bucket containing .pt files (default: game-data-manifest)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./pt_visualizations",
        help="Directory to save visualizations (default: ./pt_visualizations)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (useful for testing)",
        default=1
    )
    
    parser.add_argument(
        "--detailed-only",
        action="store_true",
        help="Create only detailed grid visualizations (skip summaries)"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Create only summary visualizations (skip detailed grids)"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    
    # Determine which .pt files to process
    if args.pt_keys:
        pt_s3_keys = args.pt_keys
        logging.info(f"Processing {len(pt_s3_keys)} specified .pt files")
    else:
        prefix = args.prefix if args.prefix else ""
        logging.info(f"Listing .pt files in bucket {args.manifest_bucket} with prefix '{prefix}'...")
        pt_s3_keys = list_pt_files_in_bucket(
            s3_client=s3_client,
            manifest_bucket=args.manifest_bucket,
            prefix=prefix
        )
        
        if args.limit and len(pt_s3_keys) > args.limit:
            pt_s3_keys = random.sample(pt_s3_keys, args.limit)
            logging.info(f"Limited to {args.limit} files")
    
    if not pt_s3_keys:
        logging.warning("No .pt files found to visualize")
        return
    
    # Determine what types of visualizations to create
    create_detailed = not args.summary_only
    create_summary = not args.detailed_only
    
    logging.info(f"Found {len(pt_s3_keys)} .pt files to visualize")
    
    # Process the files
    visualize_pt_files(
        manifest_bucket=args.manifest_bucket,
        pt_s3_keys=pt_s3_keys,
        output_dir=args.output_dir,
        create_detailed=create_detailed,
        create_summary=create_summary
    )
    
    logging.info(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    import sys 
    sys.argv[1:] = ["--list-all"]
    main()