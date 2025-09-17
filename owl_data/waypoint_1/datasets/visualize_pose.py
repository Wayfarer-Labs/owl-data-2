import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def render_pose_preview(pose_bool: torch.Tensor, frame_idx: int = 0, save_path: Optional[str] = None):
    """
    Visualize one frame from [N,1,H,W] torch.bool (unpacked).
    - pose_bool: [N,1,H,W], bool
    - frame_idx: which frame to render
    - save_path: if given, save PNG/JPEG; otherwise just return numpy image

    Returns np.ndarray [H,W] uint8 (0/255).
    """
    assert pose_bool.ndim == 4 and pose_bool.shape[1] == 1, "Expected [N,1,H,W]"
    frame = pose_bool[frame_idx,0].cpu().numpy().astype(np.uint8) * 255  # 0/255
    if save_path is not None:
        cv2.imwrite(str(save_path), frame)
        print(f"Saved pose preview to {save_path}")
    return frame


if __name__ == "__main__":
    log_path = Path('/mnt/data/sami/logs/tensors_to_pose.log')
    with open(log_path, 'r') as f: lines = f.readlines()
    # -- count total pose generated from log file 
    pose_count = sum(int('"has_pose": true' in line) for line in lines)
    no_pose_count = sum(int('"has_pose": false' in line) for line in lines)
    print(f"Total pose generated: {pose_count}")
    total_count = no_pose_count + pose_count
    print(f"Total pose generated count: {total_count}")
    all_pose_paths = list(Path('/mnt/data/waypoint_1/normalized360').glob('**/**/*_pose.pt'))
    print(f"Total pose count overall: " + str(len(all_pose_paths)))
    all_rgb_paths = list(Path('/mnt/data/waypoint_1/normalized360').glob('**/**/*_rgb.pt'))
    print(f"Total _rgb.pt count: " + str(len(all_rgb_paths)))

    unique_pose_paths = set(all_pose_paths)
    unique_rgb_paths = set(all_rgb_paths)
    # save paths
    with open('pose_paths.txt', 'w') as f:
        for pose_path in unique_pose_paths:
            f.write(str(pose_path) + '\n')
    with open('rgb_paths.txt', 'w') as f:
        for rgb_path in unique_rgb_paths:
            f.write(str(rgb_path) + '\n')

    chunk_to_rgb_path = {int(rgb_path.stem.split('_')[-2]): rgb_path for rgb_path in unique_rgb_paths} 
    chunk_to_pose_path = {int(pose_path.stem.split('_')[-2]): pose_path for pose_path in unique_pose_paths}

    with open('rgb_but_no_pose.txt', 'w') as f:
        for chunk in chunk_to_rgb_path.keys() - chunk_to_pose_path.keys():
            f.write(str(chunk_to_rgb_path[chunk]) + '\n')

    # -- count total .pt without pose
    # -- visualize random subset of things without pose in its own directory
    pass
