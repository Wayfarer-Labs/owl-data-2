import os
import csv
import json
from tqdm import tqdm

def construct_unlabelled_manifest(input_dir, output_dir, csv_path):
    """
    Scans input_dir for .mp4 files, maps them to output_dir with leading hyphens replaced by underscores,
    and writes a manifest CSV at csv_path with video info from {vid_dir}.json if available.
    """
    mp4_paths = []
    output_dir_paths = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_path = os.path.join(root, file)
                mp4_paths.append(mp4_path)

                # 1. Replace input_dir with output_dir
                rel_path = os.path.relpath(mp4_path, input_dir)
                # 2. Replace any prefixing "-" in the filename with "_"
                dirname, filename = os.path.split(rel_path)
                # Remove .mp4 suffix
                if filename.endswith('.mp4'):
                    base = filename[:-4]
                else:
                    base = filename
                # Replace leading hyphens with underscores
                while base.startswith('-'):
                    base = '_' + base[1:]
                # 3. Compose output path (as a directory, not a file)
                output_dir_path = os.path.join(output_dir, dirname, base)
                output_dir_paths.append(output_dir_path)

    fieldnames = [
        "vid_dir_path",
        "vid_duration",
        "vid_fps",
        "has_rgb_latent",
        "has_depth",
        "has_caption",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for vid_dir in tqdm(output_dir_paths, desc="Reading vid info jsons"):
            base_name = os.path.basename(vid_dir)
            #info_path = os.path.join(vid_dir, f"{base_name}.json")
            info_path = os.path.join(vid_dir, "vid_info.json")
            vid_duration = ""
            vid_fps = ""
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                        vid_duration = info.get("duration", "")
                        vid_fps = info.get("fps", "")
                except Exception:
                    continue  # skip if error reading json
            else:
                continue  # skip if no info json

            # Only write if both duration and fps are present and not empty
            if vid_duration == "" or vid_fps == "":
                continue

            writer.writerow({
                "vid_dir_path": vid_dir,
                "vid_duration": vid_duration,
                "vid_fps": vid_fps,
                "has_rgb_latent": False,
                "has_depth": False,
                "has_caption": False,
            })

def get_paths_for_depth(csv_path):
    pass