import os
import io
import tarfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import cv2
from dotenv import load_dotenv

from owl_data.waypoint_1.game_data.utils import create_s3_client, paginate_objects, ensure_dir


load_dotenv()


def list_tars_in_range(
    s3,
    bucket: str,
    prefix: Optional[str],
    min_size_mb: float,
    max_size_mb: float,
) -> List[Dict]:
    matched: List[Dict] = []
    for obj in paginate_objects(s3, bucket, prefix):
        key = obj.get("Key", "")
        if not key.endswith(".tar"):
            continue
        size_bytes = int(obj.get("Size", 0))
        size_mb = size_bytes / (1024 * 1024)
        if min_size_mb <= size_mb <= max_size_mb:
            matched.append({
                "key": key,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "last_modified": obj.get("LastModified"),
            })
    matched.sort(key=lambda r: r["size_bytes"])  # ascending
    return matched


def extract_tar_stream_to_dir(tar_bytes: bytes, out_dir: str) -> Tuple[bool, Optional[str]]:
    ensure_dir(out_dir)
    if not tar_bytes or len(tar_bytes) < 512:
        return False, f"empty_or_truncated_tar_bytes={len(tar_bytes)}"

    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        try:
            common = os.path.commonpath([abs_directory, abs_target])
        except ValueError:
            return False
        return common == abs_directory

    def safe_extract(tf: tarfile.TarFile, path: str) -> None:
        for member in tf.getmembers():
            member_name = os.path.normpath(member.name).lstrip("/")
            dest_path = os.path.join(path, member_name)
            if not is_within_directory(path, dest_path):
                continue
            if member.isdir():
                os.makedirs(dest_path, exist_ok=True)
                continue
            if member.issym() or member.islnk():
                continue
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            fsrc = tf.extractfile(member)
            if fsrc is None:
                continue
            with open(dest_path, "wb") as fdst:
                fdst.write(fsrc.read())

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            safe_extract(tf, out_dir)
            return True, None
    except (tarfile.ReadError, tarfile.TarError):
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r|*") as tf:
                safe_extract(tf, out_dir)
                return True, None
        except Exception as e2:
            return False, f"tar_read_error: {type(e2).__name__}: {e2}"


def count_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count_prop and frame_count_prop > 0:
        cap.release()
        return frame_count_prop
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count


def analyze_extracted_dir(root_dir: str) -> Dict:
    mp4_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(dirpath, name))

    total_frames = 0
    per_file_frames: List[Tuple[str, int]] = []
    for mp4 in mp4_files:
        frames = count_video_frames(mp4)
        per_file_frames.append((mp4, frames))
        total_frames += frames

    return {
        "root_dir": root_dir,
        "mp4_count": len(mp4_files),
        "total_frames": total_frames,
        "per_file_frames": per_file_frames,
    }


def parse_ranges(ranges: List[str]) -> List[Tuple[float, float]]:
    parsed: List[Tuple[float, float]] = []
    for r in ranges:
        parts = r.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid range '{r}'. Use 'min,max' in MB.")
        min_mb = float(parts[0].strip())
        max_mb = float(parts[1].strip())
        if min_mb > max_mb:
            min_mb, max_mb = max_mb, min_mb
        parsed.append((min_mb, max_mb))
    return parsed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and analyze TARs by size ranges")
    parser.add_argument("--bucket", default=os.getenv("GAME_DATA_BUCKET", "game-data"))
    parser.add_argument("--prefix", default=os.getenv("GAME_DATA_PREFIX", None))
    parser.add_argument(
        "--range",
        action="append",
        dest="ranges",
        required=True,
        help="Size range in MB as 'min,max'. Can be provided multiple times.",
    )
    parser.add_argument(
        "--download-dir",
        default=os.path.join(os.path.dirname(__file__), "tars_by_size"),
        help="Directory to store downloaded/extracted TARs",
    )
    parser.add_argument("--limit", type=int, default=50, help="Per-range limit of TARs to analyze")

    args = parser.parse_args()

    s3 = create_s3_client()
    ranges = parse_ranges(args.ranges)

    for (min_mb, max_mb) in ranges:
        print(f"\n=== Range: {min_mb} MB to {max_mb} MB ===")
        matched = list_tars_in_range(s3, args.bucket, args.prefix, min_mb, max_mb)
        if args.limit and len(matched) > args.limit:
            matched = matched[: args.limit]
        print(f"Matched TARs: {len(matched)}")

        results: List[Dict] = []

        for i, meta in enumerate(matched, start=1):
            key = meta["key"]
            size_mb = meta["size_mb"]
            lm = meta["last_modified"]
            lm_str = lm.astimezone(timezone.utc).isoformat() if isinstance(lm, datetime) else ""

            tar_bytes_io = io.BytesIO()
            print(f"[{i}/{len(matched)}] Downloading {key} ({size_mb:.3f} MB)")
            s3.download_fileobj(args.bucket, key, tar_bytes_io)
            tar_bytes = tar_bytes_io.getvalue()

            out_dir = os.path.join(
                args.download_dir,
                f"{int(min_mb)}_{int(max_mb)}MB",
                key.replace("/", "__").replace(".tar", ""),
            )
            print(f"Extracting to {out_dir}")
            ok, err = extract_tar_stream_to_dir(tar_bytes, out_dir)
            if not ok:
                print(f"  WARN: extraction failed: {err}")
                result = {
                    "key": key,
                    "size_mb": size_mb,
                    "last_modified": lm_str,
                    "mp4_count": 0,
                    "total_frames": 0,
                    "error": err,
                }
                results.append(result)
                continue

            stats = analyze_extracted_dir(out_dir)
            result = {
                "key": key,
                "size_mb": size_mb,
                "last_modified": lm_str,
                "mp4_count": stats["mp4_count"],
                "total_frames": stats["total_frames"],
            }
            results.append(result)
            print(f"  mp4_count={result['mp4_count']}, total_frames={result['total_frames']}")

        # Per-range summary
        total_mp4s = sum(r["mp4_count"] for r in results)
        total_frames = sum(r["total_frames"] for r in results)
        zero_video = sum(1 for r in results if r["mp4_count"] == 0)

        print("\nRange Summary:")
        print(f"TARs analyzed: {len(results)} | total_mp4s={total_mp4s} | total_frames={total_frames} | tar_with_no_videos={zero_video}")

        # Error breakdown per range
        error_groups: Dict[str, List[str]] = {}
        for r in results:
            err = r.get("error")
            if err:
                error_groups.setdefault(err, []).append(r["key"])

        print("Error Breakdown:")
        if not error_groups:
            print("  No errors.")
        else:
            for err, keys in sorted(error_groups.items(), key=lambda kv: len(kv[1]), reverse=True):
                print(f"  {err}: count={len(keys)}")
                examples = ", ".join(keys[:5])
                print(f"    examples: {examples}")


if __name__ == "__main__":
    import sys
    sys.argv[1:] = [
        "--bucket", "game-data",
        "--prefix", "",
        "--range", "0,5",
        # "--range", "185,200",
        "--range", "205,500",
        "--limit", "20",
    ]
    main()
