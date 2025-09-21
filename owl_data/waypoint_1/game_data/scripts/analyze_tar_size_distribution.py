import os
import json
import math
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from owl_data.waypoint_1.game_data.utils import (
    create_s3_client,
    paginate_objects,
    ensure_dir,
    write_csv,
    try_import_matplotlib,
    summarize_sizes_mb,
    bucketize_sizes_mb,
)


load_dotenv()


def collect_tar_inventory(
    s3, bucket: str, prefix: Optional[str] = None
) -> List[Dict]:
    """
    Collect metadata for .tar objects in the bucket/prefix.
    Returns list of dicts with keys: key, size_bytes, size_mb, last_modified_iso, last_modified_epoch
    """
    inventory: List[Dict] = []
    for obj in paginate_objects(s3, bucket, prefix):
        key = obj.get("Key", "")
        if not key.endswith(".tar"):
            continue
        size_bytes = int(obj.get("Size", 0))
        last_modified = obj.get("LastModified")  # tz-aware datetime
        if isinstance(last_modified, datetime):
            lm_utc = last_modified.astimezone(timezone.utc)
            last_modified_iso = lm_utc.isoformat()
            last_modified_epoch = int(lm_utc.timestamp())
        else:
            last_modified_iso = ""
            last_modified_epoch = 0

        inventory.append(
            {
                "key": key,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "last_modified_iso": last_modified_iso,
                "last_modified_epoch": last_modified_epoch,
            }
        )
    return inventory


def plot_histogram(
    sizes_mb: List[float],
    out_path: str,
    title: str,
    bins: int = 50,
    x_max_mb: Optional[float] = None,
) -> Optional[str]:
    plt = try_import_matplotlib()
    if plt is None:
        return None

    ensure_dir(os.path.dirname(out_path))

    filtered = [v for v in sizes_mb if v is not None]
    if x_max_mb is not None:
        filtered = [v for v in filtered if v <= x_max_mb]

    if not filtered:
        plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    plt.figure(figsize=(10, 6))
    plt.hist(filtered, bins=bins, color="#4B8BBE", alpha=0.8, edgecolor="white")
    plt.xlabel("TAR size (MB)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def parse_cutoff_date(cutoff_str: str) -> date:
    return datetime.strptime(cutoff_str, "%Y-%m-%d").date()


def partition_by_cutoff(
    inventory: List[Dict], cutoff_inclusive_date: date
) -> Tuple[List[Dict], List[Dict]]:
    if not inventory:
        return inventory, []

    cutoff_start_utc = datetime.combine(
        cutoff_inclusive_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )

    post_cutoff: List[Dict] = []
    for row in inventory:
        lm_iso = row.get("last_modified_iso")
        if not lm_iso:
            continue
        lm_dt = datetime.fromisoformat(lm_iso.replace("Z", "+00:00"))
        if lm_dt >= cutoff_start_utc:
            post_cutoff.append(row)

    return inventory, post_cutoff


def save_summary_json(
    out_path: str, *, all_stats: Dict, post_cutoff_stats: Dict, all_buckets: Dict, post_cutoff_buckets: Dict
) -> None:
    ensure_dir(os.path.dirname(out_path))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all": {"stats": all_stats, "buckets": all_buckets},
        "post_cutoff": {"stats": post_cutoff_stats, "buckets": post_cutoff_buckets},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Inventory .tar files in an S3-compatible bucket (Tigris), "
            "write CSV, and optionally save histograms and summary stats."
        )
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("GAME_DATA_BUCKET", "game-data"),
        help="Bucket name (default: env GAME_DATA_BUCKET or 'game-data')",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("GAME_DATA_PREFIX", None),
        help="Optional key prefix to scope the listing",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "analysis",
        ),
        help="Directory to write outputs (CSV, PNGs, JSON)",
    )
    parser.add_argument(
        "--cutoff",
        default="2025-08-27",
        help=(
            "Cutoff date (YYYY-MM-DD). Post-cutoff means strictly AFTER this date "
            "(i.e., >= next day 00:00Z). Default: 2025-08-27"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable histogram plotting (CSV and JSON summaries still produced)",
    )

    args = parser.parse_args()

    s3 = create_s3_client()

    print(f"Listing TARs from bucket='{args.bucket}' prefix='{args.prefix or ''}' ...")
    inventory = collect_tar_inventory(s3, args.bucket, args.prefix)

    out_csv = os.path.join(args.out_dir, "tar_inventory.csv")
    write_csv(inventory, out_csv)
    print(f"Wrote CSV: {out_csv} (rows={len(inventory)})")

    cutoff_date = parse_cutoff_date(args.cutoff)
    all_rows, post_rows = partition_by_cutoff(inventory, cutoff_date)

    all_sizes = [float(r["size_mb"]) for r in all_rows]
    post_sizes = [float(r["size_mb"]) for r in post_rows]

    all_stats = summarize_sizes_mb(all_sizes)
    post_stats = summarize_sizes_mb(post_sizes)

    all_buckets = bucketize_sizes_mb(all_sizes)
    post_buckets = bucketize_sizes_mb(post_sizes)

    out_json = os.path.join(args.out_dir, "tar_size_summary.json")
    save_summary_json(
        out_json,
        all_stats=all_stats,
        post_cutoff_stats=post_stats,
        all_buckets=all_buckets,
        post_cutoff_buckets=post_buckets,
    )
    print(f"Wrote summary JSON: {out_json}")

    if not args.no_plots:
        all_hist = os.path.join(args.out_dir, "hist_sizes_all.png")
        post_hist = os.path.join(args.out_dir, "hist_sizes_post_cutoff.png")

        plot_histogram(
            all_sizes,
            all_hist,
            title="TAR Sizes (All)",
        )
        plot_histogram(
            post_sizes,
            post_hist,
            title=f"TAR Sizes (Post-cutoff > {args.cutoff})",
        )

        all_hist_capped = os.path.join(args.out_dir, "hist_sizes_all_le_200mb.png")
        post_hist_capped = os.path.join(args.out_dir, "hist_sizes_post_cutoff_le_200mb.png")
        plot_histogram(
            all_sizes,
            all_hist_capped,
            title="TAR Sizes (All, <= 200 MB)",
            x_max_mb=200,
        )
        plot_histogram(
            post_sizes,
            post_hist_capped,
            title=f"TAR Sizes (Post-cutoff > {args.cutoff}, <= 200 MB)",
            x_max_mb=200,
        )

        print(
            "Histogram images saved (if matplotlib installed). Use --no-plots to skip plotting."
        )

    print("\n=== Quick Summary ===")
    print(f"All TARs: count={all_stats['count']}, zeroMB={all_stats['zero_mb_files']}, meanMB={all_stats['mean_mb']:.2f}")
    print(
        f"Post-cutoff TARs: count={post_stats['count']}, zeroMB={post_stats['zero_mb_files']}, meanMB={post_stats['mean_mb']:.2f}"
    )
    print("Size buckets (All):", all_buckets)
    print("Size buckets (Post-cutoff):", post_buckets)


if __name__ == "__main__":
    main() 