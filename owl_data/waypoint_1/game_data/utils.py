import os
import csv
import json
import math
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from dotenv import load_dotenv


load_dotenv()


def create_s3_client() -> BaseClient:
    """
    Create an S3-compatible client (Tigris) using environment variables.

    Required env vars:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_ENDPOINT_URL_S3
    - AWS_REGION (optional; defaults to 'us-east-1' if unset)
    """
    boto_config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 5, "mode": "standard"},
    )
    region = os.getenv("AWS_REGION") or "us-east-1"
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        region_name=region,
        config=boto_config,
    )
    return client


def paginate_objects(
    s3: BaseClient, bucket: str, prefix: Optional[str] = None
) -> Iterable[Dict]:
    """
    Yield all objects from list_objects_v2 with proper pagination.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pagination_params = {"Bucket": bucket}
    if prefix:
        pagination_params["Prefix"] = prefix

    for page in paginator.paginate(**pagination_params):
        contents = page.get("Contents", [])
        for obj in contents:
            yield obj


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(rows: List[Dict], out_csv_path: str) -> None:
    ensure_dir(os.path.dirname(out_csv_path))
    if not rows:
        fieldnames = [
            "key",
            "size_bytes",
            "size_mb",
            "last_modified_iso",
            "last_modified_epoch",
        ]
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def summarize_sizes_mb(sizes_mb: List[float]) -> Dict:
    sizes_sorted = sorted(sizes_mb)
    return {
        "count": len(sizes_mb),
        "mean_mb": float(sum(sizes_mb) / len(sizes_mb)) if sizes_mb else math.nan,
        "min_mb": sizes_sorted[0] if sizes_sorted else math.nan,
        "max_mb": sizes_sorted[-1] if sizes_sorted else math.nan,
        "p50_mb": percentile(sizes_sorted, 50),
        "p75_mb": percentile(sizes_sorted, 75),
        "p90_mb": percentile(sizes_sorted, 90),
        "p95_mb": percentile(sizes_sorted, 95),
        "zero_mb_files": sum(1 for v in sizes_mb if v == 0.0),
    }


def bucketize_sizes_mb(sizes_mb: List[float]) -> Dict[str, int]:
    buckets = {
        "0MB": 0,
        "(0, 10]MB": 0,
        "(10, 100]MB": 0,
        ">100MB": 0,
    }
    for v in sizes_mb:
        if v == 0:
            buckets["0MB"] += 1
        elif 0 < v <= 10:
            buckets["(0, 10]MB"] += 1
        elif 10 < v <= 100:
            buckets["(10, 100]MB"] += 1
        else:
            buckets[">100MB"] += 1
    return buckets


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