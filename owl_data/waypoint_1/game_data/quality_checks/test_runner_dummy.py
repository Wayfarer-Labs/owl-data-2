#!/usr/bin/env python3
"""
Quick local test harness for `runner.py` with dummy analyzers and synthetic TARs.

What it does
-------------
1) Generates a few **fake downsampled TARs** on your local disk with the expected layout:
   - chunks/000000.mp4 (tiny bytes, not a real video)
   - <session>.csv, <session>.json, in_video_metadata.json
2) Defines **dummy analyzers** that emit deterministic metrics/flags without reading video bytes.
3) Calls `runner.run_pipeline(...)` directly with those analyzers and writes Parquet outputs locally.

Usage
------
python test_runner_dummy.py \
  --num-tars 3 \
  --chunks-per-tar 4 \
  --workdir /tmp/gd_test \
  --out /tmp/gd_out \
  --run-id TEST-RUN \
  --writer-id node0

After running, inspect outputs under:
  /tmp/gd_out/table=chunks/run_id=TEST-RUN/
  /tmp/gd_out/table=flags/run_id=TEST-RUN/
  /tmp/gd_out/table=errors/run_id=TEST-RUN/
  /tmp/gd_out/table=runs/run_id=TEST-RUN/meta.parquet
  /tmp/gd_out/pointers/latest.json

Notes
-----
- These TARs are not real videos, but that's fine: our analyzers don't decode frames.
- If you prefer to use the CLI of runner.py instead, create a tasks.txt listing the TAR paths
  that this script prints at the end, and run runner.py with --tasks-file and --out.
"""
from __future__ import annotations

import argparse
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List

from runner import run_pipeline

# -----------------------------
# Dummy analyzers (no video decode)
# -----------------------------

class MetricFillerAnalyzer:
    """Fills fps/width/height with constants just to exercise schema."""
    name = "metric_filler"
    version = "0.0.1"

    def analyze_chunk(self, ctx, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]:
        return {"fps": 10.0, "width": 426, "height": 240, "flags": []}

class RandomFlagAnalyzer:
    """Deterministic RNG per (tar_id, chunk_idx) to emit some flags."""
    name = "random_flags"
    version = "0.0.1"

    def analyze_chunk(self, ctx, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]:
        seed = hash((ctx.tar_id, chunk_idx)) & 0xFFFFFFFF
        rng = random.Random(seed)
        flags = []
        # Simulate black frames and input inactivity with pseudo-probabilities
        pct_black = rng.uniform(0, 1)
        flags.append({
            "name": "video.black_frames",
            "value": pct_black > 0.6,
            "score": pct_black,
            "analyzer": self.name,
            "analyzer_version": self.version,
            "model": "heuristic",
            "prompt_id": None,
            "inputs_digest": "sha256:dummy-thr-0.6",
            "extras": {"thr": "0.6"},
        })
        rate = rng.uniform(0, 0.5)  # fake keypress rate Hz
        flags.append({
            "name": "input.no_activity",
            "value": rate < 0.05,
            "score": rate,
            "analyzer": self.name,
            "analyzer_version": self.version,
            "model": "heuristic",
            "prompt_id": None,
            "inputs_digest": "sha256:dummy-thr-0.05",
            "extras": {"thr": "0.05"},
        })
        return {"flags": flags}

class FakeVLMHudAnalyzer:
    """Pretends to be a VLM checking for HUD presence and emitting provenance."""
    name = "vlm_labeler"
    version = "0.0.3"

    def __init__(self, model_id: str = "gpt-4o-mini-FAKE", prompt_id: str = "hud_v1"):
        self.model_id = model_id
        self.prompt_id = prompt_id
        self._digest = "sha256:pretend-prompt-digest"

    def analyze_chunk(self, ctx, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]:
        # Simple pattern: even chunks -> HUD present
        present = (chunk_idx % 2 == 0)
        return {"flags": [{
            "name": "scene.hud_present",
            "value": present,
            "score": 0.95 if present else 0.15,
            "analyzer": self.name,
            "analyzer_version": self.version,
            "model": self.model_id,
            "prompt_id": self.prompt_id,
            "inputs_digest": self._digest,
            "extras": {"note": "dummy"},
        }]}

DUMMY_ANALYZERS = [MetricFillerAnalyzer(), RandomFlagAnalyzer(), FakeVLMHudAnalyzer()]

# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate dummy TARs and run runner with fake analyzers.")
    ap.add_argument("--num-tars", type=int, default=3)
    ap.add_argument("--chunks-per-tar", type=int, default=4)
    ap.add_argument("--workdir", type=str, default="/tmp/gd_test")
    ap.add_argument("--out", type=str, default="/tmp/gd_out")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--writer-id", type=str, default=None)
    ap.add_argument("--sample-dir", type=str, default="/mnt/data/datasets/downsampled_tars")
    ap.add_argument("--sample-size", type=int, default=10)
    args = ap.parse_args()

    work = Path(args.workdir)
    out = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Get list of TARs to process
    tars: List[str] = []
    sample_dir = Path(args.sample_dir)
    all_tars = sorted(p for p in sample_dir.rglob("*.tar") if p.is_file())

    if not all_tars:
        raise SystemExit(f"No .tar files found under {sample_dir}. Populate it first.")

    sample_n = min(args.sample_size, len(all_tars))
    tars = [str(p) for p in random.sample(all_tars, sample_n)]
    print(f"Selected {len(tars)} TARs from {sample_dir} (of {len(all_tars)} total).")
    
    # 2) Run pipeline directly (bypass analyzers.registry discovery)
    run_id = args.run_id or uuid.uuid4().hex
    print(f"RUN_ID={run_id}")
    run_pipeline(
        tasks=tars,
        out_prefix=str(out),
        analyzers=DUMMY_ANALYZERS,
        git_commit=os.getenv("GIT_COMMIT", "unknown"),
        run_id=run_id,
        writer_id=args.writer_id,
        num_workers=0,
    )

    # 3) Help the user locate artifacts
    print("\nWrote outputs to:")
    print(f"  {out}/table=chunks/run_id={run_id}")
    print(f"  {out}/table=flags/run_id={run_id}")
    print(f"  {out}/table=errors/run_id={run_id}")
    print(f"  {out}/table=runs/run_id={run_id}/meta.parquet")
    print(f"  {out}/pointers/latest.json")
    print("\nPro tip: DuckDB local query example:\n")
    print("  duckdb -c \"SELECT count(*) FROM read_parquet('" + str(out) + f"/table=chunks/run_id={run_id}/*.parquet')\"")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
