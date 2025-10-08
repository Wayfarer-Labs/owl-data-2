#!/usr/bin/env python3
"""
runner.py — Chunk analysis runner with per-flag provenance and Parquet outputs.

This version partitions outputs by a single, globally shared **run_id (UUID)**
and gives each writer a unique **writer_id** so parallel writers never clash.

Outputs (under --out, local or S3):
  table=chunks/run_id=<RUN_ID>/part-<WRITER>-00000.parquet
  table=flags/run_id=<RUN_ID>/part-<WRITER>-00000.parquet
  table=errors/run_id=<RUN_ID>/part-<WRITER>-00000.parquet
  table=runs/run_id=<RUN_ID>/meta.parquet

Assumptions about analyzers:
- discoverable via analyzers.registry.all_analyzers() OR pass your own list
- each analyzer exposes analyze_chunk(ctx, chunk_mp4_path, chunk_idx) -> dict
  where dict may include:
    { 'flags': [ { name, value, score, analyzer, analyzer_version, model,
                   prompt_id, inputs_digest, extras } ],
      'fps': float, 'width': int, 'height': int, ... }

Downsampled TAR layout:
  chunks/<000000>.mp4  (60s @ 10fps, 240p)
  <session>.csv        (controls)
  <session>.json       (metadata)
  in_video_metadata.json (ffprobe)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from owl_data.waypoint_1.game_data.quality_checks.analyzers.base import Analyzer, AnalysisContext

# --- Optional: analyzer registry ---
try:
    from analyzers.registry import all_analyzers  # type: ignore
except Exception:  # allow running with explicit analyzers passed in
    def all_analyzers():
        return []

# ================= Utilities =================

def get_git_commit_fallback() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "unknown")


def resolve_filesystem(prefix: str) -> Tuple[pafs.FileSystem, str]:
    """Return (filesystem, normalized_path) for pyarrow.fs writes.

    prefix: '/mnt/foo' -> (LocalFS, '/mnt/foo')
            's3://bkt/pfx' -> (S3FS, 'bkt/pfx')
    """
    if prefix.startswith("s3://"):
        return pafs.S3FileSystem(), prefix[len("s3://"):]
    return pafs.LocalFileSystem(), os.path.abspath(prefix)

# ================= Schemas =================

FLAG_STRUCT = pa.struct([
    pa.field("name", pa.string()),
    pa.field("value", pa.bool_()),
    pa.field("score", pa.float32()),
    pa.field("analyzer", pa.string()),
    pa.field("analyzer_version", pa.string()),
    pa.field("model", pa.string()),
    pa.field("prompt_id", pa.string()),
    pa.field("inputs_digest", pa.string()),
    pa.field("extras", pa.map_(pa.string(), pa.string())),
])

CHUNK_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("git_commit", pa.string()),
    pa.field("tar_id", pa.string()),
    pa.field("tar_s3_key", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("chunk_id", pa.string()),
    pa.field("video_path_in_tar", pa.string()),
    pa.field("created_at", pa.float64()),
    pa.field("fps", pa.float32()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("flags", pa.list_(FLAG_STRUCT)),
])

FLAGS_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("git_commit", pa.string()),
    pa.field("tar_id", pa.string()),
    pa.field("tar_s3_key", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("chunk_id", pa.string()),
    pa.field("name", pa.string()),
    pa.field("value", pa.bool_()),
    pa.field("score", pa.float32()),
    pa.field("analyzer", pa.string()),
    pa.field("analyzer_version", pa.string()),
    pa.field("model", pa.string()),
    pa.field("prompt_id", pa.string()),
    pa.field("inputs_digest", pa.string()),
    pa.field("extras", pa.map_(pa.string(), pa.string())),
    pa.field("created_at", pa.float64()),
])

ERRORS_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("tar_id", pa.string()),
    pa.field("tar_s3_key", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("stage", pa.string()),
    pa.field("error_type", pa.string()),
    pa.field("error_msg", pa.string()),
    pa.field("traceback", pa.string()),
    pa.field("created_at", pa.float64()),
])

RUNS_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("started_at", pa.float64()),
    pa.field("finished_at", pa.float64()),
    pa.field("git_commit", pa.string()),
    pa.field("analyzers", pa.string()),
    pa.field("out_prefix", pa.string()),
])

# ================= Writers =================

class ParquetBatchWriter:
    """Append-only Parquet writer that batches rows to partitioned paths.
    Partitioning is provided via `partition_kv` (e.g., {"run_id": <uuid>}).
    Each writer uses a unique `writer_id` to avoid filename collisions.
    """
    def __init__(
        self,
        schema: pa.schema,
        out_prefix: str,
        table_name: str,
        partition_kv: Dict[str, str],
        batch_size: int = 1000,
        writer_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.schema = schema
        self.batch_size = batch_size
        self.rows: List[Dict[str, Any]] = []
        self.table_name = table_name
        self.partition_kv = partition_kv
        self.logger = logger or logging.getLogger(__name__)
        self.writer_id = writer_id or uuid.uuid4().hex[:8]
        self.counter = 0

        self.fs, self.base_path = resolve_filesystem(out_prefix)
        # Ensure partition directory exists
        part_dir = self._partition_dir()
        try:
            self.fs.create_dir(part_dir, recursive=True)
        except Exception:
            pass

    def _partition_dir(self) -> str:
        # e.g., base/table=chunks/run_id=<RUN_ID>
        parts = [f"table={self.table_name}"] + [f"{k}={v}" for k, v in self.partition_kv.items()]
        return f"{self.base_path}/" + "/".join(parts)

    def _next_part_path(self) -> str:
        path = f"{self._partition_dir()}/part-{self.writer_id}-{self.counter:05d}.parquet"
        self.counter += 1
        return path

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        path = self._next_part_path()
        with self.fs.open_output_stream(path) as sink:
            pq.write_table(table, sink)
        self.rows.clear()
        if self.logger:
            self.logger.debug(f"Wrote {self.table_name} batch to {path}")

    def close(self) -> None:
        self.flush()

# ================= TAR iteration =================



def iter_tar_members(tar_bytes: bytes) -> Tuple[List[str], Callable[[str], bytes]]:
    fileobj = io.BytesIO(tar_bytes)
    tf = tarfile.open(fileobj=fileobj, mode="r:*")
    members = {m.name: m for m in tf.getmembers() if m.isfile()}

    def _get_bytes(path_in_tar: str) -> bytes:
        m = members[path_in_tar]
        f = tf.extractfile(m)
        return f.read() if f else b""

    return list(members.keys()), _get_bytes


def default_chunk_paths(member_names: Iterable[str]) -> List[str]:
    return sorted([n for n in member_names if n.lower().endswith(".mp4")])

# ================= Loaders =================

class TarLoader:
    def __call__(self, key: str) -> bytes:
        raise NotImplementedError


class LocalTarLoader(TarLoader):
    def __call__(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()


class S3TarLoader(TarLoader):
    def __init__(self, client: Optional[BaseClient] = None):
        self.client = client or boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
            region_name=os.getenv("AWS_REGION"),
        )

    def __call__(self, s3_uri: str) -> bytes:
        assert s3_uri.startswith("s3://"), f"Invalid s3 uri: {s3_uri}"
        bucket_key = s3_uri[len("s3://"):]
        bucket, key = bucket_key.split("/", 1)
        try:
            obj = self.client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError as e:
            raise RuntimeError(f"S3 get_object failed for {s3_uri}: {e}")


def pick_tar_loader(sample_key: str) -> TarLoader:
    return S3TarLoader() if sample_key.startswith("s3://") else LocalTarLoader()

# ================= Core =================

def emit_flag_rows(flags_writer: ParquetBatchWriter, base: Dict[str, Any], flags: List[Dict[str, Any]]) -> None:
    ts = time.time()
    for f in flags:
        flags_writer.add({
            "run_id": base["run_id"],
            "git_commit": base["git_commit"],
            "tar_id": base["tar_id"],
            "tar_s3_key": base["tar_s3_key"],
            "chunk_index": base["chunk_index"],
            "chunk_id": base["chunk_id"],
            "name": f.get("name"),
            "value": bool(f.get("value", False)),
            "score": f.get("score"),
            "analyzer": f.get("analyzer"),
            "analyzer_version": f.get("analyzer_version"),
            "model": f.get("model"),
            "prompt_id": f.get("prompt_id"),
            "inputs_digest": f.get("inputs_digest"),
            "extras": f.get("extras") or {},
            "created_at": ts,
        })


def run_analysis_for_tar(
    tar_id: str,
    tar_key: str,
    tar_bytes: bytes,
    run_id: str,
    git_commit: str,
    analyzers: List[Analyzer],
    chunks_writer: ParquetBatchWriter,
    flags_writer: ParquetBatchWriter,
    errors_writer: ParquetBatchWriter,
    logger: logging.Logger,
) -> None:
    try:
        names, get_bytes = iter_tar_members(tar_bytes)
    except Exception as e:
        import traceback as tb
        errors_writer.add({
            "run_id": run_id,
            "tar_id": tar_id,
            "tar_s3_key": tar_key,
            "chunk_index": None,
            "stage": "open_tar",
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": "".join(tb.format_exception(e)),
            "created_at": time.time(),
        })
        return

    chunk_paths = default_chunk_paths(names)

    ctx = AnalysisContext(
        tar_id=tar_id,
        tar_s3_key=tar_key,
        git_commit=git_commit,
        run_id=run_id,
        get_bytes=get_bytes,
        list_files=lambda: names,
    )

    for idx, mp4_path in enumerate(chunk_paths):
        base_row: Dict[str, Any] = {
            "run_id": run_id,
            "git_commit": git_commit,
            "tar_id": tar_id,
            "tar_s3_key": tar_key,
            "chunk_index": idx,
            "chunk_id": f"{tar_id}:{idx}",
            "video_path_in_tar": mp4_path,
            "created_at": time.time(),
            "fps": None,
            "width": None,
            "height": None,
            "flags": [],
        }

        merged_flags: List[Dict[str, Any]] = []
        for analyzer in analyzers:
            try:
                out = analyzer.analyze_chunk(ctx, mp4_path, idx) or {}
                for k in ("fps", "width", "height"):
                    if k in out and base_row.get(k) is None:
                        base_row[k] = out[k]
                if "flags" in out and isinstance(out["flags"], list):
                    merged_flags.extend(out["flags"])
            except Exception as e:
                import traceback as tb
                errors_writer.add({
                    "run_id": run_id,
                    "tar_id": tar_id,
                    "tar_s3_key": tar_key,
                    "chunk_index": idx,
                    "stage": "analyze",
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": "".join(tb.format_exception(e)),
                    "created_at": time.time(),
                })

        base_row["flags"] = merged_flags
        chunks_writer.add(base_row)
        if merged_flags:
            emit_flag_rows(flags_writer, base_row, merged_flags)

# ================= Orchestration =================

def load_task_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def infer_tar_id_from_key(key: str) -> str:
    name = key
    if key.startswith("s3://"):
        name = key.split("/", maxsplit=3)[-1]
    stem = Path(name).name
    if "." in stem:
        stem = stem.split(".")[0]
    return stem


def write_run_row(out_prefix: str, run_id: str, row: Dict[str, Any]) -> None:
    fs, base = resolve_filesystem(out_prefix)
    part_dir = f"{base}/table=runs/run_id={run_id}"
    try:
        fs.create_dir(part_dir, recursive=True)
    except Exception:
        pass
    path = f"{part_dir}/meta.parquet"
    table = pa.Table.from_pylist([row], schema=RUNS_SCHEMA)
    with fs.open_output_stream(path) as sink:
        pq.write_table(table, sink)


def write_latest_pointer(out_prefix: str, pointer: Dict[str, Any]) -> None:
    fs, base = resolve_filesystem(out_prefix)
    ptr_dir = f"{base}/pointers"
    try:
        fs.create_dir(ptr_dir, recursive=True)
    except Exception:
        pass

    final_path = f"{ptr_dir}/latest.json"
    data = (json.dumps(pointer, separators=(",", ":")) + "\n").encode("utf-8")

    if out_prefix.startswith("s3://"):
        # S3 PUT is atomic
        with fs.open_output_stream(final_path) as sink:
            sink.write(data)
    else:
        # Local FS: write-then-replace
        tmp_path = f"{final_path}.tmp"
        with fs.open_output_stream(tmp_path) as sink:
            sink.write(data)


def run_pipeline(
    tasks: List[str],
    out_prefix: str,
    analyzers: Optional[List[Analyzer]] = None,
    git_commit: Optional[str] = None,
    run_id: Optional[str] = None,
    writer_id: Optional[str] = None,
    num_workers: int = 0,
) -> None:
    logger = logging.getLogger("runner")
    run_id = run_id or str(uuid.uuid4())
    started_at = time.time()
    git_commit = git_commit or get_git_commit_fallback()

    analyzers = analyzers if analyzers is not None else all_analyzers()

    partition = {"run_id": run_id}
    chunks_w = ParquetBatchWriter(CHUNK_SCHEMA, out_prefix, "chunks", partition, batch_size=1000, writer_id=writer_id, logger=logger)
    flags_w  = ParquetBatchWriter(FLAGS_SCHEMA,  out_prefix, "flags",  partition, batch_size=2000, writer_id=writer_id, logger=logger)
    errors_w = ParquetBatchWriter(ERRORS_SCHEMA, out_prefix, "errors", partition, batch_size=500,  writer_id=writer_id, logger=logger)

    manifest = [
        {
            "name": a.name,
            "version": a.version,
        }
        for a in analyzers
    ]

    loader = pick_tar_loader(tasks[0]) if tasks else LocalTarLoader()

    def process_one(key: str) -> None:
        tar_id = infer_tar_id_from_key(key)
        try:
            tar_bytes = loader(key)
            run_analysis_for_tar(
                tar_id=tar_id,
                tar_key=key,
                tar_bytes=tar_bytes,
                run_id=run_id,
                git_commit=git_commit,
                analyzers=analyzers,
                chunks_writer=chunks_w,
                flags_writer=flags_w,
                errors_writer=errors_w,
                logger=logger,
            )
        except Exception as e:
            import traceback as tb
            errors_w.add({
                "run_id": run_id,
                "tar_id": tar_id,
                "tar_s3_key": key,
                "chunk_index": None,
                "stage": "download|load",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "traceback": "".join(tb.format_exception(e)),
                "created_at": time.time(),
            })

    if num_workers and num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(process_one, k) for k in tasks]
            for f in as_completed(futs):
                _ = f.result()
    else:
        for k in tasks:
            process_one(k)

    chunks_w.close(); flags_w.close(); errors_w.close()

    finished_at = time.time()
    write_run_row(out_prefix, run_id, {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": git_commit,
        "analyzers": json.dumps(manifest),
        "out_prefix": out_prefix,
    })

    # also update a simple latest pointer
    write_latest_pointer(out_prefix, {
        "run_id": run_id,
        "git_commit": git_commit,
        "started_at": started_at,
        "finished_at": finished_at,
        "out_prefix": out_prefix,
    })

# ================= CLI =================

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run chunk analyzers over downsampled TARs and write Parquet manifests (run_id partitioned).")
    p.add_argument("--tasks-file", type=str, required=True, help="Text file with one path per line (local or s3://bucket/key.tar)")
    p.add_argument("--out", type=str, required=True, help="Output prefix (local dir or s3://bucket/prefix)")
    p.add_argument("--git-commit", type=str, default=None, help="Override git commit string")
    p.add_argument("--run-id", type=str, default=None, help="Run UUID to group outputs across machines. If omitted, a new one is generated.")
    p.add_argument("--num-workers", type=int, default=0, help="Parallelism via threads (0 or 1 = sequential)")
    p.add_argument("--num_nodes", type=int, default=1, help="World size: total number of nodes participating in this run")
    p.add_argument("--node_rank", type=int, default=0, help="This node's rank in [0, num_nodes)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"]) 
    p.add_argument("--writer-id", type=str, default=None, help="Optional writer identifier for file names (defaults to random)") 

    args = p.parse_args(argv)

    # Logging with node context in the message format
    fmt = f"%(asctime)s - %(levelname)s - node {args.node_rank}/{args.num_nodes} - %(message)s"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format=fmt,
    )

    tasks = load_task_list(args.tasks_file)
    if not tasks:
        logging.warning("No tasks to process (global task list is empty).")
        return 0

    # Shard tasks across nodes using modulo of global index
    num_nodes = max(1, int(args.num_nodes))
    node_rank = int(args.node_rank)
    if node_rank < 0 or node_rank >= num_nodes:
        logging.error(f"Invalid node_rank={node_rank}; must be in [0, {num_nodes}).")
        return 2

    local_tasks = [t for i, t in enumerate(tasks) if i % num_nodes == node_rank]
    logging.info(f"Loaded {len(tasks)} tasks; this node will process {len(local_tasks)} (rank {node_rank}/{num_nodes}).")

    if not local_tasks:
        logging.info("No tasks assigned to this node after sharding. Exiting cleanly.")
        # Still record a run row to keep orchestration simple
        run_pipeline(
            tasks=[],
            out_prefix=args.out,
            analyzers=None,
            git_commit=args.git_commit,
            run_id=args.run_id,
            writer_id=args.writer_id,
            num_workers=0,
        )
        return 0

    run_pipeline(
        tasks=local_tasks,
        out_prefix=args.out,
        analyzers=None,
        git_commit=args.git_commit,
        run_id=args.run_id,
        writer_id=args.writer_id,
        num_workers=args.num_workers,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())

