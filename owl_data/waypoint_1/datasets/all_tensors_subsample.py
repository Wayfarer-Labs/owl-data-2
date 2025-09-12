import os
import ray
import tqdm
import torch
import logging
import json
from typing import Generator
from pathlib import Path
from itertools import batched

def split_by_rank(rgb_paths: list[Path], num_nodes: int, node_rank: int) -> list[Path]:
    return [path for i, path in enumerate(rgb_paths) if i % num_nodes == node_rank]

WRITE_LOG   = str(Path('/mnt/data/sami/logs') / 'tensors_subsample.log')

def rgb_paths() -> Generator[Path, None, None]:
    yield from Path('/mnt/data/waypoint_1/normalized360').glob('**/**/*_rgb.pt')

@ray.remote
def process(
    rgb_pt_dir: Path,
    old_stride_sec: float,   # e.g., 0.5
    new_stride_sec: float,   # e.g., 5.0
    num_chunks_chained: int,
    juke: bool,
) -> dict:
    rgb_pt_paths = sorted(
        rgb_pt_dir.glob('*_rgb.pt'),
        key=lambda p: int(p.stem.split('_')[0])
    )
    if not rgb_pt_paths:
        return {"ok": True, "saved_count": 0, "removed_paths": [], "dirpath": str(rgb_pt_dir), "saved_paths": []}

    assert new_stride_sec >= old_stride_sec, "This function only downsamples (new_stride_sec >= old_stride_sec)."
    ratio = new_stride_sec / old_stride_sec
    subsample_factor = max(1, int(round(ratio)))

    strided_batch_outputs: list[Path] = []
    removed_paths: list[Path] = []
    failures: list[str] = []

    def _verify_saved(new_frames: torch.Tensor, tmp_path: Path) -> tuple[bool, str | None]:
        try:
            re = torch.load(tmp_path, map_location="cpu")
        except Exception as e:
            return False, f"torch.load failed for {tmp_path}: {e}"

        if not isinstance(re, torch.Tensor):
            return False, f"Reloaded object is {type(re)}, expected torch.Tensor"
        if re.dtype != new_frames.dtype:
            return False, f"dtype mismatch: {re.dtype} vs {new_frames.dtype}"
        if re.shape != new_frames.shape:
            return False, f"shape mismatch: {tuple(re.shape)} vs {tuple(new_frames.shape)}"
        if re.numel() == 0:
            return False, "reloaded tensor is empty"

        # Spot-check some frames to avoid OOM from full equality
        T = re.shape[0]
        idxs = [0, T // 2, T - 1] if T >= 3 else list(range(T))
        for k in idxs:
            if not torch.equal(re[k], new_frames[k]):
                return False, f"content mismatch at frame {k}"
        return True, None

    for batch in batched(rgb_pt_paths, num_chunks_chained):
        # Load batch on CPU
        total_chunks = [torch.load(p, map_location="cpu") for p in batch]
        total = torch.cat(total_chunks, dim=0)

        # Downsample content
        new_frames = total[::subsample_factor]
        first_idx = int(batch[0].stem.split("_")[0])
        tmp_path = rgb_pt_dir / f"{first_idx:08d}_rgb.subsample_{new_stride_sec}.tmp.pt"
        out_path = rgb_pt_dir / f"{first_idx:08d}_rgb.subsample_{new_stride_sec}.pt"

        if not juke:
            logging.info(f"SAVE: {tmp_path}")
            torch.save(new_frames, tmp_path)

            ok, err = _verify_saved(new_frames, tmp_path)
            if not ok:
                failures.append(f"VERIFY FAILED for {tmp_path}: {err}")
                logging.error(f"VERIFY FAILED: {tmp_path} ({err}). Originals preserved.")
                # Keep tmp + originals; skip deletion and finalization for this batch
                continue

            # Finalize atomically
            Path(tmp_path).replace(out_path)
            logging.info(f"FINALIZE: {tmp_path.name} -> {out_path.name}")
        else:
            logging.info(f"JUKE SAVE: {out_path}")

        strided_batch_outputs.append(out_path)

        # Now it's safe to remove originals for this batch
        for p in batch:
            if not juke:
                p.unlink()
                removed_paths.append(p.resolve())
                logging.info(f"REMOVE: {p.name}")
            else:
                logging.info(f"JUKE REMOVE: {p.name}")

    # Renumber densely: 00000000_rgb.pt, 00000001_rgb.pt, ...
    final_paths: list[Path] = []
    for i, p in enumerate(sorted(strided_batch_outputs, key=lambda q: int(q.stem.split("_")[0]))):
        dest = p.with_name(f"{i:08d}_rgb.pt")
        if not juke:
            logging.info(f"RENAME: {p.name} -> {dest.name}")
            p.replace(dest)
        else:
            logging.info(f"JUKE RENAME: {p.name} -> {dest.name}")
        final_paths.append(dest)

    return {
        "ok": len(failures) == 0,
        "saved_count": len(final_paths),
        "removed_paths": [str(p) for p in removed_paths],
        "dirpath": str(rgb_pt_dir),
        "saved_paths": [str(p) for p in final_paths],
        "failures": failures,
    }


def all_tensors_subsample(
    rgb_paths_file: str | None,
    num_cpus: int = os.cpu_count(),
    num_nodes: int = 1,
    node_rank: int = 0,
    old_stride_sec: float = 0.5,
    new_stride_sec: float = 5.0,
    juke: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - Node {node_rank}/{num_nodes} - %(message)s',
        handlers=[
            logging.FileHandler(WRITE_LOG, mode='a'),
            logging.StreamHandler()
        ]
    )
    assert old_stride_sec < new_stride_sec

    if old_stride_sec == new_stride_sec:
        return

    num_chunks_chained = new_stride_sec / old_stride_sec
    assert num_chunks_chained.is_integer(), f"num_chunks_chained must be an integer, got {num_chunks_chained}"
    num_chunks_chained = int(num_chunks_chained)

    ray.init(num_cpus=num_cpus)
    # sort just to force a deterministic ordering
    if rgb_paths_file:
        with open(rgb_paths_file, 'r') as f:
            all_rgb_paths = [Path(path.strip()) for path in f.readlines()]
            all_rgb_paths = sorted(all_rgb_paths)
    else: all_rgb_paths = list(rgb_paths())
    logging.info(f"Processing {len(all_rgb_paths)} rgb paths")
    all_rgb_dirs = set(path.parent for path in all_rgb_paths)
    local_rgb_dirs = split_by_rank(list(all_rgb_dirs), num_nodes, node_rank)

    # process
    futures = [
        process.remote(rgb_dir, old_stride_sec, new_stride_sec, num_chunks_chained, juke)
        for rgb_dir in local_rgb_dirs
    ]

    with tqdm.tqdm(
        desc='Subsampling tensors...'
    ) as pbar:
        while futures:
            done, futures = ray.wait(futures)
            results = ray.get(done)

            for res in results:
                pbar.update(1)
                if res["ok"]:
                    logging.info(f"Processed {res['saved_count']} chunks")
                    logging.info(json.dumps(res))
                else:
                    msg = f"[FAIL] {res['dirpath']} :: {res['error']}"
                    logging.error(msg)
                    try:
                        logging.error(f"[FAIL] {res['dirpath']} :: {res['error']}")
                        logging.error(json.dumps(res))
                    except Exception: logging.error(f"[WARN] Could not write to {WRITE_LOG}: {msg}")

    ray.shutdown()
    logging.info(f"Done - Node {node_rank} of {num_nodes} finished processing {len(all_rgb_paths)} rgb paths")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_paths_file', type=str, default=None)
    parser.add_argument('--num_cpus', type=int, default=os.cpu_count())
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--old_stride_sec', type=float, default=0.5)
    parser.add_argument('--new_stride_sec', type=float, default=5.0)
    parser.add_argument('--no-juke', action='store_true', default=False)  # if juke, remove the original files or save
    args = parser.parse_args()

    all_tensors_subsample(
        rgb_paths_file=args.rgb_paths_file,
        num_cpus=args.num_cpus,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        old_stride_sec=args.old_stride_sec,
        new_stride_sec=args.new_stride_sec,
        juke=not args.no_juke,
    )
