"""
torchrun --nproc_per_node=8 mp4_to_latent.py \
  --src_root /mnt/data/waypoint_1/datasets/kinetics700/Kinetics-700/Kinetics700_part_001/test/ \
  --tgt_root /mnt/data/lapp0/test_kinetics_700 \
  --vae_cfg_path /mnt/data/shahbuland/owl-vaes/configs/waypoint_1/rgb_only.yml \
  --vae_ckpt_path /mnt/data/shahbuland/owl-vaes/checkpoints/waypoint_1_f32c64_8res_rgb_only/step_385000.pt
"""
"""
# For distilled encoder:
torchrun --nproc_per_node=8 mp4_to_latent.py \
  --src_root /mnt/data/waypoint_1/datasets/kinetics700/Kinetics-700/Kinetics700_part_001/test/ \
  --tgt_root /mnt/data/lapp0/test_kinetics_700_distilled_decoder \
  --vae_cfg_path /mnt/data/shahbuland/owl-vaes/configs/waypoint_1/rgb_enc_distill.yml \
  --vae_ckpt_path /mnt/data/shahbuland/owl-vaes/checkpoints/waypoint_1_f32c64_8res_rgb_only_distill_enc/step_15000.pt
"""


import os, glob, json, re
from typing import Iterator, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm

from nn.owl_image_vae import BatchedEncodingPipe


import decord as de
de.bridge.set_bridge('torch')  # decord returns torch tensors directly



class SequentialVideoClips(IterableDataset):
    """
    Deterministic, no-shuffle, no-overlap clip iterator.
    - Shards by global DDP rank, then by DataLoader worker.
    - Uses EVERY frame; emits consecutive, non-overlapping clips of length n_frames.
    Yields: {"frames": torch.uint8 [T,H,W,C], "metadata": {...}}
    """
    def __init__(self, root_dir: str, n_frames: int = 16,
                 resize_to: int | Tuple[int, int] = 256, suffix: str = ".mp4"):

        pat = re.compile(suffix)
        self.paths: List[str] = [
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(root_dir)
            for fn in fns
            if pat.search(fn)
        ]

        self.paths.sort()
        if not self.paths:
            raise FileNotFoundError(f"No '{suffix}' videos under {root_dir}")
        self.T = int(n_frames)
        self.size = (resize_to, resize_to) if isinstance(resize_to, int) else tuple(resize_to)

    def __iter__(self) -> Iterator[Dict]:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        print("rank:", rank, "world size:", world)
        info = get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

        # 1) shard by rank, 2) shard by worker (disjoint coverage, supports num_workers>0)
        rank_paths = self.paths[rank::world]
        # Limit effective workers to avoid empty shards.
        n = len(rank_paths)
        n_eff = min(nworkers, max(1, n))
        if wid >= n_eff:
            return  # this worker has no data on this rank
        # Assign a contiguous chunk per worker (balances better than stride when n<nworkers)
        paths = rank_paths[wid::n_eff]

        H, W = self.size
        # threads per Decord reader (keep small; you have multiple workers)
        th = int(os.environ.get("DECORD_THREADS", "1"))

        for path in paths:
            try:
                vr = de.VideoReader(path, ctx=de.cpu(0), num_threads=th)
            except Exception as e:
                print(f"[rank {rank} worker {wid}] decord open failed: {path}\n  error: {e}")
                continue

            n_frames_total = len(vr)
            # fps (may be None/0 if missing)
            try:
                fps = float(vr.get_avg_fps()) or 0.0
            except Exception:
                fps = 0.0
            fps_num = None; fps_den = None  # not exposed; keep for API parity

            idx_in_vid = 0
            # sequential, non-overlap clips of length T
            for start in range(0, max(0, n_frames_total - self.T + 1), self.T):
                idx = list(range(start, start + self.T))
                try:
                    clip = vr.get_batch(idx)  # [T,H0,W0,3], uint8, torch tensor (bridge)
                except Exception as e:
                    print(f"[rank {rank} worker {wid}] decord get_batch failed: {path} @ {start}: {e}")
                    continue
                # Resize if needed (keep uint8 output like before)
                if clip.shape[1] != H or clip.shape[2] != W:
                    clip = F.interpolate(
                        clip.permute(0,3,1,2).float(), size=(H,W), mode="bilinear", align_corners=False
                    ).round_().clamp_(0,255).to(torch.uint8).permute(0,2,3,1)

                end_frame = start + (self.T - 1)
                start_ts = (start / fps) if fps > 0 else None
                end_ts   = (end_frame / fps) if fps > 0 else None
                meta = {
                    "start_frame": int(start), "end_frame": int(end_frame),
                    "start_ts": start_ts, "end_ts": end_ts,
                    "vid_path": path, "vid_name": os.path.basename(path),
                    "vid_dir": os.path.dirname(path), "idx_in_vid": idx_in_vid,
                    "fps": fps, "fps_numer": fps_num, "fps_denom": fps_den
                }
                # Ensure contiguous [T,H,W,C] uint8 like before
                frames = clip.contiguous()
                yield {"frames": frames, "metadata": meta}
                idx_in_vid += 1


def _collate_keep_meta(batch):
    return {
        "frames": torch.stack([s["frames"] for s in batch], dim=0),  # [B,T,H,W,C]
        "metadata": [s["metadata"] for s in batch],
    }


def get_dataloader(src_root: str, batch_size: int = 32, num_workers: int = 8,
                   n_frames: int = 16, resize_to: int | Tuple[int, int] = 256,
                   suffix: str = r"\.(?:mp4|MP4)$") -> DataLoader:
    ds = SequentialVideoClips(root_dir=src_root, n_frames=n_frames,
                              resize_to=resize_to, suffix=suffix)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=8,
        pin_memory=True,
        collate_fn=_collate_keep_meta,
        multiprocessing_context="spawn",
    )


def _target_dir_for(src_root: str, tgt_root: str, video_path: str) -> str:
    # Map {src_root}/foo/bar.mp4 -> {tgt_root}/foo/bar/
    rel = os.path.relpath(video_path, src_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(tgt_root, rel_no_ext)


def run_multinode_encode_and_save(
    src_root: str,
    tgt_root: str,
    batch_size: int = 32,
    num_workers: int = 16,
    n_frames: int = 16,
    resize_to: int | Tuple[int, int] = 256,
    suffix: str = r"\.(?:mp4|MP4)$",
    vae_cfg_path: str = "",
    vae_ckpt_path: str = "",
):
    """
    Initializes DDP (nccl), builds dataloader, runs BatchedEncodingPipe in float32,
    saves ONLY:
      - latents (fp16) -> {tgt_root}/<rel/video/stem>/{idx:06d}_rgblatent.pt
      - metadata       -> {tgt_root}/<rel/video/stem>/{idx:06d}_meta.json
    """
    assert vae_cfg_path and vae_ckpt_path, "Provide --vae_cfg_path and --vae_ckpt_path"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)
    rank = dist.get_rank()

    model = BatchedEncodingPipe(vae_cfg_path, vae_ckpt_path, dtype=torch.float16)

    # Data
    loader = get_dataloader(
        src_root=src_root,
        batch_size=batch_size,
        num_workers=num_workers,
        n_frames=n_frames,
        resize_to=resize_to,
        suffix=suffix,
    )

    # Progress (rank 0 only)
    print("running on rank", rank)
    pbar = tqdm(unit="frames", dynamic_ncols=True, disable=(rank != 0))

    def _save_lat_meta(lat_tensor_fp16, lat_path, meta_dict, meta_path):
        torch.save(lat_tensor_fp16, lat_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f)

    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)                # [B,T,H,W,C] uint8
            metas = batch["metadata"]                                             # dict-of-lists or list
            B, T, H, W, C = frames.shape
            frames = frames.permute(0, 1, 4, 2, 3).contiguous().to(torch.float16)           # [B,T,C,H,W] fp32
            frames = frames.div_(255.0).mul_(2.0).sub_(1.0).view(B*T, C, H, W)    # [-1,1] fp32

            lat = model(frames)  # fp32 latents

            # If latents are per-frame, reshape to [B,T,...]
            if lat.dim() >= 2 and lat.size(0) == B * T:
                lat = lat.view(B, T, *lat.shape[1:])
            elif lat.dim() == 1 and lat.size(0) == B * T:
                lat = lat.view(B, T, 1)

            # Save latents (fp16) + metadata per clip, under mirrored target dir
            for b in range(B):
                meta = metas[b]
                out_dir = _target_dir_for(src_root, tgt_root, meta["vid_path"])
                os.makedirs(out_dir, exist_ok=True)
                stem = f"{meta['idx_in_vid']:06d}"

                lat_b = lat[b].detach().cpu().half()
                assert torch.isfinite(lat_b).all(), "Encountered non-finite values in fp16 latents"
                lat_path  = os.path.join(out_dir, f"{stem}_rgblatent.pt")
                meta_path = os.path.join(out_dir, f"{stem}_meta.json")
                # Dispatch async save to thread pool (I/O won’t block GPU compute)
                _save_lat_meta(lat_b, lat_path, meta, meta_path)

            if rank == 0:
                pbar.update(B * T)

    # Drain async saves, then cleanup
    if rank == 0:
        pbar.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="Root directory containing source videos")
    ap.add_argument("--tgt_root", required=True, help="Root directory to mirror outputs into")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--resize_to", type=int, nargs="+", default=[512])  # H W or single square
    ap.add_argument("--suffix", type=str, default=r"\.(?:mp4|MP4)$")
    ap.add_argument("--vae_cfg_path", type=str, required=True)
    ap.add_argument("--vae_ckpt_path", type=str, required=True)
    args = ap.parse_args()

    resize = tuple(args.resize_to) if len(args.resize_to) == 2 else args.resize_to[0]

    run_multinode_encode_and_save(
        src_root=args.src_root,
        tgt_root=args.tgt_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_frames=args.n_frames,
        resize_to=resize,
        suffix=args.suffix,
        vae_cfg_path=args.vae_cfg_path,
        vae_ckpt_path=args.vae_ckpt_path,
    )
