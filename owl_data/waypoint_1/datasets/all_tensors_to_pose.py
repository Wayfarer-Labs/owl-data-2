# Requirements:
#   pip install ultralytics opencv-python
# Usage:
#   from ultralytics import YOLO
#   model = YOLO('yolo11n-pose.pt').to('cuda')  # or cpu
#   pose_tensor = build_pose_sidecar_from_frames(frames_nchw, model, save_path=out_dir/'00000000_pose.pt')
import  os
import  ray
import  cv2
import  tqdm
import  logging
import  torch
import  argparse
import  traceback
import  numpy as np
import  json
from    pathlib import Path
from    typing import Optional, Generator, Literal

from ultralytics import YOLO

from owl_data.waypoint_1.datasets.utils import split_by_rank


Datasets = Literal[
    'epic_kitchens_100',
    'comma2k19',
    'egoexplore',
    'kinetics700',
    'mkif'
]

WRITE_LOG = Path("/mnt/data/sami/logs/tensors_to_pose.log")

# COCO-17 skeleton edges (adjust if your model uses a different order)
COCO_EDGES = [
    (5, 7), (7, 9),            # L-Arm
    (6, 8), (8,10),            # R-Arm
    (11,13), (13,15),          # L-Leg
    (12,14), (14,16),          # R-Leg
    (5,6), (5,11), (6,12),     # Shoulders -> hips
    (11,12),                   # Hip bridge
]


def _to_numpy_hwc_bool(frames_nchw: torch.Tensor) -> list[np.ndarray]:
    """
    Accepts [N,3,H,W] uint8 or float (0..1/0..255) and returns a list of [H,W,3] np.uint8 RGB.
    (We still need RGB uint8 for YOLO inference.)
    """
    assert frames_nchw.ndim == 4 and frames_nchw.shape[1] == 3, "Expected [N,3,H,W]"
    f = frames_nchw
    if torch.is_floating_point(f):
        maxv = float(f.max().detach().cpu())
        if maxv <= 1.5:
            f = (f * 255.0).clamp(0, 255)
        f = f.to(torch.uint8)
    elif f.dtype != torch.uint8:
        f = f.to(torch.uint8)

    # NCHW -> NHWC
    f = f.permute(0, 2, 3, 1).contiguous()  # [N,H,W,3]
    f = f.cpu().numpy()
    return [f[i] for i in range(f.shape[0])]


def _render_pose_frame_binary(
    H: int,
    W: int,
    persons: list[np.ndarray],
    edges=COCO_EDGES,
    kp_thr: float = 0.20,
    dot_radius: int = 2,
    line_thickness: int = 1,
) -> torch.Tensor:
    """
    persons: list of keypoints arrays, each [M,3] with (x,y,score) in *pixels*
    returns [1,H,W] torch.bool (True = pose pixel, False = background)
    """
    canvas = np.zeros((H, W), dtype=np.uint8)

    for kps in persons:
        # joints
        for (x, y, s) in kps:
            if s >= kp_thr:
                cv2.circle(canvas, (int(round(x)), int(round(y))), dot_radius, 1, -1)

        # bones
        for (i, j) in edges:
            xi, yi, si = kps[i]
            xj, yj, sj = kps[j]
            if si >= kp_thr and sj >= kp_thr:
                cv2.line(canvas,
                         (int(round(xi)), int(round(yi))),
                         (int(round(xj)), int(round(yj))),
                         1, line_thickness)

    return torch.from_numpy(canvas.astype(bool)).unsqueeze(0)  # [1,H,W], torch.bool


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


def rgb_paths() -> Generator[Path, None, None]:
    yield from Path('/mnt/data/waypoint_1/normalized360').glob('**/**/*_rgb.pt')

# --- a persistent GPU actor that batches frames across many chunks ---
@ray.remote(num_gpus=1)
class PoseActor:
    def __init__(self, weights: str = 'yolo11n-pose.pt', device: str = 'cuda',
                 kp_thr: float = 0.20, dot_radius: int = 2, line_thickness: int = 2,
                 max_frames_per_call: int = 2048):
        from ultralytics import YOLO
        self.model = YOLO(weights).to(device)
        self.device = device
        self.kp_thr = kp_thr
        self.dot_radius = dot_radius
        self.line_thickness = line_thickness
        self.max_frames_per_call = max_frames_per_call  # how many frames to feed per YOLO call

    def _infer_and_render(self, batch_hwc: list[np.ndarray]) -> list[torch.Tensor]:
        results = self.model(batch_hwc, verbose=False, device=self.device)
        rendered = []
        for idx, res in enumerate(results):
            persons_kps: list[np.ndarray] = []
            if hasattr(res, "keypoints") and (res.keypoints is not None):
                try:
                    xy = res.keypoints.xy
                    cf = res.keypoints.conf
                    if xy is not None:
                        xy = xy.detach().cpu().numpy()
                        cf = (cf.detach().cpu().numpy()
                            if cf is not None else np.ones(xy.shape[:2], np.float32))
                        P, M = xy.shape[:2]
                        for p in range(P):
                            kps = np.concatenate([xy[p], cf[p][..., None]], axis=-1)
                            persons_kps.append(kps.astype(np.float32))
                except Exception:
                    try:
                        kd = res.keypoints.data.detach().cpu().numpy()  # [P,M,3]
                        for p in range(kd.shape[0]):
                            persons_kps.append(kd[p].astype(np.float32))
                    except Exception:
                        persons_kps = []

            # use H,W of THIS frame (frames can differ!)
            Hi, Wi = batch_hwc[idx].shape[:2]
            pose_img = _render_pose_frame_binary(
                Hi, Wi, persons_kps,
                edges=COCO_EDGES,
                kp_thr=self.kp_thr,
                dot_radius=self.dot_radius,
                line_thickness=self.line_thickness,
            )
            rendered.append(pose_img)  # [1,Hi,Wi]
        return rendered


    def process_group(self, rgb_paths: list[str], force_overwrite: bool = False) -> list[dict]:
        """Process many *_rgb.pt paths in a single actor, batching across chunks."""
        results_meta = []
        # Preload all chunks to CPU and prepare per-chunk builders
        chunks = []
        for p in rgb_paths:
            rgb_path = Path(p)
            out_path = (rgb_path.parent / rgb_path.stem.replace('_rgb', '_pose')).with_suffix('.pt')
            if out_path.exists() and not force_overwrite:
                results_meta.append({"path": str(rgb_path), "ok": True, "skipped": True, "has_pose": None})
                continue
            frames = torch.load(rgb_path)  # [N,3,H,W]
            np_frames = _to_numpy_hwc_bool(frames)  # list of HWC uint8
            N, H, W = len(np_frames), np_frames[0].shape[0], np_frames[0].shape[1]
            chunks.append({
                "rgb_path": rgb_path, "out_path": out_path,
                "np_frames": np_frames, "N": N, "H": H, "W": W,
                "poses": [], "has_pose": False
            })

        # Megabatch frames across chunks
        staging_frames: list[np.ndarray] = []
        staging_keys: list[tuple[int,int]] = []  # (chunk_idx, frame_idx)

        def flush():
            if not staging_frames: return
            rendered = self._infer_and_render(staging_frames)  # list of [1,H,W] bool
            # demux back
            for (ci, fi), pose_img in zip(staging_keys, rendered):
                chunks[ci]["poses"].append(pose_img)
                if pose_img.any().item():
                    chunks[ci]["has_pose"] = True
            staging_frames.clear()
            staging_keys.clear()

        # Fill the staging buffer
        for ci, ch in enumerate(chunks):
            for fi, f in enumerate(ch["np_frames"]):
                staging_frames.append(f)
                staging_keys.append((ci, fi))
                if len(staging_frames) >= self.max_frames_per_call:
                    flush()
        flush()  # remaining

        # Save each chunk
        for ch in chunks:
            if len(ch["poses"]) != ch["N"]:
                results_meta.append({
                    "path": str(ch["rgb_path"]), "ok": False,
                    "error": f"pose frames mismatch: got {len(ch['poses'])}, expected {ch['N']}"
                })
                continue
            # poses were appended in order we fed them; ensure correct order per frame index
            ch["poses"] = ch["poses"]  # already ordered because we staged in increasing fi
            pose_tensor = torch.stack(ch["poses"], dim=0)  # [N,1,H,W], bool
            torch.save(pose_tensor, ch["out_path"])
            results_meta.append({
                "path": str(ch["rgb_path"]), "ok": True,
                "saved_to": str(ch["out_path"]),
                "has_pose": ch["has_pose"], "N": ch["N"], "H": ch["H"], "W": ch["W"]
            })
        return results_meta

def all_rgb_to_pose(
    num_gpus: int = 1,
    num_nodes: int = 1,
    node_rank: int = 0,
    tensors_path_file: Optional[str] = None,
    force_overwrite: bool = False,
    group_size: int = 16,              # how many chunks per actor call
    max_frames_per_call: int = 4096,   # how many frames to feed the model per YOLO call
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - Node {node_rank}/{num_nodes} - %(message)s',
        handlers=[logging.FileHandler(WRITE_LOG, mode='a'), logging.StreamHandler()]
    )
    ray.init(num_gpus=num_gpus)

    # get paths
    if tensors_path_file is None:
        all_rgb_paths = sorted(rgb_paths())
    else:
        with open(tensors_path_file, 'r') as f:
            all_rgb_paths = [Path(line.strip()) for line in f.readlines()]
    local_paths = split_by_rank(all_rgb_paths, num_nodes, node_rank)
    local_paths = [str(p) for p in local_paths]

    # one actor per GPU
    worker = PoseActor.options(num_gpus=1).remote(max_frames_per_call=max_frames_per_call)

    # group paths so each call processes many chunks and megabatches inside
    groups = [local_paths[i:i+group_size] for i in range(0, len(local_paths), group_size)]
    futures = [worker.process_group.remote(g, force_overwrite=force_overwrite) for g in groups]

    with tqdm.tqdm(total=len(groups), desc="Processing pose groups") as pbar:
        while futures:
            done, futures = ray.wait(futures)
            metas = ray.get(done)[0]
            for m in metas:
                if m.get("ok"): logging.info(json.dumps(m))
                else:           logging.error(json.dumps(m))
            pbar.update(1)

    ray.shutdown()
    logging.info(f"Done - Node {node_rank} of {num_nodes} processed {len(local_paths)} chunks")


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--num_gpus', type=int, default=1)
    args.add_argument('--num_nodes', type=int, default=1)
    args.add_argument('--node_rank', type=int, default=0)
    args.add_argument('--tensors_path_file', type=str, default='/mnt/data/sami/owl-data-2/rgb_paths.txt')
    args.add_argument('--force_overwrite', action='store_true')
    args = args.parse_args()
    all_rgb_to_pose(
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        tensors_path_file=args.tensors_path_file,
        force_overwrite=args.force_overwrite,
    )


if __name__ == '__main__':
    main()
    # # load image as tensor
    # img = cv2.imread('yoloimg.jpg')
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    # from ultralytics import YOLO
    # model = YOLO('yolo11n-pose.pt').to('cuda')  # or cpu
    # pose_tensor = build_pose_sidecar_from_frames(img_tensor, model, save_path='debug.pt')
    # render_pose_preview(pose_tensor, save_path='debug.jpeg')