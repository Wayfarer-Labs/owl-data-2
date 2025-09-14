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

model = YOLO('yolo11n-pose.pt')

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

@torch.no_grad()
def build_pose_sidecar_from_frames(
    frames_nchw: torch.Tensor,
    yolo_pose_model,                         # ultralytics.YOLO loaded with a *-pose.pt
    save_path: Optional[str] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    kp_thr: float = 0.20,
    dot_radius: int = 2,
    line_thickness: int = 2,
) -> tuple[torch.Tensor, bool]:
    """
    Converts RGB chunk [N,3,H,W] -> pose sidecar [N,1,H,W] uint8 via YOLOv11 pose.
    - yolo_pose_model: e.g., YOLO('yolo11n-pose.pt')
    - save_path: if provided, torch.save() to this path.
    Returns the [N,1,H,W] uint8 tensor.
    """
    # Convert frames to list of HWC uint8 RGB numpy arrays
    np_frames = _to_numpy_hwc_bool(frames_nchw)
    N = len(np_frames)
    H, W = np_frames[0].shape[:2]

    # Decide device
    if device is None:
        # If model already moved to cuda, use that; otherwise CPU
        try:
            device = next(yolo_pose_model.model.parameters()).device.type
        except Exception:
            device = 'cpu'

    # Inference in mini-batches
    pose_frames: list[torch.Tensor] = []
    for start in range(0, N, batch_size):
        batch = np_frames[start:start + batch_size]

        # Ultralytics accepts list of numpy RGB frames directly
        # Results is an iterable with one item per input image
        results = yolo_pose_model(
            batch,
            verbose=False,
            device=device
        )
        has_pose = False
        # Parse each frame's results -> list of persons' keypoints [M,3]
        # Ultralytics (v8/11) results: result.keypoints.xy [P,M,2], result.keypoints.conf [P,M]
        for idx, res in enumerate(results):
            persons_kps: list[np.ndarray] = []
            if hasattr(res, "keypoints") and (res.keypoints is not None):
                # Safely extract tensors to CPU numpy
                try:
                    xy = res.keypoints.xy  # (P,M,2) tensor
                    cf = res.keypoints.conf  # (P,M) tensor, may be None
                    if xy is not None:
                        xy = xy.detach().cpu().numpy()
                        if cf is not None:
                            cf = cf.detach().cpu().numpy()
                        else:
                            # fallback: uniform confidence of 1.0
                            cf = np.ones(xy.shape[:2], dtype=np.float32)
                        # Pack (x,y,score) per keypoint
                        P, M = xy.shape[:2]
                        for p in range(P):
                            kps = np.concatenate([xy[p], cf[p][..., None]], axis=-1)  # [M,3]
                            persons_kps.append(kps.astype(np.float32))
                except Exception:
                    # If API differs, try accessing .data (older ultralytics)
                    try:
                        kd = res.keypoints.data.detach().cpu().numpy()  # [P,M,3]
                        for p in range(kd.shape[0]):
                            persons_kps.append(kd[p].astype(np.float32))
                    except Exception:
                        persons_kps = []

            has_pose = len(persons_kps) > 0

            pose_img = _render_pose_frame_binary(
                H, W, persons_kps,
                edges=COCO_EDGES,
                kp_thr=kp_thr,
                dot_radius=dot_radius,
                line_thickness=line_thickness,
            )
            pose_frames.append(pose_img)

    # Stack to [N,1,H,W]
    pose_tensor = torch.stack(pose_frames, dim=0)
    assert pose_tensor.shape == (N, 1, H, W), f"Got {tuple(pose_tensor.shape)}, expected {(N,1,H,W)}"

    if save_path is not None:
        logging.info(f'Saving pose tensor to {save_path} : RGB {frames_nchw.shape} -> Pose {pose_tensor.shape}')
        torch.save(pose_tensor, save_path)

    return pose_tensor, has_pose


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


def all_rgb_to_pose(
    num_cpus: int = os.cpu_count(),
    num_nodes: int = 1,
    node_rank: int = 0,
    force_overwrite: bool = False
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - Node {node_rank}/{num_nodes} - %(message)s',
        handlers=[
            logging.FileHandler(WRITE_LOG, mode='a'),
            logging.StreamHandler()
        ]
    )

    ray.init(num_cpus=num_cpus)

    all_rgb_paths = sorted(rgb_paths())
    
    local_video_paths = split_by_rank(all_rgb_paths, num_nodes, node_rank)
    
    futures = [
        process.remote(
            path,
            force_overwrite=force_overwrite
        )
        for path in local_video_paths
    ]
    
    with tqdm.tqdm(
        desc="Processing pose tensors..."
    ) as pbar:
        while futures:
            done, futures = ray.wait(futures)
            results = ray.get(done)

            for res in results:
                pbar.update(1)
                if res["ok"]:
                    logging.info(f"Processed {res['path']}")
                    logging.info(json.dumps(res))
                else:
                    msg = f"[FAIL] {res['path']} :: {res['error']}"
                    logging.error(msg)
                    logging.error(json.dumps(res))
    
    ray.shutdown()
    logging.info(f"Done - Node {node_rank} of {num_nodes} finished processing {len(local_video_paths)} videos")


@ray.remote
def process(
    rgb_path: Path,
    force_overwrite: bool = False
):
    out_path = (rgb_path.parent / rgb_path.stem.replace('_rgb', '_pose'))\
        .with_suffix('.pt')

    try:
        frames_nchw = torch.load(rgb_path)
        ok = {"path": str("out_path"), "ok": True, "error": None, 'has_pose': False}
        pose, has_pose = build_pose_sidecar_from_frames(frames_nchw, model, save_path=str(out_path), line_thickness=2)
        ok['has_pose'] = has_pose
        if out_path.exists() and not force_overwrite: return ok
        torch.save(pose, out_path)
        return ok
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb  = "".join(traceback.format_exception_only(type(e), e)).strip()
        tb_full = traceback.format_exc(limit=3)
        return {"path": str(out_path), "ok": False, "error": f"{err} | {tb} | {tb_full}"}


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--num_cpus', type=int, default=os.cpu_count())
    args.add_argument('--num_nodes', type=int, default=1)
    args.add_argument('--node_rank', type=int, default=0)
    args.add_argument('--force_overwrite', action='store_true')
    args = args.parse_args()
    all_rgb_to_pose(
        num_cpus=args.num_cpus,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
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