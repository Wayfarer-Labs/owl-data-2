import ray
import tqdm
import torch
from pathlib import Path
import argparse
from typing import Literal, Generator
import owl_data.waypoint_1.datasets.utils as utils
import os
import itertools
import traceback

NORMALIZED_360_DIR = Path('/mnt/data/waypoint_1/normalized360')
Datasets = Literal[
    'epic_kitchens_100',
    'comma2k19',
    'egoexplore',
    'kinetics700',
    'mkif'
]
STRIDE_SEC: dict[Datasets, float] = {
    'epic_kitchens_100': 0.5,
    'comma2k19': 0.5,
    'egoexplore': 0.5,
    'kinetics700': 0.5,
    'mkif': 0.5
}


def video_paths(dataset: Datasets) -> Generator[Path, None, None]:
    match dataset:
        case 'epic_kitchens_100':
            root = Path('/mnt/data/waypoint_1/datasets') / 'epic_kitchens_100' / '2g1n6qdydwa9u22shpxqzp0t8m'
            yield from root.glob('P*/videos/*.MP4')
        case 'comma2k19':
            root = Path('/mnt/data/waypoint_1/datasets') / 'comma2k19' / 'processed'
            yield from root.glob('Chunk_*/**/**/video.hevc')
        case 'egoexplore':
            root = Path('/mnt/data/waypoint_1/datasets') / 'egoexplore' / 'videos'
            yield from root.glob('*.mp4?*')
        case 'kinetics700':
            root = Path('/mnt/data/waypoint_1/datasets') / 'kinetics700' / 'Kinetics-700'
            yield from root.glob('Kinetics700_part_*/test/*.mp4')
        case 'mkif':
            root = Path('/mnt/data/waypoint_1/datasets') / 'MKIF' / 'videos'
            yield from root.glob('*.mp4'); yield from root.glob('*.webm')
        case _:
            raise TypeError(f"Unsupported dataset: {dataset!r}")

def output_path(path: Path) -> Path:
    match path:
        case path if 'epic_kitchens_100'    in path.parts:  return NORMALIZED_360_DIR / 'epic_kitchens_100' / path.parent.name / path.stem
        case path if 'comma2k19'            in path.parts:  return NORMALIZED_360_DIR / 'comma2k19' / 'processed' / path.parent.parent.parent.name / path.parent.parent.name / path.parent.name
        case path if 'egoexplore'           in path.parts:  return NORMALIZED_360_DIR / 'egoexplore' / 'videos' / path.stem
        case path if 'kinetics700'          in path.parts:  return NORMALIZED_360_DIR / 'kinetics700' / 'Kinetics-700' / path.parent.parent.name / path.parent.name / path.stem
        case path if 'MKIF'                 in path.parts:  return NORMALIZED_360_DIR / 'MKIF' / 'videos' / path.stem
        case _:                                             raise TypeError(f"Unsupported path: {path}")

def dataset_from_path(path: Path) -> Datasets:
    match path:
        case path if 'epic_kitchens_100'    in path.parts:  return 'epic_kitchens_100'
        case path if 'comma2k19'            in path.parts:  return 'comma2k19'
        case path if 'egoexplore'           in path.parts:  return 'egoexplore'
        case path if 'kinetics700'          in path.parts:  return 'kinetics700'
        case path if 'MKIF'                 in path.parts:  return 'mkif'
        case _:                                             raise TypeError(f"Unsupported path: {path}")


FAILED_LOG = Path("/mnt/data/sami/failed_videos.txt")


@ray.remote
def process(
    path: Path,
    stride_sec: float,
    chunk_size: int = 512,
    force_overwrite: bool = False,
) -> dict:
    """
    Returns a small result dict instead of raising:
      {"path": str, "ok": bool, "saved_count": int, "saved": [str], "error": str|None}
    """
    try:
        outdir = output_path(path)
        outdir.mkdir(parents=True, exist_ok=True)

        saved = []
        for i, chunk in enumerate(utils.process_video_seek(path, stride_sec, chunk_size)):
            fp = outdir / f"{i:08d}_rgb.pt"
            if not force_overwrite and fp.exists(): continue
            torch.save(chunk, fp)
            saved.append(str(fp))
        
        return {"path": str(path), "ok": True, "saved_count": len(saved), "saved": saved, "error": None}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb  = "".join(traceback.format_exception_only(type(e), e)).strip()
        tb_full = traceback.format_exc(limit=3)
        return {"path": str(path), "ok": False, "saved_count": 0, "saved": [], "error": f"{err} | {tb} | {tb_full}", "failed_log": FAILED_LOG}


def all_videos_to_tensors(
    datasets: list[Datasets],
    num_cpus: int = os.cpu_count(),
    num_nodes: int = 1,
    node_rank: int = 0,
    force_overwrite: bool = False,
) -> None:
    ray.init(num_cpus=num_cpus)
    # sort just to force a deterministic ordering
    all_video_paths = sorted(
        itertools.chain(
        *[video_paths(dataset) for dataset in datasets]
    ))
    # stride by num_nodes starting from node-rank. e.g., rank 1 with 8 nodes:
    # 1, 9, 17 ...
    local_video_paths = all_video_paths[node_rank::num_nodes]

    futures = [
        process.remote(
            path,
            STRIDE_SEC[dataset_from_path(path)],
            force_overwrite=force_overwrite,
        )
        for path in local_video_paths
    ]

    with tqdm.tqdm(
        desc='Processing videos to tensors...'
    ) as pbar:
        while futures:
            done, futures = ray.wait(futures)
            results = ray.get(done)

            for res in results:
                pbar.update(1)
                if res["ok"]: print(f"Processed {res['path']} -> {res['saved_count']} chunks")
                else:
                    msg = f"[FAIL] {res['path']} :: {res['error']}"
                    print(msg)
                    try:
                        with FAILED_LOG.open("a") as f: f.write(res["path"] + "\t" + res["error"] + "\n")
                    except Exception: print(f"[WARN] Could not write to {FAILED_LOG}: {msg}")

    ray.shutdown()
    print(f"Done - Node {node_rank} of {num_nodes} finished processing {len(local_video_paths)} videos")
    print(f"Failures (if any) recorded in: {FAILED_LOG.resolve()}")
    
    ray.shutdown()
    print(f'Done - Node {node_rank} of {num_nodes} finished processing {len(local_video_paths)} videos')


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--datasets', type=str, nargs='+', default=['epic_kitchens_100', 'comma2k19', 'egoexplore', 'kinetics700', 'mkif'])
    args.add_argument('--num_cpus', type=int, default=os.cpu_count())
    args.add_argument('--num_nodes', type=int, default=1)
    args.add_argument('--node_rank', type=int, default=0)
    args.add_argument('--force_overwrite', action='store_true')
    args = args.parse_args()
    all_videos_to_tensors(
        datasets=args.datasets,
        num_cpus=args.num_cpus,
        num_nodes=args.num_nodes,
        node_rank=args.node_rank,
        force_overwrite=True,
    )

def test():
    local_video_paths = ['/mnt/data/waypoint_1/datasets/MKIF/videos/-6dvhAflfbc.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/-TenhotzKlQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/-XeThFZN8mc.webm', '/mnt/data/waypoint_1/datasets/MKIF/videos/00N3siEEqsQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/02ciFYAMrfw.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0RpTykaCVNQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0_qbSUGNC-M.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0a7pZaanATU.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0hPBgw_wzZI.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0piwAogCynU.webm']
    local_video_paths = [Path(path) for path in local_video_paths]
    hevc = [Path('/mnt/data/waypoint_1/datasets/comma2k19/processed/Chunk_1/b0c9d2329ad1606b|2018-08-14--20-41-07/9/video.hevc')]
    list(process(hevc[0], 0.5, force_overwrite=True))
    list(process(local_video_paths[0], 0.5, force_overwrite=True))
    list(process(local_video_paths[1], 0.5, force_overwrite=True))
    list(process(local_video_paths[2], 0.5, force_overwrite=True))
    list(process(local_video_paths[3], 0.5, force_overwrite=True))
    list(process(local_video_paths[4], 0.5, force_overwrite=True))
    list(process(local_video_paths[5], 0.5, force_overwrite=True))
    list(process(local_video_paths[6], 0.5, force_overwrite=True))
    list(process(local_video_paths[7], 0.5, force_overwrite=True))

if __name__ == '__main__':
    # test()
    main()