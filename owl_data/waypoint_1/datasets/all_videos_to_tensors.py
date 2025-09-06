import ray
import tqdm
import torch
import numpy as np
from pathlib import Path
from multimethod import multimethod, parametric
import argparse
from typing import Literal, Generator
import owl_data.waypoint_1.datasets.utils as utils
import os
import itertools

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


is_epic_kitchens_100    = lambda path: isinstance(path, Path) and 'epic_kitchens_100' in path.parts
is_comma2k19            = lambda path: isinstance(path, Path) and 'comma2k19' in path.parts
is_egoexplore           = lambda path: isinstance(path, Path) and 'egoexplore' in path.parts
is_kinetics700          = lambda path: isinstance(path, Path) and 'kinetics700' in path.parts
is_mkif                 = lambda path: isinstance(path, Path) and 'MKIF' in path.parts

EpicKitchensPath        = parametric(Path, is_epic_kitchens_100)
Comma2k19Path           = parametric(Path, is_comma2k19)
EgoexplorePath          = parametric(Path, is_egoexplore)
Kinetics700Path         = parametric(Path, is_kinetics700)
MkifPath                = parametric(Path, is_mkif)


# ---------- video_paths
@multimethod
def video_paths(dataset: str) -> Generator[Path, None, None]:
    raise NotImplementedError(f"Unsupported dataset: {dataset!r}")

@video_paths.register
def _(dataset: Literal['epic_kitchens_100']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'epic_kitchens_100' / '2g1n6qdydwa9u22shpxqzp0t8m'
    yield from root.glob('P*/videos/*.MP4')

@video_paths.register
def _(dataset: Literal['comma2k19']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'comma2k19' / 'processed'
    yield from root.glob('Chunk_*/**/**/video.hevc')

@video_paths.register
def _(dataset: Literal['egoexplore']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'egoexplore' / 'videos'
    yield from root.glob('*.mp4?*')

@video_paths.register
def _(dataset: Literal['kinetics700']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'kinetics700' / 'Kinetics-700'
    yield from root.glob('Kinetics700_part_*/test/*.mp4')

@video_paths.register
def _(dataset: Literal['mkif']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'MKIF' / 'videos'
    yield from root.glob('*.mp4')
    yield from root.glob('*.webm')


# ---------- output_path
@multimethod
def output_path(path: Path) -> Path:
    raise NotImplementedError(f"Unsupported path: {path}")

@output_path.register
def _(path: EpicKitchensPath) -> Path:
    return NORMALIZED_360_DIR / 'epic_kitchens_100' / path.parent.name / path.stem

@output_path.register
def _(path: Comma2k19Path) -> Path:
    # chunk / dongle / index
    return (NORMALIZED_360_DIR / 'comma2k19' / 'processed' /
            path.parent.parent.parent.name / path.parent.parent.name / path.parent.name)

@output_path.register
def _(path: EgoexplorePath) -> Path:
    return NORMALIZED_360_DIR / 'egoexplore' / 'videos' / path.stem

@output_path.register
def _(path: Kinetics700Path) -> Path:
    # .../Kinetics-700/Kinetics700_part_*/test/<file>.mp4
    return (NORMALIZED_360_DIR / 'kinetics700' / 'Kinetics-700' /
            path.parent.parent.name / path.parent.name / path.stem)

@output_path.register
def _(path: MkifPath) -> Path:
    return NORMALIZED_360_DIR / 'MKIF' / 'videos' / path.stem

# ---------- dataset_from_path
@multimethod
def dataset_from_path(path: Path) -> Datasets:
    raise NotImplementedError(f"Unsupported path: {path}")

@dataset_from_path.register
def _(path: EpicKitchensPath) -> Literal['epic_kitchens_100']:
    return 'epic_kitchens_100'

@dataset_from_path.register
def _(path: Comma2k19Path) -> Literal['comma2k19']:
    return 'comma2k19'

@dataset_from_path.register
def _(path: EgoexplorePath) -> Literal['egoexplore']:
    return 'egoexplore'

@dataset_from_path.register
def _(path: Kinetics700Path) -> Literal['kinetics700']:
    return 'kinetics700'

@dataset_from_path.register
def _(path: MkifPath) -> Literal['mkif']:
    return 'mkif'


@ray.remote
def process(
    path: Path,
    stride_sec: float,
    chunk_size: int = 512,
    force_overwrite: bool = False,
) -> list[Path]:
    output_path(path).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, chunk in enumerate(
        utils.process_video_seek(path, stride_sec, chunk_size)
    ):
        filepath = output_path(path) / f'{i:08d}_rgb.pt'
        
        if not force_overwrite and filepath.exists():
            continue
        
        torch.save(chunk, filepath)
        paths.append(filepath)
    
    return paths


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
            for path in ray.get(done):
                pbar.update(1)
                print(f'Processed {path}')
    
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
        force_overwrite=args.force_overwrite,
    )

def test():
    local_video_paths = ['/mnt/data/waypoint_1/datasets/MKIF/videos/-6dvhAflfbc.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/-TenhotzKlQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/-XeThFZN8mc.webm', '/mnt/data/waypoint_1/datasets/MKIF/videos/00N3siEEqsQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/02ciFYAMrfw.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0RpTykaCVNQ.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0_qbSUGNC-M.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0a7pZaanATU.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0hPBgw_wzZI.mp4', '/mnt/data/waypoint_1/datasets/MKIF/videos/0piwAogCynU.webm']
    local_video_paths = [Path(path) for path in local_video_paths]
    hevc = [Path('/mnt/data/waypoint_1/datasets/comma2k19/processed/Chunk_1/b0c9d2329ad1606b|2018-08-14--20-41-07/9/video.hevc')]
    list(process(hevc[0], 0.5, force_overwrite=True))

if __name__ == '__main__':
    # test()
    main()