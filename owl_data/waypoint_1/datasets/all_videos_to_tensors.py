import ray
import tqdm
import torch
import numpy as np
from pathlib import Path
from multimethod import multimethod, overload
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


# ----- EPIC KITCHENS

@multimethod
def video_paths(dataset: Literal['epic_kitchens_100']) -> Generator[Path, None, None]:
    yield from (
        Path('/mnt/data/waypoint_1/datasets')
        / 'epic_kitchens_100' / '2g1n6qdydwa9u22shpxqzp0t8m'
    ).glob('P*/videos/*.MP4')

@overload
def output_path(path: is_epic_kitchens_100) -> Path:
    return NORMALIZED_360_DIR / 'epic_kitchens_100' / path.parent.name / path.stem

@overload
def dataset_from_path(path: is_epic_kitchens_100) -> Literal['epic_kitchens_100']:
    return 'epic_kitchens_100'

# ----- COMMA2k19

@multimethod
def video_paths(dataset: Literal['comma2k19']) -> Generator[Path, None, None]:
    yield from (
        Path('/mnt/data/waypoint_1/datasets')
        / 'comma2k19' / 'processed'
    ).glob('Chunk_*/**/**/video.hevc')

@overload
def output_path(path: is_comma2k19) -> Path:
    return (
        NORMALIZED_360_DIR /
        'comma2k19' / 
        'processed' /
        path.parent.parent.parent.name /  # chunk
        path.parent.parent.name / # dongle
        path.parent.name # index
    )

@overload
def dataset_from_path(path: is_comma2k19) -> Literal['comma2k19']:
    return 'comma2k19'

# ----- EGOEXPLORE

@multimethod
def video_paths(dataset: Literal['egoexplore']) -> Generator[Path, None, None]:
    yield from (
        Path('/mnt/data/waypoint_1/datasets')
        / 'egoexplore' / 'videos'
    ).glob('*.mp4?*')

@overload
def dataset_from_path(path: is_egoexplore) -> Literal['egoexplore']:
    return 'egoexplore'

@overload
def output_path(path: is_egoexplore) -> Path:
    return (
        NORMALIZED_360_DIR /
        'egoexplore' / 'videos' /
        path.stem
    )

# ----- KINETICS 700

@multimethod
def video_paths(dataset: Literal['kinetics700']) -> Generator[Path, None, None]:
    yield from (
        Path('/mnt/data/waypoint_1/datasets')
        / 'kinetics700' / 'Kinetics-700'
    ).glob('Kinetics700_part_*/test/*.mp4')

@overload
def dataset_from_path(path: is_kinetics700) -> Literal['kinetics700']:
    return 'kinetics700'

@overload
def output_path(path: is_kinetics700) -> Path:
    return (
        NORMALIZED_360_DIR /
        'kinetics700' / 'Kinetics-700' /
        path.parent.parent.name / # Kinetics700_part_*
        path.parent.name / # test
        path.stem
    )

# ------ MKIF

@multimethod
def video_paths(dataset: Literal['mkif']) -> Generator[Path, None, None]:
    root = Path('/mnt/data/waypoint_1/datasets') / 'MKIF' / 'videos'
    yield from root.glob('*.mp4')
    yield from root.glob('*.webm')

@overload
def dataset_from_path(path: is_mkif) -> Literal['mkif']:
    return 'mkif'

@overload
def output_path(path: is_mkif) -> Path:
    return (
        NORMALIZED_360_DIR /
        'MKIF' / 'videos' /
        path.stem
    )

# -----

@ray.remote
def process(
    path: Path,
    stride_sec: float,
    chunk_size: int = 512,
    force_overwrite: bool = False,
) -> Generator[Path, None, None]:
    output_path(path).mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(
        utils.process_video_seek(path, stride_sec, chunk_size)
    ):
        filepath = output_path(path) / f'{i:08d}_rgb.pt'
        
        if not force_overwrite and filepath.exists():
            continue
        
        torch.save(chunk, filepath)
        yield filepath


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

if __name__ == '__main__':
    main()