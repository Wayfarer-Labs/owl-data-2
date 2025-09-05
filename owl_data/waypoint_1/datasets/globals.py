from pathlib import Path

DIMENSION_ORDER = 'NCHW'
STRIDE_SEC = 5


CHUNK_FRAME_NUM = 512
COMMA_2K19_ZIP_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/comma2k19')
COMMA_2K19_ZIP_OUT_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/processed')
COMMA_2K19_NORMALIZED_360_DIR = Path('/mnt/data/waypoint_1/normalized360/comma2k19')

MKIF_DIR = Path('/mnt/data/waypoint_1/datasets/MKIF/videos')
MKIF_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/mkif')

EGOEXPLORE_DIR = Path('/mnt/data/waypoint_1/datasets/egoexplore/videos')
EGOEXPLORE_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/egoexplore/videos')

EPIC_KITCHENS_100_DIR = Path('/mnt/data/waypoint_1/datasets/epic_kitchens_100/2g1n6qdydwa9u22shpxqzp0t8m')
EPIC_KITCHENS_100_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/epic_kitchens_100')

KINETICS_700_DIR = Path('/mnt/data/waypoint_1/datasets/kinetics700/Kinetics-700/')
KINETICS_700_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/kinetics700/Kinetics-700')