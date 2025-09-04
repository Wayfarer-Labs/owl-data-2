from pathlib import Path

DIMENSION_ORDER = 'NCHW'
STRIDE_SEC = 5


CHUNK_FRAME_NUM = 512
COMMA_2K19_ZIP_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/comma2k19')
COMMA_2K19_ZIP_OUT_DIR = Path('/mnt/data/waypoint_1/datasets/comma2k19/processed')
COMMA_2K19_NORMALIZED_360_DIR = Path('/mnt/data/waypoint_1/normalized360/comma2k19')

MKIF_DIR = Path('/mnt/data/waypoint_1/datasets/MKIF/videos')
MKIF_OUT_DIR = Path('/mnt/data/waypoint_1/normalized360/mkif')