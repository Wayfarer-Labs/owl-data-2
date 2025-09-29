import os
import csv
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from dotenv import load_dotenv

from owl_data.waypoint_1.game_data.scripts.extract_tar_to_mnt import s3_client, MNT_DST_PATH
from owl_data.waypoint_1.game_data.utils.s3_utils import download_extracted_data_from_s3
from owl_data.waypoint_1.game_data.quality_checks.quality_checks import check_darkness, check_for_menus
from owl_data.waypoint_1.game_data.constants import MENU_THRESHOLD


load_dotenv()


def _iter_extracted_roots(base_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Returns a list of extracted tar root directories.
    Structure assumed from extract_tar_to_mnt.refactor_tar_in_mnt:
      base/device_bucket/genre_bucket/extracted_dir_name
    So we pick directories exactly 3 levels below base.
    """
    # Example: /mnt/data/datasets/extracted_tars/*/*/*
    candidates = list(base_dir.glob('*/*/*'))
    return [p for p in candidates if p.is_dir()]


def _pt_key_for_dir(dirpath: pathlib.Path) -> str:
    """
    Heuristic: use the extracted dir name with '.pt' at bucket root.
    """
    return f"{dirpath.name}.pt"


def _check_flags_for_dir(dirpath: pathlib.Path, manifest_bucket: str) -> tuple[pathlib.Path, list[str]] | None:
    """
    Downloads the .pt for dirpath from manifest bucket and runs darkness/menu checks.
    Returns (dirpath, reasons) or None if .pt missing or error occurs.
    """
    try:
        client = s3_client()
        pt_key = _pt_key_for_dir(dirpath)

        data = download_extracted_data_from_s3(
            s3_client=client,
            manifest_bucket=manifest_bucket,
            pt_s3_key=pt_key
        )

        is_dark = bool(check_darkness(data))

        menu_info = check_for_menus(data)
        menu_bools = []
        menu_bools.extend(menu_info.get('stride-30_chw_menus', []))
        menu_bools.extend(menu_info.get('stride-60_chw_menus', []))
        menu_ratio = (sum(bool(x) for x in menu_bools) / max(1, len(menu_bools))) if menu_bools else 0.0
        is_menu = bool(menu_info.get('is_consecutive_menu', False) or (menu_ratio >= MENU_THRESHOLD))

        reasons = []
        if is_dark:
            reasons.append('dark')
        if is_menu:
            reasons.append('menu')

        if reasons:
            return dirpath, reasons
        return None
    except Exception as e:
        logging.warning(f"Skipping {dirpath} due to error: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Flag extracted dirs for deletion based on quality checks.")
    parser.add_argument('--manifest-bucket', type=str, default='game-data-manifest',
                        help="S3 bucket containing .pt manifests (default: game-data-manifest)")
    parser.add_argument('--mnt-dir', type=str, default=str(MNT_DST_PATH),
                        help=f"Base mount directory of extracted tars (default: {MNT_DST_PATH})")
    parser.add_argument('--output', type=str, default='to_delete.csv',
                        help="Path to write CSV with deletion flags (default: to_delete.csv)")
    parser.add_argument('--max-workers', type=int, default=1,
                        help="Max workers for parallel processing (default: 1)")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.mnt_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    dirs = _iter_extracted_roots(base_dir)
    logging.info(f"Found {len(dirs)} extracted directories under {base_dir}")

    results = []
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_check_flags_for_dir, d, args.manifest_bucket): d for d in dirs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc='Checking flags'):
                res = fut.result()
                if res is not None:
                    dirpath, reasons = res
                    results.append((str(dirpath), ';'.join(reasons)))
    else:
        for d in tqdm(dirs, desc='Checking flags'):
            res = _check_flags_for_dir(d, args.manifest_bucket)
            if res is not None:
                dirpath, reasons = res
                results.append((str(dirpath), ';'.join(reasons)))

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dirpath', 'reason'])
        for row in results:
            writer.writerow(row)

    logging.info(f"Wrote {len(results)} rows to {args.output}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
