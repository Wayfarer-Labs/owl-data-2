import io, logging, os, sys, shutil, pathlib, tarfile, functools, boto3, json, csv, torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

@functools.cache
def s3_client() -> boto3.client:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    return s3_client


def _is_kbm(inputs: dict) -> bool:
    return any('MOUSE' in k.upper() for k in inputs['event_type'])

def _is_controller(inputs: dict) -> bool:
    return any('GAMEPAD' in k.upper() for k in inputs['event_type'])

def _is_first_person_shooter(metadata: dict) -> bool:
    from fps_3ps_detector import exe_perspective
    return exe_perspective.get(metadata['game_exe']) == 'fps'

def _is_third_person(metadata: dict) -> bool:
    from fps_3ps_detector import exe_perspective
    return exe_perspective.get(metadata['game_exe']) == '3ps'


TASK_LIST_PATH = pathlib.Path('task_list.txt')
MNT_DST_PATH = pathlib.Path('/mnt/data/datasets/extracted_tars')


def _normalize_tar_filename(task_id: str) -> str:
    name = os.path.basename(task_id.rstrip('/'))
    return name if name.endswith('.tar') else f'{name}.tar'


def _read_metadata_from_tar(tf: tarfile.TarFile) -> dict:
    # Prefer a top-level metadata.json; otherwise, fall back to the first *.json file
    members = tf.getmembers()
    meta_member = None
    for m in members:
        base = pathlib.Path(m.name).name
        if base == 'metadata.json' and m.isfile():
            meta_member = m
            break
    if meta_member is None:
        for m in members:
            if m.isfile() and pathlib.Path(m.name).suffix.lower() == '.json':
                meta_member = m
                break
    if meta_member is None:
        return {}
    f = tf.extractfile(meta_member)
    return json.loads(f.read().decode('utf-8')) if f is not None else {}


def _read_inputs_from_tar(tf: tarfile.TarFile) -> dict:
    """
    Locate an inputs CSV inside `tf` and return it in column-oriented form:
        { "colA": [v1, v2, ...], "colB": [w1, w2, ...], ... }

    Heuristics:
      - Prefer a top-level 'inputs.csv'
      - Fallback to the first '*.csv'
      - If none found, return {}
      - Delimiter auto-detection among {',', '\\t', ';'} by simple frequency
    """
    members = tf.getmembers()

    # 1) Prefer exact 'inputs.csv' anywhere
    candidate = None
    for m in members:
        if m.isfile() and pathlib.Path(m.name).name.lower() == "inputs.csv":
            candidate = m
            break

    # 2) Otherwise, first '*input*.csv'
    if candidate is None:
        for m in members:
            name = pathlib.Path(m.name).name.lower()
            if m.isfile() and name.endswith(".csv"):
                candidate = m
                break

    if candidate is None:
        return {}

    f = tf.extractfile(candidate)
    if f is None:
        return {}

    # Decode, strip BOM if present
    text = f.read().decode("utf-8-sig")

    # Lightweight delimiter "sniff" (no try/except): choose the most frequent
    sample = text[:4096]
    delim_counts = {
        ",": sample.count(","),
        "\t": sample.count("\t"),
        ";": sample.count(";"),
    }
    delimiter = max(delim_counts, key=delim_counts.get) or ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        return {}

    columns = {field: [] for field in reader.fieldnames}

    for row in reader:
        for field in reader.fieldnames:
            val = row.get(field, "")
            # Normalize to string; strip surrounding whitespace
            if val is None:
                val = ""
            columns[field].append(str(val).strip())

    return columns


def download_tar(task_id: str, bucket_name: str = 'game-data') -> tuple[pathlib.Path, dict, dict]:
    """
    Downloads a tar file from the bucket and stores it under MNT_DST_PATH.
    Returns:
        (path_to_tar, metadata_dict)
    """
    MNT_DST_PATH.mkdir(parents=True, exist_ok=True)

    response = s3_client().get_object(Bucket=bucket_name, Key=task_id)
    tar_bytes = response['Body'].read()

    tar_filename = _normalize_tar_filename(task_id)
    tar_path = MNT_DST_PATH / tar_filename

    # Write tar to disk
    with open(tar_path, 'wb') as f:
        f.write(tar_bytes)

    # Read metadata from the tar (best-effort; empty dict if none)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:*') as tf:
        metadata = _read_metadata_from_tar(tf)
        inputs = _read_inputs_from_tar(tf)

    return tar_path, metadata, inputs

def extract_tar(tar_path: pathlib.Path) -> pathlib.Path:
    """
    Extracts tar into a directory with the same stem as the tar file.
    Example: /a/b/uuid.tar -> /a/b/uuid/
    """
    dst_dir = tar_path.with_suffix('')  # remove .tar
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Basic path traversal guard
    def _is_within_directory(directory: pathlib.Path, target: pathlib.Path) -> bool:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))

    with tarfile.open(tar_path, mode='r:*') as tf:
        safe_members = []
        for m in tf.getmembers():
            # Resolve extraction target and ensure it stays inside dst_dir
            target_path = (dst_dir / m.name).resolve()
            if _is_within_directory(dst_dir, target_path):
                safe_members.append(m)
        tf.extractall(path=dst_dir, members=safe_members)
    
    return dst_dir


def refactor_tar_in_mnt(extracted_dir: pathlib.Path, metadata: dict, inputs: dict) -> pathlib.Path:
    """
    `extracted_dir` is the directory returned by `extract_tar`.
    Idempotent behavior:
      - If the final destination already exists, we keep it and remove `extracted_dir`.
      - Otherwise, we move `extracted_dir` exactly to the final path (no nesting).
    """
    # Choose top-level device bucket
    if _is_controller(inputs):
        device_bucket = 'controller'
    elif _is_kbm(inputs):
        device_bucket = 'kbm'
    else:
        device_bucket = 'unknown'

    # Choose sub-genre bucket
    if _is_first_person_shooter(metadata):
        genre_bucket = 'fps'
    elif _is_third_person(metadata):
        genre_bucket = '3ps'
    else:
        genre_bucket = 'other'

    dst_base = MNT_DST_PATH / device_bucket / genre_bucket
    dst_base.mkdir(parents=True, exist_ok=True)

    dst_dir = dst_base / extracted_dir.name

    if dst_dir.exists():
        # Already organized there; delete the redundant extracted copy and return
        if extracted_dir.resolve() != dst_dir.resolve() and extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        logging.info(f'Already placed at {dst_dir}, skipping move.')
        return dst_dir

    # Destination does not exist: move exactly to dst_dir (no nesting)
    # Use rename when on same filesystem; shutil.move falls back to copy+remove cross-device.
    extracted_dir.rename(dst_dir)
    logging.info(f'Moved {extracted_dir} to {dst_dir}')
    return dst_dir


def process_entire_tar(task_id: str):
    try:
        logging.info(f'Processing {task_id}')
        tar_path, metadata, inputs = download_tar(task_id)
        logging.info(f'Downloaded {task_id}')
        extracted_dir = extract_tar(tar_path)
        logging.info(f'Extracted {task_id}')
        refactor_tar_in_mnt(extracted_dir, metadata, inputs)
        logging.info(f'Refactored {task_id}')
        os.remove(tar_path)
        logging.info(f'Removed {tar_path}')
        return True
    except Exception as e:
        import traceback
        logging.error(f'Skipping {task_id} because of error: {e}')
        logging.error(traceback.format_exc())
        return False

def main():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--task-list-path', type=str, default='task_list.txt')
    args.add_argument('--num-nodes', type=int, default=1)
    args.add_argument('--node-rank', type=int, default=0)
    args = args.parse_args()

    with open(args.task_list_path, 'r') as f:
        task_ids = [line.strip() for line in f.readlines()]

    local_task_ids = [
        t
        for i, t in enumerate(task_ids)
        if i % args.num_nodes == args.node_rank
    ]

    for task_id in tqdm(local_task_ids, desc='Processing tasks'):
        process_entire_tar(task_id)


def get_all_games(task_list_path: str) -> dict[str, int]:
    from collections import Counter

    with open(task_list_path, 'r') as f:
        task_ids = [line.strip() for line in f.readlines()]
    
    games = []
    for task_id in task_ids:
        try:
            response = s3_client().get_object(Bucket='game-data-manifest', Key=task_id.replace('.tar', '.pt'))
            pt_bytes = response['Body'].read()
            pt = torch.load(io.BytesIO(pt_bytes), weights_only=False)
            game = pt['session_metadata'].get('game_exe')
            games.append(game)
            logging.info(f'Added {game}')
        except Exception as e:
            logging.error(f'Skipping {task_id} because of error: {e}')

    
    with open('games.txt', 'w') as f:
        json.dump(games := dict(Counter(games).most_common()), f, indent=2)
        return games


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'extract_tar_to_mnt.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    main()

    # get_all_games('task_list.txt')
