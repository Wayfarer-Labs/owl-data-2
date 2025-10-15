import pathlib
import os
from datetime import datetime, timezone, timedelta

QUALITY_CHECKS_JSONL_TAR = '/mnt/data/datasets/quality_checks_tars'
DOWNSAMPLED_TARS_PATH = pathlib.Path('/mnt/data/datasets/downsampled_tars')

BANNED_EXES = (
    'obs.exe',
    'obs64.exe',
    'obs32.exe',
    'spotify.exe',
    'opera.exe',
    '',
    'discord.exe',
    'discordptb.exe',
    'OWL Control.exe',
    'firefox.exe',
    'msedge.exe',
    'searchapp.exe',
    'applicationframehost.exe',
    'discovery.exe',
    'sh.exe'
)

BANNED_EXES = tuple(exe.lower() for exe in BANNED_EXES)


BANNED_DATETIME_BEFORE = int(datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def parse_timestep(ts):
    """
    Parse the given timestep (expected as integer Unix timestamp or string representing an int)
    and return it as a UTC datetime object.
    """
    try:
        timestamp = int(ts)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid timestep for datetime parsing: {ts}") from e

from owl_data.waypoint_1.game_data.quality_checks import google_gemini

gemini = google_gemini.GoogleAuthOpenAI(project_id='openworld-main')

def run_checks(
    tasks: list[str],
    out_path_jsonl: str,
    checks: list[str] = ['menus', 'dark_screen', 'banned_exes', 'before_datetime'],
    skip_existing: bool = False
) -> None:
    from owl_data.waypoint_1.game_data.quality_checks.analyzers.hacky_menu_script import (
        yield_ext_bytes_from_tars
    )
    import json, tqdm, io, copy

    existing_json = []
    if os.path.exists(out_path_jsonl):
        with open(out_path_jsonl, 'r') as f:
            existing_json = [json.loads(line) for line in f if line.strip()]
    
    _existing_checks: set[tuple[str, str, str]] = set() # set of tuples of tarname, chunkname, check
    for jsonl in existing_json:
        quality_checks = jsonl.get('quality_checks', {})
        for check, value in quality_checks.items():
            if value is not None and value.get('error') is None:
                _existing_checks.add((jsonl['tar_name'], jsonl['chunk_name'], check))

    for tar_path in tqdm.tqdm(tasks):
        row_default = {'tar_name': tar_path, 'chunk_name': None, 'quality_checks': {}, 'error': None}
        try:
            items = list(yield_ext_bytes_from_tars([DOWNSAMPLED_TARS_PATH / tar_path], '.mp4'))
            if not items:
                print(f"No mp4 files found for {tar_path}")
                continue
        
            tar_paths, mp4_names, mp4_bytes = zip(*items)
            *_, metadata_jsonb = next(yield_ext_bytes_from_tars([DOWNSAMPLED_TARS_PATH / tar_path], '.json'))
            metadata_json = json.load(io.BytesIO(metadata_jsonb))
 
            for tar_path, mp4_name, mp4_bytes in zip(tar_paths, mp4_names, mp4_bytes):
        
                row = copy.deepcopy(row_default)
                row['chunk_name'] = mp4_name

                if 'dark_screen' in checks:
                    if skip_existing and (tar_path, mp4_name, 'dark_screen') in _existing_checks:
                        print(f"Skipping ({tar_path}, {mp4_name}, dark_screen) because it already exists")
                    else:
                        dark_screen = gemini.ask_dark_screen(mp4_bytes)
                        row['quality_checks']['dark_screen'] = dark_screen
            
                if 'menus' in checks:
                    if skip_existing and (tar_path, mp4_name, 'menus') in _existing_checks:
                        print(f"Skipping ({tar_path}, {mp4_name}, menus) because it already exists")
                    else:
                        gemini_response_menus = gemini.ask_menus(mp4_bytes)
                        row['quality_checks']['menus'] = gemini_response_menus

                if 'banned_exes' in checks:
                    exe_name = metadata_json.get('game_exe', None)
                    row['quality_checks']['banned_exes'] = {'is_banned': exe_name in BANNED_EXES, 'exe_name': exe_name, 'error': None}

                if 'before_datetime' in checks:
                    created_at = metadata_json.get('created_at', None)
                    if created_at is None:
                        row['quality_checks']['before_datetime'] = {'is_before': None, 'created_at': None, 'threshold': None, 'error': "created_at is None"}
                    else:
                        created_at = parse_timestep(created_at)
                        row['quality_checks']['before_datetime'] = {
                            'is_before': created_at < BANNED_DATETIME_BEFORE,
                            'created_at': created_at,
                            'threshold': datetime.fromtimestamp(BANNED_DATETIME_BEFORE, tz=timezone.utc),
                            'error': None
                        }

                with open(out_path_jsonl, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                    print(f"Wrote {tar_path} {mp4_name} to {out_path_jsonl}")

        except Exception as e:
            import traceback as tb
            row = copy.deepcopy(row_default)
            row['error'] = "".join(tb.format_exception(e))
            with open(out_path_jsonl, 'a') as f:
                f.write(json.dumps(row) + '\n')
                print(f"ERROR: Wrote {tar_path} {mp4_name} to {out_path_jsonl}")

        
def run_checks_wrapper(args_tuple):
    tasks, out_path_jsonl, include_checks = args_tuple
    run_checks(tasks, out_path_jsonl, include_checks)

def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run chunk analyzers over downsampled TARs and write Parquet manifests (run_id partitioned).")
    p.add_argument("--task-list", type=str, required=False,
        default='task_list.txt',
        help="Text file with one local path per line")
    p.add_argument("--out-path-jsonl", type=str, required=False,
        default=os.path.join(QUALITY_CHECKS_JSONL_TAR, "quality_checks.jsonl"),
        help="Output prefix (local dir or s3://bucket/prefix)")
    p.add_argument("--num-workers", type=int,
        default= os.cpu_count() // 4,
        help="Parallelism via threads (0 or 1 = sequential)")
    p.add_argument("--num_nodes", type=int,
        default=1,
        help="World size: total number of nodes participating in this run")
    p.add_argument("--include-checks", type=str, required=False,
        default=["menus", "dark_screen", "banned_exes", "before_datetime"],
        help="One of the above quality checks of menus, dark_screen, banned_exes, before_datetime")
    p.add_argument("--node_rank", type=int,
        default=0,
        help="This node's rank in [0, num_nodes)")

    args = p.parse_args()

    def _load_task_list(task_list_path: str) -> list[str]:
        with open(task_list_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    tasks = _load_task_list(args.task_list)
    local_tasks = [t for i, t in enumerate(tasks) if i % args.num_nodes == args.node_rank]


    import multiprocessing


    num_procs = args.num_workers
    # num_procs = 1

    # Partition local_tasks into roughly equal chunks for each process
    def chunkify(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n) if lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]]

    task_chunks = chunkify(local_tasks, num_procs)

    pool_args = [(chunk, args.out_path_jsonl, args.include_checks) for chunk in task_chunks]

    if num_procs == 1 or len(pool_args) == 1:
        # Run sequentially if only one process or one chunk
        [run_checks_wrapper(pool_args[0])]
    else:
        with multiprocessing.get_context("spawn").Pool(num_procs) as pool:
            pool.map(run_checks_wrapper, pool_args)
    

if __name__ == "__main__":
    main()