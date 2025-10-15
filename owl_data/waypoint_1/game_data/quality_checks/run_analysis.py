import pathlib
import os

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


def gemini_client():
    pass


def check_darkness(metadata_json: dict, mp4_bytes: bytes) -> dict:
    darkness_prompt = """
    You are given a single video clip.
    Your task is to identify whether the clip is of a video game,
    or if it's just a dark screen, which would indicate that the video encountered a
    crash or a recording malfunction.

    Output JSON ONLY, EXACTLY this schema:
    {"is_darkness":<bool>,"reason":<str>}
    If you found footage of a dark screen, output: {"is_darkness":True,"reason":"dark screen detected"}
    If not, output: {"is_darkness":False,"reason":"<what you see on screen>"}
    """
    if metadata_json.get('game_exe', '').lower() in BANNED_EXES:
        return {'is_darkness': True, 'reason': f'{metadata_json.get("game_exe")} detected', 'error': None}
    
    import json
    import google.genai.types as types
    from owl_data.waypoint_1.game_data.quality_checks.analyzers.hacky_menu_script import _generate_with_retries
    
    contents = types.Content(parts=[
        types.Part(text=darkness_prompt),
        types.Part(inline_data=types.Blob(data=mp4_bytes, mime_type='video/mp4')),
    ])

    try:
        response: types.GenerateContentResponse = _generate_with_retries(contents)
        part = response.candidates[0].content.parts[0]
        raw = getattr(part, "text", None) or getattr(part, "data", None)
        parsed = json.loads(raw)
        return parsed | {'error': None}
    except Exception as e:
        import traceback as tb
        return {'is_darkness': False, 'reason': f'{type(e).__name__}: {e}', 'error': "".join(tb.format_exception(e))}


def run_checks(tasks: list[str], out_path_jsonl: str, checks: list[str] = ['menus', 'dark_screen', 'deduplicated']) -> None:
    from owl_data.waypoint_1.game_data.quality_checks.analyzers.hacky_menu_script import (
        ask_gemini,
        yield_ext_bytes_from_tars
    )
    import json, tqdm, io, copy

    existing_json = []
    if os.path.exists(out_path_jsonl):
        with open(out_path_jsonl, 'r') as f:
            existing_json = [json.loads(line) for line in f if line.strip()]

    for tar_path in tqdm.tqdm(tasks):
        row_default = {'tar_name': tar_path, 'chunk_name': None, 'quality_checks': {}, 'error': None}
        rows = []
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
                    dark_screen = check_darkness(metadata_json, mp4_bytes)
                    row['quality_checks']['dark_screen'] = dark_screen
            
                if 'menus' in checks:
                    gemini_response_menus = ask_gemini(tar_path, mp4_name, mp4_bytes)
                    row['quality_checks']['menus'] = gemini_response_menus
                
                rows.append(row)

        except Exception as e:
            import traceback as tb
            row = copy.deepcopy(row_default)
            row['error'] = "".join(tb.format_exception(e))
            rows.append(row)
        
        with open(out_path_jsonl, 'a') as f:
            for row in rows: f.write(json.dumps(row) + '\n')
        
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
        default=0,
        help="Parallelism via threads (0 or 1 = sequential)")
    p.add_argument("--num_nodes", type=int,
        default=1,
        help="World size: total number of nodes participating in this run")
    p.add_argument("--include-checks", type=str, required=False,
        default=["menus", "dark_screen"],
        help="One of the above quality checks of menus, dark_screen, deduplicated")
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


    num_procs = max(1, os.cpu_count() // 4)

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