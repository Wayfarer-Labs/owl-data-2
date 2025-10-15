import os, dotenv, pathlib, typing, json, copy
from copy import deepcopy
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

dotenv.load_dotenv()

import pandas as pd
from typing import Optional, Dict, Any


# load all tar paths
TAR_PATHS = list(pathlib.Path("/mnt/data/datasets/downsampled_tars/").glob("*.tar"))
EMPTY_RESPONSE = {"intervals":[]}
# define prompt

def get_video_meta_from_bytes(mp4_bytes: bytes) -> dict:
    """Return {'fps': float, 'frames': int, 'duration_sec': float}."""
    import tempfile, cv2, numpy as np, os
    fps = 30.0
    frames = 0
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            _fps = cap.get(cv2.CAP_PROP_FPS)
            _frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if _fps and _fps > 1e-3:
                fps = float(_fps)
            if _frames and _frames > 0:
                frames = _frames
            cap.release()
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
    duration_sec = (frames / fps) if (fps > 0 and frames > 0) else 0.0
    return {"fps": fps, "frames": frames, "duration_sec": duration_sec}


def build_prompt(duration_sec: float) -> str:
    return (
        "You are given a single video clip.\n"
        f"The clip's total duration is {duration_sec:.3f} seconds.\n\n"
        "Task: Identify time intervals (in seconds) where the player has a menu screen open, or the game is in a loading screen. "
        "This is for purposes of detecting pauses in gameplay.\n\n"
        "HARD REQUIREMENTS:\n"
        f"- Every interval must satisfy 0 <= start_sec < end_sec <= {duration_sec:.3f}\n"
        "- Do not output timestamps outside the clip.\n"
        "- Merge overlapping/adjacent intervals; no overlaps.\n"
        "- Do not get mixed up between gameplay (quick movement) and loading screens and menus (maps, etc.).\n"
        "- If none found, return the empty structure below.\n\n"
        "Output JSON ONLY, EXACTLY this schema:\n"
        '{"intervals":[{"start_sec":<int>,"end_sec":<int>}]}\n'
        'If you did not find any menus, output: {"intervals":[]}\n'
    )

EMPTY_RESPONSE = {"intervals": []}

def _sanitize_intervals(
    result: dict,
    duration_sec: float,
    min_len: float = 0.05,
    tol_merge: float = 0.10
) -> dict:
    """
    - Ensure JSON shape
    - Clamp to [0, duration]
    - Drop invalid / too short
    - Merge overlaps/adjacent within tol_merge
    """
    intervals = (result or {}).get("intervals", [])
    if not isinstance(intervals, list):
        return deepcopy(EMPTY_RESPONSE)

    cleaned = []
    for it in intervals:
        try:
            s = float(it.get("start_sec", 0.0))
            e = float(it.get("end_sec", 0.0))
        except Exception:
            continue
        # clamp
        s = max(0.0, min(s, duration_sec))
        e = max(0.0, min(e, duration_sec))
        if e <= s: 
            continue
        if (e - s) < min_len:
            continue
        cleaned.append({"start_sec": s, "end_sec": e})

    if not cleaned:
        return deepcopy(EMPTY_RESPONSE)

    cleaned.sort(key=lambda x: x["start_sec"])

    merged = []
    for it in cleaned:
        if not merged:
            merged.append(it)
            continue
        last = merged[-1]
        # overlap or adjacency within tolerance
        if it["start_sec"] <= last["end_sec"] + tol_merge:
            last["end_sec"] = max(last["end_sec"], it["end_sec"])
        else:
            merged.append(it)

    return {"intervals": merged}

# define csv path
CSV_PATH = "/mnt/data/sami/menu_intervals.csv"
# csv columns
CSV_COLUMNS = [
    "tar_name",
    "mp4_chunk_name",
    "menu_start_sec",
    "menu_end_sec",
    "error"
]

# -- 
print (f"Found {len(TAR_PATHS)} tar paths")
print (f"GEMINI_API_KEY Found: {os.getenv('GEMINI_API_KEY') is not None}")


def yield_ext_bytes_from_tars(tar_paths: list[pathlib.Path], ext: str = ".mp4") -> typing.Generator[tuple[str, str, bytes], None, None]:
    """
    Yields a tuple of (tar_path, mp4_chunk_name, mp4_bytes) for each mp4 in tars
    """
    import tarfile

    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith(ext):
                    continue
                with tar.extractfile(member) as f:
                    yield (str(tar_path), member.name, f.read())

# import google.genai as genai

# CLIENT = genai.Client()
# MODEL = 'gemini-2.5-flash-lite'


# Retry helper for Gemini generate_content calls
def _is_retryable_exception(e: Exception) -> bool:
    """Return True for transient errors like timeouts and HTTP 5xx."""
    status = getattr(e, "status", None) or getattr(e, "code", None)
    if isinstance(status, int) and status in {500, 502, 503, 504}:
        return True
    msg = str(e).lower()
    retry_tokens = [
        "timeout",
        "timed out",
        "internal server error",
        "server error",
        "500",
        "502",
        "503",
        "504",
        "gateway timeout",
        "service unavailable",
    ]
    return any(tok in msg for tok in retry_tokens)


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception(_is_retryable_exception),
)
def _generate_with_retries(contents):
    import google.genai.types as types
    return CLIENT.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )


# async function that takes bytes from an mp4 and sends a query to gemini and parses the response into a list of csv rows per interval
def ask_gemini(tar_path: str, mp4_chunk_name: str, mp4_bytes: bytes, in_prompt: str = None) -> dict:
    global CLIENT, MODEL
    import google.genai.types as types

    meta = get_video_meta_from_bytes(mp4_bytes)
    prompt = in_prompt or build_prompt(meta["duration_sec"])

    contents = types.Content(parts=[
        types.Part(text=prompt),
        types.Part(inline_data=types.Blob(data=mp4_bytes, mime_type='video/mp4')),
    ])

    try:
        # Optional but helpful: force JSON output (with retries on 5xx/timeouts)
        response: types.GenerateContentResponse = _generate_with_retries(contents)
    except Exception as e:
        err = deepcopy(EMPTY_RESPONSE)
        err["error"] = f"{type(e).__name__}: {e}"
        return err

    # Some SDKs return JSON in `candidates[0].content.parts[0].text`.
    # Others put it in `.data` when JSON MIME is used—handle both.
    part = response.candidates[0].content.parts[0]
    raw = getattr(part, "text", None) or getattr(part, "data", None)

    try:
        parsed = json.loads(raw)
    except Exception as e:
        err = deepcopy(EMPTY_RESPONSE)
        err["error"] = f"JSON parse error: {e} | raw={raw!r}"
        return err

    out = _sanitize_intervals(parsed, meta["duration_sec"])
    out["error"] = None
    return out

def write_csv_rows(intervals: dict, filepath: str, tar_name: str, mp4_chunk_name: str):
    import pandas as pd
    error = intervals.get("error", None)

    rows = []
    its = intervals.get("intervals", [])
    if not its:
        rows.append({
            "tar_name": tar_name,
            "mp4_chunk_name": mp4_chunk_name,
            "menu_start_sec": None,
            "menu_end_sec": None,
            "error": error,
        })
    else:
        for it in its:
            rows.append({
                "tar_name": tar_name,
                "mp4_chunk_name": mp4_chunk_name,
                "menu_start_sec": it.get("start_sec"),
                "menu_end_sec": it.get("end_sec"),
                "error": error,
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, mode='a', header=False, index=False)


def visualize(tar_path: str, mp4_chunk_name: str, start_sec: float, end_sec: float) -> str:
    import os
    import tarfile
    from pathlib import Path
    import cv2

    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    with tarfile.open(tar_path) as tar:
        mp4_members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]
        if not mp4_members:
            raise FileNotFoundError(f"No .mp4 members found in tar: {tar_path}")

        member = next((m for m in mp4_members if m.name == mp4_chunk_name), None)
        if member is None:
            raise FileNotFoundError(
                f"'{mp4_chunk_name}' not found in tar. Available: {[m.name for m in mp4_members]}"
            )

        with tar.extractfile(member) as f:
            mp4_bytes = f.read()

    tmp_input = Path(f"/tmp/{Path(member.name).name.replace(' ', '_')}")
    with open(tmp_input, "wb") as out:
        out.write(mp4_bytes)

    cap = cv2.VideoCapture(str(tmp_input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open extracted video: {tmp_input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    if end_frame < start_frame:
        cap.release()
        raise ValueError(
            f"Computed end_frame < start_frame ({end_frame} < {start_frame}). Check start_sec/end_sec."
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(str(Path(
        f"/tmp/{Path(tar_path).stem}__{Path(member.name).stem}__{start_sec:.2f}-{end_sec:.2f}.mp4"
    )).replace(' ', '_'))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cur <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cur += 1

    writer.release()
    cap.release()

    try:
        os.remove(tmp_input)
    except Exception:
        pass

    return str(out_path)


def pick_random_interval_row(
    csv_path: str,
    min_duration: float,
    max_duration: float,
) -> Optional[Dict[str, Any]]:
    """
    Returns a random row (as dict) from the CSV where:
      - duration = menu_end_sec - menu_start_sec is between [min_duration, max_duration]
    Returns None if no rows match.
    """
    df = pd.read_csv(csv_path)

    df["duration"] = df["menu_end_sec"] - df["menu_start_sec"]

    # Filter
    mask = (
        (df["duration"] >= min_duration) &
        (df["duration"] <= max_duration)
    )
    filtered = df[mask].dropna(subset=["menu_start_sec", "menu_end_sec", "duration"])

    if filtered.empty:
        return None

    row = filtered.sample(n=1).iloc[0]
    return row.to_dict()

def write_all_csv(paths: list[Path], csv_path: str = CSV_PATH):
    for tar_path, mp4_chunk_name, mp4_bytes in yield_ext_bytes_from_tars(paths):
        intervals = ask_gemini(tar_path, mp4_chunk_name, mp4_bytes)
        
        if not os.path.exists(csv_path):
            import csv
            # write headers
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

        write_csv_rows(intervals, csv_path, tar_path, mp4_chunk_name)


if __name__ == "__main__":
    write_all_csv(TAR_PATHS)

# if __name__ == "__main__":
#     import csv
#     csv_path = "/home/sky/owl-data-2/menu_intervals_exceeds_time.csv"
#     dst_csv_path = "/home/sky/owl-data-2/menu_intervals_exceeds_time_new.csv"
#     limit = 10
#     paths = [
#         Path(row['tar_name']) for row in csv.DictReader(open(csv_path))
#     ][:limit]
#     write_all_csv(paths, dst_csv_path)


# # /home/sky/owl-data-2/5078554u5th5qyj/2025-10-05 14-32-58_180_234.mp4
# # ^ try above with gemini
# if __name__ == "__main__":
#     path = "/home/sky/owl-data-2/5078554u5th5qyj/2025-10-05 14-32-58_180_234.mp4"
#     intervals = ask_gemini(path, path, open(path, "rb").read())
#     print(intervals)
#     exit()

# if __name__ == "__main__":
#     import subprocess
#     num_samples = 10
#     out_dir = "/home/sky/owl-data-2/menu_samples/"
#     os.makedirs(out_dir, exist_ok=True)
#     rows = [
#         pick_random_interval_row(
#             "/home/sky/owl-data-2/menu_intervals_exceeds_time_new.csv",
#             min_duration=4,
#             max_duration=1000,
#         )
#         for _ in range(num_samples)
#     ]
#     tars_in_rows = set([row['tar_name'] for row in rows])
#     for row in rows:
#         print(row)
#         path = visualize(row['tar_name'], row['mp4_chunk_name'], row['menu_start_sec'], row['menu_end_sec'])
#         print(path)
#         # This line may not work as intended because: 
#         #  - subprocess.Popen runs asynchronously (you won’t see errors if the command fails)
#         #  - path may not be properly shell-escaped
#         #  - You should use a list for command and better error handling
#         #  - Use subprocess.run instead of Popen if you just want to copy and wait for it to finish
        
#         import shutil
#         shutil.copy(str(path), out_dir)