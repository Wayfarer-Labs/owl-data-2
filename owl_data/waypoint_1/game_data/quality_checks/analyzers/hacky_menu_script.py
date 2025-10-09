import os, dotenv, pathlib, typing, json
from pathlib import Path
dotenv.load_dotenv()


# load all tar paths
TAR_PATHS = list(pathlib.Path("/mnt/data/datasets/downsampled_tars/").glob("*.tar"))
EMPTY_RESPONSE = {"intervals":[],"coverage_pct":0}
# define prompt
_FIND_MENUS_PROMPT = (
    "Identify time intervals (seconds) where large non-diegetic MENU overlays are visible "
    "(pause/settings/inventory/shop/map). Respond as: "
    '{"intervals":[{"start_sec":<float>,"end_sec":<float>,"confidence":<0..1>}],'
    '"coverage_pct":<0..1>}'
    "If you did not find any menus, respond with: "
    '{"intervals":[],"coverage_pct":0}'
    'Only JSON! And make sure that the intervals do not overlap, and are merged when necessary.'
)

# define csv path
CSV_PATH = "menu_intervals.csv"
# csv columns
CSV_COLUMNS = [
    "tar_name",
    "mp4_chunk_name",
    "menu_start_sec",
    "menu_end_sec",
    "menu_confidence",
    "menu_coverage_pct",
]

# -- 
print (f"Found {len(TAR_PATHS)} tar paths")
print (f"GEMINI_API_KEY Found: {os.getenv('GEMINI_API_KEY') is not None}")


def yield_mp4_bytes_from_tars(tar_paths: list[pathlib.Path]) -> typing.Generator[tuple[str, str, bytes], None, None]:
    """
    Yields a tuple of (tar_path, mp4_chunk_name, mp4_bytes) for each mp4 in tars
    """
    import tarfile

    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".mp4"):
                    continue
                with tar.extractfile(member) as f:
                    yield (str(tar_path), member.name, f.read())

import google.genai as genai

CLIENT = genai.Client()
MODEL = 'gemini-2.5-flash-lite'


# async function that takes bytes from an mp4 and sends a query to gemini and parses the response into a list of csv rows per interval
def ask_gemini(tar_path: str, mp4_chunk_name: str, mp4_bytes: bytes) -> dict:
    global CLIENT, MODEL
    import google.genai.types as types
    
    contents = types.Content(parts=[
        types.Part(text=_FIND_MENUS_PROMPT),
        types.Part(inline_data=types.Blob(data=mp4_bytes, mime_type='video/mp4'))
    ])
    total_tokens = CLIENT.models.count_tokens(model=MODEL, contents=contents)

    print(f"Sending request to Gemini with {total_tokens} tokens")
    response: types.GenerateContentResponse = CLIENT.models.generate_content(
        model=MODEL,
        contents=contents,
    )

    output = response.candidates[0].content.parts[0].text
    try: 
        parsed = json.loads(output)
        parsed['error'] = None
        print(f"Parsed {len(parsed['intervals'])} intervals")
        return parsed
    except Exception as e:
        print (f"Error parsing output: {e} - output: {output} - for {tar_path} - {mp4_chunk_name}")
        err = EMPTY_RESPONSE
        err['error'] = str(e)
        return err

# function that writes csv rows to filepath
def write_csv_rows(intervals: dict, filepath: str, tar_name: str, mp4_chunk_name: str):
    import pandas as pd

    rows = [
        {
            "tar_name": tar_name,
            "mp4_chunk_name": mp4_chunk_name,
            "menu_start_sec":               (interval or {}).get("start_sec", None),
            "menu_end_sec":                 (interval or {}).get("end_sec", None),
            "menu_confidence":              (interval or {}).get("confidence", None),
            "menu_coverage_pct":            (interval or {}).get("coverage_pct", None),
            "error":                        (interval or {}).get("error", None),
        }
        for interval in intervals.get('intervals', [{}])
    ]

    df = pd.DataFrame(rows)
    print(f'Rows: \n {rows}')
    # append to csv instead of replacing it
    df.to_csv(filepath, mode='a', header=False, index=False)


def _merge_intervals(intervals: dict) -> dict:
    # sort by start_sec
    def _is_within(num1, num2, delta = 0.1):
        return abs(num1 - num2) <= delta

    _intervals = intervals.get('intervals', [])
    if not _intervals:
        return intervals

    _intervals.sort(key=lambda x: x.get('start_sec', 0))
    merged_intervals = []
    for interval in _intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last_interval = merged_intervals[-1]
            if _is_within(interval.get('start_sec', 0), last_interval.get('end_sec', 0)):
                merged_intervals[-1] = {
                    'start_sec': last_interval.get('start_sec'),
                    'end_sec': max(last_interval.get('end_sec', 0), interval.get('end_sec', 0))
                }
            else:
                merged_intervals.append(interval)

    intervals['intervals'] = merged_intervals
    print(f'{len(merged_intervals)} merged intervals - {len(_intervals)} original intervals')
    return intervals


for tar_path, mp4_chunk_name, mp4_bytes in yield_mp4_bytes_from_tars(TAR_PATHS):
    intervals = ask_gemini(tar_path, mp4_chunk_name, mp4_bytes)
    intervals = _merge_intervals(intervals)
    if not os.path.exists(CSV_PATH):
        import csv
        # write headers
        with open(CSV_PATH, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    write_csv_rows(intervals, CSV_PATH, tar_path, mp4_chunk_name)


