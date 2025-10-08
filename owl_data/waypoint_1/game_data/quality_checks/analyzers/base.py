# analyzers/base.py
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Protocol, Callable

@dataclass(frozen=True)
class AnalysisContext:
    tar_id: str
    tar_s3_key: str
    git_commit: str
    run_id: str
    get_bytes: Callable[[str], bytes]
    list_files: Callable[[], Iterable[str]]


class Analyzer(Protocol):
    name: str
    version: str
    requires: Iterable[str]  # e.g. ["video_mp4", "controls_csv", "session_json", "ffprobe_json"]

    def analyze_chunk(self, ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]:
        ...

