# analyzers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Dict, Any

@dataclass(frozen=True)
class AnalysisContext:
    tar_id: str
    git_commit: str
    run_id: str
    get_bytes: Callable[[str], bytes]
    list_files: Callable[[], Iterable[str]]

class Analyzer(Protocol):
    name: str
    requires: Iterable[str]
    def analyze_chunk(self, ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]: ...

class AnalyzerBase:
    name = "analyzer"
    requires: Iterable[str] = ()

