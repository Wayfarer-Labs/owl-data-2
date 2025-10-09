# analyzers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Dict, Any

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
    uses_vlm: bool
    requires: Iterable[str]
    def analyze_chunk(self, ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]: ...

class AnalyzerBase:
    name = "analyzer"
    uses_vlm = False
    requires: Iterable[str] = ()

class CallsVLM(AnalyzerBase):
    """Mix-in for analyzers that participate in a single shared VLM call per chunk."""
    uses_vlm = True
    
    # Add one instruction to the shared prompt. Use your analyzer's `name` as the response key.
    def queue_prompt(self, vlm: "VLMQuery", ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> None:
        raise NotImplementedError

    # Read the shared JSON response and return your own {'flags': [...], ...}
    def receive_response(self, shared_json: dict) -> Dict[str, Any]:
        raise NotImplementedError

