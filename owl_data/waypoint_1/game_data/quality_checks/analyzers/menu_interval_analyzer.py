# analyzers/menu_analyzer.py
from __future__ import annotations
from typing import Dict, Any, List
import json
from owl_data.waypoint_1.game_data.quality_checks.analyzers.base import AnalyzerBase, AnalysisContext
from owl_data.waypoint_1.game_data.quality_checks.analyzers.vlm_client import VLMClient, build_prompt, _sanitize_intervals, get_video_meta_from_bytes


def _coverage_from_intervals(intervals: List[dict], duration_sec: float) -> float:
    total = 0.0
    for it in intervals:
        try:
            s, e = float(it.get("start_sec", 0.0)), float(it.get("end_sec", 0.0))
        except Exception:
            continue
        if e > s:
            total += (e - s)
    if duration_sec and duration_sec > 0:
        return float(min(1.0, total / float(duration_sec)))
    return 0.0


class MenuIntervalAnalyzer(AnalyzerBase):
    name = "menu_intervals"
    version = "1.0.0"
    requires = ("video_mp4",)

    def analyze_chunk(self, ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> Dict[str, Any]:
        mp4_bytes = ctx.get_bytes(chunk_mp4_path)
        meta = get_video_meta_from_bytes(mp4_bytes)
        duration_sec = float(meta.get("duration_sec", 0.0) or 0.0)

        prompt = build_prompt(duration_sec)
        client = VLMClient()
        resp = client.ask(prompt, mp4_bytes)

        clean = _sanitize_intervals(resp.data, duration_sec).get("intervals", [])
        total = 0.0
        
        for it in clean:
            try: total += float(it.get("end_sec", 0.0)) - float(it.get("start_sec", 0.0))
            except Exception: pass

        coverage = _coverage_from_intervals(clean, duration_sec)

        return {
            "flags": [{
                "name": "scene.menu_present",
                "value": bool(clean),
                "score": coverage,
                "analyzer": self.name,
                "analyzer_version": self.version,
                "model": resp.model,
                "extras": {"intervals": json.dumps(clean, separators=(",", ":"))},
            }],
            "menu_total_seconds": total,
            "menu_coverage_pct": coverage,
        }
