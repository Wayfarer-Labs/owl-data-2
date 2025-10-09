# analyzers/menu_analyzer.py
from __future__ import annotations
from typing import Dict, Any, List
import json
from analyzers.base import CallsVLM, AnalysisContext
from analyzers.vlm_query import VLMQuery

def _pack(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))

class MenuAnalyzer(CallsVLM):
    name = "menu_intervals"
    version = "1.0.0"
    requires = ("video_mp4",)

    def queue_prompt(self, vlm: VLMQuery, ctx: AnalysisContext, chunk_mp4_path: str, chunk_idx: int) -> None:
        vlm.queue(
            key=self.name,
            bullet=(
                "Identify time intervals (seconds) where large non-diegetic MENU overlays are visible "
                "(pause/settings/inventory/shop/map). Respond as: "
                '{"intervals":[{"start_sec":<float>,"end_sec":<float>,"confidence":<0..1>}],'
                '"coverage_pct":<0..1>}'
            ),
        )

    def receive_response(self, shared_json: dict) -> Dict[str, Any]:
        data = shared_json.get(self.name, {}) or {}
        intervals = data.get("intervals", []) or []
        clean: List[dict] = []
        total = 0.0
        for it in intervals:
            try:
                s, e = float(it.get("start_sec", 0.0)), float(it.get("end_sec", 0.0))
                if e > s and (e - s) >= 1.0:
                    total += (e - s)
                    clean.append({"start_sec": s, "end_sec": e, "confidence": float(it.get("confidence", 0.0))})
            except Exception:
                pass
        coverage = float(data.get("coverage_pct", 0.0)) or (min(1.0, total / 60.0) if total else 0.0)

        prov = shared_json.get("_provenance", {})
        model = prov.get("model", "gemini-2.5-flash-video")
        digest = prov.get("inputs_digest")

        return {
            "flags": [{
                "name": "scene.menu_present",
                "value": bool(clean),
                "score": coverage,
                "analyzer": self.name,
                "analyzer_version": self.version,
                "model": model,
                "prompt_id": "multitask:menus_v1",
                "inputs_digest": digest,
                "extras": {"intervals": _pack(clean)},
            }],
            "menu_total_seconds": total,
            "menu_coverage_pct": coverage,
        }
