# analyzers/vlm_query.py
from __future__ import annotations
import json, hashlib
from typing import Callable, Dict, List, Tuple, Any

BASE_PROMPT = (
    "You are a VLM analyzing <=60s gameplay clips.\n"
    "- Use temporal reasoning across the whole minute.\n"
    "- Output strictly valid JSON only.\n"
    "- All timestamps are seconds from 0.0.\n"
)

class VLMQuery:
    """
    Per-chunk accumulator. Every CallsVLM analyzer adds a bullet under a unique key.
    send_queries() makes exactly one VLM call and returns a shared JSON object:

      {
        "<analyzer_name>": { ... that analyzer's section ... },
        "<analyzer_name_2>": { ... },
        ...
      }
    """
    def __init__(self, model_id: str, get_video_bytes: Callable[[], bytes]) -> None:
        self.model_id = model_id
        self._get_video_bytes = get_video_bytes
        self._tasks: List[Tuple[str, str]] = []  # (key, bullet)

    def queue(self, key: str, bullet: str) -> None:
        self._tasks.append((key, bullet.strip()))

    def _build_prompt(self) -> Tuple[str, str]:
        bullets = "\n".join(f"- ({k}) {b}" for k, b in self._tasks)
        prompt = (
            f"{BASE_PROMPT}\n"
            "TASKS:\n"
            f"{bullets}\n\n"
            "Return a single JSON object where each key is the task ID in parentheses above, e.g.:\n"
            "{\n"
            '  "menu_intervals": { /* your JSON for that task */ },\n'
            '  "another_task": { /* ... */ }\n'
            "}\n"
        )
        digest = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return prompt, digest

    def _call_vlm_once(self, prompt_text: str, video_bytes: bytes) -> Dict[str, Any]:
        # TODO: replace stub with Gemini 2.5 Flash Video call + json.loads(response)
        # Example shape: return { "<task_key>": {...}, "<task_key_2>": {...} }
        return {
            "menu_intervals": {
                "intervals": [{"start_sec": 1.7, "end_sec": 5.9, "confidence": 0.91}],
                "coverage_pct": 0.07
            }
        }

    def send_queries(self) -> Dict[str, Any]:
        if not self._tasks:
            return {}
        prompt, digest = self._build_prompt()
        video = self._get_video_bytes()
        shared = self._call_vlm_once(prompt, video)
        # attach provenance under a reserved key so analyzers can use it
        shared["_provenance"] = {"model": self.model_id, "inputs_digest": digest}
        return shared

