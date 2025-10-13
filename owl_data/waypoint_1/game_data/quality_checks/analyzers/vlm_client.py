from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


STUB_MODEL_ID = "vlm-stub:gemini-2.5-flash-lite"


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


def _sanitize_intervals(result: dict, duration_sec: float, min_len: float = 0.05, tol_merge: float = 0.10) -> dict:
    intervals = (result or {}).get("intervals", [])
    if not isinstance(intervals, list):
        return {"intervals": []}

    cleaned = []
    for it in intervals:
        try:
            s = float(it.get("start_sec", 0.0))
            e = float(it.get("end_sec", 0.0))
        except Exception:
            continue
        s = max(0.0, min(s, duration_sec))
        e = max(0.0, min(e, duration_sec))
        if e <= s:
            continue
        if (e - s) < min_len:
            continue
        cleaned.append({"start_sec": s, "end_sec": e, "confidence": float(it.get("confidence", 0.0))})

    if not cleaned:
        return {"intervals": []}

    cleaned.sort(key=lambda x: x["start_sec"])

    merged = []
    for it in cleaned:
        if not merged:
            merged.append(it)
            continue
        last = merged[-1]
        if it["start_sec"] <= last["end_sec"] + tol_merge:
            last["end_sec"] = max(last["end_sec"], it["end_sec"])
            last["confidence"] = max(last.get("confidence", 0.0), it.get("confidence", 0.0))
        else:
            merged.append(it)

    return {"intervals": merged}


def get_video_meta_from_bytes(mp4_bytes: bytes) -> dict:
    import tempfile, cv2, os
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
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    duration_sec = (frames / fps) if (fps > 0 and frames > 0) else 0.0
    return {"fps": fps, "frames": frames, "duration_sec": duration_sec}


@dataclass
class VLMResponse:
    data: Dict[str, Any]
    model: str


class VLMClient:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or STUB_MODEL_ID

    def _call_vlm_once(self, prompt_text: str, video_bytes: bytes) -> Dict[str, Any]:
        # Stub: mirror the old shared query shape minimally for menus
        # In future, integrate a real provider.
        return {
            "intervals": [{"start_sec": 1.7, "end_sec": 5.9, "confidence": 0.91}]
        }

    def ask(self, prompt_text: str, video_bytes: bytes) -> VLMResponse:
        raw = self._call_vlm_once(prompt_text, video_bytes)
        return VLMResponse(data=raw, model=self.model_id)


