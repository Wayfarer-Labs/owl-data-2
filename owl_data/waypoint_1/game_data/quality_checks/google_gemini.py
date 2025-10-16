import os
import vertexai
import base64, io
import google.auth
import google.auth.transport.requests
from openai import OpenAI
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from google.oauth2 import service_account

load_dotenv()

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

# credentials, project_id = google.auth.default(
#     scopes=['https://www.googleapis.com/auth/aiplatform']
# )

# # Initialize Vertex AI with the project and the specific credentials
# vertexai.init(project="openworld-main", credentials=credentials)


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


class GoogleAuthOpenAI:
    def __init__(self, project_id = 'openworld-main', location="us-central1"):
        self.project_id = project_id
        self.location = location
        self.token_expiry = datetime.now() + timedelta(hours=1)

        # Define the necessary scope to access the Vertex AI API
        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        
        # Get the service account key path from the environment
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not key_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

        # --- CORRECTED AUTHENTICATION ---
        # Create credentials that generate ACCESS TOKENS using the specified scopes.
        # This is the type of credential the API is asking for.
        self.credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=scopes
        )
        
        self.client = None
        self._refresh_client()
    
    def _refresh_client(self):
        """Refresh the ID token and recreate the OpenAI client."""
        # Use a request object to refresh the ID token
        request = Request()
        self.credentials.refresh(request)

        
        self.client = OpenAI(
            # --- CORRECTED LINE ---
            # Added /endpoints/openapi to the end of the URL
            base_url=f"https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi",
            # Use the refreshed ID token as the API key
            api_key=self.credentials.token
        )


    # --- ADDED A SIMPLE CHAT METHOD FOR THE EXAMPLE ---
    def chat_completion(self, model: str, messages: list):
        """Handle text-based chat completions."""
        # Refresh is handled by the OpenAI client library implicitly now
        # by creating a new client if needed, but we can do it explicitly
        self._refresh_client()
        
        return self.client.chat.completions.create(
            model=model,
            messages=messages
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_is_retryable_exception),
    )
    def responses_with_video(self, model: str, prompt: str, mp4_bytes: bytes):
        """
        Send prompt + video to Vertex's OpenAI-compatible API.
        This version correctly uses the chat.completions endpoint for multimodal input.
        """
        # Refresh client/token if necessary
        if datetime.now() >= self.token_expiry:
            self._refresh_client()

        # Encode the video bytes into a base64 string
        b64_video = base64.b64encode(mp4_bytes).decode("ascii")

        # Use the standard chat.completions endpoint, which supports multimodal content
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Part 1: The text prompt
                        {"type": "text", "text": prompt},
                        # Part 2: The video data, formatted correctly for this endpoint
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:video/mp4;base64,{b64_video}"
                            }
                        }
                    ],
                }
            ],
            # This keyword is supported by the chat.completions.create method
            response_format={"type": "json_object"},
        )

    def ask_menus(self, mp4_bytes: bytes) -> dict:
        import copy, json
        def _build_prompt(duration_sec: float) -> str:
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
        
        EMPTY_RESPONSE = {"intervals":[]}
        
        def _parse_response(
            raw: str,
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
            result = json.loads(raw)
            intervals = (result or {}).get("intervals", [])
            if not isinstance(intervals, list):
                return copy.deepcopy(EMPTY_RESPONSE)

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
                return copy.deepcopy(EMPTY_RESPONSE)

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


        duration_sec = get_video_meta_from_bytes(mp4_bytes)["duration_sec"]
        prompt = _build_prompt(duration_sec)

        try:
            # Prefer file upload for larger clips; set use_upload=False for small clips
            resp = self.responses_with_video(
                model="google/gemini-2.5-flash-lite",
                prompt=prompt,
                mp4_bytes=mp4_bytes,
            )
            # Responses API: text is typically in `output_text`; fall back to first text part if needed
            raw = resp.choices[0].message.content
            parsed = _parse_response(raw or "{}" , duration_sec)
            return parsed | {"error": None}
        except Exception as e:
            import traceback as tb
            err = copy.deepcopy(EMPTY_RESPONSE)
            err["error"] = f"{type(e).__name__}: {e}\n{''.join(tb.format_exception(e))}"
            return err    
    
    def ask_dark_screen(self, mp4_bytes: bytes) -> dict:
        prompt = """
            You are given a single video clip.
            Your task is to identify whether the clip is of a video game,
            or if it's just a dark screen, which would indicate that the video encountered a
            crash or a recording malfunction.

            Output JSON ONLY, EXACTLY this schema:
            {"is_darkness":<bool>}
            If you found footage of a dark screen, output: {"is_darkness":True}
            If not, output: {"is_darkness":False}
        """
        import json, copy
        
        EMPTY_RESPONSE = {"is_darkness": False, "error": None, "reason": None}

        def _parse_response(response: str) -> dict:
            return json.loads(response)

        try:
            resp = self.responses_with_video(
                model="google/gemini-2.5-flash-lite",
                prompt=prompt,
                mp4_bytes=mp4_bytes,
            )
            # Responses API: text is typically in `output_text`; fall back to first text part if needed
            raw = resp.choices[0].message.content
            parsed = _parse_response(raw or "{}" )
            return parsed | {"error": None, "reason": None} # We set reason as none cause it was taking too long
        except Exception as e:
            import traceback as tb
            err = copy.deepcopy(EMPTY_RESPONSE)
            err["error"] = f"{type(e).__name__}: {e}\n{''.join(tb.format_exception(e))}"
            return err


if __name__ == "__main__":
    # Usage
    client = GoogleAuthOpenAI()

    # This will work for hours, auto-refreshing tokens
    response = client.chat_completion(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {"role": "user", "content": "Explain how AI works"}
        ]
    )

    print(response.choices[0].message.content)  
