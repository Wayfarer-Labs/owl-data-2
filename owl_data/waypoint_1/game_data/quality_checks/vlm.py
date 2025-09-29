import os
import logging
import base64
import io
import time
import requests
import numpy as np
from PIL import Image
import dotenv
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

dotenv.load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
PROMPT = "Is this image from a video game showing active gameplay or a menu/UI screen? Answer with only the single word 'gameplay' or 'menu'."


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def _encode_frame(frame_chw: np.ndarray) -> str:
    """Converts a CHW NumPy array to a base64 encoded JPEG string."""
    frame_hwc = frame_chw.transpose(1, 2, 0)
    image = Image.fromarray(frame_hwc.astype('uint8'), 'RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@log_time
def _classify_single_frame(encoded_frame: str, max_retries: int = 3) -> str:
    """Sends a single frame to the Gemini API and returns the classification."""
    payload = {
        "contents": [{"parts": [{"text": PROMPT}, {"inlineData": {"mimeType": "image/jpeg", "data": encoded_frame}}]}]
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text'].strip().lower()
            return 'menu' if 'menu' in text else 'gameplay'
        except requests.RequestException as e:
            logging.warning(f"API request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "error"
        except (KeyError, IndexError):
            return "error"
    return "error"
