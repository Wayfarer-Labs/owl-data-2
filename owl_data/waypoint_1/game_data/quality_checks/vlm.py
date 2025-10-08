import subprocess
import tempfile
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Video processing parameters
VIDEO_PATH = "sample/vid.mp4"
t_start = 0  # Start time in seconds
t_end = 10   # End time in seconds

# Downsampling parameters
TARGET_HEIGHT = 240
FPS = 10
CRF = 28  # Compression quality (higher = more compression, 18-28 is reasonable)


def analyze_video_with_gemini(video_bytes: bytes, prompt="Describe what is happening in this video."):
    """
    Send a video to Gemini for analysis.

    :param video_path: Path to video file
    :param prompt: Prompt for Gemini
    :return: Response text from Gemini
    """
    client = genai.Client()
    model = "gemini-2.5-flash-lite"

    # Create parts for the request
    parts = [
        types.Part(text=prompt),
        types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'))
    ]

    # Count tokens
    total_tokens = client.models.count_tokens(
        model=model,
        contents=types.Content(parts=parts)
    )
    print(f"Total tokens: {total_tokens}")

    # Generate response
    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts)
    )

    output = response.candidates[0].content.parts[0].text
    return output

if __name__ == "__main__":
    # Create temporary file for downsampled video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        print(f"Creating downsampled clip from {t_start}s to {t_end}s...")
        create_downsampled_clip(
            VIDEO_PATH,
            temp_path,
            t_start,
            t_end,
            height=TARGET_HEIGHT,
            fps=FPS,
            crf=CRF
        )

        print(f"Temporary video created at: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path) / 1024:.2f} KB")

        print("\nSending to Gemini...")
        response = analyze_video_with_gemini(
            temp_path,
            prompt="Describe what is happening in this video."
        )

        print("\n=== Gemini Response ===")
        print(response)

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"\nCleaned up temporary file: {temp_path}")