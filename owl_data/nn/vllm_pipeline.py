DEFAULT_PROMPT_TEMPLATE = """
You are analyzing a short video game segment. You will be told how many frames, how much time, FPS later.
Your task is to produce FIVE SEPARATE CAPTIONS that are INDEPENDENT of one another. 
Each caption must describe a different aspect of the scene. 
If there is nothing relevant for a category, write "N/A".

1. **STYLE (visual aesthetic and camera perspective)**  
   - Example: "Borderlands cel-shaded aesthetic, first-person perspective"  
   - This controls the *visual look* of the game and whether the view is first-person, third-person, or top-down.  
   - Only describe the visual style and perspective. Do not mention HUD, player actions, or setting here.

2. **HUD ELEMENTS (on-screen interface and overlays)**  
   - Example: "An ammo counter is on the bottom right, a minimap on the top left, and a health bar at the top center"  
   - Fully describe visible HUD/UI elements and their positions on screen.  
   - Only mention HUD details. Do not describe the world, style, or player actions.

3. **PLAYER ACTION (what the player character is doing in this segment)**  
   - Example: "The player is reloading their weapon" or "The player is running up stairs."  
   - Only describe the players action.  
   - If no clear action is happening, write "N/A".  
   - Do not describe enemies, weather, or the environment.

4. **WORLD EVENT (what happens in the world that is independent of the player)**  
   - Example: "A zombie is moving toward the player" or "Lightning strikes in the distance."  
   - Only describe events caused by the environment or NPCs, not the player.  
   - If nothing happens, write "N/A".

5. **SETTING (where this is happening)**  
   - Example: "We are in a ruined city with collapsed skyscrapers and ash in the air."  
   - Only describe the overall environment/location.  
   - Do not mention HUD, style, player actions, or world events here.

Keep these all short. Just a sentence or two for each.
You may be given your past output as context as well.
If your past output is still relevant and matches what you currently see EXACTLY, just output it precisely without changing anything.
If anything has changed, please update the section, and if something has become irrelevant, replace with N/A.
If something was previously N/A, but you now see it/believe it is relevant, please add it.
---

### Required Output Format

Output exactly in the following JSON-like structure:

{{
  "style": "<caption>",
  "hud": "<caption>",
  "player_action": "<caption or N/A>",
  "world_event": "<caption or N/A>",
  "setting": "<caption>"
}}

---

### Key Rules
- Each category must be **independent** (no overlap).
- Keep descriptions concise and factual.
- Use "N/A" when appropriate.
- Never combine categories.

### Video Info:
FPS: {}
Overall Number of Frames In Window: {}
Time Elapsed Across Window: {}
### Past Output:
{}
"""

import asyncio, time, statistics as stats
from openai import AsyncOpenAI
import os
import io, base64
import numpy as np
from PIL import Image
import torch

def to_data_url_png(chw_uint8: torch.Tensor) -> str:
    # chw_uint8: [3,H,W], dtype=uint8 in [0,255]
    img = Image.fromarray(chw_uint8.permute(1,2,0).contiguous().cpu().numpy())
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def frames_uint8_to_data_urls(frames: torch.Tensor, fmt: str = "PNG"):
    """
    frames: [T,3,H,W] uint8 in [0,255]
    returns: list[str] of data URLs (one per frame)
    """
    assert frames.dtype == torch.uint8 and frames.ndim == 4 and frames.shape[1] == 3
    frames = frames.detach().cpu()
    urls = []
    for i in range(frames.shape[0]):
        chw = frames[i]                      # [3,H,W]
        hwc = chw.permute(1, 2, 0).contiguous().numpy()  # [H,W,3]
        img = Image.fromarray(hwc)           # RGB by default for 3 channels
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/{fmt.lower()};base64,{b64}")
    return urls

def create_messsages(
    video_fps_list,
    video_num_frames_list,
    video_time_elapsed_list,
    frames,
    past_output_list = None,
):
    """
    frames is [b,n,c,h,w] and b = len(video_fps_list)
    """
    if past_output_list is None:
        past_output_list = ["N/A"] * len(video_fps_list)
    messages_list = []
    for fps, num_frames, time_elapsed, past_output in zip(video_fps_list, video_num_frames_list, video_time_elapsed_list, past_output_list):
        message = DEFAULT_PROMPT_TEMPLATE.format(fps, num_frames, time_elapsed, past_output)
        messages_list.append(message)

    messages_list = [
        [{
            "role" : "user",
            "content" : [
                {
                    "type" : "text",
                    "text" : message
                }
            ] + [{"type": "image_url", "image_url": {"url": to_data_url_png(video[i])}} for i in range(video.shape[0])]
        }]
    for (video, message, fps) in zip(frames, messages_list, video_fps_list)]

    return messages_list


class VLLMPipeline:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", port=8000):
        base_url = f"http://localhost:{port}/v1"
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self.model_name = model_name

    def __call__(
        self,
        video_fps_list,
        video_num_frames_list,
        video_time_elapsed_list,
        frames,
        past_output_list = None,
    ):
        messages_list = create_messsages(
            video_fps_list,
            video_num_frames_list,
            video_time_elapsed_list,
            frames,
            past_output_list
        )
        async def one_request(messages):
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=128
            )
            return resp.choices[0].message.content

        async def run_all():
            return await asyncio.gather(*(one_request(messages) for messages in messages_list))

        return asyncio.run(run_all())