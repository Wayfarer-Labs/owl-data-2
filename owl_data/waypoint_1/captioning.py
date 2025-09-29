import argparse, os, re, sys, threading, queue, uuid, torch, asyncio, json, hashlib
from PIL import Image
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
import tqdm

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

### Required Output Format

Output exactly in the following JSON-like structure:

{{
  "style": "<caption>",
  "hud": "<caption>",
  "player_action": "<caption or N/A>",
  "world_event": "<caption or N/A>",
  "setting": "<caption>"
}}

### Key Rules
- Each category must be **independent** (no overlap).
- Keep descriptions concise and factual.
- Use "N/A" when appropriate.
- Never combine categories.

### Video Info:
FPS: {}
Overall Number of Frames In Window: {}
Time Elapsed Across Window: {}
"""

def find_files(root, node_rank=0, num_nodes=1):
    rx = re.compile(r"^\d+_rgb\.pt$")
    for dp, _, fs in os.walk(root):
        for f in fs:
            if rx.match(f):
                file_path = os.path.join(dp, f)
                # Hash the relative path for consistent node splitting
                rel_path = os.path.relpath(file_path, root)
                file_hash = hashlib.md5(rel_path.encode('utf-8')).hexdigest()
                # Convert first 8 chars of hex to int and check if it belongs to this node
                hash_int = int(file_hash[:8], 16)
                if hash_int % num_nodes == node_rank:
                    yield file_path

def frames_to_images_and_idxs(t, start_idx, kernel, window_length_frames):
    """Extract kernel number of equidistant frames from a window starting at start_idx"""
    t = t.permute(0, 2, 3, 1).contiguous()
    n = t.shape[0]
    end_idx = min(start_idx + window_length_frames, n)
    window_frames = t[start_idx:end_idx]

    window_size = end_idx - start_idx
    if window_size <= kernel:
        # If window is smaller than kernel, take all frames
        frame_indices = list(range(window_size))
    else:
        # Take kernel equidistant frames from the window
        frame_indices = [int(i * (window_size - 1) / (kernel - 1)) for i in range(kernel)]

    imgs = [Image.fromarray(window_frames[i].cpu().numpy()) for i in frame_indices]
    # Convert local indices to global indices
    global_indices = [start_idx + i for i in frame_indices]
    return imgs, global_indices

def parse_json_response(response_text):
    """Helper to parse JSON from model response"""
    try:
        # First try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Look for JSON block in markdown
        json_pattern = r'```json\s*({.*?})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for any JSON-like structure
        json_pattern = r'{[^}]*"style"[^}]*}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None

def load_vid_info(rgb_path):
    """Load vid_info.json from the directory above the rgb file"""
    parent_dir = os.path.dirname(os.path.dirname(rgb_path))
    vid_info_path = os.path.join(parent_dir, 'vid_info.json')

    if os.path.exists(vid_info_path):
        with open(vid_info_path, 'r') as f:
            return json.load(f)
    return None

p = argparse.ArgumentParser()
p.add_argument("--root", default="/mnt/data/waypoint_1/data_pt/MKIF_360P")
p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
p.add_argument("--kernel", type=int, default=5, help="Number of equidistant frames to extract from each window")
p.add_argument("--stride", type=float, default=2.5, help="Stride between windows in seconds")
p.add_argument("--window_length", type=float, default=5.0, help="Length of each window in seconds")
p.add_argument("--max_tokens", type=int, default=512)
p.add_argument("--prefetch", type=int, default=256)
p.add_argument("--node_rank", type=int, default=0)
p.add_argument("--num_nodes", type=int, default=1)
args = p.parse_args()

files = find_files(args.root, args.node_rank, args.num_nodes)
ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
if ngpu < 1: print("No CUDA GPUs visible.", file=sys.stderr); sys.exit(1)

# Producer: load tensors & prep images/idxs in background
q = queue.Queue(maxsize=args.prefetch)
def producer():
    for path in files:
        # Skip if captions file already exists
        base_name = os.path.splitext(os.path.basename(path))[0]
        captions_path = os.path.join(os.path.dirname(path), f"{base_name.replace('_rgb', '')}_captions.jsonl")
        if os.path.exists(captions_path):
            continue

        # Load video info to get fps
        vid_info = load_vid_info(path)
        fps = vid_info.get('fps', 30) if vid_info else 30

        t = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        n_frames = t.shape[0]

        # Convert seconds to frames
        window_length_frames = int(args.window_length * fps)
        stride_frames = int(args.stride * fps)

        # Generate sliding windows
        windows = []
        for start_idx in range(0, n_frames - window_length_frames + 1, stride_frames):
            if start_idx + window_length_frames > n_frames:
                break
            windows.append(start_idx)

        # If no complete windows, try one partial window
        if not windows and n_frames > 0:
            windows.append(0)

        for window_idx, start_idx in enumerate(windows):
            actual_window_length_frames = min(window_length_frames, n_frames - start_idx)
            imgs, frame_indices = frames_to_images_and_idxs(t, start_idx, args.kernel, actual_window_length_frames)

            # Calculate time info
            time_elapsed = actual_window_length_frames / fps

            # Format prompt with fps and timing info
            vision_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
            placeholders = vision_placeholder

            formatted_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                fps, len(imgs), f"{time_elapsed:.2f} seconds"
            )

            prompt_text = (
                "<|im_start|>user\n"
                f"{placeholders}\n{formatted_prompt}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            q.put((path, captions_path, window_idx, frame_indices, time_elapsed, fps, {"prompt": prompt_text, "multi_modal_data": {"video": imgs}}))

    q.put((None, None, None, None, None, None, None))
threading.Thread(target=producer, daemon=True).start()

async def main():
    eng = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        quantization="fp8",
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256",
        data_parallel_size=ngpu,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"video":1},
        dtype="auto",
        gpu_memory_utilization=0.8,
    ))
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    from tqdm import tqdm

    pbar = tqdm(desc="Processed windows", unit="window", dynamic_ncols=True)
    processed_count = 0

    while True:
        result = q.get()
        if len(result) == 7:
            path, captions_path, window_idx, frame_indices, time_elapsed, fps, req = result
        else:
            break

        if path is None:
            break

        rid = str(uuid.uuid4())
        last = None
        async for out in eng.generate(
            {"prompt": req["prompt"], "multi_modal_data": req["multi_modal_data"]},
            sp,
            rid,
        ):
            last = out

        response_text = (last.outputs[0].text or "").strip()

        # Parse JSON response
        json_data = parse_json_response(response_text)
        if json_data:
            # Add frame indices to the JSON
            json_data["frame_indices"] = frame_indices
            json_data["window_idx"] = window_idx
            json_data["fps"] = fps
            json_data["time_elapsed"] = time_elapsed

            # Append to captions file
            os.makedirs(os.path.dirname(captions_path), exist_ok=True)
            with open(captions_path, 'a') as f:
                f.write(json.dumps(json_data) + '\n')

            processed_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Last: {os.path.basename(path)}, window {window_idx}")
            # Optionally, you can print for debugging:
            # print(f"Processed {path}, window {window_idx}, frames {frame_indices}")
        else:
            print(f"Failed to parse JSON from response for {path}, window {window_idx}: {response_text}")

    pbar.close()
	
asyncio.run(main())