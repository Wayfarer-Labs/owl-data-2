import json
import torch
import tempfile
from PIL import Image
import numpy as np
import re
from typing import List
import time
from copy import deepcopy

from .vllm_pipeline_parallel import VLLMPipelineParallel

def create_sliding_windows(n_frames, fps, window_length, kernel, stride):
   """
   Create a list of indices into a tensor that is assumed of shape [n,...]

   Args:
       n_frames: Total number of frames in the tensor
       fps: Frames per second of the video
       window_length: Duration in seconds for each window
       kernel: Number of frames to sample from each window (equidistant including first and last)
       stride: Number of seconds to jump between windows

   Returns:
       List of frame indices for each window, shape [m, kernel] where m is number of windows
   """
   windows = []
   window_frames = int(window_length * fps)
   stride_frames = int(stride * fps)

   start_frame = 0
   while start_frame + window_frames <= n_frames:
      end_frame = start_frame + window_frames - 1

      # Sample kernel frames equidistantly from the window (including first and last)
      if kernel == 1:
         frame_indices = [start_frame]
      else:
         frame_indices = []
         for i in range(kernel):
            frame_idx = start_frame + int(i * (end_frame - start_frame) / (kernel - 1))
            frame_indices.append(frame_idx)

      windows.append(frame_indices)
      start_frame += stride_frames

   return windows

def safe_parse_json(text):
   # 1. First try direct
   try:
      return json.loads(text)
   except:
      pass

   # 2. Look for inline json (i.e. ```json)
   json_match = re.search(r'```json\s*({[^}]*})\s*```', text, re.DOTALL)
   if json_match:
      json_str = json_match.group(1)
      try:
         return json.loads(json_str)
      except:
         pass

   # 2. If json parsing fails, try to look for any dict
   cleaned = text.strip()
   if not cleaned.startswith('{'):
      start_idx = cleaned.find('{')
      if start_idx >= 0:
         cleaned = cleaned[start_idx:]
   if not cleaned.endswith('}'):
      end_idx = cleaned.rfind('}') + 1
      if end_idx > 0:
         cleaned = cleaned[:end_idx]

   return json.loads(cleaned)

class VLMCaptionerParallel:
   def __init__(
      self,
      window_length = 1.0,  # seconds
      kernel = 3,           # number of frames to sample from each window
      stride = 0.5,         # seconds to jump between windows
      port = 8000,
      model_name = "Qwen/Qwen2.5-VL-3B-Instruct",
      num_servers = 8
   ):

      self.pipe = VLLMPipelineParallel(base_port=port, model_name=model_name, num_servers=num_servers)
      self.window_length = window_length
      self.kernel = kernel
      self.stride = stride

   def generate_with_retries(
      self,
      fps_list,
      frames_list,
      max_retries=3
   ):
      """
      Note: past_d_list removed for true parallel processing
      """
      for _ in range(max_retries):
         try:
            text = self.pipe(
               fps_list,
               [self.kernel] * len(fps_list),
               [self.window_length] * len(fps_list),
               frames_list,
            )
            d_list = [safe_parse_json(text) for text in text]
            return d_list
         except KeyboardInterrupt:
            raise KeyboardInterrupt
         except Exception as e:
            print(f"Error: {e}")
            continue

      print("Warning: Message failed to parse from following list:")
      for i,text in enumerate(text):
         try:
            json.loads(text)
         except:
            print(f"{i}: {text}")
      raise ValueError("Failed to parse JSON after retries")

   @torch.no_grad()
   def __call__(self, x_list, fps_list):
      """
      x_list: list of video tensors of shape [n,c,h,w] where n can vary
      fps_list: list of fps values for all those videos

      :returns: list of list of dicts, each can be saved as jsonl. ret_val[i] is list of dicts/captions for x_list[i]

      Note: This parallel version processes all windows independently for maximum throughput
      """
      # x is [n,c,h,w] chunk that is mmap'd
      # x_list has a bunch of x's (varying n's possibly)
      # rgb uint8 [0,255]

      n_frames = [x.shape[0] for x in x_list]
      windows = [create_sliding_windows(
         n_f,
         fps,
         self.window_length,
         self.kernel,
         self.stride
      ) for n_f, fps in zip(n_frames, fps_list)]

      # Collect ALL windows from ALL videos for batch processing
      all_windows_data = []
      all_fps = []
      window_to_video_mapping = []  # Track which video each window belongs to

      for video_idx, (video_windows, video_tensor, fps) in enumerate(zip(windows, x_list, fps_list)):
         for window_indices in video_windows:
            frames = video_tensor[window_indices]
            all_windows_data.append(frames)
            all_fps.append(fps)
            window_to_video_mapping.append((video_idx, window_indices))

      print(f"Processing {len(all_windows_data)} total windows across {len(x_list)} videos in parallel...")

      # Process all windows in batches for maximum parallelization
      batch_size = 64  # Adjust based on your memory and server count
      all_captions = []

      start_time = time.time()
      for batch_start in range(0, len(all_windows_data), batch_size):
         batch_end = min(batch_start + batch_size, len(all_windows_data))
         batch_frames = all_windows_data[batch_start:batch_end]
         batch_fps = all_fps[batch_start:batch_end]

         batch_results = self.generate_with_retries(
            batch_fps,
            batch_frames
         )

         all_captions.extend(batch_results)

      end_time = time.time()
      print(f"Time taken: {end_time - start_time:.2f} seconds")
      print(f"Throughput: {len(all_windows_data) / (end_time - start_time):.2f} windows/second")

      # Organize results back by video
      captions = [[] for _ in range(len(x_list))]
      for caption_dict, (video_idx, frame_indices) in zip(all_captions, window_to_video_mapping):
         caption_dict["frame_indices"] = frame_indices
         captions[video_idx].append(caption_dict)

      return captions



if __name__ == "__main__":
   # Test window generation
   test_tensor = "/mnt/data/waypoint_1/data_pt/MKIF_360P/VSNETqE_rIk_000024/video/splits/00000000_rgb.pt"
   test_tensor_2 = "/mnt/data/waypoint_1/data_pt/MKIF_360P/VSNETqE_rIk_000024/video/splits/00000001_rgb.pt"

   test_tensor = torch.load(test_tensor, map_location='cpu', mmap=True)
   test_tensor_2 = torch.load(test_tensor_2, map_location='cpu', mmap=True)

   captioner = VLMCaptionerParallel(
      window_length = 5,
      kernel = 5,
      stride = 2.5,
      port = 8000,
      num_servers = 8  # Use all 8 GPUs
   )

   print("Testing parallel captioning with 2 video chunks...")
   captions = captioner([test_tensor, test_tensor_2], [60, 60])

   print(f"\nResults for first video ({len(captions[0])} windows):")
   for i, line in enumerate(captions[0][:3]):  # Show first 3
      print(f"Window {i}: {line}")

   print(f"\nResults for second video ({len(captions[1])} windows):")
   for i, line in enumerate(captions[1][:3]):  # Show first 3
      print(f"Window {i}: {line}")