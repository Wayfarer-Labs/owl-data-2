import json
import torch
import tempfile
from PIL import Image
import numpy as np
import re
from typing import List
import time
from copy import deepcopy

from .vllm_pipeline import VLLMPipeline

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

class VLMCaptioner:
   def __init__(
      self,
      window_length = 1.0,  # seconds
      kernel = 3,           # number of frames to sample from each window
      stride = 0.5,         # seconds to jump between windows
      port = 8000,
      model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
   ):

      self.pipe = VLLMPipeline(port=port, model_name=model_name)
      self.window_length = window_length
      self.kernel = kernel
      self.stride = stride
   
   def generate_with_retries(
      self,
      fps_list,
      frames_list,
      past_d_list=None,
      max_retries=3
   ):
      for _ in range(max_retries):
         try:
            text = self.pipe(
               fps_list,
               [self.kernel] * len(fps_list),
               [self.window_length] * len(fps_list),
               frames_list,
               past_output_list = past_d_list
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

      longest_windows_length = max(list(map(len, windows)))

      # Every window becomes a specific caption.
      # We need padding to account for short clips
      # Simplify by padding every windows list with its last window
      # First get actual lengths so we can extract meaningful captions out later
      actual_windows_lengths = [len(window) for window in windows]
      windows = [window + [window[-1]] * (longest_windows_length - len(window)) for window in windows]

      captions = [[] for _ in range(len(x_list))]

      past_d_list = None
      start_time = time.time()
      for i in range(longest_windows_length):
         # Get all relevant x's
         x_idx = [x_list[j][windows[j][i]] for j in range(len(x_list))]

         d_list = self.generate_with_retries(
            fps_list,
            x_idx,
            past_d_list
         )
         past_d_list = deepcopy(d_list)

         for j in range(len(d_list)):
            d_list[j]["frame_indices"] = windows[j][i]
            captions[j].append(d_list[j])

      end_time = time.time()
      print(f"Time taken: {end_time - start_time:.2f} seconds")
      for j in range(len(captions)):
         captions[j] = captions[j][:actual_windows_lengths[j]]
      return captions

if __name__ == "__main__":
   # Test window generation
   test_tensor = "/mnt/data/waypoint_1/data_pt/MKIF_360P/VSNETqE_rIk_000024/video/splits/00000000_rgb.pt"
   test_tensor_2 = "/mnt/data/waypoint_1/data_pt/MKIF_360P/VSNETqE_rIk_000024/video/splits/00000001_rgb.pt"
   
   test_tensor = torch.load(test_tensor, map_location='cpu', mmap=True)
   test_tensor_2 = torch.load(test_tensor_2, map_location='cpu', mmap=True)
   
   captioner = VLMCaptioner(
      window_length = 5,
      kernel = 5,
      stride = 2.5,
      port = 8000
   )
   captions = captioner([test_tensor], [60])
   for line in captions[0]:
      print(line)

   # Ideal size is 64