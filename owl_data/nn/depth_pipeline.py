import cv2
import os
import torch
import numpy as np
from torch.nn import functional as F
import sys

torch.backends.cuda.enable_flash_sdp(True)

sys.path.append('Depth-Anything-V2')

from depth_anything_v2.dpt import DepthAnythingV2

def download_ckpt_if_not_exists():
    os.makedirs("checkpoints/depth", exist_ok=True)
    ckpt_path = "checkpoints/depth/depth_anything_v2_vitl.pth"
    if not os.path.exists(ckpt_path):
        import urllib.request
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
        print(f"Downloading checkpoint from {url} to {ckpt_path}...")
        urllib.request.urlretrieve(url, ckpt_path)
        print("Download complete.")

def normalize_depth_np(x):  # [h,w] depth map as float np array
    # Flatten to 1D for median and MAD
    med = np.median(x)
    x_centered = x - med
    mad = np.mean(np.abs(x_centered))
    # Avoid division by zero
    if mad == 0:
        x_norm = np.zeros_like(x_centered)
    else:
        x_norm = x_centered / mad
    # Scale to [0,1]
    x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min() + 1e-8)
    # Convert to uint8 [0,255]
    x_uint8 = (x_norm * 255).astype(np.uint8)
    return x_uint8

def normalize_depth(x): # [b,1,h,w] depth map
    # Get median per batch element [b,1,1,1]
    medians = 100
    
    # Subtract medians from each element
    x_centered = x - medians
    
    # Calculate mean absolute deviation (MAD) per batch element
    mad = 100
    
    # Normalize by MAD
    x = x_centered / mad
    # Scale to [0,1] range
    x = (x - x.min()) / (x.max() - x.min())
    # Convert to uint8 [0,255]
    x = (x * 255).to(torch.uint8)
    return x    

class DepthPipeline:
    def __init__(self):
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        download_ckpt_if_not_exists()

        self.model.load_state_dict(torch.load('checkpoints/depth/depth_anything_v2_vitl.pth', map_location='cpu'))
        
        self.model.cuda().bfloat16().eval()
        self.model = torch.compile(self.model, mode='max-autotune',dynamic=False,fullgraph=True)

    @torch.no_grad()
    def __call__(self, img):
        """
        Image is uint8 BGR np array [h,w,c] [0,255]

        Returns depthmap as uint8 np array [h,w] [0,255]
        """
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            depth = self.model.infer_image(img) # HxW raw depth map
            depth = normalize_depth_np(depth)
            return depth

class BatchedDepthPipeline:
    def __init__(self, input_mode = "uint8", batch_size = 500):
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        download_ckpt_if_not_exists()

        self.model.load_state_dict(torch.load('checkpoints/depth/depth_anything_v2_vitl.pth', map_location='cpu'))
        self.model = self.model.cuda().bfloat16().eval()
        self.model = torch.compile(self.model,dynamic=False,fullgraph=True)
        self.batch_size = batch_size

        self.input_mode = input_mode # "uint8" or "bfloat16"

    @torch.compile()
    def preprocess(self, x, target_size=(518, 518)):
        """
        x is assumed [b,c,h,w] [-1,1] bfloat16 tensor (RGB)
        Returns: [b,c,h,w] float32 tensor, normalized as expected by DepthAnythingV2
        """

        x = x.cuda(non_blocking=True)

        if self.input_mode == "bfloat16":
            # Convert from [-1,1] to [0,1]
            x = (x + 1.0) / 2.0
        else:
            x = (x / 255.0).bfloat16()

        # Normalize using ImageNet mean/std for RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
        x = (x - mean) / std

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        return x

    def forward(self, batch):
        # Assume batch is input to preprocess
        _,_,h,w = batch.shape
        batch = self.preprocess(batch)
        depth = self.model(batch)
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h,w), mode='bilinear', align_corners=True)
        depth = normalize_depth(depth)
        return depth.cpu() # b1hw

    @torch.no_grad()
    def __call__(self, x):
        n = x.shape[0]
        batches = []
        for i in range(0, n, self.batch_size):
            batch = x[i:i+self.batch_size]
            if batch.shape[0] < self.batch_size:
                # Pad to full batch size
                pad_size = self.batch_size - batch.shape[0]
                pad_shape = (pad_size, *batch.shape[1:])
                pad = torch.zeros(pad_shape, device=batch.device, dtype=batch.dtype)
                batch = torch.cat([batch, pad], dim=0)
            batches.append(batch)
        outputs = []
        for i, batch in enumerate(batches):
            out = self.forward(batch)
            # If last batch and was padded, remove padding
            if i == len(batches) - 1 and n % self.batch_size != 0:
                out = out[:n % self.batch_size]
            outputs.append(out)
        return torch.cat(outputs, dim=0)


if __name__ == '__main__':
    # Generate a random noise image as a numpy array (H, W, 3) in uint8
    height, width = 480, 640
    random_img = (np.random.rand(height, width, 3) * 255).astype(np.uint8)

    depth_pipeline = DepthPipeline()
    depth = depth_pipeline(random_img)
    print("Depth output shape:", depth.shape)