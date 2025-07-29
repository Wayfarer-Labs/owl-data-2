import cv2
import torch
import numpy as np

import sys
sys.path.append('Depth-Anything-V2')

from depth_anything_v2.dpt import DepthAnythingV2

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

class DepthPipeline:
    def __init__(self):
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        self.model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
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

if __name__ == '__main__':
    # Generate a random noise image as a numpy array (H, W, 3) in uint8
    height, width = 480, 640
    random_img = (np.random.rand(height, width, 3) * 255).astype(np.uint8)

    depth_pipeline = DepthPipeline()
    depth = depth_pipeline(random_img)
    print("Depth output shape:", depth.shape)