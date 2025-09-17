import tqdm
import torch

if __name__ == "__main__":
    with open('pose_paths.txt', 'r') as f:
        for path in tqdm.tqdm(f.readlines()):
            path = path.strip()
            if not path:
                continue
            
            # Load the pose tensor
            pose_tensor = torch.load(path)
            
            # Convert bool tensor to uint8 (True -> 255, False -> 0)
            pose_uint8 = pose_tensor.to(torch.uint8) * 255
            
            # Save back to the same path
            torch.save(pose_uint8, path)