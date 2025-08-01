import torch

import sys
sys.path.append("./owl-vaes/")
from owl_vaes import from_pretrained

class EncodingPipe:
    def __init__(
        self, cfg_path, ckpt_path
    ):
        model = from_pretrained(cfg_path, ckpt_path)
        model = model.encoder.eval().bfloat16().cuda()
        self.model = torch.compile(model, mode='max-autotune', dynamic=False, fullgraph=True)
    
    @torch.no_grad()
    def __call__(self, x):
        return self.model(x)

class BatchedEncodingPipe:
    def __init__(self, cfg_path, ckpt_path, batch_size=128):
        self.pipe = EncodingPipe(cfg_path, ckpt_path)
        self.batch_size = batch_size
    
    @torch.no_grad()
    def __call__(self, x):
        n = x.shape[0]
        batches = []
        for i in range(0, n, self.batch_size):
            batch = x[i:i+self.batch_size]
            if batch.shape[0] < self.batch_size:
                # Pad the last batch with zeros
                pad_size = self.batch_size - batch.shape[0]
                pad_shape = (pad_size, *batch.shape[1:])
                pad = torch.zeros(pad_shape, dtype=batch.dtype, device=batch.device)
                batch = torch.cat([batch, pad], dim=0)
            batches.append(batch)
        out = []
        for i, batch in enumerate(batches):
            out.append(self.pipe(batch))
        out = torch.cat(out, dim=0)
        # Remove the padded samples if any
        out = out[:n]
        return out

if __name__ == "__main__":
    cfg_path = "/mnt/data/shahbuland/owl-vaes/configs/1x/base.yml"
    ckpt_path = "/mnt/data/shahbuland/owl-vaes/checkpoints/1x_rgb_depth/step_100000.pt"

    pipe = EncodingPipe(cfg_path, ckpt_path)

    import torch

    x = torch.randn(1, 4, 512, 512).cuda().bfloat16()
    out = pipe(x)
    print(out.shape)