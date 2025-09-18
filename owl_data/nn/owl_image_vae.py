import torch
import sys


class EncodingPipe:
    def __init__(
            self, cfg_path, ckpt_path, dtype=torch.bfloat16
    ):
        sys.path.append("../owl-vaes/")
        from owl_vaes import from_pretrained

        model = from_pretrained(cfg_path, ckpt_path)
        model = model.encoder.eval().to(dtype).cuda()
        self.model = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False, fullgraph=True)
        #self.model = torch.compile(model, dynamic=False, fullgraph=True)

        self.cached_shape = None

    @torch.no_grad()
    def __call__(self, x):
        if self.cached_shape is None:
            self.cached_shape = x.shape
        else:
            assert x.shape == self.cached_shape, "Shape mismatch"

        return self.model(x).clone()

class BatchedEncodingPipe:
    def __init__(self, cfg_path, ckpt_path, batch_size=128, dtype=torch.bfloat16):
        self.pipe = EncodingPipe(cfg_path, ckpt_path, dtype=dtype)
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
    cfg_path = "/mnt/data/shahbuland/owl-vaes/configs/1x/no_depth.yml"
    ckpt_path = "/mnt/data/shahbuland/owl-vaes/checkpoints/1x_rgb_no_depth/step_200000.pt"

    pipe = EncodingPipe(cfg_path, ckpt_path)

    import torch

    x = torch.randn(8, 3, 512, 512).cuda().bfloat16()
    out = pipe(x)
    print(out.shape)
