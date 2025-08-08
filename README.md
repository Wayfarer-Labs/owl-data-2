# owl-data-2
Data processing pipelines


# Video loading server

See `scripts/launch_cod_data_server.sh` for example on how to launch server. See `owl_data/video_loader.py` for example loader that makes use of this server.

# JPEG Dataset

JPEG dataset for VAE training. See `owl_data/frames_to_jpegs.py` and `owl_data/depth_from_jpegs.py` and then `upload_shuffled_tars.py`

# Quick Start Commands

### Setup
```bash
git submodule update --init --recursive
pip install -r requirements.txt torchvision
bash setup.sh
```

### Process Single Video
```bash
# Setup dirs
mkdir -p test_video/{videos,tensors,captions}
cp your_video.mp4 test_video/videos/

# Convert to tensors
python3 -m owl_data.videos_to_rgb_tensors \
    --input_dir test_video/videos \
    --output_dir test_video/tensors \
    --frame_skip 30 \
    --max_frames 50 \
    --resize_to 256 256 \
    --num_gpus 1

# Generate captions
python3 -m owl_data.captions_from_tensors \
    --input_dir test_video/tensors \
    --kernel_size 5 \
    --dilation 2 \
    --stride 3 \
    --num_gpus 1 \
    --overwrite

# View results
cat test_video/tensors/*_captions.txt
```