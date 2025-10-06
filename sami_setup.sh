uv venv venv --python 3.13
source venv/bin/activate
uv pip install -r requirements.txt
git checkout sami-dev

source .env
ln -s /mnt/data/sami/logs logs 

# Install libgl (for OpenCV, etc.)
sudo apt-get update
sudo apt-get install -y libgl1

# Install ffmpeg (for video processing)
sudo apt-get install -y ffmpeg

# Install tmux (for terminal multiplexing)
sudo apt-get install -y tmux
