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


# Generate an RSA SSH key pair (no passphrase)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Start the ssh-agent in the background
eval "$(ssh-agent -s)"

# Add the private key to the ssh-agent
ssh-add ~/.ssh/id_rsa

# Display the public key
cat ~/.ssh/id_rsa.pub

git config --global user.email "samibghanem@gmail.com"
git config --global user.name "Sami"