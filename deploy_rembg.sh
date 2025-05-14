# Install dependencies if needed
apt update && apt install -y git python3-pip python3-venv ffmpeg tmux

# Go to workspace
cd /workspace

# Clone your repo
git -c credential.helper= clone https://github.com/justinrumpf/rmvbg.git
cd rembg-server

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install --upgrade pip
pip install -r requirements.txt

# Start the server inside tmux to keep it alive
tmux new-session -d -s rembg "source venv/bin/activate && uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000"
