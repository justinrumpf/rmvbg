#!/bin/bash

set -e

echo "🔧 Updating system packages..."
apt update && apt install -y python3-pip python3-venv ffmpeg git curl tmux unzip

cd /workspace

# Clone the repo if it doesn't exist
if [ -d "rmvbg" ]; then
    echo "⚠️  Directory 'rmvbg' already exists. Skipping clone."
else
    git -c credential.helper= clone https://github.com/justinrumpf/rmvbg.git
fi

cd rmvbg

# 🔁 Prompt for Pod ID and update Python file
read -p "🌐 Enter your RunPod Pod ID (e.g., abc123): " pod_id
proxy_url="https://${pod_id}-7000.proxy.runpod.net"

echo "🛠️  Inserting public proxy URL into rembg_queue_server.py..."
sed -i "s|{public_url}|$proxy_url|g" rembg_queue_server.py

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Launching server in tmux..."
tmux kill-session -t rembg 2>/dev/null || true
tmux new-session -d -s rembg "
cd /workspace/rmvbg && \
source venv/bin/activate && \
uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000
"

echo "✅ Done! Your Rembg server is now running at: $proxy_url"
