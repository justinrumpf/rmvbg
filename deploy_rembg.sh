#!/bin/bash

set -e

echo "🔧 Updating system packages..."
apt update && apt install -y python3-pip python3-venv ffmpeg git curl tmux unzip

cd /workspace

if [ -d "rmvbg" ]; then
    echo "⚠️  Directory 'rmvbg' already exists. Skipping clone."
else
    echo "📥 Cloning repo..."
    git -c credential.helper= clone https://github.com/justinrumpf/rmvbg.git
fi

cd rmvbg

if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Launching server in tmux..."
tmux kill-session -t rembg || true
tmux new-session -d -s rembg "uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000"

echo "✅ Deployment complete. Access your server on port 7000."
