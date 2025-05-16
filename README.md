# rmvbg

RemBG implementation that gives you an HTTP server with API endpoint.

Settings for:
  -Simple Queuing
  -Concurrent Jobs
  -Watermarking
  -Image Optimization (File Size, Dimensions)

Install:

Navigate to directory where the rmvbg directory will be installed. 
Run: bash <(curl -sSL https://raw.githubusercontent.com/justinrumpf/rmvbg/main/deploy_rembg.sh)

After this, your web server will be running. You should be able to go to port 7000 (specified in code) to see a landing page. 

Attach to web server session:
tmux attach -t rembg

Kill Web Server Session:
tmux kill-session -t rembg

Run Web Server Session:
tmux new-session -d -s rembg "
cd /workspace/rmvbg && \
source venv/bin/activate && \
uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000
"
