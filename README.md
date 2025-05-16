RemBG implementation that provides an HTTP server with an API endpoint.

This setup includes settings for:

*   Simple Queuing
*   Concurrent Jobs
*   Watermarking
*   Image Optimization (File Size, Dimensions)

## Installation

1.  **Navigate to the installation directory:** In your terminal, navigate to the directory where you want the `rmvbg` directory to be installed.

2.  **Run the installation script:** Execute the following command:

    ```bash
    bash <(curl -sSL https://raw.githubusercontent.com/justinrumpf/rmvbg/main/deploy_rembg.sh)
    ```

    This script will install the necessary dependencies and configure the RemBG server.

3.  **Access the web server:** After the installation completes, your web server should be running. You can access the landing page by navigating to port 7000 (as specified in the code) in your web browser. Example: `http://your_server_ip:7000`.

## Managing the Web Server Session (Using tmux)

The installation script utilizes `tmux` to run the web server in a detached session. This allows the server to continue running even if you close your terminal. Here are the commands to manage the `tmux` session:

*   **Attach to the web server session:**

    ```bash
    tmux attach -t rembg
    ```

    This will connect you to the running `tmux` session, allowing you to view the server output and interact with it.

*   **Kill the web server session:**

    ```bash
    tmux kill-session -t rembg
    ```

    This will stop the web server process and terminate the `tmux` session.

*   **Run the web server session:**

    ```bash
    tmux new-session -d -s rembg "
    cd /workspace/rmvbg && \
    source venv/bin/activate && \
    uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000
    "
    ```

    This command does the following:

    *   `tmux new-session -d -s rembg`: Creates a new detached (`-d`) `tmux` session named `rembg` (`-s rembg`).
    *   `cd /workspace/rmvbg`: Navigates to the `rmvbg` directory (adjust this path if your installation directory is different).
    *   `source venv/bin/activate`: Activates the virtual environment created by the installation script.
    *   `uvicorn rembg_queue_server:app --host 0.0.0.0 --port 7000`: Starts the `uvicorn` web server, binding it to all available network interfaces (`0.0.0.0`) on port 7000.
