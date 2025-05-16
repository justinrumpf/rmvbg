```
 _____                   _        _  
|  __ \                 | |      / | 
| |__) |___  ___  __ _  | | ___ | | 
|  _  // _ \/ __|/ _` | | |/ _ \| | 
| | \ \  __/\__ \ (_| | | |  __/| | 
|_|  \_\___||___/\__,_| |_|\___|_|_|         IS IN DA HOUSE!
```

# RemBG implementation that provides an HTTP server with an API endpoint.

## This setup includes settings for:

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

 ## Licensing and Requirements

# 📄 Library Licensing Summary

Below is a list of the libraries and their associated open-source licenses, as commonly published on [PyPI](https://pypi.org/) and GitHub. These licenses determine how the libraries may be used, modified, and distributed in both commercial and non-commercial settings.
This list is as I know them today and there could be errors. Use and investigate licensing at your own risk. Im no attorney. 

                | Library               | License                  | Notes                                                                 |
                |-----------------------|---------------------------|-----------------------------------------------------------------------|
                | `onnxruntime-gpu`     | MIT                       | Permissive license, allows commercial use, modification, distribution. |
                | `rembg[gpu]`          | Apache 2.0                | Allows commercial use and redistribution with attribution.             |
                | `watchdog`            | Apache 2.0                | Permissive, commonly used for file system monitoring.                  |
                | `asyncer`             | MIT                       | Lightweight async utility with a permissive license.                   |
                | `aiohttp`             | Apache 2.0                | Async HTTP client/server library, widely adopted.                      |
                | `uvicorn`             | BSD 3-Clause              | Permissive license with minimal conditions.                            |
                | `fastapi`             | MIT                       | Very permissive, used for async APIs.                                  |
                | `python-multipart`    | Apache 2.0                | Handles multipart/form-data, permissive use.                           |
                | `filetype`            | MIT                       | Detect file types by binary signatures.                                |
                | `gradio`              | Apache 2.0                | For building machine learning web UIs.                                 |
                | `nano`                | GPL-3.0                   | GNU nano text editor; copyleft license, restricts proprietary reuse.   |
                | `aiofiles`            | Apache 2.0                | Asynchronous file IO operations.                                       |
                | `Pillow`              | HPND (PIL Fork License)   | Historical license, permissive; similar to MIT.                        |
                | `requests`            | Apache 2.0                | Most popular HTTP library in Python.                                   |
                | `httpx`               | BSD 3-Clause              | Async-compatible HTTP client library.                                  |

---

## ✅ Summary of Common Licenses

- **MIT / BSD / Apache 2.0**: You can use them in commercial software, modify them, redistribute them, and don’t need to open-source your own code, though attribution is often required.
- **GPL (like in `nano`)**: Requires derivative works to be licensed under the same GPL license if distributed, making it restrictive for proprietary/commercial use unless isolated.


       
