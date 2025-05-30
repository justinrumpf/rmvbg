import asyncio
import uuid
import io
import os
import aiofiles
import logging
import httpx
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# --- CREATE DIRECTORIES AT THE VERY TOP X---
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    os.makedirs(UPLOADS_DIR_STATIC, exist_ok=True)
    os.makedirs(PROCESSED_DIR_STATIC, exist_ok=True)
    os.makedirs(BASE_DIR_STATIC, exist_ok=True)
    logger.info(f"Ensured uploads directory exists: {UPLOADS_DIR_STATIC}")
    logger.info(f"Ensured processed directory exists: {PROCESSED_DIR_STATIC}")
    logger.info(f"Ensured base directory exists: {BASE_DIR_STATIC}")
except OSError as e:
    logger.error(f"CRITICAL: Error creating essential directories: {e}", exc_info=True)

from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from rembg import remove, new_session # type: ignore
from PIL import Image

app = FastAPI()

# --- ADD CORS MIDDLEWARE ---
origins = [
    "null",
    "https://localhost:44302",
    "http://127.0.0.1"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration Constants ---
MAX_CONCURRENT_TASKS = 8
MAX_QUEUE_SIZE = 5000
ESTIMATED_TIME_PER_JOB = 35
TARGET_SIZE = 1024
HTTP_CLIENT_TIMEOUT = 30.0

# Thread pool configuration
CPU_THREAD_POOL_SIZE = 4  # Adjust based on your CPU cores
PIL_THREAD_POOL_SIZE = 4  # For PIL operations

ENABLE_LOGO_WATERMARK = False
LOGO_MAX_WIDTH = 150
LOGO_MARGIN = 20
LOGO_FILENAME = "logo.png"

# --- Directory and File Paths ---
BASE_DIR = BASE_DIR_STATIC
UPLOADS_DIR = UPLOADS_DIR_STATIC
PROCESSED_DIR = PROCESSED_DIR_STATIC
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME) if ENABLE_LOGO_WATERMARK else ""

# --- Global State ---
prepared_logo_image = None
queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
results: dict = {}
EXPECTED_API_KEY = "secretApiKey"

# Thread pools for CPU-bound operations
cpu_executor: ThreadPoolExecutor = None
pil_executor: ThreadPoolExecutor = None

# Statistics and monitoring
server_start_time = time.time()
job_history = []  # List of completed jobs with stats
total_jobs_completed = 0
total_jobs_failed = 0
total_processing_time = 0.0
MAX_HISTORY_ITEMS = 50  # Keep last 50 jobs in history

MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}

# --- Pydantic Models ---
class SubmitJsonBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = "u2net"
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"

# --- Helper Functions ---
def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

def format_size(num_bytes: int) -> str:
    """Formats a byte size into a human-readable string (KB, MB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    else:
        return f"{num_bytes/1024**2:.2f} MB"

def add_job_to_history(job_id: str, status: str, total_time: float, input_size: int, output_size: int, model: str, source_type: str = "unknown", original_filename: str = ""):
    """Add completed job to history for monitoring"""
    global job_history, total_jobs_completed, total_jobs_failed, total_processing_time
    
    job_record = {
        "job_id": job_id,
        "timestamp": time.time(),
        "status": status,
        "total_time": total_time,
        "input_size": input_size,
        "output_size": output_size,
        "model": model,
        "source_type": source_type,
        "original_filename": original_filename
    }
    
    job_history.insert(0, job_record)  # Add to beginning
    if len(job_history) > MAX_HISTORY_ITEMS:
        job_history.pop()  # Remove oldest
    
    if status == "completed":
        total_jobs_completed += 1
        total_processing_time += total_time
    else:
        total_jobs_failed += 1

def format_timestamp(timestamp: float) -> str:
    """Format timestamp to readable string"""
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def get_server_stats():
    """Get current server statistics"""
    uptime = time.time() - server_start_time
    active_jobs = sum(1 for job in results.values() if job.get("status") not in ["done", "error"])
    
    return {
        "uptime": uptime,
        "queue_size": queue.qsize(),
        "active_jobs": active_jobs,
        "total_completed": total_jobs_completed,
        "total_failed": total_jobs_failed,
        "avg_processing_time": total_processing_time / max(total_jobs_completed, 1),
        "recent_jobs": job_history
    }

# --- CPU-bound functions (run in thread pool) ---
def process_rembg_sync(input_bytes: bytes, model_name: str) -> bytes:
    """Synchronous rembg processing - runs in thread pool"""
    session = new_session(model_name)
    output_bytes = remove(
        input_bytes,
        session=session,
        post_process_mask=True,
        alpha_matting=True
    )
    return output_bytes

def process_pil_sync(
    input_bytes: bytes, 
    target_size: int, 
    prepared_logo: Image.Image = None,
    enable_logo: bool = False,
    logo_margin: int = 20
) -> bytes:
    """Synchronous PIL processing - runs in thread pool"""
    # Open and convert to RGBA
    img_rgba = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    
    # Create white background and paste
    white_bg_canvas = Image.new("RGB", img_rgba.size, (255, 255, 255))
    white_bg_canvas.paste(img_rgba, (0, 0), img_rgba)
    img_on_white_bg = white_bg_canvas
    
    # Get dimensions and validate
    original_width, original_height = img_on_white_bg.size
    if original_width == 0 or original_height == 0:
        raise ValueError("Image dimensions zero after BG processing")
    
    # Resize to fit target size while maintaining aspect ratio
    ratio = min(target_size / original_width, target_size / original_height)
    new_width, new_height = int(original_width * ratio), int(original_height * ratio)
    img_resized_on_white = img_on_white_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create square canvas and center the image
    square_canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x, paste_y = (target_size - new_width) // 2, (target_size - new_height) // 2
    square_canvas.paste(img_resized_on_white, (paste_x, paste_y))
    
    # Add logo watermark if enabled
    if enable_logo and prepared_logo:
        if square_canvas.mode != 'RGBA':
            square_canvas = square_canvas.convert('RGBA')
        logo_w, logo_h = prepared_logo.size
        logo_pos_x, logo_pos_y = logo_margin, target_size - logo_h - logo_margin
        square_canvas.paste(prepared_logo, (logo_pos_x, logo_pos_y), prepared_logo)
    
    # Ensure final image is RGB
    final_image = square_canvas
    if final_image.mode == 'RGBA':
        final_opaque_canvas = Image.new("RGB", final_image.size, (255, 255, 255))
        final_opaque_canvas.paste(final_image, mask=final_image.split()[3])
        final_image = final_opaque_canvas
    
    # Convert to bytes
    output_buffer = io.BytesIO()
    final_image.save(output_buffer, 'WEBP', quality=90, background=(255, 255, 255))
    return output_buffer.getvalue()

# --- API Endpoints ---
@app.post("/submit")
async def submit_json_image_for_processing(
    request: Request,
    body: SubmitJsonBody
):
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if ENABLE_LOGO_WATERMARK and os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo watermarking enabled, logo file exists, but not loaded. Check startup.")

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    try:
        queue.put_nowait((job_id, str(body.image), body.model, True))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full. Rejecting JSON request for image {body.image}.")
        raise HTTPException(status_code=503, detail=f"Server overloaded (queue full). Max: {MAX_QUEUE_SIZE}")

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued", "input_image_url": str(body.image), "original_local_path": None,
        "processed_path": None, "error_message": None, "status_check_url": status_check_url
    }
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (JSON URL: {body.image}) enqueued. Queue size: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    return {
        "status": "processing", "job_id": job_id, "image_links": [processed_image_placeholder_url],
        "eta": eta_seconds, "status_check_url": status_check_url
    }

@app.post("/submit_form")
async def submit_form_image_for_processing(
    request: Request,
    image_file: UploadFile = File(...),
    key: str = Form(...),
    model: str = Form("u2net")
):
    if key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if ENABLE_LOGO_WATERMARK and os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo watermarking enabled, logo file exists, but not loaded. Check startup.")
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    original_filename_from_upload = image_file.filename if image_file.filename else "upload"
    content_type_from_upload = image_file.content_type.lower()
    extension = MIME_TO_EXT.get(content_type_from_upload)
    if not extension:
        _, ext_from_filename = os.path.splitext(original_filename_from_upload)
        ext_from_filename_lower = ext_from_filename.lower()
        if ext_from_filename_lower in MIME_TO_EXT.values():
            extension = ext_from_filename_lower
        else:
            extension = ".png"
            logger.warning(f"Job {job_id} (form): Could not determine ext for '{original_filename_from_upload}' from type '{content_type_from_upload}'. Defaulting to '{extension}'.")

    saved_original_filename = f"{job_id}_original{extension}"
    original_file_path = os.path.join(UPLOADS_DIR, saved_original_filename)

    try:
        async with aiofiles.open(original_file_path, 'wb') as out_file:
            file_content = await image_file.read()
            await out_file.write(file_content)
        logger.info(f"📝 Job {job_id} (Form Upload: {original_filename_from_upload}) Original image saved: {original_file_path} ({format_size(len(file_content))})")
    except Exception as e:
        logger.error(f"Error saving uploaded file {saved_original_filename} for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await image_file.close()

    file_uri_for_queue = f"file://{original_file_path}"
    try:
        queue.put_nowait((job_id, file_uri_for_queue, model, True))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full. Rejecting form request for image {original_filename_from_upload} (job {job_id}).")
        if os.path.exists(original_file_path):
            try: os.remove(original_file_path)
            except OSError as e_clean: logger.error(f"Error cleaning {original_file_path} (queue full): {e_clean}")
        raise HTTPException(status_code=503, detail=f"Server overloaded (queue full). Max: {MAX_QUEUE_SIZE}")

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued", "input_image_url": f"(form_upload: {original_filename_from_upload})",
        "original_local_path": original_file_path, "processed_path": None,
        "error_message": None, "status_check_url": status_check_url
    }

    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    original_image_served_url = f"{public_url_base}/originals/{saved_original_filename}"
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (Form Upload: {original_filename_from_upload}) enqueued. Queue size: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    return {
        "status": "processing",
        "job_id": job_id,
        "original_image_url": original_image_served_url,
        "image_links": [processed_image_placeholder_url],
        "eta": eta_seconds,
        "status_check_url": status_check_url
    }

@app.get("/job/{job_id}")
async def job_details(request: Request, job_id: str):
    """Display detailed job information with before/after images"""
    
    # Find job in history
    job_info = None
    for job in job_history:
        if job["job_id"] == job_id:
            job_info = job
            break
    
    # If not in history, check current results
    if not job_info and job_id in results:
        result = results[job_id]
        job_info = {
            "job_id": job_id,
            "timestamp": time.time(),  # Approximate
            "status": "active" if result.get("status") not in ["done", "error"] else result.get("status"),
            "total_time": 0,  # Not available for active jobs
            "input_size": 0,  # Not available
            "output_size": 0,  # Not available
            "model": "unknown",
            "source_type": "unknown",
            "original_filename": ""
        }
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    public_url_base = get_proxy_url(request)
    
    # Build image URLs
    original_image_url = None
    processed_image_url = None
    
    # Check for original image
    original_extensions = ['.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    for ext in original_extensions:
        original_filename = f"{job_id}_original{ext}"
        original_path = os.path.join(UPLOADS_DIR, original_filename)
        if os.path.exists(original_path):
            original_image_url = f"{public_url_base}/originals/{original_filename}"
            break
    
    # Check for processed image
    processed_filename = f"{job_id}.webp"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    if os.path.exists(processed_path):
        processed_image_url = f"{public_url_base}/images/{processed_filename}"
    
    # Additional job details from results dict
    result_details = results.get(job_id, {})
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Job Details - {job_id[:8]}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            .status-badge {{ padding: 5px 10px; border-radius: 15px; font-weight: bold; text-transform: uppercase; }}
            .status-completed {{ background-color: #d4edda; color: #155724; }}
            .status-failed {{ background-color: #f8d7da; color: #721c24; }}
            .status-active {{ background-color: #d1ecf1; color: #0c5460; }}
            .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .detail-card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; }}
            .detail-label {{ font-size: 12px; color: #6c757d; text-transform: uppercase; margin-bottom: 5px; }}
            .detail-value {{ font-size: 18px; font-weight: bold; color: #495057; }}
            .images-section {{ margin-top: 30px; }}
            .images-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .image-card {{ border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; background: white; }}
            .image-card h3 {{ margin-top: 0; color: #495057; }}
            .image-card img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #dee2e6; }}
            .no-image {{ color: #6c757d; font-style: italic; text-align: center; padding: 40px; background: #f8f9fa; border-radius: 4px; }}
            .back-link {{ color: #007bff; text-decoration: none; }}
            .back-link:hover {{ text-decoration: underline; }}
            @media (max-width: 768px) {{
                .images-container {{ grid-template-columns: 1fr; }}
                .header {{ flex-direction: column; align-items: flex-start; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Job Details</h1>
                <a href="/" class="back-link">← Back to Dashboard</a>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h2>Job ID: <code>{job_id}</code></h2>
                <span class="status-badge status-{job_info['status'].lower()}">{job_info['status']}</span>
            </div>
            
            <div class="details-grid">
                <div class="detail-card">
                    <div class="detail-label">Processed Time</div>
                    <div class="detail-value">{format_timestamp(job_info['timestamp'])}</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Processing Duration</div>
                    <div class="detail-value">{job_info['total_time']:.2f}s</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Model Used</div>
                    <div class="detail-value">{job_info['model']}</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Source Type</div>
                    <div class="detail-value">{job_info['source_type'].title()}</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Input Size</div>
                    <div class="detail-value">{format_size(job_info['input_size'])}</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">Output Size</div>
                    <div class="detail-value">{format_size(job_info['output_size']) if job_info['output_size'] > 0 else 'N/A'}</div>
                </div>
            </div>
            
            {f"<div class='detail-card'><div class='detail-label'>Original Filename</div><div class='detail-value'>{job_info['original_filename']}</div></div>" if job_info.get('original_filename') else ''}
            
            <div class="images-section">
                <h2>Before & After Images</h2>
                <div class="images-container">
                    <div class="image-card">
                        <h3>🔍 Original Image</h3>
                        {f'<img src="{original_image_url}" alt="Original Image" loading="lazy">' if original_image_url else '<div class="no-image">Original image not available</div>'}
                    </div>
                    <div class="image-card">
                        <h3>✨ Processed Image</h3>
                        {f'<img src="{processed_image_url}" alt="Processed Image" loading="lazy">' if processed_image_url else '<div class="no-image">Processed image not available</div>'}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 6px;">
                <h3>Technical Details</h3>
                <ul>
                    <li><strong>Current Status in System:</strong> {result_details.get('status', 'Not in active results')}</li>
                    <li><strong>Status Check URL:</strong> <a href="{result_details.get('status_check_url', '#')}" target="_blank">API Status</a></li>
                    {f"<li><strong>Error Message:</strong> {result_details.get('error_message', 'None')}</li>" if result_details.get('error_message') else ''}
                    <li><strong>Job ID:</strong> <code>{job_id}</code></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """, status_code=200)
    job_info = results.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    public_url_base = get_proxy_url(request)
    response_data = {
        "job_id": job_id, "status": job_info.get("status"),
        "input_image_url": job_info.get("input_image_url"), "status_check_url": job_info.get("status_check_url")
    }
    if job_info.get("original_local_path"):
        original_filename = os.path.basename(job_info["original_local_path"])
        response_data["original_image_url"] = f"{public_url_base}/originals/{original_filename}"
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        processed_filename = os.path.basename(job_info["processed_path"])
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
    elif job_info.get("status") == "error":
        response_data["error_message"] = job_info.get("error_message")
    return JSONResponse(content=response_data)

# --- Background Worker (now truly async) ---
async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image

    while True:
        job_id, image_source_str, model_name, _ = await queue.get()

        t_job_start = time.perf_counter()
        logger.info(f"Worker {worker_id} picked up job {job_id} for source: {image_source_str}. Model: {model_name}")

        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job ID {job_id} from queue not found in results dict. Skipping.")
            queue.task_done()
            continue

        input_bytes_for_rembg: bytes | None = None
        input_fetch_time: float = 0.0
        rembg_time: float = 0.0
        pil_time: float = 0.0
        save_time: float = 0.0
        input_size_bytes: int = 0
        output_size_bytes: int = 0

        try:
            # === PHASE 1: INPUT FETCH (I/O bound - async) ===
            t_input_fetch_start = time.perf_counter()
            if image_source_str.startswith("file://"):
                results[job_id]["status"] = "loading_file"
                local_path_from_uri = image_source_str[len("file://"):]
                if not os.path.exists(local_path_from_uri):
                    raise FileNotFoundError(f"Local file for job {job_id} not found: {local_path_from_uri}")

                async with aiofiles.open(local_path_from_uri, 'rb') as f:
                    input_bytes_for_rembg = await f.read()
                input_size_bytes = len(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (Worker {worker_id}): Loaded local file ({format_size(input_size_bytes)})")

            elif image_source_str.startswith(("http://", "https://")):
                results[job_id]["status"] = "downloading"
                logger.info(f"Job {job_id} (Worker {worker_id}): Downloading from {image_source_str}...")
                async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                    img_response = await client.get(image_source_str)
                    img_response.raise_for_status()

                input_bytes_for_rembg = await img_response.aread()
                input_size_bytes = len(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (Worker {worker_id}): Downloaded {format_size(input_size_bytes)}")

                # Handle content type detection and save original
                original_content_type_header = img_response.headers.get("content-type", "unknown")
                content_type = original_content_type_header.lower()
                if content_type == "application/octet-stream" or not content_type.startswith("image/"):
                    file_ext_from_url = os.path.splitext(urllib.parse.urlparse(image_source_str).path)[1].lower()
                    potential_ct = None
                    if file_ext_from_url == ".webp": potential_ct = "image/webp"
                    elif file_ext_from_url == ".png": potential_ct = "image/png"
                    elif file_ext_from_url in [".jpg", ".jpeg"]: potential_ct = "image/jpeg"
                    if potential_ct: content_type = potential_ct

                if not content_type.startswith("image/"):
                    raise ValueError(f"Invalid final content type '{content_type}' from URL. Not an image.")

                extension = MIME_TO_EXT.get(content_type, ".bin")
                temp_original_filename = f"{job_id}_original_downloaded{extension}"
                downloaded_original_path = os.path.join(UPLOADS_DIR, temp_original_filename)
                results[job_id]["original_local_path"] = downloaded_original_path
                async with aiofiles.open(downloaded_original_path, 'wb') as out_file:
                    await out_file.write(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (Worker {worker_id}): Saved downloaded original")
            else:
                raise ValueError(f"Unsupported image source scheme for job {job_id}: {image_source_str}")

            if input_bytes_for_rembg is None:
                raise ValueError(f"Image content for rembg is None for job {job_id}.")

            t_input_fetch_end = time.perf_counter()
            input_fetch_time = t_input_fetch_end - t_input_fetch_start

            # === PHASE 2: REMBG PROCESSING (CPU bound - thread pool) ===
            results[job_id]["status"] = "processing_rembg"
            logger.info(f"Job {job_id} (Worker {worker_id}): Starting rembg processing (model: {model_name})...")
            
            t_rembg_start = time.perf_counter()
            # Run rembg in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            output_bytes_with_alpha = await loop.run_in_executor(
                cpu_executor,
                process_rembg_sync,
                input_bytes_for_rembg,
                model_name
            )
            t_rembg_end = time.perf_counter()
            rembg_time = t_rembg_end - t_rembg_start
            logger.info(f"Job {job_id} (Worker {worker_id}): Rembg processing completed in {rembg_time:.4f}s")

            # === PHASE 3: PIL PROCESSING (CPU bound - thread pool) ===
            results[job_id]["status"] = "processing_image"
            logger.info(f"Job {job_id} (Worker {worker_id}): Starting PIL processing...")
            
            t_pil_start = time.perf_counter()
            # Run PIL processing in thread pool
            processed_image_bytes = await loop.run_in_executor(
                pil_executor,
                process_pil_sync,
                output_bytes_with_alpha,
                TARGET_SIZE,
                prepared_logo_image,
                ENABLE_LOGO_WATERMARK,
                LOGO_MARGIN
            )
            t_pil_end = time.perf_counter()
            pil_time = t_pil_end - t_pil_start
            logger.info(f"Job {job_id} (Worker {worker_id}): PIL processing completed in {pil_time:.4f}s")

            # === PHASE 4: SAVE TO DISK (I/O bound - async) ===
            results[job_id]["status"] = "saving"
            processed_filename = f"{job_id}.webp"
            processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)

            t_save_start = time.perf_counter()
            async with aiofiles.open(processed_file_path, 'wb') as out_file:
                await out_file.write(processed_image_bytes)
            t_save_end = time.perf_counter()
            save_time = t_save_end - t_save_start
            output_size_bytes = len(processed_image_bytes)

            # === JOB COMPLETION ===
            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path

            t_job_end = time.perf_counter()
            total_job_time = t_job_end - t_job_start
            
            # Determine source type for monitoring
            source_type = "url" if image_source_str.startswith(("http://", "https://")) else "upload"
            original_filename = results[job_id].get("input_image_url", "").split("/")[-1] if source_type == "url" else results[job_id].get("input_image_url", "").replace("(form_upload: ", "").replace(")", "")
            
            # Add to job history
            add_job_to_history(job_id, "completed", total_job_time, input_size_bytes, output_size_bytes, model_name, source_type, original_filename)
            
            logger.info(
                f"Job {job_id} (Worker {worker_id}) COMPLETED successfully in {total_job_time:.4f}s\n"
                f"    Input: {format_size(input_size_bytes)} → Output: {format_size(output_size_bytes)}\n"
                f"    Breakdown: Fetch={input_fetch_time:.3f}s, Rembg={rembg_time:.3f}s, PIL={pil_time:.3f}s, Save={save_time:.3f}s"
            )

        except FileNotFoundError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: FileNotFoundError: {e}", exc_info=False)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"File not found: {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: HTTPStatusError downloading: {e.response.status_code}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Download failed: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: RequestError downloading: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Network error: {type(e).__name__}"
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: Data/file processing error: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Processing error: {str(e)}"
        except Exception as e:
            logger.critical(f"Job {job_id} (Worker {worker_id}) CRITICAL Error: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Unexpected error: {str(e)}"
        finally:
            if results.get(job_id, {}).get("status") == "error":
                t_job_end_error = time.perf_counter()
                total_job_time_error = t_job_end_error - t_job_start
                
                # Add failed job to history
                source_type = "url" if image_source_str.startswith(("http://", "https://")) else "upload"
                original_filename = results[job_id].get("input_image_url", "").split("/")[-1] if source_type == "url" else results[job_id].get("input_image_url", "").replace("(form_upload: ", "").replace(")", "")
                add_job_to_history(job_id, "failed", total_job_time_error, input_size_bytes, 0, model_name, source_type, original_filename)
                
                logger.info(f"Job {job_id} (Worker {worker_id}) FAILED after {total_job_time_error:.4f}s")
            queue.task_done()

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    global prepared_logo_image, cpu_executor, pil_executor
    logger.info("Application startup event running...")

    # Initialize thread pools
    cpu_executor = ThreadPoolExecutor(
        max_workers=CPU_THREAD_POOL_SIZE,
        thread_name_prefix="RembgCPU"
    )
    pil_executor = ThreadPoolExecutor(
        max_workers=PIL_THREAD_POOL_SIZE,
        thread_name_prefix="PILCPU"
    )
    logger.info(f"Thread pools initialized: CPU={CPU_THREAD_POOL_SIZE}, PIL={PIL_THREAD_POOL_SIZE}")

    # Logo loading
    if ENABLE_LOGO_WATERMARK:
        logger.info(f"Logo watermarking ENABLED. Attempting load from: {LOGO_PATH}")
        if os.path.exists(LOGO_PATH):
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                if logo.width > LOGO_MAX_WIDTH:
                    l_ratio = LOGO_MAX_WIDTH / logo.width
                    l_new_width, l_new_height = LOGO_MAX_WIDTH, int(logo.height * l_ratio)
                    logo = logo.resize((l_new_width, l_new_height), Image.Resampling.LANCZOS)
                prepared_logo_image = logo
                logger.info(f"Logo loaded. Dimensions: {prepared_logo_image.size}")
            except Exception as e:
                logger.error(f"Failed to load logo: {e}", exc_info=True)
                prepared_logo_image = None
        else:
            logger.warning(f"Logo file not found at {LOGO_PATH}.")
            prepared_logo_image = None
    else:
        logger.info("Logo watermarking DISABLED.")
        prepared_logo_image = None

    # Start async workers
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} async workers started. Thread pools: CPU={CPU_THREAD_POOL_SIZE}, PIL={PIL_THREAD_POOL_SIZE}")

@app.on_event("shutdown")
async def shutdown_event():
    global cpu_executor, pil_executor
    logger.info("Application shutdown event running...")
    
    if cpu_executor:
        cpu_executor.shutdown(wait=True)
        logger.info("CPU thread pool shut down")
    
    if pil_executor:
        pil_executor.shutdown(wait=True)
        logger.info("PIL thread pool shut down")

# --- Static File Serving ---
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    stats = get_server_stats()
    
    logo_status = "Enabled" if ENABLE_LOGO_WATERMARK else "Disabled"
    if ENABLE_LOGO_WATERMARK and prepared_logo_image:
        logo_status += f" (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})"
    elif ENABLE_LOGO_WATERMARK and not prepared_logo_image:
        logo_status += " (Enabled but not loaded/found)"

    # Format uptime
    uptime_hours = stats["uptime"] / 3600
    uptime_str = f"{uptime_hours:.1f} hours" if uptime_hours >= 1 else f"{stats['uptime']:.0f} seconds"
    
    # Build recent jobs table
    recent_jobs_html = ""
    if stats["recent_jobs"]:
        recent_jobs_html = "<h3>Recent Jobs</h3><table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
        recent_jobs_html += "<tr style='background-color: #f0f0f0;'><th>Time</th><th>Job ID</th><th>Status</th><th>Duration</th><th>Input Size</th><th>Output Size</th><th>Model</th><th>Source</th></tr>"
        
        for job in stats["recent_jobs"][:20]:  # Show last 20 jobs
            status_color = "#4CAF50" if job["status"] == "completed" else "#f44336"
            job_link = f"/job/{job['job_id']}"
            recent_jobs_html += f"""
            <tr style="cursor: pointer;" onclick="window.location.href='{job_link}'">
                <td>{format_timestamp(job['timestamp'])}</td>
                <td style='font-family: monospace; font-size: 10px;'><a href="{job_link}" style="text-decoration: none; color: #007bff;">{job['job_id'][:8]}...</a></td>
                <td style='color: {status_color}; font-weight: bold;'>{job['status'].upper()}</td>
                <td>{job['total_time']:.2f}s</td>
                <td>{format_size(job['input_size'])}</td>
                <td>{format_size(job['output_size']) if job['output_size'] > 0 else 'N/A'}</td>
                <td>{job['model']}</td>
                <td>{job['source_type']}</td>
            </tr>
            """
        recent_jobs_html += "</table>"
    else:
        recent_jobs_html = "<h3>Recent Jobs</h3><p>No jobs processed yet.</p>"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Image API Dashboard</title>
    <style>
        body{{font-family:sans-serif;margin:20px; background-color: #f9f9f9;}} 
        .container{{max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
        .stats-grid{{display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;}}
        .stat-card{{background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; text-align: center;}}
        .stat-value{{font-size: 24px; font-weight: bold; color: #007bff; margin-bottom: 5px;}}
        .stat-label{{font-size: 14px; color: #6c757d; text-transform: uppercase;}}
        table{{font-size: 14px;}} 
        th{{background-color: #f0f0f0 !important;}}
        tr:hover{{background-color: #f8f9fa; cursor: pointer;}}
        .job-link{{color: #007bff; text-decoration: none;}}
        .job-link:hover{{text-decoration: underline;}}
        li{{margin-bottom: 5px;}}
        .status-good{{color: #28a745;}}
        .status-warning{{color: #ffc107;}}
        .status-error{{color: #dc3545;}}
    </style>
    <script>
        function refreshPage() {{
            location.reload();
        }}
        // Auto refresh every 30 seconds
        setTimeout(refreshPage, 30000);
    </script>
    </head>
    <body>
    <div class="container">
        <h1>🚀 Image Processing API Dashboard</h1>
        <p><strong>Status:</strong> <span class="status-good">RUNNING</span></p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{uptime_str}</div>
                <div class="stat-label">Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['queue_size']}</div>
                <div class="stat-label">Queue Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['active_jobs']}</div>
                <div class="stat-label">Active Jobs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value status-good">{stats['total_completed']}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value status-error">{stats['total_failed']}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['avg_processing_time']:.2f}s</div>
                <div class="stat-label">Avg Process Time</div>
            </div>
        </div>

        <h3>Configuration</h3>
        <ul>
            <li><strong>Async Workers:</strong> {MAX_CONCURRENT_TASKS}</li>
            <li><strong>CPU Thread Pool:</strong> {CPU_THREAD_POOL_SIZE}</li>
            <li><strong>Queue Capacity:</strong> {MAX_QUEUE_SIZE}</li>
            <li><strong>Logo Watermarking:</strong> {logo_status}</li>
        </ul>

        {recent_jobs_html}
        
        <p style="margin-top: 30px; font-size: 12px; color: #6c757d;">
            Page auto-refreshes every 30 seconds | Last updated: {format_timestamp(time.time())}
        </p>
    </div>
    </body></html>"""

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
