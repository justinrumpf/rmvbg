import asyncio
import uuid
import io
import os
import aiofiles
import logging
import httpx
import urllib.parse
import time
import psutil
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta

# --- CREATE DIRECTORIES AT THE VERY TOP 123---
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
    "http://localhost",
    "http://127.0.0.1",
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

# --- Monitoring Configuration ---
MONITORING_HISTORY_MINUTES = 60  # Keep 60 minutes of history
MONITORING_SAMPLE_INTERVAL = 5   # Sample every 5 seconds
MAX_MONITORING_SAMPLES = (MONITORING_HISTORY_MINUTES * 60) // MONITORING_SAMPLE_INTERVAL

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

# --- NEW: Worker and System Monitoring ---
worker_activity = defaultdict(deque)  # worker_id -> deque of (timestamp, activity_type)
system_metrics = deque(maxlen=MAX_MONITORING_SAMPLES)  # (timestamp, cpu%, memory%, gpu_info)
worker_lock = threading.Lock()

# Worker activity types
WORKER_IDLE = "idle"
WORKER_FETCHING = "fetching"
WORKER_PROCESSING_REMBG = "rembg"
WORKER_PROCESSING_PIL = "pil"
WORKER_SAVING = "saving"

def log_worker_activity(worker_id: int, activity: str):
    """Log worker activity for monitoring"""
    with worker_lock:
        worker_activity[worker_id].append((time.time(), activity))
        # Keep only recent history
        cutoff_time = time.time() - (MONITORING_HISTORY_MINUTES * 60)
        while worker_activity[worker_id] and worker_activity[worker_id][0][0] < cutoff_time:
            worker_activity[worker_id].popleft()

def get_gpu_info():
    """Get GPU information if available"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_used_mb": mem_info.used // (1024**2),
            "gpu_total_mb": mem_info.total // (1024**2),
            "gpu_utilization": utilization.gpu
        }
    except:
        return {"gpu_used_mb": 0, "gpu_total_mb": 0, "gpu_utilization": 0}

async def system_monitor():
    """Background task to collect system metrics"""
    while True:
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # GPU (if available)
            gpu_info = get_gpu_info()
            
            # Store metrics
            timestamp = time.time()
            metrics = {
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "memory_total_gb": memory_total_gb,
                **gpu_info
            }
            
            system_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        await asyncio.sleep(MONITORING_SAMPLE_INTERVAL)

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

def get_worker_activity_data():
    """Get worker activity data for charting"""
    current_time = time.time()
    cutoff_time = current_time - (MONITORING_HISTORY_MINUTES * 60)
    
    # Create time buckets (30-second intervals for smoother charts)
    bucket_size = 30  # seconds
    num_buckets = (MONITORING_HISTORY_MINUTES * 60) // bucket_size
    
    worker_data = {}
    
    with worker_lock:
        for worker_id in range(1, MAX_CONCURRENT_TASKS + 1):
            activities = worker_activity.get(worker_id, deque())
            
            # Initialize buckets
            buckets = []
            for i in range(num_buckets):
                bucket_start = cutoff_time + (i * bucket_size)
                bucket_end = bucket_start + bucket_size
                buckets.append({
                    "timestamp": bucket_start,
                    "idle": 0,
                    "fetching": 0,
                    "rembg": 0,
                    "pil": 0,
                    "saving": 0
                })
            
            # Fill buckets with activity data
            for timestamp, activity in activities:
                if timestamp >= cutoff_time:
                    bucket_index = int((timestamp - cutoff_time) // bucket_size)
                    if 0 <= bucket_index < len(buckets):
                        buckets[bucket_index][activity] += 1
            
            worker_data[f"worker_{worker_id}"] = buckets
    
    return worker_data

def get_system_metrics_data():
    """Get system metrics for charting"""
    return list(system_metrics)

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

@app.get("/api/monitoring/workers")
async def get_worker_monitoring_data():
    """API endpoint for worker activity data"""
    return get_worker_activity_data()

@app.get("/api/monitoring/system")
async def get_system_monitoring_data():
    """API endpoint for system metrics data"""
    return get_system_metrics_data()

@app.get("/status/{job_id}")
async def check_job_status(request: Request, job_id: str):
    """Check status of a specific job by ID"""
    logger.info(f"Status check requested for job_id: {job_id}")
    logger.info(f"Current results dict has {len(results)} jobs: {list(results.keys())[:5]}...")
    logger.info(f"Job history has {len(job_history)} jobs")
    
    job_info = results.get(job_id)
    logger.info(f"Job {job_id} found in results: {job_info is not None}")
    
    # If not in active results, check job history for completed/failed jobs
    if not job_info:
        historical_job = None
        for job in job_history:
            if job["job_id"] == job_id:
                historical_job = job
                break
        
        logger.info(f"Job {job_id} found in history: {historical_job is not None}")
        
        if historical_job:
            # Reconstruct response for historical job
            public_url_base = get_proxy_url(request)
            status = "done" if historical_job["status"] == "completed" else "error"
            
            response_data = {
                "job_id": job_id,
                "status": status,
                "input_image_url": f"(historical: {historical_job.get('original_filename', 'unknown')})",
                "status_check_url": f"{public_url_base}/status/{job_id}"
            }
            
            # Check if files still exist and add URLs
            # Check for original image
            original_extensions = ['.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
            for ext in original_extensions:
                original_filename = f"{job_id}_original{ext}"
                original_path = os.path.join(UPLOADS_DIR, original_filename)
                if os.path.exists(original_path):
                    response_data["original_image_url"] = f"{public_url_base}/originals/{original_filename}"
                    break
            
            # Check for processed image
            if status == "done":
                processed_filename = f"{job_id}.webp"
                processed_path = os.path.join(PROCESSED_DIR, processed_filename)
                if os.path.exists(processed_path):
                    response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
            else:
                response_data["error_message"] = f"Job failed after {historical_job['total_time']:.2f}s"
            
            logger.info(f"Returning historical job data for {job_id}")
            return JSONResponse(content=response_data)
    
    # Job not found in either location
    if not job_info:
        logger.error(f"Job {job_id} not found in results or history - returning 404")
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Handle active job (original logic)
    public_url_base = get_proxy_url(request)
    response_data = {
        "job_id": job_id, 
        "status": job_info.get("status"),
        "input_image_url": job_info.get("input_image_url"), 
        "status_check_url": job_info.get("status_check_url")
    }
    
    if job_info.get("original_local_path"):
        original_filename = os.path.basename(job_info["original_local_path"])
        response_data["original_image_url"] = f"{public_url_base}/originals/{original_filename}"
    
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        processed_filename = os.path.basename(job_info["processed_path"])
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
    elif job_info.get("status") == "error":
        response_data["error_message"] = job_info.get("error_message")
    
    logger.info(f"Returning active job data for {job_id}: status={job_info.get('status')}")
    return JSONResponse(content=response_data)

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
