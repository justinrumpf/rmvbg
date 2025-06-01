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
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta

# --- CREATE DIRECTORIES AT THE VERY TOP X4---
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence GPU monitoring warnings and psutil warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="psutil")
logging.getLogger("pynvml").setLevel(logging.ERROR)
logging.getLogger("nvidia_ml_py").setLevel(logging.ERROR)

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
ESTIMATED_TIME_PER_JOB = 35  # Will be lower with GPU
TARGET_SIZE = 1024
HTTP_CLIENT_TIMEOUT = 30.0

# --- GPU Configuration for Rembg ---
REMBG_USE_GPU = True
# Order matters: TensorRT > CUDA > DML. CPU is added as fallback.
REMBG_PREFERRED_GPU_PROVIDERS = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider']
REMBG_CPU_PROVIDERS = ['CPUExecutionProvider']


# --- Monitoring Configuration ---
MONITORING_HISTORY_MINUTES = 60
MONITORING_SAMPLE_INTERVAL = 5
MAX_MONITORING_SAMPLES = (MONITORING_HISTORY_MINUTES * 60) // MONITORING_SAMPLE_INTERVAL

# Thread pool configuration
CPU_THREAD_POOL_SIZE = 4
PIL_THREAD_POOL_SIZE = 4

ENABLE_LOGO_WATERMARK = False
LOGO_MAX_WIDTH = 150
LOGO_MARGIN = 20
LOGO_FILENAME = "logo.png"

BASE_DIR = BASE_DIR_STATIC
UPLOADS_DIR = UPLOADS_DIR_STATIC
PROCESSED_DIR = PROCESSED_DIR_STATIC
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME) if ENABLE_LOGO_WATERMARK else ""

# --- Global State ---
prepared_logo_image = None
queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
results: dict = {}
EXPECTED_API_KEY = "secretApiKey"

cpu_executor: ThreadPoolExecutor = None
pil_executor: ThreadPoolExecutor = None
active_rembg_providers: list[str] = list(REMBG_CPU_PROVIDERS) # Default, will be updated at startup

server_start_time = time.time()
job_history = []
total_jobs_completed = 0
total_jobs_failed = 0
total_processing_time = 0.0
MAX_HISTORY_ITEMS = 50

worker_activity = defaultdict(deque)
system_metrics = deque(maxlen=MAX_MONITORING_SAMPLES)
worker_lock = threading.Lock()

WORKER_IDLE = "idle"
WORKER_FETCHING = "fetching"
WORKER_PROCESSING_REMBG = "rembg"
WORKER_PROCESSING_PIL = "pil"
WORKER_SAVING = "saving"

def log_worker_activity(worker_id: int, activity: str):
    with worker_lock:
        worker_activity[worker_id].append((time.time(), activity))
        cutoff_time = time.time() - (MONITORING_HISTORY_MINUTES * 60)
        while worker_activity[worker_id] and worker_activity[worker_id][0][0] < cutoff_time:
            worker_activity[worker_id].popleft()

def get_gpu_info():
    gpu_data = {"gpu_used_mb": 0, "gpu_total_mb": 0, "gpu_utilization": 0}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assumes GPU 0
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_data["gpu_used_mb"] = mem_info.used // (1024**2)
        gpu_data["gpu_total_mb"] = mem_info.total // (1024**2)
        gpu_data["gpu_utilization"] = utilization.gpu
        
        if not hasattr(get_gpu_info, '_logged_count'):
            get_gpu_info._logged_count = 0
        if get_gpu_info._logged_count < 3:
            logger.info(f"GPU Monitor (pynvml): GPU {utilization.gpu}% | Memory {gpu_data['gpu_used_mb']}/{gpu_data['gpu_total_mb']} MB")
            get_gpu_info._logged_count += 1
            
    except ImportError:
        if not hasattr(get_gpu_info, '_import_warned'):
            logger.warning("GPU monitoring via pynvml disabled: pynvml not installed (pip install pynvml). This is mainly for NVIDIA GPUs.")
            get_gpu_info._import_warned = True
    except Exception as e: # Catches pynvml.NVMLError, etc.
        if not hasattr(get_gpu_info, '_error_warned'):
            logger.warning(f"GPU monitoring via pynvml failed: {type(e).__name__}: {e}. This might happen if no NVIDIA GPU is present or drivers are missing.")
            get_gpu_info._error_warned = True
    finally:
        # pynvml.nvmlShutdown() # Best practice to shut down, but can cause issues if called too often / from threads
        # For continuous monitoring, it's often initialized once and shut down at app exit.
        # Since this is called repeatedly, consider initializing once at startup.
        # For now, let's rely on pynvml to handle repeated Init/Shutdown or not use Shutdown here for simplicity in frequent calls.
        pass
    return gpu_data


async def system_monitor():
    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            gpu_info = get_gpu_info()
            
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
    'image/jpeg': '.jpg', 'image/png': '.png', 'image/gif': '.gif',
    'image/webp': '.webp', 'image/bmp': '.bmp', 'image/tiff': '.tiff'
}

class SubmitJsonBody(BaseModel):
    image: HttpUrl; key: str; model: str = "u2net"
    steps: int = 20; samples: int = 1; resolution: str = "1024x1024"

def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

def format_size(num_bytes: int) -> str:
    if num_bytes < 1024: return f"{num_bytes} B"
    elif num_bytes < 1024**2: return f"{num_bytes/1024:.2f} KB"
    else: return f"{num_bytes/1024**2:.2f} MB"

def add_job_to_history(job_id: str, status: str, total_time: float, input_size: int, output_size: int, model: str, source_type: str = "unknown", original_filename: str = ""):
    global job_history, total_jobs_completed, total_jobs_failed, total_processing_time
    job_record = {"job_id": job_id, "timestamp": time.time(), "status": status, "total_time": total_time,
                  "input_size": input_size, "output_size": output_size, "model": model,
                  "source_type": source_type, "original_filename": original_filename}
    job_history.insert(0, job_record)
    if len(job_history) > MAX_HISTORY_ITEMS: job_history.pop()
    if status == "completed": total_jobs_completed += 1; total_processing_time += total_time
    else: total_jobs_failed += 1

def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def get_server_stats():
    uptime = time.time() - server_start_time
    active_jobs = sum(1 for job in results.values() if job.get("status") not in ["done", "error"])
    return {"uptime": uptime, "queue_size": queue.qsize(), "active_jobs": active_jobs,
            "total_completed": total_jobs_completed, "total_failed": total_jobs_failed,
            "avg_processing_time": total_processing_time / max(total_jobs_completed, 1),
            "recent_jobs": job_history}

def get_worker_activity_data():
    current_time = time.time()
    cutoff_time = current_time - (MONITORING_HISTORY_MINUTES * 60)
    bucket_size = 30; num_buckets = (MONITORING_HISTORY_MINUTES * 60) // bucket_size
    worker_data = {}
    with worker_lock:
        for worker_id in range(1, MAX_CONCURRENT_TASKS + 1):
            activities = worker_activity.get(worker_id, deque())
            buckets = [{"timestamp": cutoff_time + (i * bucket_size), "idle": 0, "fetching": 0, "rembg": 0, "pil": 0, "saving": 0} for i in range(num_buckets)]
            for timestamp, activity in activities:
                if timestamp >= cutoff_time:
                    bucket_index = int((timestamp - cutoff_time) // bucket_size)
                    if 0 <= bucket_index < len(buckets): buckets[bucket_index][activity] += 1
            worker_data[f"worker_{worker_id}"] = buckets
    return worker_data

def get_system_metrics_data(): return list(system_metrics)

# --- CPU-bound functions (run in thread pool) ---
def process_rembg_sync(input_bytes: bytes, model_name: str) -> bytes:
    """Synchronous rembg processing - runs in thread pool. Uses globally configured providers."""
    global active_rembg_providers

    session_providers = active_rembg_providers
    try:
        session = new_session(model_name, providers=session_providers)
        current_providers_in_session = session.get_providers()
        logger.debug(f"Rembg session for model {model_name} using providers: {current_providers_in_session}")

        is_gpu_provider_configured = any(p.lower().replace("executionprovider", "") in [prov.lower().replace("executionprovider", "") for prov in REMBG_PREFERRED_GPU_PROVIDERS] for p in session_providers)
        is_gpu_provider_active = any(p.lower().replace("executionprovider", "") in [prov.lower().replace("executionprovider", "") for prov in REMBG_PREFERRED_GPU_PROVIDERS] for p in current_providers_in_session)
        
        if is_gpu_provider_configured and not is_gpu_provider_active:
             logger.warning(
                f"Rembg was configured for GPU ({session_providers}) but is using CPU providers ({current_providers_in_session}). "
                "Check ONNX Runtime GPU setup (e.g., `pip install onnxruntime-gpu`) and GPU driver compatibility."
             )
        elif is_gpu_provider_active:
            # Only log this verbosely once or a few times to avoid spam
            if not hasattr(process_rembg_sync, '_gpu_active_logged_count'):
                process_rembg_sync._gpu_active_logged_count = 0
            if process_rembg_sync._gpu_active_logged_count < 5:
                logger.info(f"Rembg is actively using GPU provider(s): {current_providers_in_session}")
                process_rembg_sync._gpu_active_logged_count +=1


    except Exception as e:
        logger.error(f"Failed to initialize rembg session with {session_providers} for model {model_name}: {e}. Falling back to CPU-only: {REMBG_CPU_PROVIDERS}")
        session_providers = list(REMBG_CPU_PROVIDERS) # Ensure it's a mutable list copy
        try:
            session = new_session(model_name, providers=session_providers)
            logger.info(f"Rembg session for model {model_name} successfully fell back to CPU providers: {session.get_providers()}")
        except Exception as e_cpu:
            logger.critical(f"CRITICAL: Failed to initialize rembg session even with CPU-only providers {session_providers} for model {model_name}: {e_cpu}", exc_info=True)
            raise # Re-raise the critical error

    output_bytes = remove(
        input_bytes,
        session=session,
        post_process_mask=True,
        alpha_matting=True
    )
    return output_bytes

def process_pil_sync(input_bytes: bytes, target_size: int, prepared_logo: Image.Image = None, enable_logo: bool = False, logo_margin: int = 20) -> bytes:
    img_rgba = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    white_bg_canvas = Image.new("RGB", img_rgba.size, (255, 255, 255))
    white_bg_canvas.paste(img_rgba, (0, 0), img_rgba)
    img_on_white_bg = white_bg_canvas
    original_width, original_height = img_on_white_bg.size
    if original_width == 0 or original_height == 0: raise ValueError("Image dimensions zero after BG processing")
    ratio = min(target_size / original_width, target_size / original_height)
    new_width, new_height = int(original_width * ratio), int(original_height * ratio)
    img_resized_on_white = img_on_white_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    square_canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x, paste_y = (target_size - new_width) // 2, (target_size - new_height) // 2
    square_canvas.paste(img_resized_on_white, (paste_x, paste_y))
    if enable_logo and prepared_logo:
        if square_canvas.mode != 'RGBA': square_canvas = square_canvas.convert('RGBA')
        logo_w, logo_h = prepared_logo.size
        logo_pos_x, logo_pos_y = logo_margin, target_size - logo_h - logo_margin
        square_canvas.paste(prepared_logo, (logo_pos_x, logo_pos_y), prepared_logo)
    final_image = square_canvas
    if final_image.mode == 'RGBA':
        final_opaque_canvas = Image.new("RGB", final_image.size, (255, 255, 255))
        final_opaque_canvas.paste(final_image, mask=final_image.split()[3])
        final_image = final_opaque_canvas
    output_buffer = io.BytesIO()
    final_image.save(output_buffer, 'WEBP', quality=90, background=(255, 255, 255))
    return output_buffer.getvalue()

@app.post("/submit")
async def submit_json_image_for_processing(request: Request, body: SubmitJsonBody):
    if body.key != EXPECTED_API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    if ENABLE_LOGO_WATERMARK and os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo watermarking enabled, logo file exists, but not loaded. Check startup.")
    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    try: queue.put_nowait((job_id, str(body.image), body.model, True))
    except asyncio.QueueFull:
        logger.warning(f"Queue full. Rejecting JSON request for {body.image}.")
        raise HTTPException(status_code=503, detail=f"Server overloaded. Max queue: {MAX_QUEUE_SIZE}")
    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {"status": "queued", "input_image_url": str(body.image), "original_local_path": None,
                       "processed_path": None, "error_message": None, "status_check_url": status_check_url}
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (JSON URL: {body.image}) enqueued. Queue: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    return {"status": "processing", "job_id": job_id, "image_links": [f"{public_url_base}/images/{job_id}.webp"],
            "eta": eta_seconds, "status_check_url": status_check_url}

@app.post("/submit_form")
async def submit_form_image_for_processing(request: Request, image_file: UploadFile = File(...), key: str = Form(...), model: str = Form("u2net")):
    if key != EXPECTED_API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    if ENABLE_LOGO_WATERMARK and os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo watermarking enabled, logo file exists, but not loaded. Check startup.")
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")
    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    original_fn = image_file.filename if image_file.filename else "upload"
    content_type = image_file.content_type.lower()
    extension = MIME_TO_EXT.get(content_type)
    if not extension:
        _, ext_fn = os.path.splitext(original_fn); ext_fn_lower = ext_fn.lower()
        if ext_fn_lower in MIME_TO_EXT.values(): extension = ext_fn_lower
        else: extension = ".png"; logger.warning(f"Job {job_id} (form): Unknown ext for '{original_fn}' from '{content_type}'. Defaulting to '{extension}'.")
    saved_fn = f"{job_id}_original{extension}"; original_path = os.path.join(UPLOADS_DIR, saved_fn)
    try:
        async with aiofiles.open(original_path, 'wb') as out_file:
            content = await image_file.read(); await out_file.write(content)
        logger.info(f"📝 Job {job_id} (Form: {original_fn}) Original saved: {original_path} ({format_size(len(content))})")
    except Exception as e:
        logger.error(f"Error saving upload {saved_fn} for job {job_id}: {e}"); raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    finally: await image_file.close()
    file_uri = f"file://{original_path}"
    try:
        queue.put_nowait((job_id, file_uri, model, True))
    except asyncio.QueueFull:
        logger.warning(f"Queue full. Rejecting form request for {original_fn} (job {job_id}).")
        if os.path.exists(original_path):
            try:
                os.remove(original_path)
            except OSError as e_clean:
                logger.error(f"Error cleaning {original_path} (queue full): {e_clean}")
        raise HTTPException(status_code=503, detail=f"Server overloaded. Max queue: {MAX_QUEUE_SIZE}")
    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {"status": "queued", "input_image_url": f"(form_upload: {original_fn})",
                       "original_local_path": original_path, "processed_path": None,
                       "error_message": None, "status_check_url": status_check_url}
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (Form: {original_fn}) enqueued. Queue: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    return {"status": "processing", "job_id": job_id, "original_image_url": f"{public_url_base}/originals/{saved_fn}",
            "image_links": [f"{public_url_base}/images/{job_id}.webp"], "eta": eta_seconds, "status_check_url": status_check_url}

@app.get("/api/monitoring/workers")
async def get_worker_monitoring_data(): return get_worker_activity_data()

@app.get("/api/monitoring/system")
async def get_system_monitoring_data(): return get_system_metrics_data()

@app.get("/api/debug/gpu")
async def debug_gpu_status():
    global active_rembg_providers
    result = {"pynvml_available": False, "gpu_detected_pynvml": False, "gpu_count_pynvml": 0,
              "error_pynvml": None, "current_metrics_pynvml": None,
              "onnxruntime_info": {"available": False, "providers": [], "error": None, "currently_configured_rembg_providers": active_rembg_providers}}
    try:
        import pynvml
        result["pynvml_available"] = True
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        result["gpu_count_pynvml"] = device_count
        if device_count > 0:
            result["gpu_detected_pynvml"] = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            device_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(device_name, bytes): device_name = device_name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result["current_metrics_pynvml"] = {
                "device_name": device_name, "memory_used_mb": mem_info.used // (1024**2),
                "memory_total_mb": mem_info.total // (1024**2), "memory_percent": (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0,
                "gpu_utilization_percent": util.gpu, "memory_utilization_percent": util.memory}
        # pynvml.nvmlShutdown() # Consider calling this at app shutdown
    except ImportError: result["error_pynvml"] = "pynvml not installed. pip install pynvml (for NVIDIA GPU stats)"
    except Exception as e: result["error_pynvml"] = f"{type(e).__name__}: {e} (pynvml error)"
    
    try:
        import onnxruntime as ort
        result["onnxruntime_info"]["available"] = True
        result["onnxruntime_info"]["providers"] = ort.get_available_providers()
    except ImportError: result["onnxruntime_info"]["error"] = "onnxruntime module not found. pip install onnxruntime or onnxruntime-gpu."
    except Exception as e: result["onnxruntime_info"]["error"] = f"Error getting ONNX Runtime info: {e}"
    return result

@app.get("/status/{job_id}")
async def check_job_status(request: Request, job_id: str):
    job_info = results.get(job_id)
    if not job_info:
        historical_job = next((job for job in job_history if job["job_id"] == job_id), None)
        if historical_job:
            public_url_base = get_proxy_url(request)
            status = "done" if historical_job["status"] == "completed" else "error"
            response_data = {"job_id": job_id, "status": status,
                             "input_image_url": f"(historical: {historical_job.get('original_filename', 'unknown')})",
                             "status_check_url": f"{public_url_base}/status/{job_id}"}
            original_extensions = ['.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
            for ext in original_extensions:
                original_filename = f"{job_id}_original{ext}"
                if os.path.exists(os.path.join(UPLOADS_DIR, original_filename)):
                    response_data["original_image_url"] = f"{public_url_base}/originals/{original_filename}"; break
            if status == "done":
                processed_filename = f"{job_id}.webp"
                if os.path.exists(os.path.join(PROCESSED_DIR, processed_filename)):
                    response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
            else: response_data["error_message"] = f"Job failed after {historical_job['total_time']:.2f}s"
            return JSONResponse(content=response_data)
        raise HTTPException(status_code=404, detail="Job not found")
    
    public_url_base = get_proxy_url(request)
    response_data = {"job_id": job_id, "status": job_info.get("status"),
                     "input_image_url": job_info.get("input_image_url"), 
                     "status_check_url": job_info.get("status_check_url")}
    if job_info.get("original_local_path"):
        response_data["original_image_url"] = f"{public_url_base}/originals/{os.path.basename(job_info['original_local_path'])}"
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        response_data["processed_image_url"] = f"{public_url_base}/images/{os.path.basename(job_info['processed_path'])}"
    elif job_info.get("status") == "error": response_data["error_message"] = job_info.get("error_message")
    return JSONResponse(content=response_data)

@app.get("/job/{job_id}")
async def job_details(request: Request, job_id: str):
    job_info = next((job for job in job_history if job["job_id"] == job_id), None)
    if not job_info and job_id in results:
        result = results[job_id]
        job_info = {"job_id": job_id, "timestamp": time.time(), "status": "active" if result.get("status") not in ["done", "error"] else result.get("status"),
                    "total_time": 0, "input_size": 0, "output_size": 0, "model": "unknown", "source_type": "unknown", "original_filename": ""}
    if not job_info: raise HTTPException(status_code=404, detail="Job not found")
    public_url_base = get_proxy_url(request)
    original_image_url, processed_image_url = None, None
    original_extensions = ['.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    for ext in original_extensions:
        original_filename = f"{job_id}_original{ext}"
        if os.path.exists(os.path.join(UPLOADS_DIR, original_filename)):
            original_image_url = f"{public_url_base}/originals/{original_filename}"; break
    processed_filename = f"{job_id}.webp"
    if os.path.exists(os.path.join(PROCESSED_DIR, processed_filename)):
        processed_image_url = f"{public_url_base}/images/{processed_filename}"
    result_details = results.get(job_id, {})
    # HTML content (same as provided, abbreviated for brevity)
    return HTMLResponse(content=f"""<!DOCTYPE html><html lang="en">
    <head><meta charset="UTF-8"><title>Job Details - {job_id[:8]}</title><style>body{{font-family:sans-serif;margin:20px;background-color:#f9f9f9;}}.container{{max-width:1200px;margin:0 auto;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}.header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;}}.status-badge{{padding:5px 10px;border-radius:15px;font-weight:bold;text-transform:uppercase;}}.status-completed{{background-color:#d4edda;color:#155724;}}.status-failed{{background-color:#f8d7da;color:#721c24;}}.status-active{{background-color:#d1ecf1;color:#0c5460;}}.details-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:20px 0;}}.detail-card{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:15px;}}.detail-label{{font-size:12px;color:#6c757d;text-transform:uppercase;margin-bottom:5px;}}.detail-value{{font-size:18px;font-weight:bold;color:#495057;}}.images-section{{margin-top:30px;}}.images-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;}}.image-card{{border:1px solid #dee2e6;border-radius:8px;padding:15px;background:white;}}.image-card h3{{margin-top:0;color:#495057;}}.image-card img{{max-width:100%;height:auto;border-radius:4px;border:1px solid #dee2e6;}}.no-image{{color:#6c757d;font-style:italic;text-align:center;padding:40px;background:#f8f9fa;border-radius:4px;}}.back-link{{color:#007bff;text-decoration:none;}}.back-link:hover{{text-decoration:underline;}}@media (max-width:768px){{.images-container{{grid-template-columns:1fr;}}.header{{flex-direction:column;align-items:flex-start;}}}}</style></head>
    <body><div class="container"><div class="header"><h1>Job Details</h1><a href="/" class="back-link">← Back to Dashboard</a></div>
    <div style="margin-bottom:20px;"><h2>Job ID: <code>{job_id}</code></h2><span class="status-badge status-{job_info['status'].lower()}">{job_info['status']}</span></div>
    <div class="details-grid"><div class="detail-card"><div class="detail-label">Processed Time</div><div class="detail-value">{format_timestamp(job_info['timestamp'])}</div></div><div class="detail-card"><div class="detail-label">Processing Duration</div><div class="detail-value">{job_info['total_time']:.2f}s</div></div><div class="detail-card"><div class="detail-label">Model Used</div><div class="detail-value">{job_info['model']}</div></div><div class="detail-card"><div class="detail-label">Source Type</div><div class="detail-value">{job_info['source_type'].title()}</div></div><div class="detail-card"><div class="detail-label">Input Size</div><div class="detail-value">{format_size(job_info['input_size'])}</div></div><div class="detail-card"><div class="detail-label">Output Size</div><div class="detail-value">{format_size(job_info['output_size']) if job_info['output_size']>0 else 'N/A'}</div></div></div>
    {f"<div class='detail-card'><div class='detail-label'>Original Filename</div><div class='detail-value'>{job_info['original_filename']}</div></div>" if job_info.get('original_filename') else ''}
    <div class="images-section"><h2>Before & After Images</h2><div class="images-container"><div class="image-card"><h3>🔍 Original Image</h3>{f'<img src="{original_image_url}" alt="Original Image" loading="lazy">' if original_image_url else '<div class="no-image">Original image not available</div>'}</div><div class="image-card"><h3>✨ Processed Image</h3>{f'<img src="{processed_image_url}" alt="Processed Image" loading="lazy">' if processed_image_url else '<div class="no-image">Processed image not available</div>'}</div></div></div>
    <div style="margin-top:30px;padding:15px;background:#f8f9fa;border-radius:6px;"><h3>Technical Details</h3><ul><li><strong>Current Status in System:</strong> {result_details.get('status','Not in active results')}</li><li><strong>Status Check URL:</strong> <a href="{result_details.get('status_check_url','#')}" target="_blank">API Status</a></li>{f"<li><strong>Error Message:</strong> {result_details.get('error_message','None')}</li>" if result_details.get('error_message') else ''}<li><strong>Job ID:</strong> <code>{job_id}</code></li></ul></div></div></body></html>""", status_code=200)

async def cleanup_old_results():
    while True:
        try:
            current_time = time.time(); expired_jobs = []
            for job_id, job_data in results.items():
                completion_time = job_data.get("completion_time")
                if completion_time and (current_time - completion_time) > 3600: expired_jobs.append(job_id)
            for job_id in expired_jobs: logger.info(f"Cleaning up old job from results: {job_id}"); del results[job_id]
            if expired_jobs: logger.info(f"Cleaned up {len(expired_jobs)} old jobs from results dict")
        except Exception as e: logger.error(f"Error in cleanup task: {e}", exc_info=True)
        await asyncio.sleep(600)

async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image
    while True:
        job_id, image_source_str, model_name, _ = await queue.get()
        t_job_start = time.perf_counter()
        logger.info(f"Worker {worker_id} picked up job {job_id} for source: {image_source_str}. Model: {model_name}")
        log_worker_activity(worker_id, WORKER_IDLE)
        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job {job_id} from queue not in results. Skipping."); queue.task_done(); continue
        input_bytes_for_rembg: bytes | None = None
        input_fetch_time, rembg_time, pil_time, save_time = 0.0, 0.0, 0.0, 0.0
        input_size_bytes, output_size_bytes = 0, 0
        try:
            log_worker_activity(worker_id, WORKER_FETCHING); t_input_fetch_start = time.perf_counter()
            if image_source_str.startswith("file://"):
                results[job_id]["status"] = "loading_file"; local_path = image_source_str[len("file://"):]
                if not os.path.exists(local_path): raise FileNotFoundError(f"Local file for job {job_id} not found: {local_path}")
                async with aiofiles.open(local_path, 'rb') as f: input_bytes_for_rembg = await f.read()
                input_size_bytes = len(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (W{worker_id}): Loaded local file ({format_size(input_size_bytes)})")
            elif image_source_str.startswith(("http://", "https://")):
                results[job_id]["status"] = "downloading"
                logger.info(f"Job {job_id} (W{worker_id}): Downloading from {image_source_str}...")
                async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                    img_response = await client.get(image_source_str); img_response.raise_for_status()
                input_bytes_for_rembg = await img_response.aread(); input_size_bytes = len(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (W{worker_id}): Downloaded {format_size(input_size_bytes)}")
                content_type = img_response.headers.get("content-type", "unknown").lower()
                if content_type == "application/octet-stream" or not content_type.startswith("image/"):
                    ext_url = os.path.splitext(urllib.parse.urlparse(image_source_str).path)[1].lower()
                    ct_map = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
                    if ext_url in ct_map: content_type = ct_map[ext_url]
                if not content_type.startswith("image/"): raise ValueError(f"Invalid final content type '{content_type}'. Not image.")
                extension = MIME_TO_EXT.get(content_type, ".bin")
                dl_original_path = os.path.join(UPLOADS_DIR, f"{job_id}_original_downloaded{extension}")
                results[job_id]["original_local_path"] = dl_original_path
                async with aiofiles.open(dl_original_path, 'wb') as out_file: await out_file.write(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (W{worker_id}): Saved downloaded original")
            else: raise ValueError(f"Unsupported image source: {image_source_str}")
            if input_bytes_for_rembg is None: raise ValueError(f"Image content is None for job {job_id}.")
            input_fetch_time = time.perf_counter() - t_input_fetch_start

            log_worker_activity(worker_id, WORKER_PROCESSING_REMBG); results[job_id]["status"] = "processing_rembg"
            logger.info(f"Job {job_id} (W{worker_id}): Starting rembg (model: {model_name}, providers: {active_rembg_providers})...")
            t_rembg_start = time.perf_counter(); loop = asyncio.get_event_loop()
            output_bytes_with_alpha = await loop.run_in_executor(cpu_executor, process_rembg_sync, input_bytes_for_rembg, model_name)
            rembg_time = time.perf_counter() - t_rembg_start
            logger.info(f"Job {job_id} (W{worker_id}): Rembg done in {rembg_time:.4f}s")

            log_worker_activity(worker_id, WORKER_PROCESSING_PIL); results[job_id]["status"] = "processing_image"
            logger.info(f"Job {job_id} (W{worker_id}): Starting PIL processing...")
            t_pil_start = time.perf_counter()
            processed_image_bytes = await loop.run_in_executor(pil_executor, process_pil_sync, output_bytes_with_alpha, TARGET_SIZE, prepared_logo_image, ENABLE_LOGO_WATERMARK, LOGO_MARGIN)
            pil_time = time.perf_counter() - t_pil_start
            logger.info(f"Job {job_id} (W{worker_id}): PIL done in {pil_time:.4f}s")

            log_worker_activity(worker_id, WORKER_SAVING); results[job_id]["status"] = "saving"
            processed_fn = f"{job_id}.webp"; processed_path = os.path.join(PROCESSED_DIR, processed_fn)
            t_save_start = time.perf_counter()
            async with aiofiles.open(processed_path, 'wb') as out_file: await out_file.write(processed_image_bytes)
            save_time = time.perf_counter() - t_save_start; output_size_bytes = len(processed_image_bytes)

            log_worker_activity(worker_id, WORKER_IDLE); results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_path
            total_job_time = time.perf_counter() - t_job_start
            source_type = "url" if image_source_str.startswith(("http:", "https:")) else "upload"
            original_fn_hist = results[job_id].get("input_image_url", "").split("/")[-1] if source_type == "url" else results[job_id].get("input_image_url", "").replace("(form_upload: ", "").replace(")", "")
            add_job_to_history(job_id, "completed", total_job_time, input_size_bytes, output_size_bytes, model_name, source_type, original_fn_hist)
            results[job_id]["completion_time"] = time.time()
            logger.info(f"Job {job_id} (W{worker_id}) COMPLETED in {total_job_time:.4f}s. Input: {format_size(input_size_bytes)} -> Output: {format_size(output_size_bytes)}. Breakdown: Fetch={input_fetch_time:.3f}s, Rembg={rembg_time:.3f}s, PIL={pil_time:.3f}s, Save={save_time:.3f}s")
        except FileNotFoundError as e: logger.error(f"Job {job_id} (W{worker_id}) Error: FileNotFoundError: {e}", exc_info=False); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"File not found: {e}"
        except httpx.HTTPStatusError as e: logger.error(f"Job {job_id} (W{worker_id}) Error: HTTPStatusError: {e.response.status_code}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Download failed: HTTP {e.response.status_code}"
        except httpx.RequestError as e: logger.error(f"Job {job_id} (W{worker_id}) Error: RequestError: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Network error: {type(e).__name__}"
        except (ValueError, IOError, OSError) as e: logger.error(f"Job {job_id} (W{worker_id}) Error: Data/file processing: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Processing error: {e}"
        except Exception as e: logger.critical(f"Job {job_id} (W{worker_id}) CRITICAL Error: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Unexpected error: {e}"
        finally:
            if results.get(job_id, {}).get("status") == "error":
                total_job_time_error = time.perf_counter() - t_job_start
                source_type = "url" if image_source_str.startswith(("http:", "https:")) else "upload"
                original_fn_hist = results[job_id].get("input_image_url", "").split("/")[-1] if source_type == "url" else results[job_id].get("input_image_url", "").replace("(form_upload: ", "").replace(")", "")
                add_job_to_history(job_id, "failed", total_job_time_error, input_size_bytes, 0, model_name, source_type, original_fn_hist)
                results[job_id]["completion_time"] = time.time()
                logger.info(f"Job {job_id} (W{worker_id}) FAILED after {total_job_time_error:.4f}s")
            log_worker_activity(worker_id, WORKER_IDLE); queue.task_done()

@app.on_event("startup")
async def startup_event():
    global prepared_logo_image, cpu_executor, pil_executor, active_rembg_providers
    logger.info("Application startup...")
    cpu_executor = ThreadPoolExecutor(max_workers=CPU_THREAD_POOL_SIZE, thread_name_prefix="RembgCPU")
    pil_executor = ThreadPoolExecutor(max_workers=PIL_THREAD_POOL_SIZE, thread_name_prefix="PILCPU")
    logger.info(f"Thread pools: CPU={CPU_THREAD_POOL_SIZE}, PIL={PIL_THREAD_POOL_SIZE}")

    if REMBG_USE_GPU:
        logger.info("REMBG_USE_GPU is True. Detecting ONNX Runtime providers...")
        try:
            import onnxruntime as ort
            available_ort_providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime available providers: {available_ort_providers}")
            chosen_providers = []
            for provider_name in REMBG_PREFERRED_GPU_PROVIDERS:
                if provider_name in available_ort_providers:
                    if provider_name not in chosen_providers: chosen_providers.append(provider_name)
            if 'CPUExecutionProvider' in chosen_providers: chosen_providers.remove('CPUExecutionProvider') # remove if added by mistake
            
            # Ensure CPUExecutionProvider is always at the end as a fallback
            if 'CPUExecutionProvider' in available_ort_providers:
                 if 'CPUExecutionProvider' not in chosen_providers: # Add if not already there
                    chosen_providers.append('CPUExecutionProvider')
            else: logger.error("'CPUExecutionProvider' not found in ONNX available_providers. This is unusual.")

            if not chosen_providers or (len(chosen_providers) == 1 and 'CPUExecutionProvider' in chosen_providers and not any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in chosen_providers)):
                logger.warning(f"Could not form valid ONNX provider list with preferred GPU providers. Falling back to CPU-only default. Configured Preferred: {REMBG_PREFERRED_GPU_PROVIDERS}, Available: {available_ort_providers}, Chosen: {chosen_providers}")
                active_rembg_providers = list(REMBG_CPU_PROVIDERS)
            else:
                active_rembg_providers = chosen_providers
            
            has_gpu_in_final = any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in active_rembg_providers if p != 'CPUExecutionProvider')
            if not has_gpu_in_final and any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in chosen_providers): # If GPU was available but not selected or CPU is the only one
                 logger.warning(f"REMBG_USE_GPU is True, but no preferred GPU providers were ultimately selected or only CPU remains. Final: {active_rembg_providers}. Check onnxruntime-gpu/drivers. Preferred: {REMBG_PREFERRED_GPU_PROVIDERS}, Available: {available_ort_providers}.")

        except ImportError: logger.warning("onnxruntime module not found. Rembg will use CPU. Install onnxruntime or onnxruntime-gpu."); active_rembg_providers = list(REMBG_CPU_PROVIDERS)
        except Exception as e: logger.error(f"Error detecting ONNX providers: {e}. Defaulting to CPU.", exc_info=True); active_rembg_providers = list(REMBG_CPU_PROVIDERS)
    else: logger.info("REMBG_USE_GPU is False. Using CPU providers for rembg."); active_rembg_providers = list(REMBG_CPU_PROVIDERS)
    logger.info(f"Rembg will attempt to use providers: {active_rembg_providers}")

    if ENABLE_LOGO_WATERMARK:
        logger.info(f"Logo watermarking ENABLED. Loading from: {LOGO_PATH}")
        if os.path.exists(LOGO_PATH):
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                if logo.width > LOGO_MAX_WIDTH:
                    l_ratio = LOGO_MAX_WIDTH / logo.width
                    logo = logo.resize((LOGO_MAX_WIDTH, int(logo.height * l_ratio)), Image.Resampling.LANCZOS)
                prepared_logo_image = logo; logger.info(f"Logo loaded: {prepared_logo_image.size}")
            except Exception as e: logger.error(f"Failed to load logo: {e}", exc_info=True); prepared_logo_image = None
        else: logger.warning(f"Logo file not found at {LOGO_PATH}."); prepared_logo_image = None
    else: logger.info("Logo watermarking DISABLED."); prepared_logo_image = None

    for i in range(MAX_CONCURRENT_TASKS): asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} async workers started.")
    asyncio.create_task(cleanup_old_results()); logger.info("Background cleanup task started.")
    asyncio.create_task(system_monitor()); logger.info("System monitoring task started.")
    # Initialize pynvml once for monitoring if available
    try:
        import pynvml
        pynvml.nvmlInit()
        logger.info("pynvml initialized for GPU monitoring.")
    except Exception:
        logger.info("pynvml could not be initialized (normal if no NVIDIA GPU or pynvml not installed).")


@app.on_event("shutdown")
async def shutdown_event():
    global cpu_executor, pil_executor
    logger.info("Application shutdown...")
    if cpu_executor: cpu_executor.shutdown(wait=True); logger.info("CPU thread pool shut down")
    if pil_executor: pil_executor.shutdown(wait=True); logger.info("PIL thread pool shut down")
    try:
        import pynvml
        pynvml.nvmlShutdown()
        logger.info("pynvml shutdown.")
    except Exception:
        pass # Ignore if not initialized or not available

app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    stats = get_server_stats()
    logo_status = "Enabled" if ENABLE_LOGO_WATERMARK else "Disabled"
    if ENABLE_LOGO_WATERMARK: logo_status += f" (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})" if prepared_logo_image else " (Enabled but not loaded/found)"
    uptime_hours = stats["uptime"] / 3600
    uptime_str = f"{uptime_hours:.1f} hours" if uptime_hours >= 1 else f"{stats['uptime']:.0f} seconds"
    current_metrics = system_metrics[-1] if system_metrics else {"cpu_percent":0,"memory_percent":0,"memory_used_gb":0,"memory_total_gb":0,"gpu_used_mb":0,"gpu_total_mb":0,"gpu_utilization":0}
    recent_jobs_html = "<h3>Recent Jobs</h3>"
    if stats["recent_jobs"]:
        recent_jobs_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse:collapse;width:100%;'><tr style='background-color:#f0f0f0;'><th>Time</th><th>Job ID</th><th>Status</th><th>Duration</th><th>Input</th><th>Output</th><th>Model</th><th>Source</th></tr>"
        for job in stats["recent_jobs"][:20]:
            status_color = "#4CAF50" if job["status"] == "completed" else "#f44336"; job_link = f"/job/{job['job_id']}"
            recent_jobs_html += f"""<tr style="cursor:pointer;" onclick="window.location.href='{job_link}'"><td>{format_timestamp(job['timestamp'])}</td><td style='font-family:monospace;font-size:10px;'><a href="{job_link}" style="text-decoration:none;color:#007bff;">{job['job_id'][:8]}...</a></td><td style='color:{status_color};font-weight:bold;'>{job['status'].upper()}</td><td>{job['total_time']:.2f}s</td><td>{format_size(job['input_size'])}</td><td>{format_size(job['output_size']) if job['output_size']>0 else 'N/A'}</td><td>{job['model']}</td><td>{job['source_type']}</td></tr>"""
        recent_jobs_html += "</table>"
    else: recent_jobs_html += "<p>No jobs processed yet.</p>"
    
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Image API Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body{{font-family:sans-serif;margin:20px; background-color: #f9f9f9;}} 
        .container{{max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
        .stats-grid{{display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;}}
        .stat-card{{background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; text-align: center;}}
        .stat-value{{font-size: 24px; font-weight: bold; color: #007bff; margin-bottom: 5px;}}
        .stat-label{{font-size: 14px; color: #6c757d; text-transform: uppercase;}}
        .monitoring-section{{margin: 30px 0;}}
        .charts-container{{display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;}}
        .chart-card{{background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;}}
        .chart-title{{font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057;}}
        .chart-container{{position: relative; height: 300px;}}
        table{{font-size: 14px;}} 
        th{{background-color: #f0f0f0 !important;}}
        tr:hover{{background-color: #f8f9fa; cursor: pointer;}}
        .job-link{{color: #007bff; text-decoration: none;}}
        .job-link:hover{{text-decoration: underline;}}
        li{{margin-bottom: 5px;}}
        .status-good{{color: #28a745;}}
        .status-warning{{color: #ffc107;}}
        .status-error{{color: #dc3545;}}
        .system-metrics{{display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;}}
        .metric-card{{background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 6px; padding: 15px;}}
        .metric-value{{font-size: 20px; font-weight: bold; margin-bottom: 5px;}}
        .metric-label{{font-size: 12px; color: #6c757d; text-transform: uppercase;}}
        @media (max-width: 1200px) {{
            .charts-container {{ grid-template-columns: 1fr; }}
        }}
    </style>
    </head>
    <body>
    <div class="container">
        <h1>🚀 Threaded Image Processing API Dashboard</h1>
        <p><strong>Status:</strong> <span class="status-good">RUNNING</span> | Background removal uses true async processing with thread pools for CPU-bound operations.</p>
        
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

        <div class="monitoring-section">
            <h2>📊 Real-time Monitoring</h2>
            
            <div class="system-metrics">
                <div class="metric-card">
                    <div class="metric-value" style="color: #dc3545;">{current_metrics['cpu_percent']:.1f}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #fd7e14;">{current_metrics['memory_percent']:.1f}%</div>
                    <div class="metric-label">Memory Usage ({current_metrics['memory_used_gb']:.1f}GB / {current_metrics['memory_total_gb']:.1f}GB)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #6f42c1;">{current_metrics['gpu_utilization']:.0f}%</div>
                    <div class="metric-label">GPU Usage ({current_metrics['gpu_used_mb']:.0f}MB / {current_metrics['gpu_total_mb']:.0f}MB)</div>
                </div>
            </div>
            
            <div class="charts-container">
                <div class="chart-card">
                    <div class="chart-title">🔧 Worker Thread Activity</div>
                    <div class="chart-container">
                        <canvas id="workerChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">💻 System Resources</div>
                    <div class="chart-container">
                        <canvas id="systemChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <h3>Configuration</h3>
        <ul>
            <li><strong>Async Workers:</strong> {MAX_CONCURRENT_TASKS}</li>
            <li><strong>CPU Thread Pool:</strong> {CPU_THREAD_POOL_SIZE}</li>
            <li><strong>PIL Thread Pool:</strong> {PIL_THREAD_POOL_SIZE}</li>
            <li><strong>Queue Capacity:</strong> {MAX_QUEUE_SIZE}</li>
            <li><strong>Logo Watermarking:</strong> {logo_status}</li>
            <li><strong>Rembg GPU Attempt:</strong> {'Enabled' if REMBG_USE_GPU else 'Disabled'}</li>
            <li><strong>Rembg Providers:</strong> {str(active_rembg_providers)}</li>
            <li><strong>GPU Monitoring (pynvml):</strong> {current_metrics['gpu_total_mb']} MB total {'(Active)' if current_metrics['gpu_total_mb'] > 0 else '(Not detected/NVIDIA pynvml)'}</li>
        </ul>
        
        <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 6px;">
            <h4>🔧 Debug Info</h4>
            <p><strong>GPU Debug:</strong> <a href="/api/debug/gpu" target="_blank">Check GPU/ONNXRT Detection Status</a></p>
            <p><strong>Worker Activity:</strong> <a href="/api/monitoring/workers" target="_blank">View Raw Worker Data</a></p>
            <p><strong>System Metrics:</strong> <a href="/api/monitoring/system" target="_blank">View Raw System Data</a></p>
        </div>

        {recent_jobs_html}
        
        <p style="margin-top: 30px; font-size: 12px; color: #6c757d;">
            Page auto-refreshes every 30 seconds | Last updated: {format_timestamp(time.time())}
        </p>
    </div>
    
    <script>
        // Chart colors for workers
        const workerColors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF' // Re-used some for more than 6 workers
        ];
        
        let workerChart, systemChart;

        // Initialize charts
        function initCharts() {{
            // Worker Activity Chart
            const workerCtx = document.getElementById('workerChart').getContext('2d');
            workerChart = new Chart(workerCtx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: []
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                boxWidth: 12,
                                // fontSize: 10 // Chart.js v3 uses `font.size`
                            }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            stacked: true, // Good for showing composition of worker activity
                            title: {{
                                display: true,
                                text: 'Active Workers / Activity Count'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time'
                            }},
                            ticks: {{
                                autoSkip: true,
                                maxTicksLimit: 15 // Adjust for readability
                            }}
                        }}
                    }},
                    elements: {{
                        line: {{
                            tension: 0.4 // Smoother lines
                        }},
                        point: {{
                            radius: 2
                        }}
                    }}
                }}
            }});

            // System Resources Chart
            const systemCtx = document.getElementById('systemChart').getContext('2d');
            systemChart = new Chart(systemCtx, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [
                        {{
                            label: 'CPU %',
                            data: [],
                            borderColor: '#dc3545', // Red
                            backgroundColor: 'rgba(220, 53, 69, 0.1)',
                            fill: false,
                            yAxisID: 'yPercent' // Assign to a Y-axis
                        }},
                        {{
                            label: 'Memory %',
                            data: [],
                            borderColor: '#fd7e14', // Orange
                            backgroundColor: 'rgba(253, 126, 20, 0.1)',
                            fill: false,
                            yAxisID: 'yPercent' // Assign to the same Y-axis
                        }},
                        {{
                            label: 'GPU %',
                            data: [],
                            borderColor: '#6f42c1', // Purple
                            backgroundColor: 'rgba(111, 66, 193, 0.1)',
                            fill: false,
                            yAxisID: 'yPercent' // Assign to the same Y-axis
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        yPercent: {{ // Define the Y-axis for percentages
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true,
                            max: 100,
                            title: {{
                                display: true,
                                text: 'Usage %'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time'
                            }},
                            ticks: {{
                                autoSkip: true,
                                maxTicksLimit: 15 // Adjust for readability
                            }}
                        }}
                    }},
                    elements: {{
                        line: {{
                            tension: 0.4 // Smoother lines
                        }},
                        point: {{
                            radius: 1
                        }}
                    }}
                }}
            }});
        }}

        // Update charts with new data
        async function updateCharts() {{
            try {{
                // Fetch worker data
                const workerResponse = await fetch('/api/monitoring/workers');
                if (!workerResponse.ok) {{
                    console.error("Failed to fetch worker data:", workerResponse.status);
                    return;
                }}
                const workerData = await workerResponse.json();
                
                // Fetch system data
                const systemResponse = await fetch('/api/monitoring/system');
                 if (!systemResponse.ok) {{
                    console.error("Failed to fetch system data:", systemResponse.status);
                    return;
                }}
                const systemData = await systemResponse.json();
                
                // Update worker chart
                updateWorkerChart(workerData);
                
                // Update system chart
                updateSystemChart(systemData);
                
            }} catch (error) {{
                console.error('Error updating charts:', error);
            }}
        }}

        function formatChartTimestamp(unixTimestamp) {{
            const date = new Date(unixTimestamp * 1000);
            return date.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
        }}

        function updateWorkerChart(data) {{
            if (!workerChart || typeof data !== 'object' || Object.keys(data).length === 0) {{
                // console.warn("Worker chart not ready or data is invalid/empty.");
                return;
            }}
            
            const workerIds = Object.keys(data).sort();
            const firstWorkerData = data[workerIds[0]];

            if (!Array.isArray(firstWorkerData) || firstWorkerData.length === 0) {{
                // console.warn("First worker data is invalid or empty for labels.");
                 workerChart.data.labels = [];
                 workerChart.data.datasets = [];
                 workerChart.update('none');
                return;
            }}
            
            const labels = firstWorkerData.map(bucket => formatChartTimestamp(bucket.timestamp));
            
            const datasets = workerIds.map((workerId, index) => {{
                const workerBuckets = data[workerId] || []; // Ensure it's an array
                // Summing up activities for simplicity, could be stacked bars for details
                const totalActivity = workerBuckets.map(bucket => 
                    (bucket.fetching || 0) + (bucket.rembg || 0) + (bucket.pil || 0) + (bucket.saving || 0)
                );
                
                return {{
                    label: workerId.replace('worker_', 'Worker '),
                    data: totalActivity,
                    borderColor: workerColors[index % workerColors.length],
                    backgroundColor: workerColors[index % workerColors.length] + '33', // More transparent fill
                    fill: true, // Can set to true for area chart style
                    tension: 0.4
                }};
            }});
            
            workerChart.data.labels = labels;
            workerChart.data.datasets = datasets;
            workerChart.update('none'); // 'none' for no animation, quicker updates
        }}

        function updateSystemChart(data) {{
            if (!systemChart || !Array.isArray(data) || data.length === 0) {{
                // console.warn("System chart not ready or data is invalid/empty.");
                return;
            }}
            
            const labels = data.map(metric => formatChartTimestamp(metric.timestamp));
            
            const cpuData = data.map(metric => metric.cpu_percent);
            const memoryData = data.map(metric => metric.memory_percent);
            const gpuData = data.map(metric => metric.gpu_utilization || 0); // Default to 0 if undefined
            
            systemChart.data.labels = labels;
            systemChart.data.datasets[0].data = cpuData;
            systemChart.data.datasets[1].data = memoryData;
            systemChart.data.datasets[2].data = gpuData;
            systemChart.update('none'); // 'none' for no animation
        }}

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initCharts();
            updateCharts(); // Initial fetch
            
            // Update charts every 10 seconds
            setInterval(updateCharts, {MONITORING_SAMPLE_INTERVAL * 2 * 1000}); // Update slightly less frequently than data collection
        }});

        // Page refresh function
        function refreshPage() {{
            // Only refresh if the page is visible to avoid unnecessary reloads
            if (document.visibilityState === 'visible') {{
                // console.log("Refreshing page content...");
                // Instead of full reload, one could fetch stats and update specific DOM elements
                // For now, keep it simple with a full reload if desired
                // location.reload(); 
                
                // Or, to just re-fetch and update stats without full page reload:
                 fetch('/')
                    .then(response => response.text())
                    .then(html => {{
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        // Update specific parts, e.g., stats grid, recent jobs table
                        const newStatsGrid = doc.querySelector('.stats-grid');
                        const currentStatsGrid = document.querySelector('.stats-grid');
                        if (newStatsGrid && currentStatsGrid) {{
                            currentStatsGrid.innerHTML = newStatsGrid.innerHTML;
                        }}
                        // Add similar updates for other dynamic parts like recent_jobs_html
                        const newRecentJobs = doc.querySelector('.monitoring-section + h3 + table, .monitoring-section + h3 + p');
                        const currentRecentJobsContainer = document.querySelector('.monitoring-section').nextElementSibling; //h3
                        if(currentRecentJobsContainer && currentRecentJobsContainer.nextElementSibling){
                            let currentJobsDisplay = currentRecentJobsContainer.nextElementSibling;
                             if (newRecentJobs && currentJobsDisplay) {{
                                currentJobsDisplay.outerHTML = newRecentJobs.outerHTML;
                            }}
                        }
                        
                        // Update "Last updated" timestamp
                        const lastUpdatedP = document.querySelector('p[style*="font-size: 12px"]');
                        if(lastUpdatedP) {{
                             lastUpdatedP.innerHTML = `Page data refreshed: {format_timestamp(time.time())} | Auto-refresh active`;
                        }}
                    }})
                    .catch(err => console.error("Error refreshing page content:", err));
            }}
        }}
        
        // Auto refresh page content (not full reload) every 30 seconds
        setInterval(refreshPage, 30000); 
    </script>
    </body></html>"""


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
