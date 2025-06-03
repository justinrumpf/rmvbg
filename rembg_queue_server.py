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

# --- CREATE DIRECTORIES AT THE VERY TOP X99---
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
ESTIMATED_TIME_PER_JOB = 15
TARGET_SIZE = 1024
HTTP_CLIENT_TIMEOUT = 30.0
DEFAULT_MODEL_NAME = "birefnet"

# --- GPU Configuration for Rembg ---
REMBG_USE_GPU = True
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
active_rembg_providers: list[str] = list(REMBG_CPU_PROVIDERS)

server_start_time = time.time()
job_history = []
total_jobs_completed = 0
total_jobs_failed = 0
total_processing_time = 0.0
MAX_HISTORY_ITEMS = 1000 # MODIFIED: Increased history items

worker_activity = defaultdict(deque)
system_metrics = deque(maxlen=MAX_MONITORING_SAMPLES)
worker_lock = threading.Lock()

# --- IP Traffic Statistics ---
ip_traffic_stats = defaultdict(lambda: {
    "requests": 0,
    "total_input_bytes": 0,
    "total_output_bytes": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "last_seen": 0.0
})
ip_traffic_lock = threading.Lock()

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
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        gpu_data["gpu_used_mb"] = mem_info.used // (1024**2)
        gpu_data["gpu_total_mb"] = mem_info.total // (1024**2)
        gpu_data["gpu_utilization"] = utilization.gpu

        if not hasattr(get_gpu_info, '_logged_count'):
            get_gpu_info._logged_count = 0
        if get_gpu_info._logged_count < 3:
            logger.debug(f"GPU Monitor (pynvml): GPU {utilization.gpu}% | Memory {gpu_data['gpu_used_mb']}/{gpu_data['gpu_total_mb']} MB")
            get_gpu_info._logged_count += 1

    except ImportError:
        if not hasattr(get_gpu_info, '_import_warned'):
            logger.warning("GPU monitoring via pynvml disabled: pynvml not installed (pip install pynvml). This is mainly for NVIDIA GPUs.")
            get_gpu_info._import_warned = True
    except Exception as e:
        if not hasattr(get_gpu_info, '_error_warned'):
            logger.warning(f"GPU monitoring via pynvml failed: {type(e).__name__}: {e}. This might happen if no NVIDIA GPU is present or drivers are missing.")
            get_gpu_info._error_warned = True
    finally:
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
    image: HttpUrl
    key: str
    model: str = DEFAULT_MODEL_NAME
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"

def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

def format_size(num_bytes: int) -> str:
    if num_bytes < 0: return "N/A"
    if num_bytes < 1024: return f"{num_bytes} B"
    elif num_bytes < 1024**2: return f"{num_bytes/1024:.2f} KB"
    else: return f"{num_bytes/1024**2:.2f} MB"

# MODIFIED: Added requester_ip parameter
def add_job_to_history(job_id: str, status: str, total_time: float, input_size: int, output_size: int, model: str, source_type: str = "unknown", original_filename: str = "", requester_ip: str = "unknown"):
    global job_history, total_jobs_completed, total_jobs_failed, total_processing_time
    job_record = {
        "job_id": job_id, "timestamp": time.time(), "status": status, "total_time": total_time,
        "input_size": input_size, "output_size": output_size, "model": model,
        "source_type": source_type, "original_filename": original_filename,
        "requester_ip": requester_ip # MODIFIED: Storing IP
    }
    job_history.insert(0, job_record)
    if len(job_history) > MAX_HISTORY_ITEMS: job_history.pop()
    if status == "completed": total_jobs_completed += 1; total_processing_time += total_time
    else: total_jobs_failed += 1

def format_timestamp(timestamp: float) -> str:
    if timestamp == 0.0: return "Never"
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

def get_requester_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(',')[0].strip()
        # logger.debug(f"Derived client IP from X-Forwarded-For: {client_ip} (Full header: '{x_forwarded_for}')")
        return client_ip
    if request.client and request.client.host:
        client_ip = request.client.host
        # logger.debug(f"Using client IP from request.client.host: {client_ip}")
        return client_ip
    # logger.warning("Could not determine client IP. Using 'unknown_client'.")
    return "unknown_client"

def process_rembg_sync(input_bytes: bytes, model_name: str) -> bytes:
    global active_rembg_providers
    session_wrapper = None
    providers_to_attempt = list(active_rembg_providers)
    try:
        # logger.info(f"Rembg: Attempting to initialize session for model '{model_name}' with providers: {providers_to_attempt}")
        session_wrapper = new_session(model_name, providers=providers_to_attempt)
        if session_wrapper is None:
            err_msg_session_none = f"CRITICAL: rembg.new_session returned None for model '{model_name}' with providers {providers_to_attempt}. This indicates a failure in session creation."
            logger.critical(err_msg_session_none)
            raise RuntimeError(err_msg_session_none)
        # logger.debug(f"Rembg: Successfully called new_session. Type of session_wrapper object: {type(session_wrapper)}")
        onnx_inference_session = None
        if hasattr(session_wrapper, 'inner_session'):
            onnx_inference_session = session_wrapper.inner_session
            # logger.debug("Rembg: Accessed 'inner_session' from rembg session wrapper.")
        elif hasattr(session_wrapper, 'sess'):
            onnx_inference_session = session_wrapper.sess
            # logger.debug("Rembg: Accessed 'sess' from rembg session wrapper.")
        else:
            logger.warning(
                f"Rembg: Could not find 'inner_session' or 'sess' attribute on rembg session wrapper (type: {type(session_wrapper)}). "
                "Attempting to treat the wrapper itself as the ONNX session for get_providers(). This might fail."
            )
            onnx_inference_session = session_wrapper
        if onnx_inference_session is None:
            err_msg_no_onnx_session = (
                f"Rembg: Failed to retrieve the underlying ONNX InferenceSession from the rembg session wrapper "
                f"(type: {type(session_wrapper)}) for model '{model_name}'. Cannot verify providers."
            )
            logger.error(err_msg_no_onnx_session)
            actual_session_providers = ["Error:CouldNotAccessONNXSession"]
        else:
            # logger.debug(f"Rembg: Object being used for get_providers(): {type(onnx_inference_session)}")
            actual_session_providers = []
            try:
                actual_session_providers = onnx_inference_session.get_providers()
                if not actual_session_providers:
                     logger.warning(f"Rembg: onnx_inference_session.get_providers() returned an empty list for model '{model_name}'.")
                     actual_session_providers = ["Error:GetProvidersReturnedEmpty"]
            except AttributeError:
                logger.error(
                    f"Rembg: The object (type: {type(onnx_inference_session)}) used for provider checking "
                    f"does NOT have 'get_providers()' method. Intended rembg wrapper type: {type(session_wrapper)}."
                )
                actual_session_providers = ["Error:GetProvidersMethodMissingOnObject"]
            except Exception as e_get_providers:
                logger.error(
                    f"Rembg: Error calling get_providers() on object (type: {type(onnx_inference_session)}) for model '{model_name}': {type(e_get_providers).__name__}: {e_get_providers}."
                )
                actual_session_providers = [f"Error:GetProvidersCallFailed_{type(e_get_providers).__name__}"]
        # logger.info(f"Rembg: Session for model '{model_name}'. Intended providers: {providers_to_attempt}, Actual providers reported by session: {actual_session_providers}")
        if REMBG_USE_GPU:
            if not providers_to_attempt or not any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in providers_to_attempt):
                logger.critical(
                    f"CRITICAL LOGIC FLAW: REMBG_USE_GPU is True, but providers_to_attempt ({providers_to_attempt}) "
                    f"does not reflect a GPU intention. Check startup provider configuration."
                )
            is_any_preferred_gpu_in_actual = any(p in actual_session_providers for p in REMBG_PREFERRED_GPU_PROVIDERS)
            is_cpu_in_actual = 'CPUExecutionProvider' in actual_session_providers
            if any("Error:" in p for p in actual_session_providers):
                err_msg = (
                    f"FORCED GPU FAILED (Provider Detection Issue): Rembg session for model '{model_name}'. "
                    f"Could not reliably determine actual providers (reported: {actual_session_providers}). "
                    f"Intended providers were {providers_to_attempt}. Cannot confirm GPU usage."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            if is_cpu_in_actual and not is_any_preferred_gpu_in_actual:
                err_msg = (
                    f"FORCED GPU FAILED (CPU Fallback): Rembg session for model '{model_name}' is confirmed to be using CPUExecutionProvider "
                    f"(actual: {actual_session_providers}) and NO preferred GPU provider is active, despite GPU being intended with {providers_to_attempt}."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            elif not is_any_preferred_gpu_in_actual and any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in providers_to_attempt):
                err_msg = (
                    f"FORCED GPU FAILED (No Preferred GPU Active): Rembg session for model '{model_name}' did not activate any of the "
                    f"intended preferred GPU providers ({providers_to_attempt}). Actual providers reported by session: {actual_session_providers}."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            elif is_cpu_in_actual and is_any_preferred_gpu_in_actual:
                 logger.info(
                     f"Rembg: A preferred GPU provider is active in session ({actual_session_providers}), "
                     f"and CPUExecutionProvider is also present. This is typical. Intended: {providers_to_attempt}."
                 )
            elif is_any_preferred_gpu_in_actual:
                logger.info(f"Rembg: Successfully using a preferred GPU provider. Actual: {actual_session_providers}, Intended: {providers_to_attempt}")
            else:
                 logger.warning(
                    f"Rembg: No preferred GPU provider was intended by 'providers_to_attempt' ({providers_to_attempt}) or "
                    f"none are active in session. Actual providers: {actual_session_providers}. "
                    "If REMBG_USE_GPU is True, this state likely indicates a startup misconfiguration of providers, "
                    "or the ONNX session did not initialize with any of the preferred GPU providers successfully."
                )
    except Exception as e:
        log_message = (
            f"CRITICAL: Failed to initialize or verify rembg session for model '{model_name}' with "
            f"intended providers {providers_to_attempt}. Error: {type(e).__name__}: {e}. "
        )
        if REMBG_USE_GPU and "FORCED GPU FAILED" in str(e):
            log_message += "NO FALLBACK TO CPU. This job will fail as per 'force GPU' policy."
        elif REMBG_USE_GPU:
             log_message += "REMBG_USE_GPU was True. An error occurred before or during provider verification. NO FALLBACK TO CPU. This job will fail."
        else:
            log_message += "REMBG_USE_GPU was False. Error occurred during CPU or configured provider processing."
        logger.critical(log_message, exc_info=True)
        raise
    output_bytes = remove(
        input_bytes,
        session=session_wrapper,
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

    requester_ip = get_requester_ip(request)

    with ip_traffic_lock:
        ip_traffic_stats[requester_ip]["requests"] += 1
        ip_traffic_stats[requester_ip]["last_seen"] = time.time()

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    model_to_use = body.model if body.model else DEFAULT_MODEL_NAME
    try:
        queue.put_nowait((job_id, str(body.image), model_to_use, True, requester_ip))
    except asyncio.QueueFull:
        logger.warning(f"Queue full for IP {requester_ip}. Rejecting JSON request for {body.image}.")
        raise HTTPException(status_code=503, detail=f"Server overloaded. Max queue: {MAX_QUEUE_SIZE}")

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {"status": "queued", "input_image_url": str(body.image), "original_local_path": None,
                       "processed_path": None, "error_message": None, "status_check_url": status_check_url, "requester_ip": requester_ip}
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (JSON URL: {body.image}, Model: {model_to_use}, IP: {requester_ip}) enqueued. Queue: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    return {"status": "processing", "job_id": job_id, "image_links": [f"{public_url_base}/images/{job_id}.webp"],
            "eta": eta_seconds, "status_check_url": status_check_url}

@app.post("/submit_form")
async def submit_form_image_for_processing(request: Request, image_file: UploadFile = File(...), key: str = Form(...), model: str = Form(DEFAULT_MODEL_NAME)):
    if key != EXPECTED_API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    if ENABLE_LOGO_WATERMARK and os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo watermarking enabled, logo file exists, but not loaded. Check startup.")
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")

    requester_ip = get_requester_ip(request)

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
    file_content_length = 0
    try:
        content = await image_file.read()
        file_content_length = len(content)
        async with aiofiles.open(original_path, 'wb') as out_file:
            await out_file.write(content)
        logger.info(f"📝 Job {job_id} (Form: {original_fn}, IP: {requester_ip}) Original saved: {original_path} ({format_size(file_content_length)})")
    except Exception as e:
        logger.error(f"Error saving upload {saved_fn} for job {job_id} from IP {requester_ip}: {e}"); raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    finally:
        await image_file.close()

    with ip_traffic_lock:
        ip_traffic_stats[requester_ip]["requests"] += 1
        ip_traffic_stats[requester_ip]["total_input_bytes"] += file_content_length
        ip_traffic_stats[requester_ip]["last_seen"] = time.time()

    file_uri = f"file://{original_path}"
    model_to_use = model if model else DEFAULT_MODEL_NAME

    try:
        queue.put_nowait((job_id, file_uri, model_to_use, True, requester_ip))
    except asyncio.QueueFull:
        logger.warning(f"Queue full for IP {requester_ip}. Rejecting form request for {original_fn} (job {job_id}).")
        if os.path.exists(original_path):
            try: os.remove(original_path)
            except OSError as e_clean: logger.error(f"Error cleaning {original_path} (queue full): {e_clean}")
        raise HTTPException(status_code=503, detail=f"Server overloaded. Max queue: {MAX_QUEUE_SIZE}")

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {"status": "queued", "input_image_url": f"(form_upload: {original_fn})",
                       "original_local_path": original_path, "processed_path": None,
                       "error_message": None, "status_check_url": status_check_url, "requester_ip": requester_ip}
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    logger.info(f"Job {job_id} (Form: {original_fn}, Model: {model_to_use}, IP: {requester_ip}) enqueued. Queue: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
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
              "onnxruntime_info": {"available": False, "providers": [], "error": None,
                                   "rembg_use_gpu_config": REMBG_USE_GPU,
                                   "rembg_preferred_gpu_providers_config": REMBG_PREFERRED_GPU_PROVIDERS,
                                   "currently_active_rembg_providers_for_workers": active_rembg_providers}}
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
                             "status_check_url": f"{public_url_base}/status/{job_id}",
                             "requester_ip": historical_job.get("requester_ip", "unknown")}
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
                     "status_check_url": job_info.get("status_check_url"),
                     "requester_ip": job_info.get("requester_ip", "unknown")}
    if job_info.get("original_local_path"):
        response_data["original_image_url"] = f"{public_url_base}/originals/{os.path.basename(job_info['original_local_path'])}"
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        response_data["processed_image_url"] = f"{public_url_base}/images/{os.path.basename(job_info['processed_path'])}"
    elif job_info.get("status") == "error": response_data["error_message"] = job_info.get("error_message")
    return JSONResponse(content=response_data)

@app.get("/job/{job_id}")
async def job_details(request: Request, job_id: str):
    job_info_hist = next((job for job in job_history if job["job_id"] == job_id), None)
    result_details = results.get(job_id, {})

    display_job_info = {}
    if job_info_hist:
        display_job_info = job_info_hist.copy()
    elif result_details:
        display_job_info = {
            "job_id": job_id,
            "timestamp": time.time(),
            "status": result_details.get("status", "unknown"),
            "total_time": 0, "input_size": 0, "output_size": 0,
            "model": "N/A (active)", "source_type": "N/A (active)",
            "original_filename": result_details.get("input_image_url", "N/A (active)").split('/')[-1],
            "requester_ip": result_details.get("requester_ip", "N/A (active)")
        }
    else:
        raise HTTPException(status_code=404, detail="Job not found in active results or history")

    public_url_base = get_proxy_url(request)
    original_image_url, processed_image_url = None, None

    if result_details.get("original_local_path"):
         original_image_url = f"{public_url_base}/originals/{os.path.basename(result_details['original_local_path'])}"
    elif display_job_info.get("source_type") == "upload":
        original_fn_guess = display_job_info.get('original_filename', '')
        original_extensions = ['.webp', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        found_orig = False
        for ext in original_extensions:
            potential_orig_fn = f"{job_id}_original{ext}"
            if os.path.exists(os.path.join(UPLOADS_DIR, potential_orig_fn)):
                original_image_url = f"{public_url_base}/originals/{potential_orig_fn}"
                found_orig = True
                break
        if not found_orig and original_fn_guess:
             pass

    processed_filename = f"{job_id}.webp"
    if os.path.exists(os.path.join(PROCESSED_DIR, processed_filename)):
        processed_image_url = f"{public_url_base}/images/{processed_filename}"
    elif result_details.get("processed_path"):
        processed_image_url = f"{public_url_base}/images/{os.path.basename(result_details['processed_path'])}"


    return HTMLResponse(content=f"""<!DOCTYPE html><html lang="en">
    <head><meta charset="UTF-8"><title>Job Details - {job_id[:8]}</title><style>body{{font-family:sans-serif;margin:20px;background-color:#f9f9f9;}}.container{{max-width:1200px;margin:0 auto;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}.header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;}}.status-badge{{padding:5px 10px;border-radius:15px;font-weight:bold;text-transform:uppercase;}}.status-completed{{background-color:#d4edda;color:#155724;}}.status-failed,.status-error{{background-color:#f8d7da;color:#721c24;}}.status-active,.status-queued,.status-processing_rembg,.status-processing_image,.status-saving,.status-loading_file,.status-downloading{{background-color:#d1ecf1;color:#0c5460;}}.details-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:20px 0;}}.detail-card{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:15px;}}.detail-label{{font-size:12px;color:#6c757d;text-transform:uppercase;margin-bottom:5px;}}.detail-value{{font-size:18px;font-weight:bold;color:#495057;}}.images-section{{margin-top:30px;}}.images-container{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;}}.image-card{{border:1px solid #dee2e6;border-radius:8px;padding:15px;background:white;}}.image-card h3{{margin-top:0;color:#495057;}}.image-card img{{max-width:100%;height:auto;border-radius:4px;border:1px solid #dee2e6;}}.no-image{{color:#6c757d;font-style:italic;text-align:center;padding:40px;background:#f8f9fa;border-radius:4px;}}.back-link{{color:#007bff;text-decoration:none;}}.back-link:hover{{text-decoration:underline;}}@media (max-width:768px){{.images-container{{grid-template-columns:1fr;}}.header{{flex-direction:column;align-items:flex-start;}}}}</style></head>
    <body><div class="container"><div class="header"><h1>Job Details</h1><a href="/" class="back-link">← Back to Dashboard</a></div>
    <div style="margin-bottom:20px;"><h2>Job ID: <code>{job_id}</code></h2><span class="status-badge status-{display_job_info['status'].lower()}">{display_job_info['status']}</span></div>
    <div class="details-grid">
        <div class="detail-card"><div class="detail-label">Time</div><div class="detail-value">{format_timestamp(display_job_info['timestamp'])}</div></div>
        <div class="detail-card"><div class="detail-label">Processing Duration</div><div class="detail-value">{display_job_info['total_time']:.2f}s</div></div>
        <div class="detail-card"><div class="detail-label">Model Used</div><div class="detail-value">{display_job_info['model']}</div></div>
        <div class="detail-card"><div class="detail-label">Source Type</div><div class="detail-value">{str(display_job_info['source_type']).title()}</div></div>
        <div class="detail-card"><div class="detail-label">Input Size</div><div class="detail-value">{format_size(display_job_info['input_size'])}</div></div>
        <div class="detail-card"><div class="detail-label">Output Size</div><div class="detail-value">{format_size(display_job_info['output_size']) if display_job_info['output_size']>0 else 'N/A'}</div></div>
        <div class="detail-card"><div class="detail-label">Requester IP</div><div class="detail-value">{display_job_info.get('requester_ip', 'N/A')}</div></div>
    </div>
    {f"<div class='detail-card' style='grid-column: span / auto;'><div class='detail-label'>Original Filename/URL</div><div class='detail-value' style='word-break:break-all;'>{display_job_info.get('original_filename', result_details.get('input_image_url','N/A'))}</div></div>" if display_job_info.get('original_filename') or result_details.get('input_image_url') else ''}
    <div class="images-section"><h2>Before & After Images</h2><div class="images-container"><div class="image-card"><h3>🔍 Original Image</h3>{f'<img src="{original_image_url}" alt="Original Image" loading="lazy">' if original_image_url else '<div class="no-image">Original image not available or not yet processed</div>'}</div><div class="image-card"><h3>✨ Processed Image</h3>{f'<img src="{processed_image_url}" alt="Processed Image" loading="lazy">' if processed_image_url else '<div class="no-image">Processed image not available or job failed/pending</div>'}</div></div></div>
    <div style="margin-top:30px;padding:15px;background:#f8f9fa;border-radius:6px;"><h3>Technical Details (Live Job Data if Active)</h3><ul>
        <li><strong>Current Status in System:</strong> {result_details.get('status','Not in active results (check history details above)')}</li>
        <li><strong>Requester IP (if tracked):</strong> {result_details.get('requester_ip', display_job_info.get('requester_ip', 'N/A'))}</li>
        <li><strong>Status Check URL:</strong> <a href="{result_details.get('status_check_url', f'{public_url_base}/status/{job_id}')}" target="_blank">API Status</a></li>
        {f"<li><strong>Error Message:</strong> {result_details.get('error_message','None')}</li>" if result_details.get('error_message') else ''}
        <li><strong>Job ID:</strong> <code>{job_id}</code></li>
    </ul></div></div></body></html>""", status_code=200)


async def cleanup_old_results():
    while True:
        try:
            current_time = time.time(); expired_jobs = []
            for job_id, job_data in list(results.items()):
                completion_time = job_data.get("completion_time")
                if job_data.get("status") in ["done", "error"] and completion_time and (current_time - completion_time) > 3600 :
                     expired_jobs.append(job_id)

            for job_id in expired_jobs:
                logger.info(f"Cleaning up old job from active results: {job_id}")
                del results[job_id]
            if expired_jobs:
                logger.info(f"Cleaned up {len(expired_jobs)} old jobs from active results dict")
        except Exception as e: logger.error(f"Error in cleanup task: {e}", exc_info=True)
        await asyncio.sleep(600)

async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image
    while True:
        job_id, image_source_str, model_name, _unused_flag, requester_ip = await queue.get()

        t_job_start = time.perf_counter()
        logger.info(f"Worker {worker_id} picked up job {job_id} for source: {image_source_str} (IP: {requester_ip}). Model: {model_name}")
        log_worker_activity(worker_id, WORKER_IDLE)

        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job {job_id} (from queue) not found in 'results' dict. Skipping.");
            queue.task_done(); continue

        input_bytes_for_rembg: bytes | None = None
        input_fetch_time, rembg_time, pil_time, save_time = 0.0, 0.0, 0.0, 0.0
        input_size_bytes, output_size_bytes = 0, 0
        original_fn_for_history = image_source_str.split('/')[-1]
        source_type_for_history = "url" if image_source_str.startswith(("http:", "https:")) else "upload"

        try:
            results[job_id]["status"] = "fetching_input"
            log_worker_activity(worker_id, WORKER_FETCHING); t_input_fetch_start = time.perf_counter()

            if image_source_str.startswith("file://"):
                results[job_id]["status"] = "loading_file";
                local_path = image_source_str[len("file://"):]
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Local file for job {job_id} not found: {local_path}")
                async with aiofiles.open(local_path, 'rb') as f: input_bytes_for_rembg = await f.read()
                input_size_bytes = len(input_bytes_for_rembg)
                original_fn_for_history = os.path.basename(local_path)
                logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Loaded local file '{original_fn_for_history}' ({format_size(input_size_bytes)})")
            elif image_source_str.startswith(("http://", "https://")):
                results[job_id]["status"] = "downloading"
                logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Downloading from {image_source_str}...")
                async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                    img_response = await client.get(image_source_str); img_response.raise_for_status()
                input_bytes_for_rembg = await img_response.aread(); input_size_bytes = len(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Downloaded {format_size(input_size_bytes)}")

                with ip_traffic_lock:
                    ip_traffic_stats[requester_ip]["total_input_bytes"] += input_size_bytes
                    ip_traffic_stats[requester_ip]["last_seen"] = time.time()

                content_type = img_response.headers.get("content-type", "unknown").lower()
                parsed_url_path = urllib.parse.urlparse(image_source_str).path
                _, url_ext = os.path.splitext(parsed_url_path)
                extension = MIME_TO_EXT.get(content_type, url_ext if url_ext else ".bin")

                dl_original_fn = f"{job_id}_original_downloaded{extension}"
                dl_original_path = os.path.join(UPLOADS_DIR, dl_original_fn)
                results[job_id]["original_local_path"] = dl_original_path
                async with aiofiles.open(dl_original_path, 'wb') as out_file: await out_file.write(input_bytes_for_rembg)
                original_fn_for_history = dl_original_fn
                logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Saved downloaded original as '{dl_original_fn}'")
            else:
                raise ValueError(f"Unsupported image source format: {image_source_str}")

            if input_bytes_for_rembg is None:
                raise ValueError(f"Image content is None for job {job_id} after fetch attempt.")
            input_fetch_time = time.perf_counter() - t_input_fetch_start

            log_worker_activity(worker_id, WORKER_PROCESSING_REMBG); results[job_id]["status"] = "processing_rembg"
            logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Starting rembg (model: {model_name})...")
            t_rembg_start = time.perf_counter(); loop = asyncio.get_event_loop()
            output_bytes_with_alpha = await loop.run_in_executor(cpu_executor, process_rembg_sync, input_bytes_for_rembg, model_name)
            rembg_time = time.perf_counter() - t_rembg_start
            logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Rembg done in {rembg_time:.4f}s")

            log_worker_activity(worker_id, WORKER_PROCESSING_PIL); results[job_id]["status"] = "processing_image"
            logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): Starting PIL processing...")
            t_pil_start = time.perf_counter()
            processed_image_bytes = await loop.run_in_executor(pil_executor, process_pil_sync, output_bytes_with_alpha, TARGET_SIZE, prepared_logo_image, ENABLE_LOGO_WATERMARK, LOGO_MARGIN)
            pil_time = time.perf_counter() - t_pil_start
            logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}): PIL done in {pil_time:.4f}s")

            log_worker_activity(worker_id, WORKER_SAVING); results[job_id]["status"] = "saving"
            processed_fn = f"{job_id}.webp"; processed_path = os.path.join(PROCESSED_DIR, processed_fn)
            t_save_start = time.perf_counter()
            async with aiofiles.open(processed_path, 'wb') as out_file: await out_file.write(processed_image_bytes)
            save_time = time.perf_counter() - t_save_start; output_size_bytes = len(processed_image_bytes)

            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_path
            total_job_time = time.perf_counter() - t_job_start
            # MODIFIED: Pass requester_ip to add_job_to_history
            add_job_to_history(job_id, "completed", total_job_time, input_size_bytes, output_size_bytes, model_name, source_type_for_history, original_fn_for_history, requester_ip)
            results[job_id]["completion_time"] = time.time()

            with ip_traffic_lock:
                ip_traffic_stats[requester_ip]["total_output_bytes"] += output_size_bytes
                ip_traffic_stats[requester_ip]["completed_jobs"] += 1
                ip_traffic_stats[requester_ip]["last_seen"] = time.time()

            logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) COMPLETED in {total_job_time:.4f}s. Input: {format_size(input_size_bytes)} -> Output: {format_size(output_size_bytes)}. Breakdown: Fetch={input_fetch_time:.3f}s, Rembg={rembg_time:.3f}s, PIL={pil_time:.3f}s, Save={save_time:.3f}s")

        except FileNotFoundError as e:
            logger.error(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) Error: FileNotFoundError: {e}", exc_info=False); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"File not found: {e}"
        except httpx.HTTPStatusError as e:
            logger.error(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) Error: HTTPStatusError downloading {image_source_str}: {e.response.status_code}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Download failed: HTTP {e.response.status_code} for {image_source_str}"
        except httpx.RequestError as e:
            logger.error(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) Error: httpx.RequestError downloading {image_source_str}: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Network error during download: {type(e).__name__}"
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) Error: Data/file processing error: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Processing error: {e}"
        except RuntimeError as e:
            logger.critical(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) CRITICAL RuntimeError: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Critical runtime error: {e}"
        except Exception as e:
            logger.critical(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) UNHANDLED CRITICAL Error: {e}", exc_info=True); results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Unexpected critical error: {e}"
        finally:
            if results.get(job_id, {}).get("status") == "error":
                total_job_time_error = time.perf_counter() - t_job_start
                # MODIFIED: Pass requester_ip to add_job_to_history
                add_job_to_history(job_id, "failed", total_job_time_error, input_size_bytes, 0, model_name, source_type_for_history, original_fn_for_history, requester_ip)
                results[job_id]["completion_time"] = time.time()

                with ip_traffic_lock:
                    ip_traffic_stats[requester_ip]["failed_jobs"] += 1
                    ip_traffic_stats[requester_ip]["last_seen"] = time.time()

                logger.info(f"Job {job_id} (W{worker_id}, IP: {requester_ip}) FAILED after {total_job_time_error:.4f}s. Error: {results[job_id].get('error_message', 'Unknown error')}")

            log_worker_activity(worker_id, WORKER_IDLE)
            queue.task_done()

@app.on_event("startup")
async def startup_event():
    global prepared_logo_image, cpu_executor, pil_executor, active_rembg_providers
    logger.info("Application startup...")
    cpu_executor = ThreadPoolExecutor(max_workers=CPU_THREAD_POOL_SIZE, thread_name_prefix="RembgCPU")
    pil_executor = ThreadPoolExecutor(max_workers=PIL_THREAD_POOL_SIZE, thread_name_prefix="PILCPU")
    logger.info(f"Thread pools initialized: RembgCPU Bound={CPU_THREAD_POOL_SIZE}, PILCPU Bound={PIL_THREAD_POOL_SIZE}")

    available_ort_providers = []
    try:
        import onnxruntime as ort
        available_ort_providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime available providers: {available_ort_providers}")
    except ImportError:
        logger.error("onnxruntime module not found. Rembg processing will likely fail. Install onnxruntime or onnxruntime-gpu.")
    except Exception as e:
        logger.error(f"Error getting ONNX Runtime available providers: {e}. Rembg processing may be unstable.", exc_info=True)

    if REMBG_USE_GPU:
        logger.info(f"REMBG_USE_GPU is True. Configuring rembg to use ONLY preferred GPU providers: {REMBG_PREFERRED_GPU_PROVIDERS}")
        if not REMBG_PREFERRED_GPU_PROVIDERS:
            logger.critical(
                "CRITICAL MISCONFIGURATION: REMBG_USE_GPU is True, but REMBG_PREFERRED_GPU_PROVIDERS is empty. "
                "Cannot force GPU without specifying preferred GPU providers. "
                "Setting active providers to a dummy value to force failure rather than allow ONNX default to CPU."
            )
            active_rembg_providers = ["MisconfiguredForceGPUErrProvider"]
        else:
            active_rembg_providers = list(REMBG_PREFERRED_GPU_PROVIDERS)
            actually_available_gpus = [p for p in active_rembg_providers if p in available_ort_providers]
            if not actually_available_gpus:
                logger.warning(
                    f"REMBG_USE_GPU is True. While forcing preferred GPU providers {active_rembg_providers}, "
                    f"note that NONE of them seem to be available in ONNX Runtime ({available_ort_providers}). "
                    "Rembg session initialization with these providers is expected to FAIL."
                )
            else:
                logger.info(
                    f"REMBG_USE_GPU is True. Will attempt to use providers {active_rembg_providers}. "
                    f"Of these, the following are reported as available by ONNX Runtime: {actually_available_gpus}."
                )
    else:
        logger.info("REMBG_USE_GPU is False. Configuring CPU-only providers for rembg.")
        if REMBG_CPU_PROVIDERS[0] in available_ort_providers:
            active_rembg_providers = list(REMBG_CPU_PROVIDERS)
        else:
            logger.error(
                f"CRITICAL: REMBG_USE_GPU is False, but the CPU provider ({REMBG_CPU_PROVIDERS[0]}) "
                f"is not in ONNX available_providers ({available_ort_providers}). Rembg CPU processing will likely fail."
            )
            active_rembg_providers = []

    logger.info(f"Final 'active_rembg_providers' determined at startup (will be used by workers): {active_rembg_providers}")

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
        else: logger.warning(f"Logo file not found at {LOGO_PATH}. Watermarking will be skipped."); prepared_logo_image = None
    else: logger.info("Logo watermarking DISABLED."); prepared_logo_image = None

    for i in range(MAX_CONCURRENT_TASKS): asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} async image processing workers started.")
    asyncio.create_task(cleanup_old_results()); logger.info("Background cleanup task for old job results started.")
    asyncio.create_task(system_monitor()); logger.info("System monitoring task started.")

    try:
        import pynvml
        pynvml.nvmlInit()
        logger.info("pynvml initialized for GPU monitoring (if NVIDIA GPU is present).")
    except Exception as e:
        logger.info(f"pynvml could not be initialized at startup (this is normal if no NVIDIA GPU or pynvml not installed): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global cpu_executor, pil_executor
    logger.info("Application shutdown sequence initiated...")

    if cpu_executor:
        cpu_executor.shutdown(wait=True)
        logger.info("RembgCPU thread pool shut down.")
    if pil_executor:
        pil_executor.shutdown(wait=True)
        logger.info("PILCPU thread pool shut down.")

    try:
        import pynvml
        pynvml.nvmlShutdown()
        logger.info("pynvml shutdown.")
    except ImportError:
        logger.info("pynvml not imported, no shutdown needed.")
    except Exception as e:
        logger.info(f"Error during pynvml shutdown (may be benign): {e}")
    logger.info("Application shutdown complete.")

app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Root Endpoint (Index Page) ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    stats = get_server_stats()

    logo_status = "Enabled" if ENABLE_LOGO_WATERMARK else "Disabled"
    if ENABLE_LOGO_WATERMARK and prepared_logo_image:
        logo_status += f" (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})"
    elif ENABLE_LOGO_WATERMARK and not prepared_logo_image:
        logo_status += " (Enabled but not loaded/found)"

    uptime_seconds = stats["uptime"]
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str_parts = []
    if days > 0: uptime_str_parts.append(f"{int(days)}d")
    if hours > 0: uptime_str_parts.append(f"{int(hours)}h")
    if minutes > 0: uptime_str_parts.append(f"{int(minutes)}m")
    uptime_str_parts.append(f"{int(seconds)}s")
    uptime_str = " ".join(uptime_str_parts) if uptime_str_parts else "0s"


    current_metrics = system_metrics[-1] if system_metrics else {
        "cpu_percent": 0, "memory_percent": 0, "memory_used_gb": 0,
        "memory_total_gb": 0, "gpu_used_mb": 0, "gpu_total_mb": 0, "gpu_utilization": 0
    }

    recent_jobs_html = ""
    if stats["recent_jobs"]:
        # Note: Sorting for recent_jobs is handled by JavaScript on the client-side
        recent_jobs_html += """
        <div class="table-responsive scrollable-table-container">
            <table class="styled-table" id="jobHistoryTable">
                <thead>
                    <tr>
                        <th data-sort-key="timestamp" data-sort-type="numeric">Time <span class="sort-arrow"></span></th>
                        <th>Job ID</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Input</th>
                        <th>Output</th>
                        <th>Model</th>
                        <th>Source</th>
                        <th data-sort-key="requester_ip" data-sort-type="string">IP Address <span class="sort-arrow"></span></th>
                        <th>Filename</th>
                    </tr>
                </thead>
                <tbody>"""

        for job in stats["recent_jobs"]: # Display all, JS will handle initial view if needed
            status_class = "status-completed" if job["status"] == "completed" else "status-failed"
            job_link = f"/job/{job['job_id']}"
            orig_fn_display = job.get('original_filename', '')
            if len(orig_fn_display) > 30:
                orig_fn_display = orig_fn_display[:15] + "..." + orig_fn_display[-12:]

            recent_jobs_html += f"""
                    <tr onclick="window.location.href='{job_link}'" style="cursor:pointer;">
                        <td data-timestamp="{job['timestamp']}">{format_timestamp(job['timestamp'])}</td>
                        <td><a href="{job_link}" class="job-link-id">{job['job_id'][:8]}...</a></td>
                        <td><span class="status-badge {status_class}">{job['status'].upper()}</span></td>
                        <td>{job['total_time']:.2f}s</td>
                        <td>{format_size(job['input_size'])}</td>
                        <td>{format_size(job['output_size']) if job['output_size'] > 0 else 'N/A'}</td>
                        <td>{job['model']}</td>
                        <td>{job['source_type']}</td>
                        <td>{job.get('requester_ip', 'N/A')}</td>
                        <td title="{job.get('original_filename', '')}">{orig_fn_display}</td>
                    </tr>"""
        recent_jobs_html += """
                </tbody>
            </table>
        </div>"""
    else:
        recent_jobs_html = "<p>No jobs processed yet.</p>"

    ip_stats_html = ''
    with ip_traffic_lock:
        sorted_ip_stats = sorted(
            ip_traffic_stats.items(),
            key=lambda item: item[1]["requests"],
            reverse=True
        )

    if sorted_ip_stats:
        ip_stats_html += """
        <div class="table-responsive">
            <table class="styled-table">
                <thead>
                    <tr>
                        <th>IP Address</th>
                        <th>Total Requests</th>
                        <th>Completed</th>
                        <th>Failed</th>
                        <th>Total Input</th>
                        <th>Total Output</th>
                        <th>Last Seen</th>
                    </tr>
                </thead>
                <tbody>"""
        for ip, stats_data in sorted_ip_stats:
            ip_stats_html += f"""
                    <tr>
                        <td>{ip}</td>
                        <td>{stats_data['requests']}</td>
                        <td>{stats_data['completed_jobs']}</td>
                        <td>{stats_data['failed_jobs']}</td>
                        <td>{format_size(stats_data['total_input_bytes'])}</td>
                        <td>{format_size(stats_data['total_output_bytes'])}</td>
                        <td>{format_timestamp(stats_data['last_seen'])}</td>
                    </tr>"""
        ip_stats_html += """
                </tbody>
            </table>
        </div>"""
    else:
        ip_stats_html += "<p>No IP-specific traffic data recorded yet.</p>"


    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing API Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            margin: 0;
            background-color: #f4f6f9;
            color: #343a40;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 30px auto;
            padding: 25px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        }}
        h1 {{
            color: #0056b3;
            margin-bottom: 10px;
            font-size: 2.2em;
            font-weight: 600;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #555;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .status-good {{ color: #28a745; font-weight: bold; }}
        .status-warning {{ color: #ffc107; font-weight: bold; }}
        .status-error {{ color: #dc3545; font-weight: bold; }}

        .section-title {{
            font-size: 1.6em;
            color: #333;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}

        .stats-grid, .system-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card, .metric-card {{
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .stat-card:hover, .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        }}
        .stat-value, .metric-value {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 8px;
            color: #007bff;
        }}
        .stat-label, .metric-label {{
            font-size: 0.90em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .metric-value {{ color: #343a40; }}
        .metric-card .cpu {{ color: #dc3545; }}
        .metric-card .memory {{ color: #fd7e14; }}
        .metric-card .gpu {{ color: #6f42c1; }}


        .charts-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }}
        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #495057;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}

        .config-list {{
            list-style: none;
            padding: 0;
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            font-size: 0.95em;
        }}
        .config-list li {{
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
        }}
        .config-list li:last-child {{
            border-bottom: none;
        }}
        .config-list strong {{
            color: #0056b3;
            margin-right: 10px;
        }}
        .config-list span {{
            text-align: right;
            word-break: break-all;
        }}


        .debug-info {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .debug-info h4 {{ margin-top: 0; color: #343a40; }}
        .debug-info p a {{
            color: #007bff;
            text-decoration: none;
            font-weight: 500;
        }}
        .debug-info p a:hover {{
            text-decoration: underline;
        }}

        .table-responsive {{
            overflow-x: auto;
        }}
        .scrollable-table-container {{ /* For job history table */
            max-height: 600px; /* Adjust as needed */
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0; /* Reset for scrollable container */
            font-size: 0.9em;
            /* box-shadow: 0 2px 8px rgba(0,0,0,0.07); */ /* Shadow on container now */
            /* border-radius: 8px; */ /* Radius on container now */
            /* overflow: hidden; */ /* Overflow on container now */
        }}
        .styled-table thead tr {{
            background-color: #007bff;
            color: #ffffff;
            text-align: left;
            font-weight: bold;
            position: sticky; /* Make header sticky */
            top: 0; /* Stick to top of scrollable container */
            z-index: 10;
        }}
        .styled-table th, .styled-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #dddddd;
            white-space: nowrap;
        }}
        .styled-table th[data-sort-key] {{ cursor: pointer; }}
        .styled-table th .sort-arrow {{
            display: inline-block;
            width: 0;
            height: 0;
            margin-left: 5px;
            vertical-align: middle;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
        }}
        .styled-table th .sort-arrow.asc {{ border-bottom: 5px solid #fff; }}
        .styled-table th .sort-arrow.desc {{ border-top: 5px solid #fff; }}

         .styled-table td:nth-child(10) {{ /* Last column (Filename in Job History) */
            white-space: normal;
            word-break: break-all;
        }}
        .styled-table tbody tr {{
            background-color: #fdfdfd;
            transition: background-color 0.2s ease;
        }}
        .styled-table tbody tr:nth-of-type(even) {{
            background-color: #f3f3f3;
        }}
        .styled-table tbody tr:hover {{
            background-color: #e9ecef;
        }}
        .styled-table tbody tr:last-of-type td {{
            /* border-bottom: 2px solid #007bff; */ /* No special border for last if scrolling */
        }}
        .job-link-id {{
            font-family: monospace;
            font-size: 0.95em;
            color: #0056b3;
            text-decoration: none;
        }}
        .job-link-id:hover {{ text-decoration: underline; }}

        .status-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
            color: white;
            display: inline-block;
        }}
        .status-badge.status-completed {{ background-color: #28a745; }}
        .status-badge.status-failed {{ background-color: #dc3545; }}
        .status-badge.status-error {{ background-color: #dc3545; }}
        .status-badge.status-active {{ background-color: #17a2b8; }}
        .status-badge.status-queued,
        .status-badge.status-processing_rembg, .status-badge.status-processing_image,
        .status-badge.status-saving, .status-badge.status-loading_file,
        .status-badge.status-downloading, .status-badge.status-fetching_input {{ background-color: #ffc107; color: #212529;}}


        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            font-size: 0.9em;
            color: #6c757d;
        }}

        @media (max-width: 992px) {{
            .charts-container {{
                grid-template-columns: 1fr;
            }}
            h1 {{ font-size: 1.8em; }}
            .section-title {{ font-size: 1.4em; }}
        }}
        @media (max-width: 768px) {{
            .stats-grid, .system-metrics {{
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            }}
            .stat-value, .metric-value {{ font-size: 1.5em; }}
            .container {{ padding: 15px; margin: 15px; }}
            .config-list li {{ flex-direction: column; align-items: flex-start; }}
            .config-list span {{ text-align: left; margin-top: 4px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Image Processing API Dashboard</h1>
        <p class="subtitle">
            <strong>Status:</strong> <span class="status-good">RUNNING</span> | Real-time monitoring of image processing tasks.
        </p>

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

        <h2 class="section-title">📊 Real-time Monitoring</h2>

        <div class="system-metrics">
            <div class="metric-card">
                <div class="metric-value cpu">{current_metrics['cpu_percent']:.1f}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value memory">{current_metrics['memory_percent']:.1f}%</div>
                <div class="metric-label">Memory ({current_metrics['memory_used_gb']:.1f}GB / {current_metrics['memory_total_gb']:.1f}GB)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value gpu">{current_metrics['gpu_utilization']:.0f}%</div>
                <div class="metric-label">GPU ({current_metrics['gpu_used_mb']:.0f}MB / {current_metrics['gpu_total_mb']:.0f}MB)</div>
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

        <h2 class="section-title">⚙️ Configuration & Debug</h2>
        <ul class="config-list">
            <li><strong>Async Workers:</strong> <span>{MAX_CONCURRENT_TASKS}</span></li>
            <li><strong>Rembg Thread Pool:</strong> <span>{CPU_THREAD_POOL_SIZE}</span></li>
            <li><strong>PIL Thread Pool:</strong> <span>{PIL_THREAD_POOL_SIZE}</span></li>
            <li><strong>Queue Capacity:</strong> <span>{MAX_QUEUE_SIZE}</span></li>
            <li><strong>Logo Watermarking:</strong> <span>{logo_status}</span></li>
            <li><strong>Force GPU (REMBG_USE_GPU):</strong> <span style="font-weight:bold; color: { 'green' if REMBG_USE_GPU else 'orange' };">{'Enabled' if REMBG_USE_GPU else 'Disabled'}</span></li>
            <li><strong>Preferred GPU Providers (Config):</strong> <span>{str(REMBG_PREFERRED_GPU_PROVIDERS)}</span></li>
            <li><strong>Active Rembg Providers (Runtime):</strong> <span style="font-weight:bold;">{str(active_rembg_providers)}</span></li>
            <li><strong>GPU Monitoring (pynvml):</strong> <span>{current_metrics['gpu_total_mb']} MB total {'(Active)' if current_metrics['gpu_total_mb'] > 0 else '(Not detected/NVIDIA pynvml required)'}</span></li>
        </ul>

        <div class="debug-info">
            <h4>🔧 Debug Links</h4>
            <p><a href="/api/debug/gpu" target="_blank">Check GPU/ONNXRT Detection Status & Rembg Provider Config</a></p>
            <p><a href="/api/monitoring/workers" target="_blank">View Raw Worker Data (JSON)</a></p>
            <p><a href="/api/monitoring/system" target="_blank">View Raw System Data (JSON)</a></p>
        </div>

        <h2 class="section-title">🚦 IP Traffic Monitor</h2>
        {ip_stats_html}

        <h2 class="section-title">📋 Job History (Last {MAX_HISTORY_ITEMS})</h2>
        {recent_jobs_html}

        <div class="footer">
            Page auto-refreshes every 30 seconds (charts update every {MONITORING_SAMPLE_INTERVAL}s) | Last updated: {format_timestamp(time.time())}
        </div>
    </div>

    <script>
        const workerColors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF', '#77DD77'];
        let workerChart, systemChart;

        function initCharts() {{
            const commonChartOptions = (yLabel, xLabel = 'Time') => ({{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 15 }} }} }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: yLabel }} }},
                    x: {{ title: {{ display: true, text: xLabel }}, ticks: {{ autoSkip: true, maxTicksLimit: 12, maxRotation: 0, minRotation: 0 }} }}
                }},
                elements: {{ line: {{ tension: 0.25 }}, point: {{ radius: 1.5 }} }},
                animation: {{ duration: 400, easing: 'easeInOutQuad' }}
            }});

            workerChart = new Chart(document.getElementById('workerChart').getContext('2d'), {{
                type: 'line', data: {{ labels: [], datasets: [] }},
                options: commonChartOptions('Active Tasks per Worker')
            }});
            systemChart = new Chart(document.getElementById('systemChart').getContext('2d'), {{
                type: 'line', data: {{ labels: [], datasets: [
                    {{ label: 'CPU %', data: [], borderColor: '#dc3545', backgroundColor: 'rgba(220, 53, 69, 0.1)', fill: 'origin' }},
                    {{ label: 'Memory %', data: [], borderColor: '#fd7e14', backgroundColor: 'rgba(253, 126, 20, 0.1)', fill: 'origin' }},
                    {{ label: 'GPU %', data: [], borderColor: '#6f42c1', backgroundColor: 'rgba(111, 66, 193, 0.1)', fill: 'origin' }}
                ]}},
                options: {{ ...commonChartOptions('Usage %'), scales: {{ ...commonChartOptions('Usage %').scales, y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Usage %' }} }} }} }}
            }});
        }}

        async function updateCharts() {{
            try {{
                const [workerResponse, systemResponse] = await Promise.all([
                    fetch('/api/monitoring/workers'),
                    fetch('/api/monitoring/system')
                ]);
                if (!workerResponse.ok || !systemResponse.ok) {{
                    console.error('Failed to fetch monitoring data:', workerResponse.status, systemResponse.status);
                    return;
                }}
                const workerData = await workerResponse.json();
                const systemData = await systemResponse.json();

                updateWorkerChart(workerData);
                updateSystemChart(systemData);
            }} catch (error) {{ console.error('Error updating charts:', error); }}
        }}

        function formatChartTimestamp(timestamp) {{
            return new Date(timestamp * 1000).toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
        }}

        function updateWorkerChart(data) {{
            const workerIds = Object.keys(data).sort();
            if (!workerIds.length || !data[workerIds[0]] || !data[workerIds[0]].length) {{
                workerChart.data.labels = []; workerChart.data.datasets = [];
                workerChart.update('none'); return;
            }}

            const labels = data[workerIds[0]].map(bucket => formatChartTimestamp(bucket.timestamp));
            const datasets = workerIds.map((workerId, index) => {{
                const buckets = data[workerId] || [];
                const totalActivity = buckets.map(b => (b.fetching||0) + (b.rembg||0) + (b.pil||0) + (b.saving||0));
                return {{
                    label: workerId.replace('worker_', 'W'), data: totalActivity,
                    borderColor: workerColors[index % workerColors.length],
                    backgroundColor: workerColors[index % workerColors.length] + '22', fill: 'origin'
                }};
            }});
            workerChart.data.labels = labels; workerChart.data.datasets = datasets;
            workerChart.update('none');
        }}

        function updateSystemChart(data) {{
            if (!data || !data.length) {{
                systemChart.data.labels = []; systemChart.data.datasets.forEach(ds => ds.data = []);
                systemChart.update('none'); return;
            }}
            systemChart.data.labels = data.map(m => formatChartTimestamp(m.timestamp));
            systemChart.data.datasets[0].data = data.map(m => m.cpu_percent);
            systemChart.data.datasets[1].data = data.map(m => m.memory_percent);
            systemChart.data.datasets[2].data = data.map(m => m.gpu_utilization);
            systemChart.update('none');
        }}

        // --- Job History Table Sorting ---
        let jobHistorySortState = {{ column: 0, direction: 'desc' }}; // Default sort by Time (index 0), descending

        function sortJobHistoryTable(columnIndex, sortType) {{
            const table = document.getElementById('jobHistoryTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('thead th');

            const direction = (jobHistorySortState.column === columnIndex && jobHistorySortState.direction === 'asc') ? 'desc' : 'asc';
            jobHistorySortState = {{ column: columnIndex, direction: direction }};

            rows.sort((a, b) => {{
                let valA, valB;
                if (sortType === 'numeric') {{ // For Time (timestamp)
                    valA = parseFloat(a.cells[columnIndex].dataset.timestamp);
                    valB = parseFloat(b.cells[columnIndex].dataset.timestamp);
                }} else {{ // For IP Address (string) or other text
                    valA = a.cells[columnIndex].textContent.trim().toLowerCase();
                    valB = b.cells[columnIndex].textContent.trim().toLowerCase();
                }}

                if (valA < valB) return direction === 'asc' ? -1 : 1;
                if (valA > valB) return direction === 'asc' ? 1 : -1;
                return 0;
            }});

            // Clear and re-append sorted rows
            tbody.innerHTML = '';
            rows.forEach(row => tbody.appendChild(row));

            // Update sort arrows
            headers.forEach(th => {{
                const arrow = th.querySelector('.sort-arrow');
                if (arrow) {{
                    arrow.className = 'sort-arrow'; // Reset
                }}
            }});
            const currentHeaderArrow = headers[columnIndex].querySelector('.sort-arrow');
            if (currentHeaderArrow) {{
                currentHeaderArrow.classList.add(direction);
            }}
        }}


        document.addEventListener('DOMContentLoaded', function() {{
            initCharts(); updateCharts();
            setInterval(updateCharts, {MONITORING_SAMPLE_INTERVAL * 1000});

            // Add click listeners to sortable headers in Job History
            const jobHistoryTable = document.getElementById('jobHistoryTable');
            if (jobHistoryTable) {{
                jobHistoryTable.querySelectorAll('thead th[data-sort-key]').forEach((th, index) => {{
                    // Find the actual column index in the full header row
                    let trueColumnIndex = 0;
                    let currentElement = th;
                    while(currentElement.previousElementSibling) {{
                        trueColumnIndex++;
                        currentElement = currentElement.previousElementSibling;
                    }}

                    th.addEventListener('click', () => {{
                        const sortType = th.dataset.sortType || 'string';
                        sortJobHistoryTable(trueColumnIndex, sortType);
                    }});
                }});
                // Initial sort for job history (by Time, descending)
                sortJobHistoryTable(0, 'numeric'); // Time column
                 // Call it again to ensure 'desc' is set, as first call might make it 'asc'
                if(jobHistorySortState.direction === 'asc') sortJobHistoryTable(0, 'numeric');
            }}
        }});

        setTimeout(() => {{ document.querySelector('.footer').textContent = 'Refreshing...'; location.reload(); }}, 30000);

    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development on http://0.0.0.0:7000 ...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
