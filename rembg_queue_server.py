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
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from rembg import remove, new_session  # type: ignore
from PIL import Image

# --- CREATE DIRECTORIES AT THE VERY TOP ---
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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

# --- Configuration Constants ---
MAX_CONCURRENT_TASKS = 8
MAX_QUEUE_SIZE = 5000
ESTIMATED_TIME_PER_JOB = 15
TARGET_SIZE = 1024
HTTP_CLIENT_TIMEOUT = 30.0
DEFAULT_MODEL_NAME = "birefnet"
MAX_JOBS_PER_IP_IN_QUEUE = 50  # Max jobs per IP allowed in queue at once
IP_CLEANUP_INTERVAL = 300  # 5 minutes - clean up old IP data
IP_STATS_RETENTION_HOURS = 24  # Keep IP stats for 24 hours
REMBG_USE_GPU = True  # Force GPU usage
REMBG_PREFERRED_GPU_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
]
REMBG_CPU_PROVIDERS = ["CPUExecutionProvider"]
MONITORING_HISTORY_MINUTES = 60
MONITORING_SAMPLE_INTERVAL = 5
MAX_MONITORING_SAMPLES = (MONITORING_HISTORY_MINUTES * 60) // MONITORING_SAMPLE_INTERVAL

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

EXPECTED_API_KEY = "secretApiKey"

# MIME types mapping
MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

# --- Fair Queue System ---
class FairQueue:
    def __init__(self):
        # map each IP → deque of job_data tuples
        self.ip_queues: Dict[str, deque] = defaultdict(deque)
        # per-IP statistics
        self.ip_stats: Dict[str, dict] = defaultdict(lambda: {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "current_queue_size": 0,
            "total_processing_time": 0.0,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "bytes_processed": 0,
        })
        # how many active jobs currently processing per IP
        self.ip_active_jobs: Dict[str, int] = defaultdict(int)

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_ip_index = 0

    def can_add_job(self, ip: str) -> bool:
        """Return False if this IP’s queue is at the per-IP limit."""
        with self.lock:
            return len(self.ip_queues[ip]) < MAX_JOBS_PER_IP_IN_QUEUE

    def add_job(self, ip: str, job_data: tuple) -> bool:
        """
        Enqueue job_data under this IP if under limit.
        Returns False if IP’s queue is already full.
        """
        with self.condition:
            if not self.can_add_job(ip):
                return False

            self.ip_queues[ip].append(job_data)
            self.ip_stats[ip]["total_jobs"] += 1
            self.ip_stats[ip]["current_queue_size"] = len(self.ip_queues[ip])
            self.ip_stats[ip]["last_seen"] = time.time()
            # wake any waiting workers
            self.condition.notify_all()
            logger.debug(
                f"Added job for IP {ip}. Queue size now: {self._total_queue_size_unsafe()}"
            )
            return True

    def get_next_job(self) -> Optional[Tuple[str, tuple]]:
        """
        Block until at least one IP queue is non-empty.
        Then pick exactly one job from the next non-empty IP in round-robin order.
        Returns (ip, job_data) or None if no jobs exist.
        """
        with self.condition:
            # wait for something to enqueue
            while self._total_queue_size_unsafe() == 0:
                logger.debug("Worker waiting for jobs...")
                self.condition.wait()

            ips_with_jobs = [ip for ip, q in self.ip_queues.items() if q]
            if not ips_with_jobs:
                return None

            start_idx = self.next_ip_index % len(ips_with_jobs)
            for i in range(len(ips_with_jobs)):
                idx = (start_idx + i) % len(ips_with_jobs)
                ip = ips_with_jobs[idx]
                if self.ip_queues[ip]:
                    job_data = self.ip_queues[ip].popleft()
                    self.ip_stats[ip]["current_queue_size"] = len(self.ip_queues[ip])
                    self.ip_active_jobs[ip] += 1
                    self.next_ip_index = (idx + 1) % len(ips_with_jobs)
                    logger.debug(
                        f"Worker got job for IP {ip}. Remaining total queue: {self._total_queue_size_unsafe()}"
                    )
                    return ip, job_data

            return None

    def job_completed(self, ip: str, success: bool, processing_time: float, bytes_processed: int):
        """
        Must be called when a worker finishes a job for the given IP.
        Updates stats: decrement active, increment completed/failed, accumulate processing time, bytes.
        """
        with self.lock:
            self.ip_active_jobs[ip] = max(0, self.ip_active_jobs[ip] - 1)
            if success:
                self.ip_stats[ip]["completed_jobs"] += 1
            else:
                self.ip_stats[ip]["failed_jobs"] += 1
            self.ip_stats[ip]["total_processing_time"] += processing_time
            self.ip_stats[ip]["bytes_processed"] += bytes_processed
            self.ip_stats[ip]["last_seen"] = time.time()

    def _total_queue_size_unsafe(self) -> int:
        """Sum of lengths of all per-IP queues. Caller must hold self.lock."""
        return sum(len(q) for q in self.ip_queues.values())

    def get_total_queue_size(self) -> int:
        """Thread-safe total pending jobs across all IPs."""
        with self.lock:
            return self._total_queue_size_unsafe()

    def get_ip_stats(self) -> Dict[str, dict]:
        """
        Return a snapshot of per-IP statistics, including:
          - total_jobs, completed_jobs, failed_jobs
          - current_queue_size, active_jobs, avg_processing_time, success_rate, bytes_processed
          - hours_since_first_seen, minutes_since_last_seen
        """
        with self.lock:
            now = time.time()
            snapshot = {}
            for ip, data in self.ip_stats.items():
                total = data["total_jobs"]
                completed = data["completed_jobs"]
                avg_time = data["total_processing_time"] / max(completed, 1)
                success_rate = (completed / total * 100) if total > 0 else 0.0
                snapshot[ip] = {
                    "total_jobs": total,
                    "completed_jobs": completed,
                    "failed_jobs": data["failed_jobs"],
                    "current_queue_size": data["current_queue_size"],
                    "active_jobs": self.ip_active_jobs[ip],
                    "avg_processing_time": avg_time,
                    "success_rate": success_rate,
                    "bytes_processed": data["bytes_processed"],
                    "hours_since_first_seen": (now - data["first_seen"]) / 3600,
                    "minutes_since_last_seen": (now - data["last_seen"]) / 60,
                }
            return snapshot

    def cleanup_old_ips(self):
        """
        Remove any IPs that:
          - last_seen is older than retention window
          - current_queue_size == 0
          - active_jobs == 0
        """
        cutoff = time.time() - (IP_STATS_RETENTION_HOURS * 3600)
        with self.lock:
            to_delete = []
            for ip, stats in self.ip_stats.items():
                if (
                    stats["last_seen"] < cutoff
                    and self.ip_active_jobs[ip] == 0
                    and len(self.ip_queues[ip]) == 0
                ):
                    to_delete.append(ip)

            for ip in to_delete:
                del self.ip_stats[ip]
                del self.ip_active_jobs[ip]
                del self.ip_queues[ip]
                logger.info(f"Cleaned up old IP stats for: {ip}")


# --- Global State ---
app = FastAPI()

# Add CORS middleware
origins = ["null", "http://localhost", "http://127.0.0.1"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

fair_queue = FairQueue()
results: dict = {}
server_start_time = time.time()
job_history: List[dict] = []
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

prepared_logo_image: Optional[Image.Image] = None
cpu_executor: Optional[ThreadPoolExecutor] = None
pil_executor: Optional[ThreadPoolExecutor] = None
active_rembg_providers: List[str] = list(REMBG_CPU_PROVIDERS)


# --- Helper Functions ---
def get_client_ip(request: Request) -> str:
    """
    Extract client IP from headers, respecting x-forwarded-for or x-real-ip if present.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def log_worker_activity(worker_id: int, activity: str):
    """
    Append a timestamped activity (idle/fetching/rembg/pil/saving) for the given worker.
    Maintain only the last MONITORING_HISTORY_MINUTES of data.
    """
    with worker_lock:
        worker_activity[worker_id].append((time.time(), activity))
        cutoff = time.time() - (MONITORING_HISTORY_MINUTES * 60)
        while worker_activity[worker_id] and worker_activity[worker_id][0][0] < cutoff:
            worker_activity[worker_id].popleft()


def get_gpu_info():
    """
    Attempt to query NVIDIA GPU metrics via pynvml. Return a dict with
    gpu_used_mb, gpu_total_mb, gpu_utilization. Suppress repeated warnings.
    """
    gpu_data = {"gpu_used_mb": 0, "gpu_total_mb": 0, "gpu_utilization": 0}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        gpu_data["gpu_used_mb"] = mem_info.used // (1024 ** 2)
        gpu_data["gpu_total_mb"] = mem_info.total // (1024 ** 2)
        gpu_data["gpu_utilization"] = utilization.gpu

        if not hasattr(get_gpu_info, "_logged_count"):
            get_gpu_info._logged_count = 0
        if get_gpu_info._logged_count < 3:
            logger.debug(
                f"GPU Monitor (pynvml): GPU {utilization.gpu}% | "
                f"Memory {gpu_data['gpu_used_mb']}/{gpu_data['gpu_total_mb']} MB"
            )
            get_gpu_info._logged_count += 1

    except ImportError:
        if not hasattr(get_gpu_info, "_import_warned"):
            logger.warning(
                "GPU monitoring via pynvml disabled: pynvml not installed. "
                "Install with `pip install pynvml`."
            )
            get_gpu_info._import_warned = True
    except Exception as e:
        if not hasattr(get_gpu_info, "_error_warned"):
            logger.warning(
                f"GPU monitoring via pynvml failed: {type(e).__name__}: {e}. "
                "This may occur if no NVIDIA GPU or missing drivers."
            )
            get_gpu_info._error_warned = True
    return gpu_data


async def system_monitor():
    """
    Background task: every MONITORING_SAMPLE_INTERVAL seconds, record CPU/memory/GPU usage into system_metrics.
    """
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
                **gpu_info,
            }
            system_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        await asyncio.sleep(MONITORING_SAMPLE_INTERVAL)


async def ip_cleanup_task():
    """
    Periodic background task: clean up old IP stats every IP_CLEANUP_INTERVAL seconds.
    """
    while True:
        try:
            fair_queue.cleanup_old_ips()
        except Exception as e:
            logger.error(f"Error in IP cleanup task: {e}")
        await asyncio.sleep(IP_CLEANUP_INTERVAL)


class SubmitJsonBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = DEFAULT_MODEL_NAME
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"


def get_proxy_url(request: Request) -> str:
    """
    Construct base URL (scheme + host) for generating public links.
    """
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"


def format_size(num_bytes: int) -> str:
    """
    Human-readable size: B, KB, or MB.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    else:
        return f"{num_bytes/1024**2:.2f} MB"


def add_job_to_history(
    job_id: str,
    status: str,
    total_time: float,
    input_size: int,
    output_size: int,
    model: str,
    source_type: str = "unknown",
    original_filename: str = "",
    client_ip: str = "unknown",
):
    """
    Record one job’s outcome into job_history list (most recent first).
    """
    global job_history, total_jobs_completed, total_jobs_failed, total_processing_time

    record = {
        "job_id": job_id,
        "timestamp": time.time(),
        "status": status,
        "total_time": total_time,
        "input_size": input_size,
        "output_size": output_size,
        "model": model,
        "source_type": source_type,
        "original_filename": original_filename,
        "client_ip": client_ip,
    }
    job_history.insert(0, record)
    if len(job_history) > MAX_HISTORY_ITEMS:
        job_history.pop()
    if status == "completed":
        total_jobs_completed += 1
        total_processing_time += total_time
    else:
        total_jobs_failed += 1


def format_timestamp(timestamp: float) -> str:
    """
    Format a UNIX timestamp as "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def get_server_stats() -> dict:
    """
    Return a dict of high-level server stats:
     - uptime, queue_size, active_jobs, total_completed, total_failed, avg_processing_time, recent_jobs
    """
    uptime = time.time() - server_start_time
    active_jobs = sum(
        1 for job in results.values() if job.get("status") not in ["done", "error"]
    )
    return {
        "uptime": uptime,
        "queue_size": fair_queue.get_total_queue_size(),
        "active_jobs": active_jobs,
        "total_completed": total_jobs_completed,
        "total_failed": total_jobs_failed,
        "avg_processing_time": total_processing_time / max(total_jobs_completed, 1),
        "recent_jobs": job_history,
    }


def get_worker_activity_data() -> dict:
    """
    Return a dict mapping worker IDs to a list of time-bucketed activity counts.
    Each bucket is a 30-second slice over the last MONITORING_HISTORY_MINUTES.
    """
    current_time = time.time()
    cutoff = current_time - (MONITORING_HISTORY_MINUTES * 60)
    bucket_size = 30
    num_buckets = (MONITORING_HISTORY_MINUTES * 60) // bucket_size
    worker_data = {}

    with worker_lock:
        for worker_id in range(1, MAX_CONCURRENT_TASKS + 1):
            activities = worker_activity.get(worker_id, deque())
            # Initialize buckets
            buckets = [
                {"timestamp": cutoff + (i * bucket_size), "idle": 0, "fetching": 0,
                 "rembg": 0, "pil": 0, "saving": 0}
                for i in range(num_buckets)
            ]
            for ts, act in activities:
                if ts >= cutoff:
                    idx = int((ts - cutoff) // bucket_size)
                    if 0 <= idx < len(buckets):
                        buckets[idx][act] += 1
            worker_data[f"worker_{worker_id}"] = buckets

    return worker_data


def get_system_metrics_data() -> List[dict]:
    """
    Return a list of recent system metrics (each with timestamp, cpu_percent, memory_percent, etc.).
    """
    return list(system_metrics)


def process_rembg_sync(input_bytes: bytes, model_name: str) -> bytes:
    """
    Synchronous rembg processing. Called inside a ThreadPoolExecutor.

    1) Create a rembg session with active_rembg_providers.
    2) Verify that forced GPU (if REMBG_USE_GPU=True) is actually used by checking onnxrt providers.
    3) Call rembg.remove(...) with that session.
    """
    global active_rembg_providers
    session_wrapper = None
    providers_to_attempt = list(active_rembg_providers)

    try:
        logger.info(
            f"Rembg: Attempting new_session(model='{model_name}', providers={providers_to_attempt})"
        )
        session_wrapper = new_session(model_name, providers=providers_to_attempt)
        if session_wrapper is None:
            raise RuntimeError(
                f"rembg.new_session returned None for model '{model_name}' with providers {providers_to_attempt}"
            )

        # Try to extract the underlying ONNX InferenceSession
        onnx_session = None
        if hasattr(session_wrapper, "inner_session"):
            onnx_session = session_wrapper.inner_session
        elif hasattr(session_wrapper, "sess"):
            onnx_session = session_wrapper.sess
        else:
            onnx_session = session_wrapper  # fallback; may not have get_providers

        if onnx_session is None:
            actual_providers = ["Error:CouldNotAccessONNXSession"]
        else:
            try:
                actual_providers = onnx_session.get_providers()
                if not actual_providers:
                    actual_providers = ["Error:GetProvidersReturnedEmpty"]
            except AttributeError:
                actual_providers = ["Error:GetProvidersMethodMissing"]
            except Exception as e:
                actual_providers = [f"Error:GetProvidersFailed_{type(e).__name__}"]

        logger.info(
            f"Rembg session for '{model_name}': intended={providers_to_attempt}, actual={actual_providers}"
        )

        if REMBG_USE_GPU:
            # If forced GPU, ensure at least one preferred GPU provider is active
            is_gpu_active = any(p in actual_providers for p in REMBG_PREFERRED_GPU_PROVIDERS)
            is_cpu_active = "CPUExecutionProvider" in actual_providers

            if any("Error:" in p for p in actual_providers):
                raise RuntimeError(
                    f"FORCED GPU FAILED: Could not verify providers. actual_providers={actual_providers}"
                )
            if is_cpu_active and not is_gpu_active:
                raise RuntimeError(
                    f"FORCED GPU FAILED: session fell back to CPU. actual_providers={actual_providers}"
                )
            if not is_gpu_active and any(p in REMBG_PREFERRED_GPU_PROVIDERS for p in providers_to_attempt):
                raise RuntimeError(
                    f"FORCED GPU FAILED: no preferred GPU active. actual_providers={actual_providers}"
                )
            if is_gpu_active:
                logger.info(
                    f"Rembg: Preferred GPU provider active: {actual_providers}"
                )
        else:
            # CPU-only case
            if REMBG_CPU_PROVIDERS[0] not in actual_providers:
                logger.error(
                    f"CRITICAL: CPU provider '{REMBG_CPU_PROVIDERS[0]}' not in ONNX providers {actual_providers}"
                )
                active_rembg_providers = []

    except Exception as e:
        log_msg = (
            f"CRITICAL: Failed to init/verify rembg session for '{model_name}' with providers {providers_to_attempt}. "
            f"Error: {type(e).__name__}: {e}."
        )
        if REMBG_USE_GPU:
            log_msg += " REMBG_USE_GPU=True, no fallback to CPU. Job will fail."
        else:
            log_msg += " REMBG_USE_GPU=False, CPU-only error."
        logger.critical(log_msg, exc_info=True)
        raise

    # Finally, call rembg.remove(...)
    output_bytes = remove(
        input_bytes,
        session=session_wrapper,
        post_process_mask=True,
        alpha_matting=True,
    )
    return output_bytes


def process_pil_sync(
    input_bytes: bytes,
    target_size: int,
    prepared_logo: Optional[Image.Image] = None,
    enable_logo: bool = False,
    logo_margin: int = 20,
) -> bytes:
    """
    Synchronous PIL processing: 
    1) Paste transparent PNG (from rembg) onto white background
    2) Resize to fit within target_size, then center on a square canvas of size target_size x target_size
    3) Optionally paste a logo watermark onto the bottom-left
    4) Save as WebP with white background
    """
    img_rgba = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    white_bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
    white_bg.paste(img_rgba, (0, 0), img_rgba)
    img_on_white = white_bg

    orig_w, orig_h = img_on_white.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Image dimensions zero after background removal")

    ratio = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    img_resized = img_on_white.resize((new_w, new_h), Image.Resampling.LANCZOS)

    square_canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    square_canvas.paste(img_resized, (paste_x, paste_y))

    if enable_logo and prepared_logo:
        if square_canvas.mode != "RGBA":
            square_canvas = square_canvas.convert("RGBA")
        logo_w, logo_h = prepared_logo.size
        logo_x = logo_margin
        logo_y = target_size - logo_h - logo_margin
        square_canvas.paste(prepared_logo, (logo_x, logo_y), prepared_logo)

    final_img = square_canvas
    if final_img.mode == "RGBA":
        opaque_canvas = Image.new("RGB", final_img.size, (255, 255, 255))
        opaque_canvas.paste(final_img, mask=final_img.split()[3])
        final_img = opaque_canvas

    buf = io.BytesIO()
    final_img.save(buf, "WEBP", quality=90, background=(255, 255, 255))
    return buf.getvalue()


# --- FastAPI Endpoints ---

@app.post("/submit")
async def submit_json_image_for_processing(request: Request, body: SubmitJsonBody):
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client_ip = get_client_ip(request)
    if not fair_queue.can_add_job(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many jobs queued for your IP. Maximum {MAX_JOBS_PER_IP_IN_QUEUE} jobs allowed.",
        )

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    model_to_use = body.model or DEFAULT_MODEL_NAME
    job_data = (job_id, str(body.image), model_to_use, True, client_ip)

    if not fair_queue.add_job(client_ip, job_data):
        raise HTTPException(
            status_code=503,
            detail=f"Server overloaded for your IP. Max {MAX_JOBS_PER_IP_IN_QUEUE} jobs.",
        )

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued",
        "input_image_url": str(body.image),
        "original_local_path": None,
        "processed_path": None,
        "error_message": None,
        "status_check_url": status_check_url,
        "client_ip": client_ip,
    }

    total_queue_size = fair_queue.get_total_queue_size()
    eta_seconds = (total_queue_size * ESTIMATED_TIME_PER_JOB) / MAX_CONCURRENT_TASKS

    logger.info(
        f"Job {job_id} (JSON: {body.image}, Model: {model_to_use}, IP: {client_ip}) enqueued. "
        f"Queue: {total_queue_size}. ETA: {eta_seconds:.2f}s"
    )

    return {
        "status": "processing",
        "job_id": job_id,
        "image_links": [f"{public_url_base}/images/{job_id}.webp"],
        "eta": eta_seconds,
        "status_check_url": status_check_url,
        "queue_position_info": f"Total jobs in queue: {total_queue_size}",
    }


@app.post("/submit_form")
async def submit_form_image_for_processing(
    request: Request,
    image_file: UploadFile = File(...),
    key: str = Form(...),
    model: str = Form(DEFAULT_MODEL_NAME),
):
    if key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client_ip = get_client_ip(request)
    if not fair_queue.can_add_job(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many jobs queued for your IP. Maximum {MAX_JOBS_PER_IP_IN_QUEUE} jobs.",
        )

    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    original_fn = image_file.filename or "upload"
    content_type = image_file.content_type.lower()
    extension = MIME_TO_EXT.get(content_type)

    if not extension:
        _, ext_fn = os.path.splitext(original_fn)
        ext_fn_lower = ext_fn.lower()
        if ext_fn_lower in MIME_TO_EXT.values():
            extension = ext_fn_lower
        else:
            extension = ".png"
            logger.warning(
                f"Job {job_id} (form): Unknown ext for '{original_fn}' from '{content_type}'. Defaulting to '{extension}'."
            )

    saved_fn = f"{job_id}_original{extension}"
    original_path = os.path.join(UPLOADS_DIR, saved_fn)

    try:
        async with aiofiles.open(original_path, "wb") as out_file:
            content = await image_file.read()
            await out_file.write(content)
        logger.info(
            f"📝 Job {job_id} (Form: {original_fn}, IP: {client_ip}) Original saved: "
            f"{original_path} ({format_size(len(content))})"
        )
    except Exception as e:
        logger.error(f"Error saving upload {saved_fn} for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    finally:
        await image_file.close()

    file_uri = f"file://{original_path}"
    model_to_use = model or DEFAULT_MODEL_NAME
    job_data = (job_id, file_uri, model_to_use, True, client_ip)

    if not fair_queue.add_job(client_ip, job_data):
        if os.path.exists(original_path):
            try:
                os.remove(original_path)
            except OSError as e_clean:
                logger.error(f"Error cleaning {original_path} (queue full): {e_clean}")
        raise HTTPException(
            status_code=503,
            detail=f"Server overloaded for your IP. Max {MAX_JOBS_PER_IP_IN_QUEUE} jobs.",
        )

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued",
        "input_image_url": f"(form_upload: {original_fn})",
        "original_local_path": original_path,
        "processed_path": None,
        "error_message": None,
        "status_check_url": status_check_url,
        "client_ip": client_ip,
    }

    total_queue_size = fair_queue.get_total_queue_size()
    eta_seconds = (total_queue_size * ESTIMATED_TIME_PER_JOB) / MAX_CONCURRENT_TASKS

    logger.info(
        f"Job {job_id} (Form: {original_fn}, Model: {model_to_use}, IP: {client_ip}) enqueued. "
        f"Queue: {total_queue_size}. ETA: {eta_seconds:.2f}s"
    )

    return {
        "status": "processing",
        "job_id": job_id,
        "original_image_url": f"{public_url_base}/originals/{saved_fn}",
        "image_links": [f"{public_url_base}/images/{job_id}.webp"],
        "eta": eta_seconds,
        "status_check_url": status_check_url,
        "queue_position_info": f"Total jobs in queue: {total_queue_size}",
    }


@app.get("/api/monitoring/workers")
async def get_worker_monitoring_data():
    return get_worker_activity_data()


@app.get("/api/monitoring/system")
async def get_system_monitoring_data():
    return get_system_metrics_data()


@app.get("/api/monitoring/ips")
async def get_ip_monitoring_data():
    """
    Returns live per-IP stats, including queue size, active jobs, total jobs, success rate, etc.
    """
    return fair_queue.get_ip_stats()


@app.get("/api/debug/gpu")
async def debug_gpu_status():
    """
    Return debugging info about:
     - pynvml availability & metrics
     - onnxruntime available providers & current rembg provider configuration
    """
    global active_rembg_providers
    result = {
        "pynvml_available": False,
        "gpu_detected_pynvml": False,
        "gpu_count_pynvml": 0,
        "error_pynvml": None,
        "current_metrics_pynvml": None,
        "onnxruntime_info": {
            "available": False,
            "providers": [],
            "error": None,
            "rembg_use_gpu_config": REMBG_USE_GPU,
            "rembg_preferred_gpu_providers_config": REMBG_PREFERRED_GPU_PROVIDERS,
            "currently_active_rembg_providers_for_workers": active_rembg_providers,
        },
    }

    # Check pynvml
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
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result["current_metrics_pynvml"] = {
                "device_name": device_name,
                "memory_used_mb": mem_info.used // (1024 ** 2),
                "memory_total_mb": mem_info.total // (1024 ** 2),
                "memory_percent": (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0,
                "gpu_utilization_percent": util.gpu,
                "memory_utilization_percent": util.memory,
            }
    except ImportError:
        result["error_pynvml"] = "pynvml not installed. Install with `pip install pynvml`."
    except Exception as e:
        result["error_pynvml"] = f"{type(e).__name__}: {e} (pynvml error)"

    # Check onnxruntime
    try:
        import onnxruntime as ort

        result["onnxruntime_info"]["available"] = True
        result["onnxruntime_info"]["providers"] = ort.get_available_providers()
    except ImportError:
        result["onnxruntime_info"]["error"] = (
            "onnxruntime module not found. Install with `pip install onnxruntime` or `onnxruntime-gpu`."
        )
    except Exception as e:
        result["onnxruntime_info"]["error"] = f"Error getting ONNX Runtime info: {e}"

    return result


@app.get("/status/{job_id}")
async def check_job_status(request: Request, job_id: str):
    """
    Return JSON status for the given job_id. If not in active results, check history.
    """
    job_info = results.get(job_id)
    if not job_info:
        # Maybe it's in history
        historical = next((j for j in job_history if j["job_id"] == job_id), None)
        if historical:
            public_url = get_proxy_url(request)
            status = "done" if historical["status"] == "completed" else "error"
            resp = {
                "job_id": job_id,
                "status": status,
                "input_image_url": f"(historical: {historical.get('original_filename', 'unknown')})",
                "status_check_url": f"{public_url}/status/{job_id}",
                "client_ip": historical.get("client_ip", "unknown"),
            }
            # Check if original exists
            for ext in [".webp", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
                orig_fn = f"{job_id}_original{ext}"
                if os.path.exists(os.path.join(UPLOADS_DIR, orig_fn)):
                    resp["original_image_url"] = f"{public_url}/originals/{orig_fn}"
                    break
            if status == "done":
                proc_fn = f"{job_id}.webp"
                if os.path.exists(os.path.join(PROCESSED_DIR, proc_fn)):
                    resp["processed_image_url"] = f"{public_url}/images/{proc_fn}"
            else:
                resp["error_message"] = f"Job failed after {historical['total_time']:.2f}s"
            return JSONResponse(content=resp)
        raise HTTPException(status_code=404, detail="Job not found")

    public_url_base = get_proxy_url(request)
    response_data = {
        "job_id": job_id,
        "status": job_info.get("status"),
        "input_image_url": job_info.get("input_image_url"),
        "status_check_url": job_info.get("status_check_url"),
        "client_ip": job_info.get("client_ip", "unknown"),
    }
    if job_info.get("original_local_path"):
        response_data["original_image_url"] = (
            f"{public_url_base}/originals/{os.path.basename(job_info['original_local_path'])}"
        )
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        response_data["processed_image_url"] = (
            f"{public_url_base}/images/{os.path.basename(job_info['processed_path'])}"
        )
    elif job_info.get("status") == "error":
        response_data["error_message"] = job_info.get("error_message")
    return JSONResponse(content=response_data)


@app.get("/job/{job_id}")
async def job_details(request: Request, job_id: str):
    """
    Return an HTML page showing detailed info (timestamps, sizes, images) for this job_id.
    """
    job_info_hist = next((j for j in job_history if j["job_id"] == job_id), None)
    result_details = results.get(job_id, {})

    display_info = {}
    if job_info_hist:
        display_info = job_info_hist.copy()
    elif result_details:
        display_info = {
            "job_id": job_id,
            "timestamp": time.time(),
            "status": result_details.get("status", "unknown"),
            "total_time": 0.0,
            "input_size": 0,
            "output_size": 0,
            "model": "N/A (active)",
            "source_type": "N/A (active)",
            "original_filename": result_details.get("input_image_url", "N/A").split("/")[-1],
            "client_ip": result_details.get("client_ip", "unknown"),
        }
    else:
        raise HTTPException(status_code=404, detail="Job not found in active results or history")

    public_url_base = get_proxy_url(request)

    # Try to find original image URL
    original_image_url = None
    if result_details.get("original_local_path"):
        original_image_url = f"{public_url_base}/originals/{os.path.basename(result_details['original_local_path'])}"
    else:
        if display_info.get("source_type") == "upload":
            for ext in [".webp", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
                fn = f"{job_id}_original{ext}"
                if os.path.exists(os.path.join(UPLOADS_DIR, fn)):
                    original_image_url = f"{public_url_base}/originals/{fn}"
                    break

    # Processed image URL
    processed_img_url = None
    proc_fn = f"{job_id}.webp"
    if os.path.exists(os.path.join(PROCESSED_DIR, proc_fn)):
        processed_img_url = f"{public_url_base}/images/{proc_fn}"
    elif result_details.get("processed_path"):
        processed_img_url = f"{public_url_base}/images/{os.path.basename(result_details['processed_path'])}"

    # Build HTML response
    html_content = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><title>Job Details - {job_id[:8]}</title><style>
body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
.header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
.status-badge {{ padding: 5px 10px; border-radius: 15px; font-weight: bold; text-transform: uppercase; }}
.status-completed {{ background-color: #d4edda; color: #155724; }}
.status-failed, .status-error {{ background-color: #f8d7da; color: #721c24; }}
.status-active, .status-queued, .status-processing_rembg, .status-processing_image, .status-saving, .status-loading_file, .status-downloading {{ background-color: #d1ecf1; color: #0c5460; }}
.details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 15px; margin: 20px 0; }}
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
</style></head>
<body><div class="container">
  <div class="header">
    <h1>Job Details</h1>
    <a href="/" class="back-link">← Back to Dashboard</a>
  </div>
  <div style="margin-bottom:20px;">
    <h2>Job ID: <code>{job_id}</code></h2>
    <span class="status-badge status-{display_info['status'].lower()}">{display_info['status']}</span>
  </div>
  <div class="details-grid">
    <div class="detail-card">
      <div class="detail-label">Time</div>
      <div class="detail-value">{format_timestamp(display_info['timestamp'])}</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Processing Duration</div>
      <div class="detail-value">{display_info['total_time']:.2f}s</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Model Used</div>
      <div class="detail-value">{display_info['model']}</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Source Type</div>
      <div class="detail-value">{str(display_info['source_type']).title()}</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Client IP</div>
      <div class="detail-value">{display_info.get('client_ip', 'unknown')}</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Input Size</div>
      <div class="detail-value">{format_size(display_info.get('input_size', 0))}</div>
    </div>
    <div class="detail-card">
      <div class="detail-label">Output Size</div>
      <div class="detail-value">
        {format_size(display_info.get('output_size', 0)) if display_info.get('output_size', 0) > 0 else 'N/A'}
      </div>
    </div>
  </div>
  {(
    f"<div class='detail-card' style='grid-column: span / auto;'>"
    f"<div class='detail-label'>Original Filename/URL</div>"
    f"<div class='detail-value' style='word-break:break-all;'>{display_info.get('original_filename', result_details.get('input_image_url','N/A'))}</div>"
    f"</div>"
  ) if display_info.get("original_filename") or result_details.get("input_image_url") else ""}
  <div class="images-section">
    <h2>Before & After Images</h2>
    <div class="images-container">
      <div class="image-card">
        <h3>🔍 Original Image</h3>
        {f'<img src="{original_image_url}" alt="Original Image" loading="lazy">' if original_image_url else '<div class="no-image">Original image not available or not yet processed</div>'}
      </div>
      <div class="image-card">
        <h3>✨ Processed Image</h3>
        {f'<img src="{processed_img_url}" alt="Processed Image" loading="lazy">' if processed_img_url else '<div class="no-image">Processed image not available or job failed/pending</div>'}
      </div>
    </div>
  </div>
  <div style="margin-top:30px; padding:15px; background:#f8f9fa; border-radius:6px;">
    <h3>Technical Details (Live Job Data if Active)</h3>
    <ul>
      <li><strong>Current Status in System:</strong> {result_details.get("status", "Not in active results")}</li>
      <li><strong>Status Check URL:</strong> 
        <a href="{result_details.get("status_check_url", f"{public_url_base}/status/{job_id}")}" target="_blank">API Status</a>
      </li>
      {f"<li><strong>Error Message:</strong> {result_details.get('error_message', 'None')}</li>" if result_details.get("error_message") else ""}
      <li><strong>Job ID:</strong> <code>{job_id}</code></li>
    </ul>
  </div>
</div></body></html>"""

    return HTMLResponse(content=html_content, status_code=200)


async def cleanup_old_results():
    """
    Every 10 minutes, remove from 'results' any job that is in status 'done' or 'error' AND completed > 1 hour ago.
    """
    while True:
        try:
            now = time.time()
            to_remove = []
            for job_id, job_data in list(results.items()):
                completion = job_data.get("completion_time")
                if job_data.get("status") in ["done", "error"] and completion and (now - completion) > 3600:
                    to_remove.append(job_id)
            for jid in to_remove:
                logger.info(f"Cleaning up old job from active results: {jid}")
                del results[jid]
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old jobs from active results")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)
        await asyncio.sleep(600)  # run every 10 minutes


async def image_processing_worker(worker_id: int):
    """
    Each worker repeatedly:
      1) Calls fair_queue.get_next_job() inside an executor (blocks)
      2) Unpacks (client_ip, (job_id, image_source_str, model_name, _, _))
      3) Fetches input (from file:// or HTTP)
      4) process_rembg_sync(...) in cpu_executor
      5) process_pil_sync(...) in pil_executor
      6) Save final WebP to disk
      7) Update results[job_id], call fair_queue.job_completed(...)
      8) add_job_to_history(...)
    """
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image

    while True:
        try:
            # Block until a job is available
            loop = asyncio.get_event_loop()
            job_result = await loop.run_in_executor(None, fair_queue.get_next_job)

            if job_result is None:
                await asyncio.sleep(0.1)
                continue

            client_ip, (job_id, image_source_str, model_name, _, _) = job_result
            t_job_start = time.perf_counter()
            logger.info(f"Worker {worker_id} picked job {job_id} from IP {client_ip}")
            log_worker_activity(worker_id, WORKER_IDLE)

            if job_id not in results:
                logger.error(
                    f"Worker {worker_id}: Job {job_id} not found in results. Skipping."
                )
                fair_queue.job_completed(client_ip, False, 0.0, 0)
                continue

            input_bytes_for_rembg: Optional[bytes] = None
            input_fetch_time = rembg_time = pil_time = save_time = 0.0
            input_size_bytes = output_size_bytes = 0
            original_fn_for_history = image_source_str.split("/")[-1]
            source_type_for_history = (
                "url" if image_source_str.startswith(("http://", "https://")) else "upload"
            )

            try:
                # --- FETCH INPUT BYTES ---
                results[job_id]["status"] = "fetching_input"
                log_worker_activity(worker_id, WORKER_FETCHING)
                t_fetch_start = time.perf_counter()

                if image_source_str.startswith("file://"):
                    # Load from local path
                    results[job_id]["status"] = "loading_file"
                    local_path = image_source_str[len("file://") :]
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(f"Local file not found: {local_path}")
                    async with aiofiles.open(local_path, "rb") as f:
                        input_bytes_for_rembg = await f.read()
                    input_size_bytes = len(input_bytes_for_rembg)
                    original_fn_for_history = os.path.basename(local_path)
                    logger.info(
                        f"Job {job_id} (W{worker_id}): Loaded local file '{original_fn_for_history}' ({format_size(input_size_bytes)})"
                    )
                elif image_source_str.startswith(("http://", "https://")):
                    # Download via HTTP
                    results[job_id]["status"] = "downloading"
                    logger.info(f"Job {job_id} (W{worker_id}): Downloading {image_source_str}...")
                    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                        img_response = await client.get(image_source_str)
                        img_response.raise_for_status()
                    input_bytes_for_rembg = await img_response.aread()
                    input_size_bytes = len(input_bytes_for_rembg)
                    logger.info(
                        f"Job {job_id} (W{worker_id}): Downloaded {format_size(input_size_bytes)}"
                    )

                    # Save downloaded copy for record
                    content_type = img_response.headers.get("content-type", "unknown").lower()
                    parsed_path = urllib.parse.urlparse(image_source_str).path
                    _, url_ext = os.path.splitext(parsed_path)
                    extension = MIME_TO_EXT.get(content_type, url_ext if url_ext else ".bin")

                    dl_fn = f"{job_id}_original_downloaded{extension}"
                    dl_path = os.path.join(UPLOADS_DIR, dl_fn)
                    results[job_id]["original_local_path"] = dl_path
                    async with aiofiles.open(dl_path, "wb") as out_file:
                        await out_file.write(input_bytes_for_rembg)
                    original_fn_for_history = dl_fn
                    logger.info(f"Job {job_id} (W{worker_id}): Saved downloaded as '{dl_fn}'")
                else:
                    raise ValueError(f"Unsupported image source: {image_source_str}")

                if input_bytes_for_rembg is None:
                    raise ValueError(f"Image content is None for job {job_id}")
                input_fetch_time = time.perf_counter() - t_fetch_start

                # --- RUN REMBG (GPU/CPU) ---
                log_worker_activity(worker_id, WORKER_PROCESSING_REMBG)
                results[job_id]["status"] = "processing_rembg"
                logger.info(f"Job {job_id} (W{worker_id}): Starting rembg (model={model_name})...")
                t_rembg_start = time.perf_counter()
                loop = asyncio.get_event_loop()
                output_bytes_with_alpha = await loop.run_in_executor(
                    cpu_executor, process_rembg_sync, input_bytes_for_rembg, model_name
                )
                rembg_time = time.perf_counter() - t_rembg_start
                logger.info(
                    f"Job {job_id} (W{worker_id}): Rembg done in {rembg_time:.4f}s"
                )

                # --- RUN PIL PROCESSING ---
                log_worker_activity(worker_id, WORKER_PROCESSING_PIL)
                results[job_id]["status"] = "processing_image"
                logger.info(f"Job {job_id} (W{worker_id}): Starting PIL processing...")
                t_pil_start = time.perf_counter()
                processed_image_bytes = await loop.run_in_executor(
                    pil_executor,
                    process_pil_sync,
                    output_bytes_with_alpha,
                    TARGET_SIZE,
                    prepared_logo_image,
                    ENABLE_LOGO_WATERMARK,
                    LOGO_MARGIN,
                )
                pil_time = time.perf_counter() - t_pil_start
                logger.info(
                    f"Job {job_id} (W{worker_id}): PIL done in {pil_time:.4f}s"
                )

                # --- SAVE PROCESSED IMAGE ---
                log_worker_activity(worker_id, WORKER_SAVING)
                results[job_id]["status"] = "saving"
                processed_fn = f"{job_id}.webp"
                processed_path = os.path.join(PROCESSED_DIR, processed_fn)
                t_save_start = time.perf_counter()
                async with aiofiles.open(processed_path, "wb") as out_file:
                    await out_file.write(processed_image_bytes)
                save_time = time.perf_counter() - t_save_start
                output_size_bytes = len(processed_image_bytes)

                # Mark job as done
                results[job_id]["status"] = "done"
                results[job_id]["processed_path"] = processed_path
                total_job_time = time.perf_counter() - t_job_start

                fair_queue.job_completed(
                    client_ip, True, total_job_time, input_size_bytes + output_size_bytes
                )
                add_job_to_history(
                    job_id,
                    "completed",
                    total_job_time,
                    input_size_bytes,
                    output_size_bytes,
                    model_name,
                    source_type_for_history,
                    original_fn_for_history,
                    client_ip,
                )
                results[job_id]["completion_time"] = time.time()

                logger.info(
                    f"Job {job_id} (W{worker_id}) COMPLETED in {total_job_time:.4f}s. "
                    f"Sizes: {format_size(input_size_bytes)} -> {format_size(output_size_bytes)}. "
                    f"Breakdown: Fetch={input_fetch_time:.3f}s, Rembg={rembg_time:.3f}s, PIL={pil_time:.3f}s, Save={save_time:.3f}s"
                )

            except FileNotFoundError as e:
                logger.error(f"Job {job_id} (W{worker_id}) Error: FileNotFoundError: {e}", exc_info=False)
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"File not found: {e}"
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Job {job_id} (W{worker_id}) Error: HTTPStatusError downloading {image_source_str}: {e.response.status_code}",
                    exc_info=True,
                )
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"Download failed: HTTP {e.response.status_code}"
            except httpx.RequestError as e:
                logger.error(
                    f"Job {job_id} (W{worker_id}) Error: httpx.RequestError downloading {image_source_str}: {e}",
                    exc_info=True,
                )
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"Network error: {type(e).__name__}"
            except (ValueError, IOError, OSError) as e:
                logger.error(
                    f"Job {job_id} (W{worker_id}) Error: Data/file processing error: {e}",
                    exc_info=True,
                )
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"Processing error: {e}"
            except RuntimeError as e:
                logger.critical(
                    f"Job {job_id} (W{worker_id}) CRITICAL RuntimeError: {e}",
                    exc_info=True,
                )
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"Critical runtime error: {e}"
            except Exception as e:
                logger.critical(
                    f"Job {job_id} (W{worker_id}) UNHANDLED CRITICAL Error: {e}",
                    exc_info=True,
                )
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = f"Unexpected critical error: {e}"
            finally:
                if results.get(job_id, {}).get("status") == "error":
                    total_error_time = time.perf_counter() - t_job_start
                    fair_queue.job_completed(client_ip, False, total_error_time, input_size_bytes)
                    add_job_to_history(
                        job_id,
                        "failed",
                        total_error_time,
                        input_size_bytes,
                        0,
                        model_name,
                        source_type_for_history,
                        original_fn_for_history,
                        client_ip,
                    )
                    results[job_id]["completion_time"] = time.time()
                    logger.info(
                        f"Job {job_id} (W{worker_id}) FAILED after {total_error_time:.4f}s. "
                        f"Error: {results[job_id].get('error_message', 'Unknown')}"
                    )

                log_worker_activity(worker_id, WORKER_IDLE)

        except Exception as e:
            logger.error(f"Worker {worker_id} encountered an error: {e}", exc_info=True)
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    """
    Initialize:
      - Thread pools for rembg (CPU) and PIL
      - Determine active_rembg_providers via onnxruntime.get_available_providers()
      - Load logo if watermarking enabled
      - Spawn worker tasks and background tasks
      - Optionally initialize pynvml for GPU monitoring
    """
    global prepared_logo_image, cpu_executor, pil_executor, active_rembg_providers

    logger.info("Application startup...")

    cpu_executor = ThreadPoolExecutor(
        max_workers=CPU_THREAD_POOL_SIZE, thread_name_prefix="RembgCPU"
    )
    pil_executor = ThreadPoolExecutor(
        max_workers=PIL_THREAD_POOL_SIZE, thread_name_prefix="PILCPU"
    )
    logger.info(
        f"Thread pools initialized: RembgCPU Bound={CPU_THREAD_POOL_SIZE}, PILCPU Bound={PIL_THREAD_POOL_SIZE}"
    )

    # Determine active_rembg_providers
    available_ort_providers = []
    try:
        import onnxruntime as ort

        available_ort_providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {available_ort_providers}")
    except ImportError:
        logger.error(
            "onnxruntime module not found. Rembg processing will likely fail. "
            "Install `onnxruntime` or `onnxruntime-gpu`."
        )
    except Exception as e:
        logger.error(
            f"Error getting ONNX Runtime providers: {e}. Rembg processing may be unstable.",
            exc_info=True,
        )

    if REMBG_USE_GPU:
        logger.info(
            f"REMBG_USE_GPU=True. Forcing providers: {REMBG_PREFERRED_GPU_PROVIDERS}"
        )
        if not REMBG_PREFERRED_GPU_PROVIDERS:
            logger.critical(
                "CRITICAL MISCONFIGURATION: REMBG_USE_GPU=True but no preferred GPU providers specified."
            )
            active_rembg_providers = ["MisconfiguredForceGPUErrProvider"]
        else:
            active_rembg_providers = list(REMBG_PREFERRED_GPU_PROVIDERS)
            actually_available = [
                p for p in active_rembg_providers if p in available_ort_providers
            ]
            if not actually_available:
                logger.warning(
                    f"None of the forced GPU providers {active_rembg_providers} "
                    f"are in ONNX providers {available_ort_providers}. Sessions may fail."
                )
            else:
                logger.info(f"Will attempt GPU providers: {actually_available}")
    else:
        logger.info("REMBG_USE_GPU=False. Using CPU providers.")
        if REMBG_CPU_PROVIDERS[0] in available_ort_providers:
            active_rembg_providers = list(REMBG_CPU_PROVIDERS)
        else:
            logger.error(
                f"CPU provider {REMBG_CPU_PROVIDERS[0]} not in ONNX providers {available_ort_providers}. "
                "Rembg CPU processing will likely fail."
            )
            active_rembg_providers = []

    logger.info(f"Final active_rembg_providers: {active_rembg_providers}")

    # Load logo if watermarking enabled
    if ENABLE_LOGO_WATERMARK:
        logger.info(f"Logo watermarking ENABLED. Loading from: {LOGO_PATH}")
        if os.path.exists(LOGO_PATH):
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                if logo.width > LOGO_MAX_WIDTH:
                    ratio = LOGO_MAX_WIDTH / logo.width
                    logo = logo.resize((LOGO_MAX_WIDTH, int(logo.height * ratio)), Image.Resampling.LANCZOS)
                prepared_logo_image = logo
                logger.info(f"Logo loaded: {prepared_logo_image.size}")
            except Exception as e:
                logger.error(f"Failed to load logo: {e}", exc_info=True)
                prepared_logo_image = None
        else:
            logger.warning(f"Logo not found at {LOGO_PATH}. Watermarking skipped.")
            prepared_logo_image = None
    else:
        logger.info("Logo watermarking DISABLED.")
        prepared_logo_image = None

    # Spawn worker coroutines
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i + 1))
    logger.info(f"{MAX_CONCURRENT_TASKS} async image processing workers started.")

    # Background cleanup tasks
    asyncio.create_task(cleanup_old_results())
    logger.info("Background cleanup task for old results started.")
    asyncio.create_task(system_monitor())
    logger.info("System monitoring task started.")
    asyncio.create_task(ip_cleanup_task())
    logger.info("IP cleanup task started.")

    # Initialize pynvml once (optional)
    try:
        import pynvml

        pynvml.nvmlInit()
        logger.info("pynvml initialized for GPU monitoring.")
    except Exception as e:
        logger.info(f"pynvml init failed (may be normal if no GPU or not installed): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    On shutdown, cleanly shut down thread pools and pynvml.
    """
    global cpu_executor, pil_executor
    logger.info("Application shutdown initiated...")

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
        logger.info("pynvml not imported; no shutdown needed.")
    except Exception as e:
        logger.info(f"pynvml shutdown error (likely benign): {e}")

    logger.info("Application shutdown complete.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """
    Return an HTML dashboard showing:
     - Server stats (uptime, queue size, active IPs, completed/failed counts, avg time)
     - Real-time system metrics (CPU%, Memory%, GPU%)
     - Charts for worker activity & system metrics (via Chart.js)
     - Table of per-IP stats
     - Configuration & debug links
     - Recent job history table (last MAX_HISTORY_ITEMS)
    """
    stats = get_server_stats()
    ip_stats = fair_queue.get_ip_stats()

    logo_status = "Enabled" if ENABLE_LOGO_WATERMARK else "Disabled"
    if ENABLE_LOGO_WATERMARK and prepared_logo_image:
        logo_status += f" (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})"
    elif ENABLE_LOGO_WATERMARK:
        logo_status += " (Enabled but not loaded)"

    uptime_seconds = stats["uptime"]
    days, rem = divmod(uptime_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    uptime_parts = []
    if days > 0:
        uptime_parts.append(f"{int(days)}d")
    if hours > 0:
        uptime_parts.append(f"{int(hours)}h")
    if minutes > 0:
        uptime_parts.append(f"{int(minutes)}m")
    uptime_parts.append(f"{int(seconds)}s")
    uptime_str = " ".join(uptime_parts) if uptime_parts else "0s"

    current_metrics = system_metrics[-1] if system_metrics else {
        "cpu_percent": 0,
        "memory_percent": 0,
        "memory_used_gb": 0,
        "memory_total_gb": 0,
        "gpu_used_mb": 0,
        "gpu_total_mb": 0,
        "gpu_utilization": 0,
    }

    # Build IP stats HTML
    ip_stats_html = "<h3>Client IP Statistics</h3>"
    if ip_stats:
        sorted_ips = sorted(ip_stats.items(), key=lambda x: x[1]["total_jobs"], reverse=True)
        ip_stats_html += """
        <div class="table-responsive">
          <table class="styled-table">
            <thead>
              <tr>
                <th>Client IP</th>
                <th>Active Jobs</th>
                <th>Queue Size</th>
                <th>Total Jobs</th>
                <th>Completed</th>
                <th>Failed</th>
                <th>Success Rate</th>
                <th>Avg Time</th>
                <th>Data Processed</th>
                <th>First Seen</th>
                <th>Last Seen</th>
              </tr>
            </thead>
            <tbody>
        """
        for ip, data in sorted_ips:
            success_class = (
                "status-good" if data["success_rate"] > 90
                else "status-warning" if data["success_rate"] > 70
                else "status-error"
            )
            active_class = "status-warning" if data["active_jobs"] > 3 else ""
            queue_class = "status-warning" if data["current_queue_size"] > 10 else ""
            ip_stats_html += f"""
              <tr>
                <td><code>{ip}</code></td>
                <td><span class="{active_class}">{data['active_jobs']}</span></td>
                <td><span class="{queue_class}">{data['current_queue_size']}</span></td>
                <td>{data['total_jobs']}</td>
                <td class="status-good">{data['completed_jobs']}</td>
                <td class="status-error">{data['failed_jobs']}</td>
                <td><span class="{success_class}">{data['success_rate']:.1f}%</span></td>
                <td>{data['avg_processing_time']:.2f}s</td>
                <td>{format_size(data['bytes_processed'])}</td>
                <td>{data['hours_since_first_seen']:.1f}h ago</td>
                <td>{data['minutes_since_last_seen']:.1f}m ago</td>
              </tr>
            """
        ip_stats_html += """
            </tbody>
          </table>
        </div>
        """
    else:
        ip_stats_html += "<p>No client activity yet.</p>"

    # Build recent jobs HTML
    recent_jobs_html = "<h3>Recent Jobs</h3>"
    if stats["recent_jobs"]:
        recent_jobs_html += """
        <div class="table-responsive">
          <table class="styled-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Job ID</th>
                <th>Client IP</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Input</th>
                <th>Output</th>
                <th>Model</th>
                <th>Source</th>
                <th>Filename</th>
              </tr>
            </thead>
            <tbody>
        """
        for job in stats["recent_jobs"][:20]:
            status_cls = "status-completed" if job["status"] == "completed" else "status-failed"
            job_link = f"/job/{job['job_id']}"
            orig_fn = job.get("original_filename", "")
            if len(orig_fn) > 30:
                orig_fn = orig_fn[:15] + "..." + orig_fn[-12:]
            recent_jobs_html += f"""
              <tr onclick="window.location.href='{job_link}'" style="cursor:pointer;">
                <td>{format_timestamp(job['timestamp'])}</td>
                <td><a href="{job_link}" class="job-link-id">{job['job_id'][:8]}...</a></td>
                <td><code>{job.get('client_ip', 'unknown')}</code></td>
                <td><span class="status-badge {status_cls}">{job['status'].upper()}</span></td>
                <td>{job['total_time']:.2f}s</td>
                <td>{format_size(job['input_size'])}</td>
                <td>{format_size(job['output_size']) if job['output_size'] > 0 else 'N/A'}</td>
                <td>{job['model']}</td>
                <td>{job['source_type']}</td>
                <td title="{job.get('original_filename', '')}">{orig_fn}</td>
              </tr>
            """
        recent_jobs_html += """
            </tbody>
          </table>
        </div>
        """
    else:
        recent_jobs_html += "<p>No jobs processed yet.</p>"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fair Queue Image Processing API Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      background-color: #f4f6f9;
      color: #343a40;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1800px;
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
    .config-list li:last-child {{ border-bottom: none; }}
    .config-list strong {{ color: #0056b3; margin-right: 10px; }}
    .config-list span {{ text-align: right; word-break: break-all; }}

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
    .debug-info p a:hover {{ text-decoration: underline; }}

    .table-responsive {{ overflow-x: auto; }}
    .styled-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.85em;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
      border-radius: 8px;
      overflow: hidden;
    }}
    .styled-table thead tr {{
      background-color: #007bff;
      color: #ffffff;
      text-align: left;
      font-weight: bold;
    }}
    .styled-table th, .styled-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid #dddddd;
      white-space: nowrap;
    }}
    .styled-table td:last-child {{
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
      border-bottom: 2px solid #007bff;
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
    .status-badge.status-downloading, .status-badge.status-fetching_input {{
      background-color: #ffc107;
      color: #212529;
    }}

    .fair-queue-info {{
      background: #d1ecf1;
      border: 1px solid #bee5eb;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 30px;
    }}
    .fair-queue-info h4 {{
      margin-top: 0;
      color: #0c5460;
    }}
    .fair-queue-info ul {{ margin-bottom: 0; }}

    .footer {{
      text-align: center;
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid #e9ecef;
      font-size: 0.9em;
      color: #6c757d;
    }}

    @media (max-width: 992px) {{
      .charts-container {{ grid-template-columns: 1fr; }}
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
    <h1>🚀 Fair Queue Image Processing API</h1>
    <p class="subtitle">
      <strong>Status:</strong> <span class="status-good">RUNNING</span> | Real-time monitoring with fair resource allocation per IP.
    </p>

    <div class="fair-queue-info">
      <h4>🔄 Fair Queue System Active</h4>
      <ul>
        <li><strong>Round-robin processing:</strong> Each IP gets equal access.</li>
        <li><strong>Per-IP queue limit:</strong> Maximum {MAX_JOBS_PER_IP_IN_QUEUE} jobs per IP.</li>
        <li><strong>Active unique IPs:</strong> {len(ip_stats)}</li>
        <li><strong>Total queued jobs:</strong> {stats['queue_size']}</li>
      </ul>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{uptime_str}</div>
        <div class="stat-label">Uptime</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['queue_size']}</div>
        <div class="stat-label">Total Queue Size</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{len(ip_stats)}</div>
        <div class="stat-label">Active IPs</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{stats['active_jobs']}</div>
        <div class="stat-label">Processing Jobs</div>
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

    <h2 class="section-title">🌐 Client IP Activity & Fair Share</h2>
    {ip_stats_html}

    <h2 class="section-title">⚙️ Configuration & Debug</h2>
    <ul class="config-list">
      <li><strong>Async Workers:</strong> <span>{MAX_CONCURRENT_TASKS}</span></li>
      <li><strong>Fair Queue System:</strong> <span>Enabled (Round-robin per IP)</span></li>
      <li><strong>Max Jobs per IP:</strong> <span>{MAX_JOBS_PER_IP_IN_QUEUE}</span></li>
      <li><strong>IP Stats Retention:</strong> <span>{IP_STATS_RETENTION_HOURS} hours</span></li>
      <li><strong>Rembg Thread Pool:</strong> <span>{CPU_THREAD_POOL_SIZE}</span></li>
      <li><strong>PIL Thread Pool:</strong> <span>{PIL_THREAD_POOL_SIZE}</span></li>
      <li><strong>Queue Capacity:</strong> <span>{MAX_QUEUE_SIZE}</span></li>
      <li><strong>Logo Watermarking:</strong> <span>{logo_status}</span></li>
      <li><strong>Force GPU (REMBG_USE_GPU):</strong> 
        <span style="font-weight:bold; color: {'green' if REMBG_USE_GPU else 'orange'};">
          {'Enabled' if REMBG_USE_GPU else 'Disabled'}
        </span>
      </li>
      <li><strong>Preferred GPU Providers (Config):</strong> <span>{REMBG_PREFERRED_GPU_PROVIDERS}</span></li>
      <li><strong>Active Rembg Providers (Runtime):</strong> <span style="font-weight:bold;">{active_rembg_providers}</span></li>
      <li><strong>GPU Monitoring (pynvml):</strong> 
        <span>{current_metrics['gpu_total_mb']} MB total 
        {'(Active)' if current_metrics['gpu_total_mb'] > 0 else '(Not detected/NVIDIA pynvml required)'}</span>
      </li>
    </ul>

    <div class="debug-info">
      <h4>🔧 Debug Links</h4>
      <p><a href="/api/debug/gpu" target="_blank">Check GPU/ONNXRT Detection Status & Rembg Provider Config</a></p>
      <p><a href="/api/monitoring/workers" target="_blank">View Raw Worker Data (JSON)</a></p>
      <p><a href="/api/monitoring/system" target="_blank">View Raw System Data (JSON)</a></p>
      <p><a href="/api/monitoring/ips" target="_blank">View Raw IP Statistics (JSON)</a></p>
    </div>

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
      const commonOpts = (yLabel, xLabel = 'Time') => ({{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 15 }} }},
        }},
        scales: {{
          y: {{ beginAtZero: true, title: {{ display: true, text: yLabel }} }},
          x: {{ title: {{ display: true, text: xLabel }}, ticks: {{ autoSkip: true, maxTicksLimit: 12, maxRotation: 0, minRotation: 0 }} }},
        }},
        elements: {{ line: {{ tension: 0.25 }}, point: {{ radius: 1.5 }} }},
        animation: {{ duration: 400, easing: 'easeInOutQuad' }},
      }});

      workerChart = new Chart(document.getElementById('workerChart').getContext('2d'), {{
        type: 'line',
        data: {{ labels: [], datasets: [] }},
        options: commonOpts('Active Tasks per Worker'),
      }});

      systemChart = new Chart(document.getElementById('systemChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: [],
          datasets: [
            {{ label: 'CPU %', data: [], borderColor: '#dc3545', backgroundColor: 'rgba(220, 53, 69, 0.1)', fill: 'origin' }},
            {{ label: 'Memory %', data: [], borderColor: '#fd7e14', backgroundColor: 'rgba(253, 126, 20, 0.1)', fill: 'origin' }},
            {{ label: 'GPU %', data: [], borderColor: '#6f42c1', backgroundColor: 'rgba(111, 66, 193, 0.1)', fill: 'origin' }},
          ],
        }},
        options: {{
          ...commonOpts('Usage %'),
          scales: {{
            ...commonOpts('Usage %').scales,
            y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Usage %' }} }},
          }},
        }},
      }});
    }}

    async function updateCharts() {{
      try {{
        const [wRes, sRes] = await Promise.all([
          fetch('/api/monitoring/workers'),
          fetch('/api/monitoring/system')
        ]);
        if (!wRes.ok || !sRes.ok) {{
          console.error('Failed to fetch monitoring data:', wRes.status, sRes.status);
          return;
        }}
        const wData = await wRes.json();
        const sData = await sRes.json();
        updateWorkerChart(wData);
        updateSystemChart(sData);
      }} catch (err) {{
        console.error('Error updating charts:', err);
      }}
    }}

    function formatChartTimestamp(ts) {{
      return new Date(ts * 1000).toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
    }}

    function updateWorkerChart(data) {{
      const workerIds = Object.keys(data).sort();
      if (!workerIds.length || !data[workerIds[0]].length) {{
        workerChart.data.labels = [];
        workerChart.data.datasets = [];
        workerChart.update('none');
        return;
      }}
      const labels = data[workerIds[0]].map(bucket => formatChartTimestamp(bucket.timestamp));
      const datasets = workerIds.map((wid, idx) => {{
        const buckets = data[wid] || [];
        const totalActivity = buckets.map(b => (b.fetching||0) + (b.rembg||0) + (b.pil||0) + (b.saving||0));
        return {{
          label: wid.replace('worker_', 'W'),
          data: totalActivity,
          borderColor: workerColors[idx % workerColors.length],
          backgroundColor: workerColors[idx % workerColors.length] + '22',
          fill: 'origin'
        }};
      }});
      workerChart.data.labels = labels;
      workerChart.data.datasets = datasets;
      workerChart.update('none');
    }}

    function updateSystemChart(data) {{
      if (!data || !data.length) {{
        systemChart.data.labels = [];
        systemChart.data.datasets.forEach(ds => ds.data = []);
        systemChart.update('none');
        return;
      }}
      systemChart.data.labels = data.map(m => formatChartTimestamp(m.timestamp));
      systemChart.data.datasets[0].data = data.map(m => m.cpu_percent);
      systemChart.data.datasets[1].data = data.map(m => m.memory_percent);
      systemChart.data.datasets[2].data = data.map(m => m.gpu_utilization);
      systemChart.update('none');
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      initCharts();
      updateCharts();
      setInterval(updateCharts, {MONITORING_SAMPLE_INTERVAL * 1000});
    }});

    // Auto-refresh non-chart data every 30s
    setTimeout(() => {{ location.reload(); }}, 30000);
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Fair Queue Uvicorn server at http://0.0.0.0:7000 ...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
