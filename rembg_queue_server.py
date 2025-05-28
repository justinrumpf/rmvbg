import asyncio
import uuid
import io
import os
import aiofiles
import logging
import httpx
import urllib.parse
import time # <--- ADDED IMPORT

# --- CREATE DIRECTORIES AT THE VERY TOP ---
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Added format
logger = logging.getLogger(__name__) # This will use the module name, e.g., '__main__' or your script's filename

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
ESTIMATED_TIME_PER_JOB = 35 # Keep this, but actual times will be logged
TARGET_SIZE = 1024
HTTP_CLIENT_TIMEOUT = 30.0

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
        queue.put_nowait((job_id, str(body.image), body.model, True)) # True for ignored post_process flag
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
        queue.put_nowait((job_id, file_uri_for_queue, model, True)) # True for ignored post_process flag
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

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
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
        response_data["downloaded_original_image_url"] = f"{public_url_base}/originals/{original_filename}"
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        processed_filename = os.path.basename(job_info["processed_path"])
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
    elif job_info.get("status") == "error":
        response_data["error_message"] = job_info.get("error_message")
    return JSONResponse(content=response_data)

# --- Background Worker ---
async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image

    while True:
        job_id, image_source_str, model_name, _ = await queue.get() # 4th item (post_process_flag) is ignored

        t_job_start = time.perf_counter()
        logger.info(f"Worker {worker_id} picked up job {job_id} for source: {image_source_str}. Model: {model_name}. Applying Alpha Matting & Post-Processing.")

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
            t_input_fetch_start = time.perf_counter()
            if image_source_str.startswith("file://"):
                results[job_id]["status"] = "processing_local_file"
                local_path_from_uri = image_source_str[len("file://"):]
                if not os.path.exists(local_path_from_uri):
                    raise FileNotFoundError(f"Local file for job {job_id} not found: {local_path_from_uri}")

                async with aiofiles.open(local_path_from_uri, 'rb') as f:
                    input_bytes_for_rembg = await f.read()
                input_size_bytes = len(input_bytes_for_rembg)
                t_input_fetch_end = time.perf_counter()
                input_fetch_time = t_input_fetch_end - t_input_fetch_start
                logger.info(f"Job {job_id} (Worker {worker_id}): Read local file {local_path_from_uri} ({format_size(input_size_bytes)}) in {input_fetch_time:.4f}s.")

            elif image_source_str.startswith(("http://", "https://")):
                results[job_id]["status"] = "downloading"
                logger.info(f"Job {job_id} (Worker {worker_id}): Downloading from {image_source_str}...")
                async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                    img_response = await client.get(image_source_str)
                    img_response.raise_for_status()

                input_bytes_for_rembg = await img_response.aread()
                input_size_bytes = len(input_bytes_for_rembg)
                t_input_fetch_end = time.perf_counter()
                input_fetch_time = t_input_fetch_end - t_input_fetch_start
                logger.info(f"Job {job_id} (Worker {worker_id}): Downloaded {format_size(input_size_bytes)} from {image_source_str} in {input_fetch_time:.4f}s.")

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
                async with aiofiles.open(downloaded_original_path, 'wb') as out_file: # Save downloaded original
                    await out_file.write(input_bytes_for_rembg)
                logger.info(f"Job {job_id} (Worker {worker_id}): Saved downloaded original to {downloaded_original_path}")
            else:
                raise ValueError(f"Unsupported image source scheme for job {job_id}: {image_source_str}")

            if input_bytes_for_rembg is None:
                raise ValueError(f"Image content for rembg is None for job {job_id}.")

            results[job_id]["status"] = "processing_rembg"
            logger.info(f"Job {job_id} (Worker {worker_id}): Starting rembg processing (model: {model_name})...")
            t_rembg_start = time.perf_counter()
            session = new_session(model_name)
            output_bytes_with_alpha = remove(
                input_bytes_for_rembg,
                session=session,
                post_process_mask=True,
                alpha_matting=True
            )
            t_rembg_end = time.perf_counter()
            rembg_time = t_rembg_end - t_rembg_start
            logger.info(f"Job {job_id} (Worker {worker_id}): Rembg processing completed in {rembg_time:.4f}s.")

            results[job_id]["status"] = "processing_pil"
            logger.info(f"Job {job_id} (Worker {worker_id}): Starting PIL processing (resize, white BG, watermark)...")
            t_pil_start = time.perf_counter()
            img_rgba = Image.open(io.BytesIO(output_bytes_with_alpha)).convert("RGBA")

            white_bg_canvas = Image.new("RGB", img_rgba.size, (255, 255, 255))
            white_bg_canvas.paste(img_rgba, (0, 0), img_rgba)
            img_on_white_bg = white_bg_canvas

            original_width, original_height = img_on_white_bg.size
            if original_width == 0 or original_height == 0:
                raise ValueError(f"Image dimensions zero after BG processing for job {job_id}.")

            ratio = min(TARGET_SIZE / original_width, TARGET_SIZE / original_height)
            new_width, new_height = int(original_width * ratio), int(original_height * ratio)
            img_resized_on_white = img_on_white_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)

            square_canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
            paste_x, paste_y = (TARGET_SIZE - new_width) // 2, (TARGET_SIZE - new_height) // 2
            square_canvas.paste(img_resized_on_white, (paste_x, paste_y))

            if ENABLE_LOGO_WATERMARK and prepared_logo_image:
                if square_canvas.mode != 'RGBA':
                    square_canvas = square_canvas.convert('RGBA')
                logo_w, logo_h = prepared_logo_image.size
                logo_pos_x, logo_pos_y = LOGO_MARGIN, TARGET_SIZE - logo_h - LOGO_MARGIN
                square_canvas.paste(prepared_logo_image, (logo_pos_x, logo_pos_y), prepared_logo_image)

            final_image_to_save = square_canvas
            if final_image_to_save.mode == 'RGBA':
                 final_opaque_canvas = Image.new("RGB", final_image_to_save.size, (255,255,255))
                 final_opaque_canvas.paste(final_image_to_save, mask=final_image_to_save.split()[3])
                 final_image_to_save = final_opaque_canvas
            t_pil_end = time.perf_counter()
            pil_time = t_pil_end - t_pil_start
            logger.info(f"Job {job_id} (Worker {worker_id}): PIL processing completed in {pil_time:.4f}s.")

            processed_filename = f"{job_id}.webp"
            processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)

            logger.info(f"Job {job_id} (Worker {worker_id}): Saving processed image to {processed_file_path}...")
            t_save_start = time.perf_counter()
            final_image_to_save.save(processed_file_path, 'WEBP', quality=90, background=(255,255,255))
            t_save_end = time.perf_counter()
            save_time = t_save_end - t_save_start
            output_size_bytes = os.path.getsize(processed_file_path)
            logger.info(f"Job {job_id} (Worker {worker_id}): Saved processed image ({format_size(output_size_bytes)}) in {save_time:.4f}s.")

            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path

            t_job_end = time.perf_counter()
            total_job_time = t_job_end - t_job_start
            logger.info(
                f"Job {job_id} (Worker {worker_id}) COMPLETED successfully. Processed: {processed_file_path}\n"
                f"    Input Size: {format_size(input_size_bytes)}, Output Size: {format_size(output_size_bytes)}\n"
                f"    Timings: InputFetch={input_fetch_time:.4f}s, Rembg={rembg_time:.4f}s, PIL={pil_time:.4f}s, Save={save_time:.4f}s\n"
                f"    Total Job Time: {total_job_time:.4f}s"
            )

        except FileNotFoundError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: FileNotFoundError: {e}", exc_info=False)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"File not found: {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: HTTPStatusError downloading {image_source_str}: {e.response.status_code}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Download failed: HTTP {e.response.status_code} from {image_source_str}."
        except httpx.RequestError as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: RequestError downloading {image_source_str}: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Network error downloading from {image_source_str}: {type(e).__name__}."
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: Data/file processing error: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Data or file error: {str(e)}"
        except Exception as e:
            logger.critical(f"Job {job_id} (Worker {worker_id}) CRITICAL Error: Unexpected processing error: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Unexpected processing error: {str(e)}"
        finally:
            if results.get(job_id, {}).get("status") == "error":
                t_job_end_error = time.perf_counter()
                total_job_time_error = t_job_end_error - t_job_start
                logger.info(f"Job {job_id} (Worker {worker_id}) FAILED. Total time before failure: {total_job_time_error:.4f}s")
            queue.task_done()

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    global prepared_logo_image
    logger.info("Application startup event running...")

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

    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} workers started. Queue max size: {MAX_QUEUE_SIZE}. ETA per job (rough estimate): {ESTIMATED_TIME_PER_JOB}s.")

# --- Static File Serving ---
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    logo_status = "Enabled" if ENABLE_LOGO_WATERMARK else "Disabled"
    if ENABLE_LOGO_WATERMARK and prepared_logo_image:
        logo_status += f" (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})"
    elif ENABLE_LOGO_WATERMARK and not prepared_logo_image:
        logo_status += " (Enabled but not loaded/found)"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Image API</title>
    <style>body{{font-family:sans-serif;margin:20px}} li{{margin-bottom: 5px;}}</style></head>
    <body><h1>Image Processing API Running</h1><p>Background removal now <b>always uses alpha matting and post-processing</b> for highest quality.</p>
    <p>Settings:<ul>
    <li>Workers: {MAX_CONCURRENT_TASKS}</li>
    <li>Queue Capacity: {MAX_QUEUE_SIZE}</li>
    <li>Est. Time per Job: {ESTIMATED_TIME_PER_JOB} seconds (actual times logged per job)</li>
    <li>Logo Watermarking: {logo_status}</li>
    </ul></p></body></html>"""

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
