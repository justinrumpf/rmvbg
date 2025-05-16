import asyncio
import uuid
import io
import os # Make sure os is imported very early
import aiofiles
import logging
import httpx
import urllib.parse

# --- CREATE DIRECTORIES AT THE VERY TOP ---
# Define necessary paths first
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg" # If BASE_DIR is needed for LOGO_PATH before FastAPI app

# Configure logging early as well so directory creation attempts are logged
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Get logger after basicConfig

try:
    os.makedirs(UPLOADS_DIR_STATIC, exist_ok=True)
    os.makedirs(PROCESSED_DIR_STATIC, exist_ok=True)
    # Also create BASE_DIR if it's used for the logo and might not exist
    os.makedirs(BASE_DIR_STATIC, exist_ok=True)
    logger.info(f"Ensured uploads directory exists: {UPLOADS_DIR_STATIC}")
    logger.info(f"Ensured processed directory exists: {PROCESSED_DIR_STATIC}")
    logger.info(f"Ensured base directory exists: {BASE_DIR_STATIC}")
except OSError as e:
    logger.error(f"CRITICAL: Error creating essential directories: {e}", exc_info=True)
    # For such a critical step, you might want the application to exit if directories can't be made.
    # import sys
    # sys.exit(f"CRITICAL: Could not create essential directories: {e}")
    # For now, we'll let it continue and see if FastAPI/Starlette complains later.

# Now import FastAPI and other components that might depend on these paths indirectly
# or for app.mount
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from rembg import remove, new_session
from PIL import Image

app = FastAPI() # FastAPI app instance created AFTER directories are handled

# --- Configuration Constants (can use the _STATIC versions or redefine) ---
MAX_CONCURRENT_TASKS = 8
MAX_QUEUE_SIZE = 5000
ESTIMATED_TIME_PER_JOB = 13
TARGET_SIZE = 1024
LOGO_MAX_WIDTH = 150
LOGO_MARGIN = 20
HTTP_CLIENT_TIMEOUT = 30.0

# --- Directory and File Paths (using the _STATIC versions or re-assigning) ---
BASE_DIR = BASE_DIR_STATIC
UPLOADS_DIR = UPLOADS_DIR_STATIC
PROCESSED_DIR = PROCESSED_DIR_STATIC # Crucial: use the one that was created
LOGO_FILENAME = "logo.png"
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME)


# --- Global State ---
prepared_logo_image = None
queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
results = {}
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
class SubmitRequestBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = "u2net"
    post_process: bool = False
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"

# --- Helper Functions ---
def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

# --- API Endpoints ---
@app.post("/submit")
async def submit_image_for_processing(
    request: Request,
    body: SubmitRequestBody
):
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo file exists at startup but was not loaded. Watermarking may fail or be skipped. Check startup logs.")
    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    try:
        queue.put_nowait((job_id, str(body.image), body.model, body.post_process))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full (max size: {MAX_QUEUE_SIZE}). Rejecting request for image {body.image}.")
        raise HTTPException(status_code=503, detail=f"Server is temporarily overloaded (queue full). Please try again later. Max queue size: {MAX_QUEUE_SIZE}")
    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued", "input_image_url": str(body.image), "original_local_path": None,
        "processed_path": None, "error_message": None, "status_check_url": status_check_url
    }
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB
    return {
        "status": "processing", "job_id": job_id, "image_links": [processed_image_placeholder_url],
        "eta": eta_seconds, "status_check_url": status_check_url
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
        job_id, image_url_str, model_name, post_process_flag = await queue.get()
        logger.info(f"Worker {worker_id} picked up job {job_id} for URL: {image_url_str}")
        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job ID {job_id} from queue not found in results dict. Skipping.")
            queue.task_done()
            continue
        original_file_path = None
        results[job_id]["status"] = "downloading"
        try:
            async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                img_response = await client.get(image_url_str)
                img_response.raise_for_status()
            
            image_content = await img_response.aread()
            original_content_type_header = img_response.headers.get("content-type", "unknown")
            content_type = original_content_type_header.lower()
            logger.info(f"Job {job_id}: Received initial Content-Type='{original_content_type_header}' for URL {image_url_str}")
            if content_type == "application/octet-stream" or not content_type.startswith("image/"):
                if content_type == "application/octet-stream":
                    logger.warning(f"Job {job_id}: Original Content-Type is 'application/octet-stream'. Attempting to infer from URL file extension.")
                else:
                    logger.warning(f"Job {job_id}: Original Content-Type is '{content_type}', which is not 'image/*'. Attempting to infer from URL file extension as a fallback.")
                file_extension_from_url = os.path.splitext(urllib.parse.urlparse(image_url_str).path)[1].lower()
                logger.info(f"Job {job_id}: Parsed file extension from URL: '{file_extension_from_url}'")
                potential_new_content_type = None
                if file_extension_from_url == ".webp": potential_new_content_type = "image/webp"
                elif file_extension_from_url == ".png": potential_new_content_type = "image/png"
                elif file_extension_from_url in [".jpg", ".jpeg"]: potential_new_content_type = "image/jpeg"
                elif file_extension_from_url == ".gif": potential_new_content_type = "image/gif"
                elif file_extension_from_url == ".bmp": potential_new_content_type = "image/bmp"
                elif file_extension_from_url in [".tif", ".tiff"]: potential_new_content_type = "image/tiff"
                if potential_new_content_type:
                    logger.info(f"Job {job_id}: Overriding Content-Type from '{original_content_type_header}' to '{potential_new_content_type}' based on URL extension '{file_extension_from_url}'.")
                    content_type = potential_new_content_type
                else:
                    logger.warning(f"Job {job_id}: URL file extension '{file_extension_from_url}' is not in the recognized list for Content-Type override. Original Content-Type '{original_content_type_header}' will be used for the final check.")
            if not content_type.startswith("image/"):
                logger.error(f"Job {job_id}: FINAL Content-Type check FAILED. Content-Type is '{content_type}'. URL: {image_url_str}")
                raise ValueError(f"Invalid content type '{content_type}' from URL. Not an image.")
            logger.info(f"Job {job_id}: Proceeding with Content-Type '{content_type}'.")

            extension = MIME_TO_EXT.get(content_type)
            if not extension:
                parsed_url_path_for_save_ext = urllib.parse.urlparse(image_url_str).path
                _, ext_from_url_for_save = os.path.splitext(parsed_url_path_for_save_ext)
                ext_from_url_for_save = ext_from_url_for_save.lower()
                if ext_from_url_for_save and ext_from_url_for_save in MIME_TO_EXT.values():
                    extension = ext_from_url_for_save
                    logger.info(f"Job {job_id}: Using URL extension '{extension}' for saving original file as Content-Type '{content_type}' was not directly in MIME_TO_EXT map.")
                else:
                    extension = ".bin"
                    logger.warning(f"Job {job_id}: Could not determine specific file extension for saving original from Content-Type '{content_type}' or URL. Defaulting to '{extension}'.")
            original_filename = f"{job_id}_original{extension}"
            original_file_path = os.path.join(UPLOADS_DIR, original_filename)
            results[job_id]["original_local_path"] = original_file_path
            async with aiofiles.open(original_file_path, 'wb') as out_file:
                await out_file.write(image_content)
            logger.info(f"Worker {worker_id} saved original image for job {job_id} to {original_file_path}")
            results[job_id]["status"] = "processing"
            with open(original_file_path, 'rb') as i:
                input_bytes = i.read()
            session = new_session(model_name)
            output_bytes = remove(input_bytes, session=session, post_process_mask=post_process_flag)
            img_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            original_width, original_height = img_no_bg.size
            if original_width == 0 or original_height == 0:
                raise ValueError("Image dimensions are zero after background removal.")
            ratio = min(TARGET_SIZE / original_width, TARGET_SIZE / original_height)
            new_width, new_height = int(original_width * ratio), int(original_height * ratio)
            img_resized = img_no_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
            square_canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
            paste_x, paste_y = (TARGET_SIZE - new_width) // 2, (TARGET_SIZE - new_height) // 2
            square_canvas.paste(img_resized, (paste_x, paste_y), img_resized)
            if prepared_logo_image:
                logo_w, logo_h = prepared_logo_image.size
                logo_pos_x, logo_pos_y = LOGO_MARGIN, TARGET_SIZE - logo_h - LOGO_MARGIN
                square_canvas.paste(prepared_logo_image, (logo_pos_x, logo_pos_y), prepared_logo_image)
            else:
                logger.info(f"Job {job_id}: Skipping watermark as logo is not available/loaded.")
            final_image = square_canvas
            processed_filename = f"{job_id}.webp"
            processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
            final_image.save(processed_file_path, 'WEBP', quality=90)
            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path
            logger.info(f"Worker {worker_id} finished job {job_id}. Processed image: {processed_file_path}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Worker {worker_id} HTTP error {e.response.status_code} for job {job_id} from {image_url_str}: {e.response.text}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Failed to download image: HTTP {e.response.status_code} from {image_url_str}."
        except httpx.RequestError as e:
            logger.error(f"Worker {worker_id} Network error downloading for job {job_id} from {image_url_str}: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Network error downloading image from {image_url_str}: {type(e).__name__}."
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Worker {worker_id} data/file error for job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Data or file error: {str(e)}"
        except Exception as e:
            logger.error(f"Worker {worker_id} critical error processing job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"; results[job_id]["error_message"] = f"Unexpected processing error: {str(e)}"
        finally:
            queue.task_done()

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    global prepared_logo_image
    logger.info("Application startup event running...")

    # Directories should have been created at module level.
    # This is just an additional check or for clarity if someone looks here.
    if not os.path.isdir(UPLOADS_DIR):
        logger.warning(f"Uploads directory {UPLOADS_DIR} was not found at startup event, trying to create again.")
        os.makedirs(UPLOADS_DIR, exist_ok=True)
    if not os.path.isdir(PROCESSED_DIR):
        logger.warning(f"Processed directory {PROCESSED_DIR} was not found at startup event, trying to create again.")
        os.makedirs(PROCESSED_DIR, exist_ok=True)


    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            if logo.width > LOGO_MAX_WIDTH:
                l_ratio = LOGO_MAX_WIDTH / logo.width
                l_new_width, l_new_height = LOGO_MAX_WIDTH, int(logo.height * l_ratio)
                logo = logo.resize((l_new_width, l_new_height), Image.Resampling.LANCZOS)
            prepared_logo_image = logo
            logger.info(f"Logo loaded. Dimensions: {prepared_logo_image.size if prepared_logo_image else 'None'}")
        except Exception as e:
            logger.error(f"Failed to load logo from {LOGO_PATH}: {e}", exc_info=True); prepared_logo_image = None
    else:
        logger.warning(f"Logo file not found at {LOGO_PATH}. Watermarking will be skipped."); prepared_logo_image = None
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} workers started. Queue max size: {MAX_QUEUE_SIZE}.")

# --- Static File Serving ---
# These must come AFTER app = FastAPI() is defined AND directories exist
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing API</title><style>body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
    .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
    h1 {{ color: #0056b3; }}</style></head><body><div class="container"><h1>Resale1... In The House!</h1>
    <p>This service provides background removal and image processing capabilities.</p><p>Current settings:<ul>
    <li>Max Concurrent Workers: {MAX_CONCURRENT_TASKS}</li><li>Max Queue Size: {MAX_QUEUE_SIZE}</li></ul></p></div></body></html>"""

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
