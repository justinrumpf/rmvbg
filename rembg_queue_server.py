from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from rembg import remove, new_session
from PIL import Image

import asyncio
import uuid
import io
import os
import aiofiles
import logging
# from typing import List, Optional # Not strictly needed now
import httpx
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_CONCURRENT_TASKS = 6
ESTIMATED_TIME_PER_JOB = 13 # May increase slightly due to download in worker
TARGET_SIZE = 1024
LOGO_MAX_WIDTH = 150
LOGO_MARGIN = 20

BASE_DIR = "/workspace/rmvbg"
UPLOADS_DIR = "/workspace/uploads"
PROCESSED_DIR = "/workspace/processed"
LOGO_FILENAME = "logo.png"
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME)

prepared_logo_image = None

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

queue = asyncio.Queue()
results = {} # job_id -> { "status": ..., "input_image_url": ..., "original_local_path": ..., "processed_path": ..., "error_message": ... }


EXPECTED_API_KEY = "secretApiKey"

MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
}

class SubmitRequestBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = "u2net"
    post_process: bool = False
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"


def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

@app.post("/submit")
async def submit_image_for_processing(
    request: Request,
    body: SubmitRequestBody
):
    # 1. API Key Validation (Quick)
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Logo Availability Check (Quick local check)
    # This check ensures that if a logo is configured and expected, it's loaded.
    # If LOGO_PATH exists but prepared_logo_image is None, it implies a startup loading issue.
    if os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo file exists but was not loaded. Watermarking may fail or be skipped.")
        # Depending on policy, you might raise an error or allow proceeding with a warning.
        # For now, let's allow it to proceed, worker will handle missing logo.
        # If logo is critical, uncomment:
        # raise HTTPException(status_code=500, detail="Server configuration error: Logo not available for watermarking.")
    
    # 3. Job ID Generation (Quick)
    job_id = str(uuid.uuid4())
    
    # 4. Add job to queue with URL, not local path (Quick)
    # The worker will handle downloading and saving.
    await queue.put((job_id, str(body.image), body.model, body.post_process))
    
    # 5. Initialize job status (Quick)
    results[job_id] = {
        "status": "queued",
        "input_image_url": str(body.image), # Store the input URL
        "original_local_path": None,      # Will be filled by worker
        "processed_path": None,
        "error_message": None
    }

    # 6. Generate response data (Quick)
    public_url_base = get_proxy_url(request)
    # Placeholder for the *processed* image. Output is still expected to be job_id.webp
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB 

    # 7. Return response (Quick)
    return {
        "status": "processing", # User sees "processing" even if it's just queued
        "image_links": [processed_image_placeholder_url],
        "etc": eta_seconds
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    job_info = results.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response_data = {
        "job_id": job_id, 
        "status": job_info["status"],
        "input_image_url": job_info.get("input_image_url")
    }
    public_url_base = get_proxy_url(request)

    if job_info.get("original_local_path"):
         # Provide a link to the downloaded original if it was saved
        original_filename = os.path.basename(job_info["original_local_path"])
        response_data["downloaded_original_image_url"] = f"{public_url_base}/originals/{original_filename}"

    if job_info["status"] == "done" and job_info.get("processed_path"):
        processed_filename = os.path.basename(job_info["processed_path"])
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
    elif job_info["status"] == "error":
        response_data["error_message"] = job_info["error_message"]
    
    return JSONResponse(content=response_data)


async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started.")
    global prepared_logo_image

    while True:
        job_id, image_url_str, model_name, post_process_flag = await queue.get()
        logger.info(f"Worker {worker_id} picked up job {job_id} for URL: {image_url_str}")
        
        original_file_path = None # Initialize
        results[job_id]["status"] = "downloading"

        try:
            # --- 1. Download Image ---
            async with httpx.AsyncClient(timeout=30.0) as client: # Added timeout
                img_response = await client.get(image_url_str)
                img_response.raise_for_status()
            
            image_content = await img_response.aread()
            content_type = img_response.headers.get("content-type", "").lower()

            if not content_type.startswith("image/"):
                raise ValueError(f"Invalid content type '{content_type}'. URL does not point to an image.")

            # --- 2. Determine Extension & Save Original Image ---
            extension = MIME_TO_EXT.get(content_type)
            if not extension:
                parsed_url_path = urllib.parse.urlparse(image_url_str).path
                _, ext_from_url = os.path.splitext(parsed_url_path)
                if ext_from_url and ext_from_url.lower() in MIME_TO_EXT.values():
                    extension = ext_from_url.lower()
                else:
                    extension = ".png" # Default
                    logger.warning(f"Job {job_id}: Could not determine extension for {image_url_str} (Content-Type: {content_type}). Defaulting to '.png'.")
            
            original_filename = f"{job_id}_original{extension}"
            original_file_path = os.path.join(UPLOADS_DIR, original_filename)
            results[job_id]["original_local_path"] = original_file_path # Update results

            async with aiofiles.open(original_file_path, 'wb') as out_file:
                await out_file.write(image_content)
            logger.info(f"Worker {worker_id} saved original image for job {job_id} to {original_file_path}")
            
            # --- 3. Process Image (rembg, square, watermark) ---
            results[job_id]["status"] = "processing" # More specific status
            
            with open(original_file_path, 'rb') as i:
                input_bytes = i.read()
            
            session = new_session(model_name)
            output_bytes = remove(
                input_bytes,
                session=session,
                post_process_mask=post_process_flag
            )
            
            img_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

            original_width, original_height = img_no_bg.size
            ratio = min(TARGET_SIZE / original_width, TARGET_SIZE / original_height) if original_width > 0 and original_height > 0 else 1
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            img_resized = img_no_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
            square_canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
            paste_x = (TARGET_SIZE - new_width) // 2
            paste_y = (TARGET_SIZE - new_height) // 2
            square_canvas.paste(img_resized, (paste_x, paste_y), img_resized)

            if prepared_logo_image:
                logo_w, logo_h = prepared_logo_image.size
                logo_pos_x = LOGO_MARGIN
                logo_pos_y = TARGET_SIZE - logo_h - LOGO_MARGIN
                square_canvas.paste(prepared_logo_image, (logo_pos_x, logo_pos_y), prepared_logo_image)

            final_image = square_canvas
            processed_filename = f"{job_id}.webp" # Output is always webp
            processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
            final_image.save(processed_file_path, 'WEBP', quality=90)

            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path
            logger.info(f"Worker {worker_id} finished job {job_id}. Processed image: {processed_file_path}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Worker {worker_id} HTTP error downloading for job {job_id} from {image_url_str}: {e.response.status_code} - {e.response.text}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Failed to download image: HTTP {e.response.status_code}."
        except (httpx.RequestError, ValueError, IOError, OSError) as e: # Catch download, save, or image format errors
            logger.error(f"Worker {worker_id} error during download/save/initial open for job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = str(e)
        except Exception as e: # Catch-all for other processing errors
            logger.error(f"Worker {worker_id} critical error processing job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Processing failed: {str(e)}"
        finally:
            queue.task_done()
            # Clean up original downloaded file if it exists and an error occurred *after* saving it
            # but *before* successful processing.
            # This is optional and depends on your cleanup strategy.
            if results[job_id]["status"] == "error" and original_file_path and os.path.exists(original_file_path):
                try:
                    # os.remove(original_file_path)
                    # logger.info(f"Cleaned up original file {original_file_path} for failed job {job_id}")
                    pass # Decide on cleanup
                except OSError as e_clean:
                    logger.error(f"Error cleaning up original file {original_file_path} for job {job_id}: {e_clean}")


@app.on_event("startup")
async def startup_event():
    global prepared_logo_image
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            if logo.width > LOGO_MAX_WIDTH:
                l_ratio = LOGO_MAX_WIDTH / logo.width
                l_new_width = LOGO_MAX_WIDTH
                l_new_height = int(logo.height * l_ratio)
                logo = logo.resize((l_new_width, l_new_height), Image.Resampling.LANCZOS)
            prepared_logo_image = logo
            logger.info(f"Logo loaded and prepared from {LOGO_PATH}. Dimensions: {prepared_logo_image.size if prepared_logo_image else 'None'}")
        except Exception as e:
            logger.error(f"Failed to load or prepare logo from {LOGO_PATH}: {e}")
            prepared_logo_image = None
    else:
        logger.warning(f"Logo file not found at {LOGO_PATH}. Watermarking will be skipped.")
        prepared_logo_image = None

    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} worker(s) started.")

app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><head><title>Image Processing API</title></head>
    <body><h1>Image Processing API is running</h1>
    <p>Use the /submit endpoint to process images.</p></body></html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
