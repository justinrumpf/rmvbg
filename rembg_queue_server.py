from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse # FileResponse not needed for root anymore
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from rembg import remove, new_session # Make sure you have rembg[gpu] installed for GPU usage
from PIL import Image

import asyncio
import uuid
import io
import os
import aiofiles
import logging
import httpx
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Configuration Constants ---
# YOU MUST TUNE MAX_CONCURRENT_TASKS BASED ON YOUR HARDWARE AND TESTING
MAX_CONCURRENT_TASKS = 8  # Example: Start with number of CPU cores or based on GPU capacity
MAX_QUEUE_SIZE = 5000     # Max number of jobs to hold in memory if workers are busy
ESTIMATED_TIME_PER_JOB = 13 # Estimated seconds per job (download, process, save) - re-evaluate with more workers
TARGET_SIZE = 1024        # Target dimension for squared image
LOGO_MAX_WIDTH = 150      # Max width for the logo overlay
LOGO_MARGIN = 20          # Margin for the logo from image edges
HTTP_CLIENT_TIMEOUT = 30.0 # Timeout for downloading images from URLs

# --- Directory and File Paths ---
BASE_DIR = "/workspace/rmvbg"       # For logo primarily
UPLOADS_DIR = "/workspace/uploads"  # For downloaded original images
PROCESSED_DIR = "/workspace/processed" # For final processed images
LOGO_FILENAME = "logo.png"
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME)

# --- Global State ---
prepared_logo_image = None
# Initialize queue with maxsize for backpressure
queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
# Stores job_id -> { "status": ..., "input_image_url": ..., ..., "status_check_url": ... }
results = {}

EXPECTED_API_KEY = "secretApiKey"  # Replace in production

MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'image/webp': '.webp',
}

# --- Pydantic Models ---
class SubmitRequestBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = "u2net"
    post_process: bool = False
    steps: int = 20 # Accepted but not used by rembg
    samples: int = 1 # Accepted but not used by rembg
    resolution: str = "1024x1024" # Effectively enforced by TARGET_SIZE

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
    # 1. API Key Validation (Quick)
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Logo Configuration Check (Quick local check)
    if os.path.exists(LOGO_PATH) and not prepared_logo_image:
        logger.error("Logo file exists at startup but was not loaded. Watermarking may fail or be skipped. Check startup logs.")
        # Decide if this is critical enough to stop submission. For now, allows proceeding.
    
    # 3. Job ID Generation (Quick)
    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    
    # 4. Add job to queue (Quick, with backpressure)
    try:
        # The worker will handle downloading and saving.
        queue.put_nowait((job_id, str(body.image), body.model, body.post_process))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full (max size: {MAX_QUEUE_SIZE}). Rejecting request for image {body.image}.")
        raise HTTPException(status_code=503, detail=f"Server is temporarily overloaded (queue full). Please try again later. Max queue size: {MAX_QUEUE_SIZE}")
    
    # 5. Initialize job status in results (Quick)
    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued",
        "input_image_url": str(body.image),
        "original_local_path": None,
        "processed_path": None,
        "error_message": None,
        "status_check_url": status_check_url
    }

    # 6. Generate response data (Quick)
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    # ETA based on current queue size. If queue is often full, this might be optimistic for new requests.
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB 

    # 7. Return response (Quick)
    return {
        "status": "processing", # User sees "processing"
        "job_id": job_id,
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
        "job_id": job_id, 
        "status": job_info.get("status"),
        "input_image_url": job_info.get("input_image_url"),
        "status_check_url": job_info.get("status_check_url")
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
    global prepared_logo_image # Access pre-loaded logo

    while True:
        job_id, image_url_str, model_name, post_process_flag = await queue.get()
        logger.info(f"Worker {worker_id} picked up job {job_id} for URL: {image_url_str}")
        
        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job ID {job_id} from queue not found in results dict. This shouldn't happen. Skipping.")
            queue.task_done()
            continue

        original_file_path = None # Initialize for cleanup logic
        results[job_id]["status"] = "downloading"

        try:
            # 1. Download Image
            async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
                img_response = await client.get(image_url_str)
                img_response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            
            image_content = await img_response.aread()
            content_type = img_response.headers.get("content-type", "").lower()


        try:
            # 1. Download Image and Initial Content-Type Check
            image_content = await img_response.aread()
            original_content_type_header = img_response.headers.get("content-type", "unknown") # Get original for logging
            content_type = original_content_type_header.lower() # Work with lowercase
            
            logger.info(f"Job {job_id}: Received initial Content-Type='{original_content_type_header}' for URL {image_url_str}")

            # Attempt to infer from URL if original Content-Type is problematic (octet-stream or not image/*)
            if content_type == "application/octet-stream" or not content_type.startswith("image/"):
                if content_type == "application/octet-stream":
                    logger.warning(f"Job {job_id}: Original Content-Type is 'application/octet-stream'. Attempting to infer from URL file extension.")
                else: # It's not "image/" and also not "application/octet-stream", but some other type
                    logger.warning(f"Job {job_id}: Original Content-Type is '{content_type}', which is not 'image/*'. Attempting to infer from URL file extension as a fallback.")

                file_extension_from_url = os.path.splitext(urllib.parse.urlparse(image_url_str).path)[1].lower()
                logger.info(f"Job {job_id}: Parsed file extension from URL: '{file_extension_from_url}'")

                potential_new_content_type = None
                if file_extension_from_url == ".webp":
                    potential_new_content_type = "image/webp"
                elif file_extension_from_url == ".png":
                    potential_new_content_type = "image/png"
                elif file_extension_from_url in [".jpg", ".jpeg"]: # Common variations for JPEG
                    potential_new_content_type = "image/jpeg"
                elif file_extension_from_url == ".gif":
                    potential_new_content_type = "image/gif"
                elif file_extension_from_url == ".bmp":
                    potential_new_content_type = "image/bmp"
                elif file_extension_from_url in [".tif", ".tiff"]: # Common variations for TIFF
                    potential_new_content_type = "image/tiff"
                # Add other common image extensions if needed

                if potential_new_content_type:
                    logger.info(f"Job {job_id}: Overriding Content-Type from '{original_content_type_header}' to '{potential_new_content_type}' based on URL extension '{file_extension_from_url}'.")
                    content_type = potential_new_content_type # Update content_type
                else:
                    logger.warning(f"Job {job_id}: URL file extension '{file_extension_from_url}' is not in the recognized list for Content-Type override. Original Content-Type '{original_content_type_header}' will be used for the final check.")
                    # If no override, content_type remains what it was (e.g., 'application/octet-stream' or other non-image type)

            # Final validation after potential override (this is effectively your line 180)
            if not content_type.startswith("image/"):
                logger.error(f"Job {job_id}: FINAL Content-Type check FAILED. Content-Type is '{content_type}'. URL: {image_url_str}")
                raise ValueError(f"Invalid content type '{content_type}' from URL. Not an image.")
            
            logger.info(f"Job {job_id}: Proceeding with Content-Type '{content_type}'.")

            # 2. Determine Extension for saving & Save Original Image to UPLOADS_DIR
            # This 'extension' is for the *saved file*, derived from the final 'content_type'
            saved_file_extension = MIME_TO_EXT.get(content_type) 
            if not saved_file_extension:
                # If MIME_TO_EXT doesn't have it (e.g. image/svg+xml), try to use the URL's extension if it's a known one
                # Or fallback to a default like .bin or raise an error if strictness is needed
                parsed_url_path_for_save_ext = urllib.parse.urlparse(image_url_str).path
                _, ext_from_url_for_save = os.path.splitext(parsed_url_path_for_save_ext)
                ext_from_url_for_save = ext_from_url_for_save.lower()

                if ext_from_url_for_save and ext_from_url_for_save in MIME_TO_EXT.values(): # Check if this suffix is something we map
                    saved_file_extension = ext_from_url_for_save
                    logger.info(f"Job {job_id}: Using URL extension '{saved_file_extension}' for saving original file as Content-Type '{content_type}' was not in MIME_TO_EXT map.")
                else:
                    saved_file_extension = ".bin" # A generic binary extension if truly unknown
                    logger.warning(f"Job {job_id}: Could not determine specific file extension for saving original from Content-Type '{content_type}' or URL. Defaulting to '{saved_file_extension}'.")
            
            original_filename = f"{job_id}_original{saved_file_extension}"
            original_file_path = os.path.join(UPLOADS_DIR, original_filename)
            results[job_id]["original_local_path"] = original_file_path # Update results with local path

            async with aiofiles.open(original_file_path, 'wb') as out_file:
                await out_file.write(image_content)
            logger.info(f"Worker {worker_id} saved original image for job {job_id} to {original_file_path}")
            
            # 3. Process Image (rembg, square, watermark)
            results[job_id]["status"] = "processing"
            
            # Use a blocking open here as rembg/PIL will be blocking anyway for this part
            with open(original_file_path, 'rb') as i:
                input_bytes = i.read()
            
            # --- rembg: Background Removal ---
            session = new_session(model_name) # Create session per model or reuse if thread-safe and efficient
            output_bytes = remove(
                input_bytes,
                session=session,
                post_process_mask=post_process_flag
            )
            
            img_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

            # --- Pillow: Squaring ---
            original_width, original_height = img_no_bg.size
            if original_width == 0 or original_height == 0: # Handle invalid image dimensions
                raise ValueError("Image dimensions are zero after background removal.")

            ratio = min(TARGET_SIZE / original_width, TARGET_SIZE / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            img_resized = img_no_bg.resize((new_width, new_height), Image.Resampling.LANCZOS)
            square_canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0)) # Transparent canvas
            paste_x = (TARGET_SIZE - new_width) // 2
            paste_y = (TARGET_SIZE - new_height) // 2
            square_canvas.paste(img_resized, (paste_x, paste_y), img_resized) # Paste using its own alpha

            # --- Pillow: Watermarking ---
            if prepared_logo_image:
                logo_w, logo_h = prepared_logo_image.size
                logo_pos_x = LOGO_MARGIN
                logo_pos_y = TARGET_SIZE - logo_h - LOGO_MARGIN
                square_canvas.paste(prepared_logo_image, (logo_pos_x, logo_pos_y), prepared_logo_image)
            else:
                logger.info(f"Job {job_id}: Skipping watermark as logo is not available/loaded.")

            final_image = square_canvas
            processed_filename = f"{job_id}.webp" # Save final output as WEBP
            processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
            
            # Save the final image
            final_image.save(processed_file_path, 'WEBP', quality=90) # Adjust quality if needed

            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path
            logger.info(f"Worker {worker_id} finished job {job_id}. Processed image: {processed_file_path}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Worker {worker_id} HTTP error {e.response.status_code} for job {job_id} from {image_url_str}: {e.response.text}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Failed to download image: HTTP {e.response.status_code} from {image_url_str}."
        except httpx.RequestError as e: # Covers DNS, connection, timeout errors
            logger.error(f"Worker {worker_id} Network error downloading for job {job_id} from {image_url_str}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Network error downloading image from {image_url_str}: {type(e).__name__}."
        except (ValueError, IOError, OSError) as e: # Covers invalid image data, file save errors etc.
            logger.error(f"Worker {worker_id} data/file error for job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Data or file error: {str(e)}"
        except Exception as e: # Catch-all for unexpected processing errors
            logger.error(f"Worker {worker_id} critical error processing job {job_id}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = f"Unexpected processing error: {str(e)}"
        finally:
            queue.task_done() # Crucial: signal that this queue item is processed
            # Optional: Cleanup original downloaded file on error
            # if results[job_id]["status"] == "error" and original_file_path and os.path.exists(original_file_path):
            #     try:
            #         os.remove(original_file_path)
            #         logger.info(f"Cleaned up original file {original_file_path} for failed job {job_id}")
            #     except OSError as e_clean:
            #         logger.error(f"Error cleaning up original file {original_file_path} for job {job_id}: {e_clean}")

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    global prepared_logo_image
    logger.info("Application startup...")

    # Create directories if they don't exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"Uploads directory: {UPLOADS_DIR}, Processed directory: {PROCESSED_DIR}")

    # Load and prepare the logo
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
            logger.error(f"Failed to load or prepare logo from {LOGO_PATH}: {e}", exc_info=True)
            prepared_logo_image = None # Ensure it's None on failure
    else:
        logger.warning(f"Logo file not found at {LOGO_PATH}. Watermarking will be skipped.")
        prepared_logo_image = None

    # Start worker tasks
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} image processing worker(s) started.")
    logger.info(f"Job queue max size: {MAX_QUEUE_SIZE}.")

# --- Static File Serving ---
# Serves processed images (e.g., /images/job_id.webp)
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
# Serves original downloaded images (e.g., /originals/job_id_original.png)
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False) # Exclude from OpenAPI docs
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Processing API</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
            .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Resale1... In The House!</h1>
            <p>This service provides background removal and image processing capabilities.</p>
            <p>Current settings:
                <ul>
                    <li>Max Concurrent Workers: """ + str(MAX_CONCURRENT_TASKS) + """</li>
                    <li>Max Queue Size: """ + str(MAX_QUEUE_SIZE) + """</li>
                </ul>
            </p>
        </div>
    </body>
    </html>
    """

# --- Main Execution (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    # For production, use a proper ASGI server like Uvicorn with Gunicorn workers:
    # gunicorn -w (num_workers) -k uvicorn.workers.UvicornWorker main:app
    # Number of Gunicorn workers is typically (2 * CPU_CORES) + 1.
    # Each Gunicorn worker will run its own FastAPI app instance with MAX_CONCURRENT_TASKS asyncio workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)
