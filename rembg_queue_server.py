from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageEnhance
import asyncio
import uuid
import io
import os
import aiofiles
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_CONCURRENT_TASKS = 1
ESTIMATED_TIME_PER_JOB = 9  # Rough estimate, can be fine-tuned

# Define directories
BASE_DIR = "/workspace/rmvbg"
UPLOADS_DIR = "/workspace/uploads"
PROCESSED_DIR = "/workspace/processed"
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# In-memory job tracking
# queue stores (job_id, original_image_path, model_name, post_process_flag)
queue = asyncio.Queue()
# results stores job_id -> { "status": "processing/done/error", "original_path": "...", "processed_path": "...", "error_message": "..." }
results = {}


EXPECTED_API_KEY = "secretApiKey"  # Replace in production

def get_proxy_url(request: Request):
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    # For services like Codeanywhere, the port might be included in x-forwarded-host or host
    # Ensure the scheme matches what the proxy expects (http vs https)
    # If running behind a proxy that terminates SSL, x-forwarded-proto is crucial.
    return f"{scheme}://{host}"

@app.post("/submit")
async def submit_image_for_processing(
    request: Request,
    image_file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("u2net"),
    post_process: bool = Form(False)
):
    if api_key != EXPECTED_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not image_file.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid file type. Please upload an image."}, status_code=400)

    job_id = str(uuid.uuid4())
    original_filename = f"{job_id}_{image_file.filename}"
    original_file_path = os.path.join(UPLOADS_DIR, original_filename)

    try:
        # Save the uploaded original file
        async with aiofiles.open(original_file_path, 'wb') as out_file:
            content = await image_file.read()
            await out_file.write(content)
        logger.info(f"📝 Original image saved: {original_file_path}")

    except Exception as e:
        logger.error(f"Error saving uploaded file {original_filename}: {e}")
        return JSONResponse({"error": f"Failed to save uploaded file: {e}"}, status_code=500)

    # Add job to queue
    await queue.put((job_id, original_file_path, model, post_process))
    
    # Initialize job status
    results[job_id] = {
        "status": "queued",
        "original_path": original_file_path,
        "processed_path": None,
        "error_message": None
    }

    public_url_base = get_proxy_url(request)
    original_image_url = f"{public_url_base}/originals/{original_filename}"
    
    # Calculate ETA
    # Position in queue (0-indexed for current job if it's next, +1 for 1-indexed queue position)
    # queue.qsize() includes the current job if it's just added and not yet picked by worker.
    # A more accurate ETA would be (items_before_this_in_queue + 1) * ESTIMATED_TIME_PER_JOB
    # For simplicity, we'll use queue.qsize() as an approximation of jobs ahead or currently processing.
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB 

    return {
        "status": "queued",
        "job_id": job_id,
        "original_image_url": original_image_url,
        "processed_image_url_placeholder": f"{public_url_base}/images/{job_id}.webp", # Placeholder
        "eta": eta_seconds,
        "message": "Image queued for processing."
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    job_info = results.get(job_id)
    public_url_base = get_proxy_url(request)

    if not job_info:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    original_filename = os.path.basename(job_info["original_path"])
    original_image_url = f"{public_url_base}/originals/{original_filename}"
    processed_image_url = f"{public_url_base}/images/{job_id}.webp" if job_info["status"] == "done" else None
    
    # Calculate ETA if still in queue or processing
    eta_seconds = 0
    if job_info["status"] in ["queued", "processing"]:
        # Find position in the actual queue._queue (internal list of asyncio.Queue)
        try:
            # This is a bit of a hack to inspect the queue content.
            # A more robust way would be to maintain a separate list of job_ids in order.
            queued_job_ids = [item[0] for item in list(queue._queue)]
            if job_id in queued_job_ids:
                position = queued_job_ids.index(job_id) + 1 # 1-indexed position
                eta_seconds = position * ESTIMATED_TIME_PER_JOB
        except Exception: # Handle cases where queue might be empty or job not found
            eta_seconds = ESTIMATED_TIME_PER_JOB # Default if something goes wrong with inspection

    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "original_image_url": original_image_url,
        "processed_image_url": processed_image_url,
        "eta": eta_seconds
    }
    if job_info["status"] == "error":
        response["error_message"] = job_info.get("error_message", "An unknown error occurred.")
    
    return response

async def image_processing_worker():
    logo_image = None
    if os.path.exists(LOGO_PATH):
        try:
            logo_image = Image.open(LOGO_PATH).convert("RGBA")
            logger.info(f"🖼️ Logo loaded from {LOGO_PATH}")
        except Exception as e:
            logger.error(f"Failed to load logo: {e}")
            logo_image = None
    else:
        logger.warning(f"Logo not found at {LOGO_PATH}")


    while True:
        job_id, original_image_path, model_name, post_process_flag = await queue.get()
        logger.info(f"🚀 Processing job {job_id} for image {original_image_path}")
        
        results[job_id]["status"] = "processing"

        try:
            # Load original image
            input_image = Image.open(original_image_path).convert("RGBA")
            
            # Initialize rembg session
            session = new_session(model_name=model_name)
            
            # Remove background
            removed_bg_image = remove(input_image, session=session, post_process_remove=post_process_flag)

            # --- Image post-processing (resize, canvas, logo) ---
            max_size = 1024
            width, height = removed_bg_image.size
            
            if width == 0 or height == 0: # Handle cases where rembg might return an empty image
                raise ValueError("Background removal resulted in an empty image.")

            scale = min(max_size / width, max_size / height) if width > 0 and height > 0 else 1.0
            new_size = (int(width * scale), int(height * scale))
            
            # Ensure new_size dimensions are at least 1x1
            new_size = (max(1, new_size[0]), max(1, new_size[1]))

            resized_image = removed_bg_image.resize(new_size, Image.Resampling.LANCZOS)

            # Create square canvas with white background
            # Ensure canvas is RGBA if pasting an RGBA image, then convert to RGB later if needed
            canvas = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255)) # White, fully opaque
            offset_x = (max_size - new_size[0]) // 2
            offset_y = (max_size - new_size[1]) // 2
            canvas.paste(resized_image, (offset_x, offset_y), resized_image) # Use resized_image as mask for transparency

            # Add logo if available
            if logo_image:
                logo_max_width = int(max_size * 0.20) # Logo width 20% of canvas
                logo_scale = min(1.0, logo_max_width / logo_image.width) if logo_image.width > 0 else 1.0
                
                logo_new_width = int(logo_image.width * logo_scale)
                logo_new_height = int(logo_image.height * logo_scale)

                if logo_new_width > 0 and logo_new_height > 0:
                    logo_resized = logo_image.resize(
                        (logo_new_width, logo_new_height),
                        Image.Resampling.LANCZOS
                    )
                    # Position logo (e.g., bottom-left with padding)
                    padding = 20
                    logo_pos_x = padding
                    logo_pos_y = max_size - logo_resized.height - padding
                    canvas.paste(logo_resized, (logo_pos_x, logo_pos_y), logo_resized) # Use logo_resized as mask

            # Convert final image to RGB before saving as WebP (WebP can handle alpha, but often RGB is fine)
            final_image_to_save = canvas.convert("RGB") 
            # If you need transparency in WebP, keep it as RGBA:
            # final_image_to_save = canvas 


            # Save as WebP, aiming for < 500KB
            processed_file_path = os.path.join(PROCESSED_DIR, f"{job_id}.webp")
            saved_successfully = False
            for quality in range(95, 25, -5): # Try qualities from 95 down to 30
                buffer = io.BytesIO()
                final_image_to_save.save(buffer, format="WEBP", quality=quality, lossless=False) # Use lossless=False for better compression
                if buffer.tell() < 500 * 1024:
                    with open(processed_file_path, "wb") as f:
                        f.write(buffer.getvalue())
                    logger.info(f"✅ Saved {job_id}.webp (Quality: {quality}, Size: {buffer.tell() / 1024:.2f} KB)")
                    saved_successfully = True
                    break
            
            if not saved_successfully: # Fallback if >500KB even at lowest quality tested
                final_image_to_save.save(processed_file_path, format="WEBP", quality=25, lossless=False)
                logger.warning(f"⚠️ Saved {job_id}.webp with fallback quality 25 (Size: {os.path.getsize(processed_file_path) / 1024:.2f} KB)")

            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_file_path

        except Exception as e:
            logger.error(f"❌ Error processing job {job_id} for {original_image_path}: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = str(e)
        finally:
            queue.task_done()

@app.on_event("startup")
async def startup_event():
    # Start worker tasks
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker())
        logger.info(f"Worker {i+1} started.")

# Serve static files
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# Serve root index.html
@app.get("/", response_class=FileResponse)
async def serve_index_html():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# Placeholder for the "upload item" page
@app.get("/upload-item", response_class=HTMLResponse)
async def upload_item_page(request: Request, images: Optional[List[str]] = None):
    if images is None:
        images = []
    
    image_list_html = "<ul>"
    for img_url in images:
        image_list_html += f"<li><a href='{img_url}' target='_blank'>{img_url}</a></li>"
    image_list_html += "</ul>"

    return f"""
    <html>
        <head><title>Upload Item</title></head>
        <body>
            <h1>Item Upload Page</h1>
            <p>This is a placeholder page. You would integrate your item creation form here.</p>
            <h2>Selected Images:</h2>
            {image_list_html if images else "<p>No images selected.</p>"}
            <p><a href="/">Back to Image Processing</a></p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    # This part is for local development. 
    # In a typical containerized deployment, Gunicorn or Uvicorn would be run directly.
    # Make sure BASE_DIR points to where your index.html and CM.png are.
    # For Codeanywhere, /workspace/rmvbg/ seems correct.
    uvicorn.run(app, host="0.0.0.0", port=8000)
