from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl # HttpUrl for URL validation
from rembg import remove, new_session # Assuming these are used by the worker
# from PIL import Image, ImageOps, ImageEnhance # Assuming these are used by the worker

import asyncio
import uuid
import io
import os
import aiofiles
import logging
from typing import List, Optional
import httpx # For downloading image from URL
import urllib.parse # For parsing URL to get extension

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MAX_CONCURRENT_TASKS = 1
ESTIMATED_TIME_PER_JOB = 9  # Rough estimate, can be fine-tuned

# Define directories
BASE_DIR = "/workspace/rmvbg" # Adjust if your base directory is different
UPLOADS_DIR = "/workspace/uploads"
PROCESSED_DIR = "/workspace/processed"
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# In-memory job tracking
queue = asyncio.Queue()
results = {}

EXPECTED_API_KEY = "secretApiKey"  # Replace in production

# Helper for MIME types, you can expand this
MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
}

# Pydantic model for the new request body
class SubmitRequestBody(BaseModel):
    image: HttpUrl  # Validates if it's a URL
    key: str
    model: str = "u2net"
    post_process: bool = False
    # These fields are part of the API spec but not directly used by rembg
    # They are included here to match the desired API request structure.
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
    body: SubmitRequestBody # Use the Pydantic model for the request body
):
    if body.key != EXPECTED_API_KEY:
        # Use HTTPException for standard error responses
        raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid.uuid4())
    
    # Download the image from the URL
    try:
        async with httpx.AsyncClient() as client:
            img_response = await client.get(str(body.image)) # Convert HttpUrl to str for httpx
            img_response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    except httpx.RequestError as e:
        logger.error(f"Error downloading image from {body.image}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading image {body.image}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching image from URL: {e.response.reason_phrase}")


    image_content = await img_response.aread()
    content_type = img_response.headers.get("content-type", "").lower()

    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. URL does not point to an image.")

    # Determine file extension
    extension = MIME_TO_EXT.get(content_type)
    if not extension:
        # Fallback: try to get extension from URL path
        parsed_url_path = urllib.parse.urlparse(str(body.image)).path
        _, ext_from_url = os.path.splitext(parsed_url_path)
        if ext_from_url and ext_from_url.lower() in MIME_TO_EXT.values():
            extension = ext_from_url
        else:
            extension = ".png" # Default extension if cannot be determined
            logger.warning(f"Could not determine extension from Content-Type '{content_type}' or URL '{body.image}'. Defaulting to '.png'.")
    
    original_filename = f"{job_id}_original{extension}"
    original_file_path = os.path.join(UPLOADS_DIR, original_filename)

    try:
        # Save the downloaded original file
        async with aiofiles.open(original_file_path, 'wb') as out_file:
            await out_file.write(image_content)
        logger.info(f"📝 Original image saved: {original_file_path} from URL {body.image}")

    except Exception as e:
        logger.error(f"Error saving downloaded file {original_filename}: {e}")
        # Clean up if save fails? (e.g., os.remove(original_file_path)) - consider atomicity
        raise HTTPException(status_code=500, detail=f"Failed to save downloaded file: {e}")

    # Add job to queue using data from the JSON body
    await queue.put((job_id, original_file_path, body.model, body.post_process))
    
    results[job_id] = {
        "status": "queued", # Internally, it's "queued"
        "original_path": original_file_path,
        "processed_path": None,
        "error_message": None,
        # You might want to store other info from body if needed by worker or status endpoint
        # "model_used": body.model, 
        # "post_process_applied": body.post_process
    }

    public_url_base = get_proxy_url(request)
    # The placeholder URL for the processed image. Assumes output is .webp
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.webp"
    
    eta_seconds = (queue.qsize()) * ESTIMATED_TIME_PER_JOB 

    # Return the new JSON response format
    return {
        "status": "processing", # As per your API spec, even though it's queued
        "image_links": [processed_image_placeholder_url],
        "etc": eta_seconds  # "etc" for Estimated Time of Completion
    }

# --- The rest of your existing code ---
# (Make sure the image_processing_worker and startup_event are correctly implemented)

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    job_info = results.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Construct a proper response. For example:
    response_data = {"job_id": job_id, "status": job_info["status"]}
    if job_info["status"] == "done":
        public_url_base = get_proxy_url(request)
        # Assuming processed files are named job_id.webp as per placeholder
        processed_filename = f"{job_id}.webp" 
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
    elif job_info["status"] == "error":
        response_data["error_message"] = job_info["error_message"]
    
    # If you want to mirror the example for completed jobs more closely:
    # {
    #   "status": "success", // "success" if done, "processing", "failed"
    #   "id": "some_id_maybe_job_id_or_from_upstream",
    #   "output": ["url_to_processed_image.png"],
    #   "generationTime": 1.23, // If you track this
    #   "meta": { ... } // Any other metadata
    # }
    # This would require changes to how results are stored and presented.
    # For now, returning basic job_info:
    return JSONResponse(content=response_data)


# Placeholder for the actual image processing worker
# This needs to be properly implemented to fetch from queue, process, and update results
async def image_processing_worker(worker_id: int):
    logger.info(f"Worker {worker_id} started.")
    while True:
        try:
            job_id, original_file_path, model_name, post_process_flag = await queue.get()
            logger.info(f"Worker {worker_id} picked up job {job_id}")
            results[job_id]["status"] = "processing"
            
            # --- Actual Processing Logic using rembg ---
            # This is a simplified example; adapt as needed.
            try:
                with open(original_file_path, 'rb') as i:
                    input_bytes = i.read()
                
                # Choose session based on model_name
                session = new_session(model_name) # Or however you manage sessions

                output_bytes = remove(
                    input_bytes,
                    session=session,
                    post_process_mask=post_process_flag
                    # Add other rembg parameters if needed, e.g., alpha_matting, etc.
                )
                
                # Save processed image (e.g., as WEBP)
                processed_filename = f"{job_id}.webp" # Matches placeholder
                processed_file_path = os.path.join(PROCESSED_DIR, processed_filename)
                
                # Convert to PIL Image to save as WEBP if output_bytes is PNG
                img = Image.open(io.BytesIO(output_bytes))
                img.save(processed_file_path, 'WEBP')

                results[job_id]["status"] = "done"
                results[job_id]["processed_path"] = processed_file_path
                logger.info(f"Worker {worker_id} finished job {job_id}. Processed image: {processed_file_path}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing job {job_id}: {e}")
                results[job_id]["status"] = "error"
                results[job_id]["error_message"] = str(e)
            # --- End of Processing Logic ---
            
            queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} stopping.")
            break
        except Exception as e: # Catch-all for unexpected errors in worker loop
            logger.error(f"Critical error in worker {worker_id}: {e}")
            # Potentially re-queue the job or mark as failed permanently
            # For now, just log and continue to next job attempt.
            if 'job_id' in locals() and job_id in results: # If job_id was retrieved
                 results[job_id]["status"] = "error"
                 results[job_id]["error_message"] = "Worker failed unexpectedly."
                 queue.task_done() # Ensure task_done is called if job was retrieved
            await asyncio.sleep(1) # Avoid tight loop on persistent errors


@app.on_event("startup")
async def startup_event():
    # Import PIL Image here if not already at top, to be used in worker
    global Image
    from PIL import Image

    # Start worker tasks
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    logger.info(f"{MAX_CONCURRENT_TASKS} worker(s) started.")


# Serve static files
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# Serve root index.html
@app.get("/", response_class=FileResponse)
async def serve_index_html():
    # Ensure index.html is in BASE_DIR or adjust path
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.warning(f"index.html not found at {index_path}. Serving basic message.")
        return HTMLResponse("<html><body><h1>Image Processing Service</h1><p>index.html not found.</p></body></html>")
    return FileResponse(index_path)

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
    # Make sure BASE_DIR points to where your index.html is.
    # For example, if your script is /workspace/rmvbg/main.py and index.html is in the same folder,
    # BASE_DIR = "/workspace/rmvbg" is correct.
    uvicorn.run(app, host="0.0.0.0", port=8000)
