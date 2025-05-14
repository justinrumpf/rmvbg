from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageEnhance
import asyncio, uuid, io, os, requests

app = FastAPI()

MAX_CONCURRENT_TASKS = 1
ESTIMATED_TIME_PER_JOB = 9
PROCESSED_DIR = "/workspace/processed"
LOGO_PATH = "/workspace/rmvbg/CM.png"  # Updated to use bundled logo

os.makedirs(PROCESSED_DIR, exist_ok=True)

queue = asyncio.Queue()
results = {}

class ImageRequest(BaseModel):
    image: str
    key: str
    model: str = "u2net"
    post_process: bool = False
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"

EXPECTED_API_KEY = "secretApiKey"  # Replace in production

def get_proxy_url(request: Request):
    host = request.headers.get("host", "localhost")
    scheme = request.url.scheme
    return f"{scheme}://{host}"

@app.post("/submit")
async def submit_image(request: Request, data: ImageRequest):
    if data.key != EXPECTED_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = str(uuid.uuid4())
    queue.put_nowait((job_id, data.image, data.model, data.post_process))
    results[job_id] = None

    eta_seconds = queue.qsize() * ESTIMATED_TIME_PER_JOB
    public_url = get_proxy_url(request)

    return {
        "status": "processing",
        "image_links": [f"{public_url}/images/{job_id}.webp"],
        "eta": eta_seconds,
        "id": job_id
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    result = results.get(job_id)
    public_url = get_proxy_url(request)
    image_url = f"{public_url}/images/{job_id}.webp"

    job_keys = list(queue._queue)
    position = next((i for i, (k, *_ ) in enumerate(job_keys) if k == job_id), None)
    eta_seconds = (position * ESTIMATED_TIME_PER_JOB) if position is not None else 0

    if result is None:
        return {
            "status": "processing",
            "image_links": [image_url],
            "eta": eta_seconds,
            "id": job_id
        }
    elif result == "error":
        return {
            "status": "error",
            "image_links": [],
            "eta": 0,
            "id": job_id
        }
    else:
        return {
            "status": "done",
            "image_links": [image_url],
            "eta": 0,
            "id": job_id
        }

async def worker():
    logo = Image.open(LOGO_PATH).convert("RGBA") if os.path.exists(LOGO_PATH) else None

    while True:
        job_id, image_url, model_name, post_process = await queue.get()
        try:
            print(f"📥 Downloading image from {image_url}...")
            response = requests.get(image_url)
            response.raise_for_status()

            input_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            session = new_session(model_name=model_name)
            removed = remove(input_image, session=session, post_process=post_process)

            # Resize to fit within 1024x1024 while preserving aspect ratio
            max_size = 1024
            width, height = removed.size
            scale = min(max_size / width, max_size / height)
            new_size = (int(width * scale), int(height * scale))
            resized = removed.resize(new_size, Image.LANCZOS)

            # Create square canvas with white background
            canvas = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
            offset_x = (max_size - new_size[0]) // 2
            offset_y = (max_size - new_size[1]) // 2
            canvas.paste(resized, (offset_x, offset_y), resized)

            # Add logo if available
            if logo:
                logo_max_width = int(max_size * 0.25)
                logo_scale = min(1.0, logo_max_width / logo.width)
                logo_resized = logo.resize(
                    (int(logo.width * logo_scale), int(logo.height * logo_scale)),
                    Image.LANCZOS
                )
                logo_offset_x = 20
                logo_offset_y = max_size - logo_resized.height - 20
                canvas.paste(logo_resized, (logo_offset_x, logo_offset_y), logo_resized)

            final_image = canvas.convert("RGB")

            # Save WebP under 500KB
            out_path = os.path.join(PROCESSED_DIR, f"{job_id}.webp")
            for quality in range(90, 30, -5):
                buffer = io.BytesIO()
                final_image.save(buffer, format="WEBP", quality=quality)
                if buffer.tell() < 500 * 1024:
                    with open(out_path, "wb") as f:
                        f.write(buffer.getvalue())
                    print(f"✅ Saved {job_id}.webp at quality={quality} ({buffer.tell()} bytes)")
                    break
            else:
                final_image.save(out_path, format="WEBP", quality=30)
                print(f"⚠️ Saved {job_id}.webp at fallback quality=30")

            results[job_id] = out_path

        except Exception as e:
            print(f"❌ Error processing job {job_id}: {e}")
            results[job_id] = "error"
        finally:
            queue.task_done()

@app.on_event("startup")
async def start_workers():
    for _ in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(worker())

# Serve processed images
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="images")

# Serve root index.html from repo root
@app.get("/")
async def serve_index():
    return FileResponse("/workspace/rmvbg/index.html")
