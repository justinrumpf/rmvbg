from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import asyncio, uuid, io, os
import requests

# Replace this with your actual key validation logic if needed
EXPECTED_API_KEY = "secretApiKey"

class ImageRequest(BaseModel):
    image: str
    key: str
    steps: int = 20
    samples: int = 1
    resolution: str = "1024x1024"

app = FastAPI()

MAX_CONCURRENT_TASKS = 1
ESTIMATED_TIME_PER_JOB = 5
PROCESSED_DIR = "/workspace/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

queue = asyncio.Queue()
results = {}

def get_proxy_url(request: Request):
    host = request.headers.get("host", "localhost")
    scheme = request.url.scheme
    return f"{scheme}://{host}"

@app.post("/submit")
async def submit_image(request: Request, data: ImageRequest):
    if data.key != EXPECTED_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    job_id = str(uuid.uuid4())
    queue.put_nowait((job_id, data.image))  # just pass the URL
    results[job_id] = None

    eta_seconds = queue.qsize() * ESTIMATED_TIME_PER_JOB
    public_url = get_proxy_url(request)

    return {
        "status": "processing",
        "image_links": [f"{public_url}/images/{job_id}.png"],
        "eta": eta_seconds,
        "id": job_id
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    result = results.get(job_id)
    public_url = get_proxy_url(request)
    image_url = f"{public_url}/images/{job_id}.png"

    job_keys = list(queue._queue)
    position = next((i for i, (k, _) in enumerate(job_keys) if k == job_id), None)
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
    while True:
        job_id, image_url = await queue.get()
        try:
            print(f"Downloading image from {image_url}...")
            response = requests.get(image_url)
            response.raise_for_status()

            input_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            output_image = remove(input_image)

            out_path = os.path.join(PROCESSED_DIR, f"{job_id}.png")
            output_image.save(out_path)
            results[job_id] = out_path
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            results[job_id] = "error"
        finally:
            queue.task_done()

@app.on_event("startup")
async def start_workers():
    for _ in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(worker())

app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="images")
