from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove
from PIL import Image
import asyncio, uuid, io, os

app = FastAPI()

# Config
MAX_CONCURRENT_TASKS = 1
ESTIMATED_TIME_PER_JOB = 5  # seconds
PROCESSED_DIR = "/workspace/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

queue = asyncio.Queue()
results = {}

def get_proxy_url(request: Request):
    """
    Constructs a RunPod proxy-safe public URL using the host header.
    """
    host = request.headers.get("host", "localhost")
    scheme = request.url.scheme
    return f"{scheme}://{host}"

@app.post("/submit")
async def submit_image(request: Request, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    contents = await file.read()
    await queue.put((job_id, contents))
    results[job_id] = None

    position = queue.qsize()
    eta_seconds = position * ESTIMATED_TIME_PER_JOB

    public_base_url = get_proxy_url(request)

    return {
        "status": "processing",
        "image_links": [f"{public_base_url}/images/{job_id}.png"],
        "eta": eta_seconds,
        "id": job_id
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    result = results.get(job_id)
    public_base_url = get_proxy_url(request)
    image_url = f"{public_base_url}/images/{job_id}.png"

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
        job_id, contents = await queue.get()
        try:
            input_image = Image.open(io.BytesIO(contents))
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

# Serve static processed images
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="images")
