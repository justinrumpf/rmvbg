import asyncio
import uuid
import io
import os
import aiofiles
import logging
import httpx
import urllib.parse
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

# --- CREATE DIRECTORIES AT THE VERY TOP ---
UPLOADS_DIR_STATIC = "/workspace/uploads"
PROCESSED_DIR_STATIC = "/workspace/processed"
BASE_DIR_STATIC = "/workspace/rmvbg"
STATS_DIR_STATIC = "/workspace/stats"

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{BASE_DIR_STATIC}/api.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Create all necessary directories
REQUIRED_DIRS = [UPLOADS_DIR_STATIC, PROCESSED_DIR_STATIC, BASE_DIR_STATIC, STATS_DIR_STATIC]
for directory in REQUIRED_DIRS:
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    except OSError as e:
        logger.error(f"CRITICAL: Error creating directory {directory}: {e}", exc_info=True)

from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter
import pillow_heif
import numpy as np
from scipy import ndimage
import cv2

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(lifespan=lifespan)

# Enhanced CORS setup
origins = [
    "null",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "https://*.vercel.app",
    "https://*.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Enhanced Configuration Constants ---
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_WORKERS", "8"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "5000"))
ESTIMATED_TIME_PER_JOB = 35
TARGET_SIZE = int(os.getenv("TARGET_SIZE", "1024"))
HTTP_CLIENT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default

# Enhanced image processing options
ENABLE_LOGO_WATERMARK = os.getenv("ENABLE_LOGO", "false").lower() == "true"
LOGO_MAX_WIDTH = 150
LOGO_MARGIN = 20
LOGO_FILENAME = "logo.png"

# Available models, output formats, edge processing types, and shadow removal options
AVAILABLE_MODELS = [
    "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", 
    "silueta", "isnet-general-use", "isnet-anime", "sam",
    "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
    "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive"
]
OUTPUT_FORMATS = ["webp", "png", "jpg", "jpeg"]
BACKGROUND_COLORS = ["white", "transparent", "black", "custom"]
EDGE_PROCESSING_TYPES = ["none", "sharpness", "edge_detect_sobel", "edge_detect_canny", "edge_enhance", "unsharp_mask"]
SHADOW_REMOVAL_METHODS = ["none", "fill_holes", "enhance_shadows", "hybrid"]

# --- Directory and File Paths ---
BASE_DIR = BASE_DIR_STATIC
UPLOADS_DIR = UPLOADS_DIR_STATIC
PROCESSED_DIR = PROCESSED_DIR_STATIC
STATS_DIR = STATS_DIR_STATIC
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILENAME) if ENABLE_LOGO_WATERMARK else ""

# --- Global State ---
prepared_logo_image = None
queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
results: Dict[str, Dict[str, Any]] = {}
processing_stats: Dict[str, Any] = {
    "total_processed": 0,
    "total_errors": 0,
    "average_processing_time": 0,
    "model_usage": {},
    "daily_stats": {}
}
EXPECTED_API_KEY = os.getenv("API_KEY", "secretApiKey")

# Enhanced MIME type mapping
MIME_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'image/heic': '.heic',
    'image/heif': '.heif',
    'image/avif': '.avif'
}

# --- Enhanced Pydantic Models ---
class ProcessingOptions(BaseModel):
    resize_to: Optional[int] = Field(default=1024, ge=256, le=4096, description="Target size for resizing")
    background_color: str = Field(default="white", description="Background color: white, transparent, black, or hex code")
    output_format: str = Field(default="webp", description="Output format: webp, png, jpg")
    quality: int = Field(default=90, ge=1, le=100, description="Output quality for lossy formats")
    edge_processing: str = Field(default="none", description="Edge processing type")
    # Shadow removal options
    shadow_removal: str = Field(default="none", description="Shadow removal method")
    shadow_fill_kernel_size: int = Field(default=15, ge=5, le=50, description="Kernel size for morphological operations")
    shadow_blur_radius: float = Field(default=2.0, ge=0.5, le=10.0, description="Blur radius for shadow smoothing")
    shadow_brightness_boost: float = Field(default=1.2, ge=1.0, le=2.0, description="Brightness enhancement factor")
    shadow_contrast_reduction: float = Field(default=0.9, ge=0.5, le=1.0, description="Contrast reduction factor")
    # Sharpness enhancement options
    sharpness_factor: float = Field(default=1.5, ge=0.5, le=3.0, description="Sharpness enhancement factor")
    # Edge detection options
    canny_low_threshold: int = Field(default=50, ge=10, le=200, description="Canny edge detection low threshold")
    canny_high_threshold: int = Field(default=150, ge=50, le=300, description="Canny edge detection high threshold")
    # Unsharp mask options
    unsharp_radius: float = Field(default=2.0, ge=0.5, le=5.0, description="Unsharp mask radius")
    unsharp_percent: float = Field(default=150, ge=50, le=300, description="Unsharp mask strength percentage")
    unsharp_threshold: int = Field(default=3, ge=0, le=10, description="Unsharp mask threshold")
    # Edge overlay options for detection methods
    edge_overlay_opacity: float = Field(default=1.0, ge=0.1, le=1.0, description="Edge overlay opacity (for detection methods)")
    preserve_original: bool = Field(default=False, description="Overlay edges on original image instead of replacing")

class SubmitJsonBody(BaseModel):
    image: HttpUrl
    key: str
    model: str = Field(default="u2net", description="Background removal model")
    options: Optional[ProcessingOptions] = None

class BatchProcessRequest(BaseModel):
    images: List[HttpUrl] = Field(..., max_items=10, description="List of image URLs to process")
    key: str
    model: str = Field(default="u2net", description="Background removal model")
    options: Optional[ProcessingOptions] = None

class JobResponse(BaseModel):
    status: str
    job_id: str
    image_links: List[str]
    eta: float
    status_check_url: str
    original_image_url: Optional[str] = None

# --- Enhanced Helper Functions ---
def get_proxy_url(request: Request) -> str:
    """Get the proxy URL for the request."""
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{scheme}://{host}"

def format_size(num_bytes: int) -> str:
    """Format byte size into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    else:
        return f"{num_bytes/1024**2:.2f} MB"

def validate_file_size(file_size: int) -> None:
    """Validate uploaded file size."""
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {format_size(MAX_FILE_SIZE)}"
        )

def apply_shadow_removal(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """
    Apply shadow removal post-processing to handle internal shadows and holes.
    
    Methods:
    - none: No shadow removal
    - fill_holes: Fill holes in alpha mask using morphological operations
    - enhance_shadows: Brighten and soften shadows instead of removing
    - hybrid: Combination of fill_holes and enhance_shadows
    """
    if options.shadow_removal == "none":
        return image
    
    logger.info(f"Applying shadow removal method: {options.shadow_removal}")
    
    if options.shadow_removal == "fill_holes":
        return fill_internal_holes(image, options)
    elif options.shadow_removal == "enhance_shadows":
        return enhance_shadow_areas(image, options)
    elif options.shadow_removal == "hybrid":
        # First enhance shadows, then fill holes
        enhanced = enhance_shadow_areas(image, options)
        return fill_internal_holes(enhanced, options)
    else:
        return image

def fill_internal_holes(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """
    Fill holes in the alpha mask to remove unwanted transparent areas like shadows.
    """
    if image.mode != 'RGBA':
        # If no alpha channel, convert and create one based on non-white pixels
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create alpha mask based on non-white pixels
        img_array = np.array(image)
        # Consider pixels as foreground if they're not close to white
        mask = np.any(img_array < 240, axis=2).astype(np.uint8) * 255
        
        # Add alpha channel
        alpha = Image.fromarray(mask, mode='L')
        image.putalpha(alpha)
    
    # Convert PIL to OpenCV for processing
    img_array = np.array(image)
    
    if img_array.shape[2] == 4:  # RGBA
        bgr = cv2.cvtColor(img_array[:,:,:3], cv2.COLOR_RGB2BGR)
        alpha = img_array[:,:,3]
    else:
        return image
    
    # Create binary mask from alpha channel
    _, binary_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    
    # Find external contours (main object boundary)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create filled mask
        filled_mask = np.zeros_like(alpha)
        cv2.fillPoly(filled_mask, [largest_contour], 255)
        
        # Smooth the filled mask to avoid hard edges
        kernel_size = max(3, min(options.shadow_fill_kernel_size, 50))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd number
        filled_mask = cv2.GaussianBlur(filled_mask, (kernel_size, kernel_size), 0)
        
        # Combine original alpha with filled mask (take maximum)
        new_alpha = np.maximum(alpha, filled_mask)
        
        # Create final image
        result_array = np.dstack([img_array[:,:,:3], new_alpha])
        result = Image.fromarray(result_array, 'RGBA')
        
        logger.info("Successfully filled internal holes in alpha mask")
        return result
    
    return image

def enhance_shadow_areas(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """
    Enhance shadow areas by brightening and reducing harsh contrasts.
    """
    # Work with RGBA if available, otherwise convert
    if image.mode != 'RGBA':
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Create alpha channel
        alpha = Image.new('L', image.size, 255)
        image.putalpha(alpha)
    
    # Separate channels
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Convert to numpy for advanced processing
    img_array = np.array(rgb_image)
    
    # Identify shadow areas (darker regions)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    shadow_mask = gray < np.percentile(gray[gray > 0], 30)  # Bottom 30% of non-background pixels
    
    # Apply brightness boost to shadow areas
    brightness_factor = options.shadow_brightness_boost
    enhanced_array = img_array.astype(np.float32)
    
    # Apply brightness boost only to shadow areas
    for i in range(3):  # RGB channels
        channel = enhanced_array[:,:,i]
        channel[shadow_mask] = np.clip(channel[shadow_mask] * brightness_factor, 0, 255)
    
    enhanced_array = enhanced_array.astype(np.uint8)
    enhanced_rgb = Image.fromarray(enhanced_array)
    
    # Apply contrast reduction for smoother shadows
    enhancer = ImageEnhance.Contrast(enhanced_rgb)
    contrast_adjusted = enhancer.enhance(options.shadow_contrast_reduction)
    
    # Apply slight blur to soften harsh shadow edges
    blur_radius = min(options.shadow_blur_radius, 5.0)
    softened = contrast_adjusted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Combine back with original alpha
    final_image = Image.merge('RGBA', (*softened.split(), a))
    
    logger.info("Successfully enhanced shadow areas")
    return final_image

def apply_morphological_cleanup(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological operations to clean up the mask.
    """
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close operation to fill small holes
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Optional: Opening to remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel//2)
    
    return opened

def apply_edge_processing(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """
    Apply various edge processing techniques to the image.
    
    Available techniques:
    - none: No edge processing
    - sharpness: Simple sharpness enhancement
    - edge_detect_sobel: Sobel edge detection
    - edge_detect_canny: Canny edge detection
    - edge_enhance: PIL's edge enhancement filter
    - unsharp_mask: Professional unsharp masking
    """
    if options.edge_processing == "none":
        return image
    
    # Convert RGBA to RGB for processing if needed
    processing_image = image
    if image.mode == "RGBA":
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        processing_image = rgb_image
    
    if options.edge_processing == "sharpness":
        enhancer = ImageEnhance.Sharpness(processing_image)
        result = enhancer.enhance(options.sharpness_factor)
        
    elif options.edge_processing == "edge_detect_sobel":
        result = apply_sobel_edge_detection(processing_image, options)
        
    elif options.edge_processing == "edge_detect_canny":
        result = apply_canny_edge_detection(processing_image, options)
        
    elif options.edge_processing == "edge_enhance":
        result = processing_image.filter(ImageFilter.EDGE_ENHANCE)
        
    elif options.edge_processing == "unsharp_mask":
        result = apply_unsharp_mask(processing_image, options)
        
    else:
        result = processing_image
    
    # If original image had alpha channel, preserve it
    if image.mode == "RGBA" and result.mode == "RGB":
        result_rgba = result.convert("RGBA")
        # Use original alpha channel
        result_rgba.putalpha(image.split()[3])
        return result_rgba
    
    return result

def apply_sobel_edge_detection(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply Sobel edge detection algorithm."""
    # Convert to grayscale for edge detection
    gray = image.convert('L')
    img_array = np.array(gray)
    
    # Apply Sobel edge detection
    sobel_x = ndimage.sobel(img_array, axis=1)
    sobel_y = ndimage.sobel(img_array, axis=0)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_magnitude = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
    edge_image = Image.fromarray(sobel_magnitude).convert('RGB')
    
    if options.preserve_original:
        # Overlay edges on original image
        return blend_edge_overlay(image, edge_image, options.edge_overlay_opacity)
    else:
        return edge_image

def apply_canny_edge_detection(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply Canny edge detection algorithm."""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, options.canny_low_threshold, options.canny_high_threshold)
        edge_image = Image.fromarray(edges).convert('RGB')
        
        if options.preserve_original:
            # Overlay edges on original image
            return blend_edge_overlay(image, edge_image, options.edge_overlay_opacity)
        else:
            return edge_image
            
    except Exception as e:
        logger.warning(f"OpenCV not available or error in Canny detection: {e}. Falling back to Sobel.")
        return apply_sobel_edge_detection(image, options)

def apply_unsharp_mask(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply unsharp masking for professional edge enhancement."""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create blurred version
    blurred = image.filter(ImageFilter.GaussianBlur(radius=options.unsharp_radius))
    blurred_array = np.array(blurred)
    
    # Create unsharp mask
    mask = img_array.astype(float) - blurred_array.astype(float)
    
    # Apply threshold
    mask = np.where(np.abs(mask) < options.unsharp_threshold, 0, mask)
    
    # Apply the mask
    sharpened = img_array.astype(float) + (mask * options.unsharp_percent / 100.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sharpened)

def blend_edge_overlay(original: Image.Image, edge_image: Image.Image, opacity: float) -> Image.Image:
    """Blend edge detection results with the original image."""
    # Convert edge image to have white background and black edges
    edge_array = np.array(edge_image.convert('L'))
    # Invert so edges are white on black background
    edge_inverted = 255 - edge_array
    edge_colored = Image.fromarray(edge_inverted).convert('RGB')
    
    # Blend with original
    if original.mode == "RGBA":
        original_rgb = Image.new("RGB", original.size, (255, 255, 255))
        original_rgb.paste(original, mask=original.split()[3])
    else:
        original_rgb = original.convert('RGB')
    
    # Use PIL's blend function
    blended = Image.blend(original_rgb, edge_colored, opacity * 0.3)  # Reduce opacity for better visibility
    
    return blended
    """
    Apply various edge processing techniques to the image.
    
    Available techniques:
    - none: No edge processing
    - sharpness: Simple sharpness enhancement
    - edge_detect_sobel: Sobel edge detection
    - edge_detect_canny: Canny edge detection
    - edge_enhance: PIL's edge enhancement filter
    - unsharp_mask: Professional unsharp masking
    """
    if options.edge_processing == "none":
        return image
    
    # Convert RGBA to RGB for processing if needed
    processing_image = image
    if image.mode == "RGBA":
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        processing_image = rgb_image
    
    if options.edge_processing == "sharpness":
        enhancer = ImageEnhance.Sharpness(processing_image)
        result = enhancer.enhance(options.sharpness_factor)
        
    elif options.edge_processing == "edge_detect_sobel":
        result = apply_sobel_edge_detection(processing_image, options)
        
    elif options.edge_processing == "edge_detect_canny":
        result = apply_canny_edge_detection(processing_image, options)
        
    elif options.edge_processing == "edge_enhance":
        result = processing_image.filter(ImageFilter.EDGE_ENHANCE)
        
    elif options.edge_processing == "unsharp_mask":
        result = apply_unsharp_mask(processing_image, options)
        
    else:
        result = processing_image
    
    # If original image had alpha channel, preserve it
    if image.mode == "RGBA" and result.mode == "RGB":
        result_rgba = result.convert("RGBA")
        # Use original alpha channel
        result_rgba.putalpha(image.split()[3])
        return result_rgba
    
    return result

def apply_sobel_edge_detection(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply Sobel edge detection algorithm."""
    # Convert to grayscale for edge detection
    gray = image.convert('L')
    img_array = np.array(gray)
    
    # Apply Sobel edge detection
    sobel_x = ndimage.sobel(img_array, axis=1)
    sobel_y = ndimage.sobel(img_array, axis=0)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255 range
    sobel_magnitude = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)
    edge_image = Image.fromarray(sobel_magnitude).convert('RGB')
    
    if options.preserve_original:
        # Overlay edges on original image
        return blend_edge_overlay(image, edge_image, options.edge_overlay_opacity)
    else:
        return edge_image

def apply_canny_edge_detection(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply Canny edge detection algorithm."""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, options.canny_low_threshold, options.canny_high_threshold)
        edge_image = Image.fromarray(edges).convert('RGB')
        
        if options.preserve_original:
            # Overlay edges on original image
            return blend_edge_overlay(image, edge_image, options.edge_overlay_opacity)
        else:
            return edge_image
            
    except Exception as e:
        logger.warning(f"OpenCV not available or error in Canny detection: {e}. Falling back to Sobel.")
        return apply_sobel_edge_detection(image, options)

def apply_unsharp_mask(image: Image.Image, options: ProcessingOptions) -> Image.Image:
    """Apply unsharp masking for professional edge enhancement."""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create blurred version
    blurred = image.filter(ImageFilter.GaussianBlur(radius=options.unsharp_radius))
    blurred_array = np.array(blurred)
    
    # Create unsharp mask
    mask = img_array.astype(float) - blurred_array.astype(float)
    
    # Apply threshold
    mask = np.where(np.abs(mask) < options.unsharp_threshold, 0, mask)
    
    # Apply the mask
    sharpened = img_array.astype(float) + (mask * options.unsharp_percent / 100.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sharpened)

def blend_edge_overlay(original: Image.Image, edge_image: Image.Image, opacity: float) -> Image.Image:
    """Blend edge detection results with the original image."""
    # Convert edge image to have white background and black edges
    edge_array = np.array(edge_image.convert('L'))
    # Invert so edges are white on black background
    edge_inverted = 255 - edge_array
    edge_colored = Image.fromarray(edge_inverted).convert('RGB')
    
    # Blend with original
    if original.mode == "RGBA":
        original_rgb = Image.new("RGB", original.size, (255, 255, 255))
        original_rgb.paste(original, mask=original.split()[3])
    else:
        original_rgb = original.convert('RGB')
    
    # Use PIL's blend function
    blended = Image.blend(original_rgb, edge_colored, opacity * 0.3)  # Reduce opacity for better visibility
    
    return blended

def parse_background_color(color_str: str) -> tuple:
    """Parse background color string to RGB tuple."""
    color_str = color_str.lower()
    if color_str == "white":
        return (255, 255, 255)
    elif color_str == "black":
        return (0, 0, 0)
    elif color_str == "transparent":
        return None
    elif color_str.startswith("#"):
        try:
            hex_color = color_str[1:]
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (255, 255, 255)
    else:
        return (255, 255, 255)
    """Parse background color string to RGB tuple."""
    color_str = color_str.lower()
    if color_str == "white":
        return (255, 255, 255)
    elif color_str == "black":
        return (0, 0, 0)
    elif color_str == "transparent":
        return None
    elif color_str.startswith("#"):
        try:
            hex_color = color_str[1:]
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (255, 255, 255)
    else:
        return (255, 255, 255)

def update_processing_stats(processing_time: float, model_name: str, success: bool) -> None:
    """Update global processing statistics."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Update daily stats
    if today not in processing_stats["daily_stats"]:
        processing_stats["daily_stats"][today] = {
            "processed": 0, "errors": 0, "total_time": 0
        }
    
    daily = processing_stats["daily_stats"][today]
    
    if success:
        processing_stats["total_processed"] += 1
        daily["processed"] += 1
        daily["total_time"] += processing_time
        
        # Update average processing time
        total_time = sum(day["total_time"] for day in processing_stats["daily_stats"].values())
        total_jobs = processing_stats["total_processed"]
        processing_stats["average_processing_time"] = total_time / total_jobs if total_jobs > 0 else 0
    else:
        processing_stats["total_errors"] += 1
        daily["errors"] += 1
    
    # Update model usage stats
    if model_name not in processing_stats["model_usage"]:
        processing_stats["model_usage"][model_name] = {"count": 0, "success": 0, "errors": 0}
    
    processing_stats["model_usage"][model_name]["count"] += 1
    if success:
        processing_stats["model_usage"][model_name]["success"] += 1
    else:
        processing_stats["model_usage"][model_name]["errors"] += 1

def cleanup_old_files(background_tasks: BackgroundTasks) -> None:
    """Schedule cleanup of old files."""
    background_tasks.add_task(perform_cleanup)

async def perform_cleanup() -> None:
    """Clean up old files (older than 24 hours)."""
    cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
    
    for directory in [UPLOADS_DIR, PROCESSED_DIR]:
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup in {directory}: {e}")

# --- Enhanced API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "queue_size": queue.qsize(),
        "max_queue_size": MAX_QUEUE_SIZE,
        "active_workers": MAX_CONCURRENT_TASKS,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/shadow-removal")
async def list_shadow_removal_options():
    """List available shadow removal options with descriptions."""
    return {
        "available_methods": SHADOW_REMOVAL_METHODS,
        "default_method": "none",
        "descriptions": {
            "none": "No shadow removal applied",
            "fill_holes": "Fill holes in alpha mask caused by internal shadows - best for bags, containers",
            "enhance_shadows": "Brighten and soften shadow areas instead of removing them",
            "hybrid": "Combination of fill_holes and enhance_shadows for comprehensive shadow handling"
        },
        "use_cases": {
            "none": "When shadow removal is not needed",
            "fill_holes": "Handbags, containers, objects with internal shadows creating unwanted holes",
            "enhance_shadows": "When you want to preserve shadow details but reduce harshness",
            "hybrid": "Complex objects with both internal holes and harsh shadows"
        },
        "parameters": {
            "shadow_fill_kernel_size": "15 (5-50) - Size of morphological operations for hole filling",
            "shadow_blur_radius": "2.0 (0.5-10.0) - Blur radius for shadow edge softening",
            "shadow_brightness_boost": "1.2 (1.0-2.0) - How much to brighten shadow areas",
            "shadow_contrast_reduction": "0.9 (0.5-1.0) - Reduce contrast in shadow areas"
        },
        "recommended_combinations": {
            "handbags_purses": {
                "shadow_removal": "fill_holes",
                "model": "isnet-general-use",
                "edge_processing": "unsharp_mask"
            },
            "product_photography": {
                "shadow_removal": "enhance_shadows",
                "model": "u2net",
                "background_color": "white"
            },
            "complex_objects": {
                "shadow_removal": "hybrid",
                "model": "isnet-general-use",
                "edge_processing": "edge_enhance"
            }
        }
    }
    """List available edge processing options with descriptions."""
    return {
        "available_options": EDGE_PROCESSING_TYPES,
        "default_option": "none",
        "descriptions": {
            "none": "No edge processing applied",
            "sharpness": "Simple sharpness enhancement - makes image appear more crisp",
            "edge_detect_sobel": "Sobel edge detection - creates edge map showing boundaries",
            "edge_detect_canny": "Canny edge detection - cleaner edge detection with less noise",
            "edge_enhance": "PIL edge enhancement filter - strengthens existing edges",
            "unsharp_mask": "Professional unsharp masking - used in photo editing software"
        },
        "use_cases": {
            "none": "Default for most applications",
            "sharpness": "General image improvement, product photography",
            "edge_detect_sobel": "Computer vision, artistic effects, technical analysis",
            "edge_detect_canny": "Technical drawings, precise edge detection",
            "edge_enhance": "Photography where natural appearance is important",
            "unsharp_mask": "Professional photography, print quality images"
        },
        "parameters": {
            "sharpness_factor": "1.5 (0.5-3.0) - Higher values = more sharpening",
            "canny_low_threshold": "50 (10-200) - Lower threshold for edge detection",
            "canny_high_threshold": "150 (50-300) - Upper threshold for edge detection",
            "unsharp_radius": "2.0 (0.5-5.0) - Blur radius for unsharp mask",
            "unsharp_percent": "150 (50-300) - Strength of unsharp effect",
            "unsharp_threshold": "3 (0-10) - Minimum difference for edge detection",
            "edge_overlay_opacity": "1.0 (0.1-1.0) - Opacity when overlaying edges",
            "preserve_original": "false - Overlay edges on original vs replace image"
        }
    }
    """List available background removal models."""
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": "u2net",
        "model_descriptions": {
            "u2net": "General purpose model, good for most images",
            "u2net_human_seg": "Optimized for human subjects",
            "silueta": "High quality general purpose model",
            "isnet-general-use": "Latest general purpose model with improved accuracy",
            "sam": "Segment Anything Model for precise segmentation"
        }
    }

@app.get("/stats")
async def get_stats(detailed: bool = Query(False, description="Include detailed daily stats")):
    """Get processing statistics."""
    stats = processing_stats.copy()
    if not detailed:
        # Remove detailed daily stats for cleaner response
        stats["daily_stats"] = {
            "today": stats["daily_stats"].get(datetime.now().strftime("%Y-%m-%d"), {})
        }
    return stats

@app.post("/submit", response_model=JobResponse)
async def submit_json_image_for_processing(
    request: Request,
    body: SubmitJsonBody,
    background_tasks: BackgroundTasks
):
    """Submit image URL for background removal processing."""
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if body.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model. Available models: {', '.join(AVAILABLE_MODELS)}"
        )

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    
    try:
        queue.put_nowait((job_id, str(body.image), body.model, body.options))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full. Rejecting JSON request for image {body.image}.")
        raise HTTPException(
            status_code=503, 
            detail=f"Server overloaded (queue full). Max: {MAX_QUEUE_SIZE}"
        )

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued",
        "input_image_url": str(body.image),
        "original_local_path": None,
        "processed_path": None,
        "error_message": None,
        "status_check_url": status_check_url,
        "created_at": datetime.now().isoformat(),
        "options": body.options.dict() if body.options else None
    }
    
    output_format = body.options.output_format if body.options else "webp"
    processed_image_placeholder_url = f"{public_url_base}/images/{job_id}.{output_format}"
    eta_seconds = queue.qsize() * ESTIMATED_TIME_PER_JOB
    
    # Schedule cleanup
    cleanup_old_files(background_tasks)
    
    logger.info(f"Job {job_id} (JSON URL: {body.image}) enqueued. Queue size: {queue.qsize()}. ETA: {eta_seconds:.2f}s")
    
    return JobResponse(
        status="processing",
        job_id=job_id,
        image_links=[processed_image_placeholder_url],
        eta=eta_seconds,
        status_check_url=status_check_url
    )

@app.post("/submit_form", response_model=JobResponse)
async def submit_form_image_for_processing(
    request: Request,
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(...),
    key: str = Form(...),
    model: str = Form("u2net"),
    resize_to: int = Form(1024),
    background_color: str = Form("white"),
    output_format: str = Form("webp"),
    quality: int = Form(90),
    shadow_removal: str = Form("none"),
    shadow_fill_kernel_size: int = Form(15),
    shadow_blur_radius: float = Form(2.0),
    shadow_brightness_boost: float = Form(1.2),
    shadow_contrast_reduction: float = Form(0.9),
    edge_processing: str = Form("none"),
    sharpness_factor: float = Form(1.5),
    canny_low_threshold: int = Form(50),
    canny_high_threshold: int = Form(150),
    unsharp_radius: float = Form(2.0),
    unsharp_percent: float = Form(150),
    unsharp_threshold: int = Form(3),
    edge_overlay_opacity: float = Form(1.0),
    preserve_original: bool = Form(False)
):
    """Submit image file for background removal processing."""
    if key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    
    if shadow_removal not in SHADOW_REMOVAL_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shadow removal method. Available methods: {', '.join(SHADOW_REMOVAL_METHODS)}"
        )
    
    if edge_processing not in EDGE_PROCESSING_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid edge processing type. Available options: {', '.join(EDGE_PROCESSING_TYPES)}"
        )
    
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Read file to check size
    file_content = await image_file.read()
    validate_file_size(len(file_content))
    await image_file.seek(0)  # Reset file pointer

    job_id = str(uuid.uuid4())
    public_url_base = get_proxy_url(request)
    
    # Create processing options
    options = ProcessingOptions(
        resize_to=resize_to,
        background_color=background_color,
        output_format=output_format,
        quality=quality,
        shadow_removal=shadow_removal,
        shadow_fill_kernel_size=shadow_fill_kernel_size,
        shadow_blur_radius=shadow_blur_radius,
        shadow_brightness_boost=shadow_brightness_boost,
        shadow_contrast_reduction=shadow_contrast_reduction,
        edge_processing=edge_processing,
        sharpness_factor=sharpness_factor,
        canny_low_threshold=canny_low_threshold,
        canny_high_threshold=canny_high_threshold,
        unsharp_radius=unsharp_radius,
        unsharp_percent=unsharp_percent,
        unsharp_threshold=unsharp_threshold,
        edge_overlay_opacity=edge_overlay_opacity,
        preserve_original=preserve_original
    )

    # Save uploaded file
    original_filename = image_file.filename or "upload"
    content_type = image_file.content_type.lower()
    extension = MIME_TO_EXT.get(content_type, ".png")
    
    saved_original_filename = f"{job_id}_original{extension}"
    original_file_path = os.path.join(UPLOADS_DIR, saved_original_filename)

    try:
        async with aiofiles.open(original_file_path, 'wb') as out_file:
            await out_file.write(file_content)
        logger.info(f"Job {job_id}: Saved uploaded file {original_file_path} ({format_size(len(file_content))})")
    except Exception as e:
        logger.error(f"Error saving uploaded file for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        await image_file.close()

    file_uri_for_queue = f"file://{original_file_path}"
    try:
        queue.put_nowait((job_id, file_uri_for_queue, model, options))
    except asyncio.QueueFull:
        logger.warning(f"Queue is full. Rejecting form request for job {job_id}.")
        if os.path.exists(original_file_path):
            try:
                os.remove(original_file_path)
            except OSError as e_clean:
                logger.error(f"Error cleaning {original_file_path}: {e_clean}")
        raise HTTPException(status_code=503, detail=f"Server overloaded (queue full). Max: {MAX_QUEUE_SIZE}")

    status_check_url = f"{public_url_base}/status/{job_id}"
    results[job_id] = {
        "status": "queued",
        "input_image_url": f"(form_upload: {original_filename})",
        "original_local_path": original_file_path,
        "processed_path": None,
        "error_message": None,
        "status_check_url": status_check_url,
        "created_at": datetime.now().isoformat(),
        "options": options.dict()
    }

    processed_image_url = f"{public_url_base}/images/{job_id}.{output_format}"
    original_image_served_url = f"{public_url_base}/originals/{saved_original_filename}"
    eta_seconds = queue.qsize() * ESTIMATED_TIME_PER_JOB
    
    # Schedule cleanup
    cleanup_old_files(background_tasks)
    
    logger.info(f"Job {job_id} (Form Upload: {original_filename}) enqueued. Queue size: {queue.qsize()}.")
    
    return JobResponse(
        status="processing",
        job_id=job_id,
        original_image_url=original_image_served_url,
        image_links=[processed_image_url],
        eta=eta_seconds,
        status_check_url=status_check_url
    )

@app.post("/batch")
async def submit_batch_processing(
    request: Request,
    body: BatchProcessRequest,
    background_tasks: BackgroundTasks
):
    """Submit multiple images for batch processing."""
    if body.key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if body.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {', '.join(AVAILABLE_MODELS)}"
        )

    if len(body.images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    batch_id = str(uuid.uuid4())
    job_ids = []
    public_url_base = get_proxy_url(request)
    
    for image_url in body.images:
        job_id = str(uuid.uuid4())
        job_ids.append(job_id)
        
        try:
            queue.put_nowait((job_id, str(image_url), body.model, body.options))
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=503,
                detail=f"Server overloaded (queue full). Max: {MAX_QUEUE_SIZE}"
            )
        
        results[job_id] = {
            "status": "queued",
            "input_image_url": str(image_url),
            "batch_id": batch_id,
            "original_local_path": None,
            "processed_path": None,
            "error_message": None,
            "status_check_url": f"{public_url_base}/status/{job_id}",
            "created_at": datetime.now().isoformat(),
            "options": body.options.dict() if body.options else None
        }
    
    # Schedule cleanup
    cleanup_old_files(background_tasks)
    
    return {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "status": "processing",
        "total_jobs": len(job_ids),
        "batch_status_url": f"{public_url_base}/batch/{batch_id}",
        "eta": queue.qsize() * ESTIMATED_TIME_PER_JOB
    }

@app.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of batch processing."""
    batch_jobs = {k: v for k, v in results.items() if v.get("batch_id") == batch_id}
    
    if not batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    completed = sum(1 for job in batch_jobs.values() if job["status"] == "done")
    failed = sum(1 for job in batch_jobs.values() if job["status"] == "error")
    processing = len(batch_jobs) - completed - failed
    
    return {
        "batch_id": batch_id,
        "total_jobs": len(batch_jobs),
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "jobs": batch_jobs
    }

@app.get("/status/{job_id}")
async def check_status(request: Request, job_id: str):
    """Check the status of a processing job."""
    job_info = results.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    public_url_base = get_proxy_url(request)
    response_data = {
        "job_id": job_id,
        "status": job_info.get("status"),
        "input_image_url": job_info.get("input_image_url"),
        "status_check_url": job_info.get("status_check_url"),
        "created_at": job_info.get("created_at"),
        "processing_options": job_info.get("options")
    }
    
    if job_info.get("original_local_path"):
        original_filename = os.path.basename(job_info["original_local_path"])
        response_data["downloaded_original_image_url"] = f"{public_url_base}/originals/{original_filename}"
    
    if job_info.get("status") == "done" and job_info.get("processed_path"):
        processed_filename = os.path.basename(job_info["processed_path"])
        response_data["processed_image_url"] = f"{public_url_base}/images/{processed_filename}"
        response_data["download_url"] = f"{public_url_base}/download/{job_id}"
    elif job_info.get("status") == "error":
        response_data["error_message"] = job_info.get("error_message")
    
    return JSONResponse(content=response_data)

@app.get("/download/{job_id}")
async def download_processed_image(job_id: str):
    """Download processed image directly."""
    job_info = results.get(job_id)
    if not job_info or job_info.get("status") != "done":
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    processed_path = job_info.get("processed_path")
    if not processed_path or not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="Processed image file not found")
    
    filename = os.path.basename(processed_path)
    return FileResponse(
        processed_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job_info = results.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up files
    for path_key in ["original_local_path", "processed_path"]:
        file_path = job_info.get(path_key)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except OSError as e:
                logger.error(f"Error deleting file {file_path}: {e}")
    
    # Remove from results
    del results[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

# --- Enhanced Background Worker ---
async def image_processing_worker(worker_id: int):
    """Enhanced background worker with improved processing options."""
    logger.info(f"Worker {worker_id} started. Listening for jobs...")
    global prepared_logo_image

    while True:
        job_id, image_source_str, model_name, options = await queue.get()
        
        t_job_start = time.perf_counter()
        logger.info(f"Worker {worker_id} picked up job {job_id} for source: {image_source_str}. Model: {model_name}")

        if job_id not in results:
            logger.error(f"Worker {worker_id}: Job ID {job_id} not found in results dict. Skipping.")
            queue.task_done()
            continue

        success = False
        try:
            # Process options
            if options is None:
                options = ProcessingOptions()
            elif isinstance(options, dict):
                options = ProcessingOptions(**options)
            
            # Fetch image data
            input_bytes = await fetch_image_data(job_id, image_source_str, worker_id)
            if input_bytes is None:
                raise ValueError("Failed to fetch image data")
            
            # Remove background
            results[job_id]["status"] = "processing_rembg"
            output_bytes_with_alpha = await remove_background(input_bytes, model_name, job_id, worker_id)
            
            # Process image with PIL
            results[job_id]["status"] = "processing_pil"
            final_image = await process_image_with_options(
                output_bytes_with_alpha, options, prepared_logo_image, job_id, worker_id
            )
            
            # Save processed image
            processed_path = await save_processed_image(final_image, job_id, options)
            
            results[job_id]["status"] = "done"
            results[job_id]["processed_path"] = processed_path
            success = True
            
            t_job_end = time.perf_counter()
            total_time = t_job_end - t_job_start
            
            logger.info(f"Job {job_id} (Worker {worker_id}) COMPLETED successfully in {total_time:.4f}s")
            update_processing_stats(total_time, model_name, True)

        except Exception as e:
            logger.error(f"Job {job_id} (Worker {worker_id}) Error: {e}", exc_info=True)
            results[job_id]["status"] = "error"
            results[job_id]["error_message"] = str(e)
            
            t_job_end = time.perf_counter()
            total_time = t_job_end - t_job_start
            update_processing_stats(total_time, model_name, False)
        
        finally:
            queue.task_done()

async def fetch_image_data(job_id: str, image_source_str: str, worker_id: int) -> Optional[bytes]:
    """Fetch image data from URL or local file."""
    if image_source_str.startswith("file://"):
        results[job_id]["status"] = "processing_local_file"
        local_path = image_source_str[len("file://"):]
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        async with aiofiles.open(local_path, 'rb') as f:
            return await f.read()
    
    elif image_source_str.startswith(("http://", "https://")):
        results[job_id]["status"] = "downloading"
        logger.info(f"Job {job_id} (Worker {worker_id}): Downloading from {image_source_str}")
        
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT) as client:
            response = await client.get(image_source_str)
            response.raise_for_status()
            
            image_data = await response.aread()
            
            # Save downloaded original
            content_type = response.headers.get("content-type", "").lower()
            extension = MIME_TO_EXT.get(content_type, ".bin")
            
            original_filename = f"{job_id}_original_downloaded{extension}"
            original_path = os.path.join(UPLOADS_DIR, original_filename)
            results[job_id]["original_local_path"] = original_path
            
            async with aiofiles.open(original_path, 'wb') as f:
                await f.write(image_data)
            
            return image_data
    
    else:
        raise ValueError(f"Unsupported image source: {image_source_str}")

async def remove_background(input_bytes: bytes, model_name: str, job_id: str, worker_id: int) -> bytes:
    """Remove background from image using specified model."""
    logger.info(f"Job {job_id} (Worker {worker_id}): Starting rembg processing (model: {model_name})")
    
    t_start = time.perf_counter()
    session = new_session(model_name)
    output_bytes = remove(
        input_bytes,
        session=session,
        post_process_mask=True,
        alpha_matting=True
    )
    t_end = time.perf_counter()
    
    logger.info(f"Job {job_id} (Worker {worker_id}): Rembg processing completed in {t_end - t_start:.4f}s")
    return output_bytes

async def process_image_with_options(
    image_bytes: bytes, 
    options: ProcessingOptions, 
    logo_image: Optional[Image.Image],
    job_id: str,
    worker_id: int
) -> Image.Image:
    """Process image with enhanced options."""
    logger.info(f"Job {job_id} (Worker {worker_id}): Starting PIL processing with options")
    
    # Load image with alpha channel
    img_rgba = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Apply background color
    bg_color = parse_background_color(options.background_color)
    if bg_color is None:  # Transparent background
        final_image = img_rgba
    else:
        # Create background with specified color
        bg_canvas = Image.new("RGB", img_rgba.size, bg_color)
        bg_canvas.paste(img_rgba, (0, 0), img_rgba)
        final_image = bg_canvas
    
    # Resize image
    if options.resize_to and options.resize_to != max(final_image.size):
        original_width, original_height = final_image.size
        if original_width == 0 or original_height == 0:
            raise ValueError("Image has zero dimensions")
        
        # Calculate new dimensions maintaining aspect ratio
        ratio = min(options.resize_to / original_width, options.resize_to / original_height)
        new_width, new_height = int(original_width * ratio), int(original_height * ratio)
        
        # Resize with high quality
        final_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create square canvas if needed
        if options.background_color != "transparent":
            square_canvas = Image.new("RGB", (options.resize_to, options.resize_to), bg_color or (255, 255, 255))
            paste_x = (options.resize_to - new_width) // 2
            paste_y = (options.resize_to - new_height) // 2
            square_canvas.paste(final_image, (paste_x, paste_y))
            final_image = square_canvas
    
    # Apply shadow removal if requested (before edge processing for better results)
    if options.shadow_removal and options.shadow_removal != "none":
        logger.info(f"Job {job_id} (Worker {worker_id}): Applying shadow removal: {options.shadow_removal}")
        final_image = apply_shadow_removal(final_image, options)
    
    # Apply edge processing if requested
    if options.edge_processing and options.edge_processing != "none":
        logger.info(f"Job {job_id} (Worker {worker_id}): Applying edge processing: {options.edge_processing}")
        final_image = apply_edge_processing(final_image, options)
    
    # Apply logo watermark if enabled
    if ENABLE_LOGO_WATERMARK and logo_image and options.background_color != "transparent":
        if final_image.mode != 'RGBA':
            final_image = final_image.convert('RGBA')
        
        logo_w, logo_h = logo_image.size
        img_w, img_h = final_image.size
        
        # Position logo in bottom-left corner
        logo_pos_x = LOGO_MARGIN
        logo_pos_y = img_h - logo_h - LOGO_MARGIN
        
        # Ensure logo fits within image bounds
        if logo_pos_x + logo_w <= img_w and logo_pos_y >= 0:
            final_image.paste(logo_image, (logo_pos_x, logo_pos_y), logo_image)
    
    # Convert back to RGB if needed for certain output formats
    if options.output_format.lower() in ["jpg", "jpeg"] and final_image.mode in ["RGBA", "P"]:
        rgb_canvas = Image.new("RGB", final_image.size, (255, 255, 255))
        if final_image.mode == "RGBA":
            rgb_canvas.paste(final_image, mask=final_image.split()[3])
        else:
            rgb_canvas.paste(final_image)
        final_image = rgb_canvas
    
    return final_image

async def save_processed_image(image: Image.Image, job_id: str, options: ProcessingOptions) -> str:
    """Save processed image with specified format and quality."""
    output_format = options.output_format.lower()
    if output_format not in OUTPUT_FORMATS:
        output_format = "webp"
    
    # Map format names
    format_mapping = {
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "webp": "WEBP"
    }
    
    pil_format = format_mapping.get(output_format, "WEBP")
    extension = f".{output_format}"
    processed_filename = f"{job_id}{extension}"
    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
    
    # Save with appropriate options
    save_kwargs = {}
    if pil_format == "JPEG":
        save_kwargs.update({"quality": options.quality, "optimize": True})
    elif pil_format == "WEBP":
        save_kwargs.update({"quality": options.quality, "optimize": True})
    elif pil_format == "PNG":
        save_kwargs.update({"optimize": True})
    
    image.save(processed_path, pil_format, **save_kwargs)
    
    file_size = os.path.getsize(processed_path)
    logger.info(f"Saved processed image: {processed_path} ({format_size(file_size)})")
    
    return processed_path

# --- Application Startup and Shutdown ---
async def startup_event():
    """Enhanced startup event with better error handling."""
    global prepared_logo_image
    logger.info("Application startup event running...")

    # Load logo if watermarking is enabled
    if ENABLE_LOGO_WATERMARK:
        logger.info(f"Logo watermarking ENABLED. Attempting load from: {LOGO_PATH}")
        if os.path.exists(LOGO_PATH):
            try:
                logo = Image.open(LOGO_PATH).convert("RGBA")
                if logo.width > LOGO_MAX_WIDTH:
                    ratio = LOGO_MAX_WIDTH / logo.width
                    new_width = LOGO_MAX_WIDTH
                    new_height = int(logo.height * ratio)
                    logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
                prepared_logo_image = logo
                logger.info(f"Logo loaded successfully. Dimensions: {prepared_logo_image.size}")
            except Exception as e:
                logger.error(f"Failed to load logo: {e}", exc_info=True)
                prepared_logo_image = None
        else:
            logger.warning(f"Logo file not found at {LOGO_PATH}")
            prepared_logo_image = None
    else:
        logger.info("Logo watermarking DISABLED")
        prepared_logo_image = None

    # Start worker tasks
    for i in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(image_processing_worker(worker_id=i+1))
    
    logger.info(f"Started {MAX_CONCURRENT_TASKS} workers. Queue max size: {MAX_QUEUE_SIZE}")
    
    # Load existing stats if available
    await load_processing_stats()

async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutdown event running...")
    
    # Save processing stats
    await save_processing_stats()
    
    # Wait for queue to be processed
    if not queue.empty():
        logger.info(f"Waiting for {queue.qsize()} remaining jobs to complete...")
        await queue.join()
    
    logger.info("Application shutdown completed")

async def load_processing_stats():
    """Load processing statistics from file."""
    stats_file = os.path.join(STATS_DIR, "processing_stats.json")
    try:
        if os.path.exists(stats_file):
            async with aiofiles.open(stats_file, 'r') as f:
                content = await f.read()
                loaded_stats = json.loads(content)
                processing_stats.update(loaded_stats)
            logger.info("Loaded processing statistics from file")
    except Exception as e:
        logger.error(f"Error loading processing stats: {e}")

async def save_processing_stats():
    """Save processing statistics to file."""
    stats_file = os.path.join(STATS_DIR, "processing_stats.json")
    try:
        async with aiofiles.open(stats_file, 'w') as f:
            await f.write(json.dumps(processing_stats, indent=2))
        logger.info("Saved processing statistics to file")
    except Exception as e:
        logger.error(f"Error saving processing stats: {e}")

# --- Static File Serving ---
app.mount("/images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/originals", StaticFiles(directory=UPLOADS_DIR), name="original_images")

# --- Enhanced Root Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Enhanced root endpoint with better UI and information."""
    logo_status = "Disabled"
    if ENABLE_LOGO_WATERMARK:
        if prepared_logo_image:
            logo_status = f"Enabled (Loaded, {prepared_logo_image.width}x{prepared_logo_image.height})"
        else:
            logo_status = "Enabled (Not loaded/found)"

    current_stats = processing_stats
    today = datetime.now().strftime("%Y-%m-%d")
    today_stats = current_stats["daily_stats"].get(today, {"processed": 0, "errors": 0})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Background Removal API</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #3498db;
        }}
        .card h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.3em;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .stat:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            color: #7f8c8d;
        }}
        .stat-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .status-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-healthy {{
            background-color: #27ae60;
        }}
        .status-warning {{
            background-color: #f39c12;
        }}
        .endpoints {{
            background: #2c3e50;
            color: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }}
        .endpoints h3 {{
            margin-top: 0;
        }}
        .endpoint {{
            background: #34495e;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-family: monospace;
        }}
        .method {{
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        .method.post {{
            background: #27ae60;
        }}
        .method.delete {{
            background: #e74c3c;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Enhanced Background Removal API</h1>
        <p class="subtitle">High-quality AI-powered background removal with advanced processing options</p>
        
        <div class="grid">
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="stat">
                    <span class="stat-label">
                        <span class="status-indicator status-healthy"></span>Service Status
                    </span>
                    <span class="stat-value">Online</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Queue Size</span>
                    <span class="stat-value">{queue.qsize()}/{MAX_QUEUE_SIZE}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Active Workers</span>
                    <span class="stat-value">{MAX_CONCURRENT_TASKS}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Max File Size</span>
                    <span class="stat-value">{format_size(MAX_FILE_SIZE)}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Processing Stats</h3>
                <div class="stat">
                    <span class="stat-label">Total Processed</span>
                    <span class="stat-value">{current_stats['total_processed']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Today's Jobs</span>
                    <span class="stat-value">{today_stats['processed']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Total Errors</span>
                    <span class="stat-value">{current_stats['total_errors']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Processing Time</span>
                    <span class="stat-value">{current_stats['average_processing_time']:.2f}s</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ Configuration</h3>
                <div class="stat">
                    <span class="stat-label">Available Models</span>
                    <span class="stat-value">{len(AVAILABLE_MODELS)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Output Formats</span>
                    <span class="stat-value">{', '.join(OUTPUT_FORMATS)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Edge Processing Types</span>
                    <span class="stat-value">{len(EDGE_PROCESSING_TYPES)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Shadow Removal Methods</span>
                    <span class="stat-value">{len(SHADOW_REMOVAL_METHODS)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Logo Watermark</span>
                    <span class="stat-value">{logo_status}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Target Size</span>
                    <span class="stat-value">{TARGET_SIZE}px</span>
                </div>
            </div>
        </div>
        
        <div class="endpoints">
            <h3>🛠️ API Endpoints</h3>
            <div class="endpoint">
                <span class="method post">POST</span>/submit - Submit image URL for processing
            </div>
            <div class="endpoint">
                <span class="method post">POST</span>/submit_form - Submit image file for processing
            </div>
            <div class="endpoint">
                <span class="method post">POST</span>/batch - Submit multiple images for batch processing
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/status/{{job_id}} - Check job status
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/batch/{{batch_id}} - Check batch status
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/download/{{job_id}} - Download processed image
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/models - List available AI models
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/edge-processing - List edge processing options and parameters
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/shadow-removal - List shadow removal methods and parameters
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/stats - Get detailed processing statistics
            </div>
            <div class="endpoint">
                <span class="method">GET</span>/health - Health check endpoint
            </div>
            <div class="endpoint">
                <span class="method delete">DELETE</span>/job/{{job_id}} - Delete job and files
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 Enhanced with true edge detection and intelligent shadow removal for professional results</p>
            <p>🎯 Shadow Removal: {', '.join(SHADOW_REMOVAL_METHODS)} | Edge Processing: {', '.join(EDGE_PROCESSING_TYPES)}</p>
            <p>Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
    </div>
</body>
</html>"""

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced Background Removal API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", "7000")),
        workers=1,
        access_log=True
    )
