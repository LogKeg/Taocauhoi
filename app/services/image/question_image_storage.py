"""
Image storage service for question images.

Handles saving, retrieving, and deleting images associated with questions.
Storage structure: uploads/images/{question_id}/main.{ext}
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import httpx

# Base directory for question images
IMAGES_DIR = Path(__file__).parent.parent.parent.parent / "uploads" / "images"

# Supported image extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}

# Max image size (5MB)
MAX_IMAGE_SIZE = 5 * 1024 * 1024


def ensure_images_dir():
    """Ensure the images directory exists."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_question_image_dir(question_id: int) -> Path:
    """Get the directory path for a question's images."""
    return IMAGES_DIR / str(question_id)


def get_question_image_path(question_id: int, filename: str = "main.png") -> Path:
    """Get the full path for a question's image."""
    return get_question_image_dir(question_id) / filename


def save_question_image(
    question_id: int,
    image_bytes: bytes,
    original_filename: str = "image.png"
) -> Optional[str]:
    """
    Save an image for a question.

    Args:
        question_id: The question ID
        image_bytes: The image data as bytes
        original_filename: Original filename to extract extension

    Returns:
        Relative path to the saved image (for storing in DB), or None if failed
    """
    if not image_bytes:
        return None

    if len(image_bytes) > MAX_IMAGE_SIZE:
        print(f"Image too large: {len(image_bytes)} bytes > {MAX_IMAGE_SIZE}")
        return None

    # Get extension from original filename
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        ext = ".png"  # Default to PNG

    # Create question image directory
    question_dir = get_question_image_dir(question_id)
    question_dir.mkdir(parents=True, exist_ok=True)

    # Save as main.{ext}
    filename = f"main{ext}"
    file_path = question_dir / filename

    try:
        file_path.write_bytes(image_bytes)
        # Return relative path for DB storage
        return f"images/{question_id}/{filename}"
    except Exception as e:
        print(f"Error saving image for question {question_id}: {e}")
        return None


def get_question_image(question_id: int) -> Optional[Tuple[bytes, str]]:
    """
    Get the image data for a question.

    Returns:
        Tuple of (image_bytes, content_type) or None if not found
    """
    question_dir = get_question_image_dir(question_id)
    if not question_dir.exists():
        return None

    # Look for main.* file
    for ext in SUPPORTED_EXTENSIONS:
        file_path = question_dir / f"main{ext}"
        if file_path.exists():
            content_type = _get_content_type(ext)
            return file_path.read_bytes(), content_type

    return None


def delete_question_images(question_id: int) -> bool:
    """
    Delete all images for a question.

    Returns:
        True if deleted successfully, False otherwise
    """
    import shutil

    question_dir = get_question_image_dir(question_id)
    if not question_dir.exists():
        return True  # Already deleted

    try:
        shutil.rmtree(question_dir)
        return True
    except Exception as e:
        print(f"Error deleting images for question {question_id}: {e}")
        return False


async def download_image(url: str, timeout: float = 10.0) -> Optional[bytes]:
    """
    Download an image from a URL.

    Args:
        url: The image URL
        timeout: Request timeout in seconds

    Returns:
        Image bytes or None if failed
    """
    if not url:
        return None

    # Normalize URL
    if url.startswith("//"):
        url = "https:" + url

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or _is_image_url(url):
                    return response.content
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")

    return None


def download_image_sync(url: str, timeout: float = 10.0) -> Optional[bytes]:
    """
    Synchronous version of download_image.
    """
    if not url:
        return None

    # Normalize URL
    if url.startswith("//"):
        url = "https:" + url

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or _is_image_url(url):
                    return response.content
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")

    return None


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute MD5 hash of image for deduplication."""
    return hashlib.md5(image_bytes).hexdigest()


def _get_content_type(ext: str) -> str:
    """Get MIME content type for file extension."""
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    return content_types.get(ext.lower(), "application/octet-stream")


def _is_image_url(url: str) -> bool:
    """Check if URL looks like an image based on extension."""
    url_lower = url.lower().split("?")[0]  # Remove query params
    return any(url_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def get_image_stats() -> dict:
    """Get statistics about stored images."""
    ensure_images_dir()

    stats = {
        "total_questions_with_images": 0,
        "total_size_bytes": 0,
        "by_extension": {},
    }

    if not IMAGES_DIR.exists():
        return stats

    for question_dir in IMAGES_DIR.iterdir():
        if question_dir.is_dir():
            for img_file in question_dir.iterdir():
                if img_file.is_file():
                    stats["total_questions_with_images"] += 1
                    size = img_file.stat().st_size
                    stats["total_size_bytes"] += size

                    ext = img_file.suffix.lower()
                    if ext not in stats["by_extension"]:
                        stats["by_extension"][ext] = {"count": 0, "size": 0}
                    stats["by_extension"][ext]["count"] += 1
                    stats["by_extension"][ext]["size"] += size

    return stats


# Initialize directory on import
ensure_images_dir()
