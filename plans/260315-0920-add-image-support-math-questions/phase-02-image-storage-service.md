---
phase: 2
title: "Image Storage Service"
status: pending
effort: 1h
---

# Phase 2: Image Storage Service

## Context

- Existing storage: `app/services/local_storage.py` (templates by subject/grade)
- Existing image dir: `app/services/image/` (currently has OMR-related code)
- Uploads root: `uploads/`

## Requirements

1. Save images for questions to disk
2. Download images from external URLs
3. Delete images when question deleted
4. Return relative paths for DB storage

## Architecture

```
uploads/
├── images/
│   ├── {question_id}/
│   │   └── main.{ext}
│   └── ...
├── math/ (existing)
└── english/ (existing)
```

## Related Code Files

**Create:**
- `/Users/long/Downloads/Tạo đề online/app/services/image/question-image-storage.py`

**Modify:**
- `/Users/long/Downloads/Tạo đề online/app/services/image/__init__.py`

## Implementation Steps

1. Create `question-image-storage.py`:

```python
"""
Image storage service for question images.
Stores images in uploads/images/{question_id}/ directory.
"""
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "uploads"
IMAGES_DIR = UPLOAD_DIR / "images"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB


def ensure_images_dir():
    """Create images directory if not exists."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_question_image_dir(question_id: int) -> Path:
    """Get directory for question images."""
    return IMAGES_DIR / str(question_id)


def save_question_image(
    question_id: int,
    image_bytes: bytes,
    filename: str = "main.png"
) -> str:
    """
    Save image for a question.
    Returns relative path for DB storage.
    """
    ensure_images_dir()

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".png"
        filename = "main.png"

    # Create question dir
    question_dir = get_question_image_dir(question_id)
    question_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = question_dir / filename
    file_path.write_bytes(image_bytes)

    # Return relative path
    return f"images/{question_id}/{filename}"


def download_image(url: str, timeout: int = 30) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download image from URL.
    Returns (bytes, error_message).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; QuestionBank/1.0)"
        }
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()

            # Check content type
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None, f"Not an image: {content_type}"

            # Check size
            if len(resp.content) > MAX_IMAGE_SIZE:
                return None, f"Image too large: {len(resp.content)} bytes"

            return resp.content, None
    except Exception as e:
        return None, str(e)


def get_extension_from_url(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".png"


def delete_question_images(question_id: int) -> bool:
    """Delete all images for a question."""
    question_dir = get_question_image_dir(question_id)
    if question_dir.exists():
        import shutil
        shutil.rmtree(question_dir)
        return True
    return False


def image_hash(image_bytes: bytes) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(image_bytes).hexdigest()
```

2. Update `__init__.py` to export:
```python
from .question_image_storage import (
    save_question_image,
    download_image,
    delete_question_images,
    get_extension_from_url,
)
```

## TODO

- [ ] Create question-image-storage.py
- [ ] Add httpx dependency check (already in requirements.txt)
- [ ] Update __init__.py exports
- [ ] Test save/download/delete functions
- [ ] Add to question delete handler

## Success Criteria

- Images save to correct directory structure
- Download works for common image hosts
- Delete cleans up all question images
- Relative paths returned correctly
