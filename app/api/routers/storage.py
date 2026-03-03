"""
API endpoints for local file storage management.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import os

from app.services.local_storage import (
    save_file,
    list_files,
    get_file,
    delete_file,
    move_file,
    get_stats,
    get_file_path,
    SUBJECTS,
    GRADES,
)

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/subjects")
def get_subjects():
    """Get list of available subjects."""
    return [{"key": k, "label": v} for k, v in SUBJECTS.items()]


@router.get("/grades")
def get_grades():
    """Get list of available grades."""
    return [{"key": k, "label": v} for k, v in GRADES.items()]


@router.get("/stats")
def get_storage_stats():
    """Get storage statistics."""
    return get_stats()


@router.get("/files")
def list_storage_files(
    subject: Optional[str] = None,
    grade: Optional[str] = None,
):
    """List files, optionally filtered by subject and/or grade."""
    files = list_files(subject=subject, grade=grade)
    return {"files": files, "count": len(files)}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    subject: str = Form(...),
    grade: str = Form(...),
    description: Optional[str] = Form(None),
):
    """Upload a file to local storage."""
    # Validate file type
    allowed_extensions = {".docx", ".doc", ".pdf", ".xlsx", ".xls", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read file content
    content = await file.read()

    # Prepare metadata
    metadata = {
        "description": description,
        "content_type": file.content_type,
        "original_size": len(content),
    }

    # Save file
    result = save_file(
        file_content=content,
        filename=file.filename,
        subject=subject,
        grade=grade,
        metadata=metadata,
    )

    return {"success": True, "file": result}


@router.get("/download/{subject}/{grade}/{filename}")
def download_file(subject: str, grade: str, filename: str):
    """Download a file from storage."""
    file_path = get_file_path(subject, grade, filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.delete("/files/{subject}/{grade}/{filename}")
def delete_storage_file(subject: str, grade: str, filename: str):
    """Delete a file from storage."""
    success = delete_file(subject, grade, filename)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "message": f"Deleted {filename}"}


@router.post("/move")
def move_storage_file(
    subject: str = Form(...),
    grade: str = Form(...),
    filename: str = Form(...),
    new_subject: str = Form(...),
    new_grade: str = Form(...),
):
    """Move a file to a different subject/grade."""
    result = move_file(subject, grade, filename, new_subject, new_grade)
    if not result:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "file": result}


@router.get("/tree")
def get_file_tree():
    """Get hierarchical file tree structure."""
    files = list_files()

    # Build tree structure
    tree = {}
    for f in files:
        subj = f["subject"]
        grade = f["grade"]

        if subj not in tree:
            tree[subj] = {
                "label": SUBJECTS.get(subj, subj),
                "grades": {},
                "count": 0,
            }

        if grade not in tree[subj]["grades"]:
            tree[subj]["grades"][grade] = {
                "label": GRADES.get(grade, grade),
                "files": [],
            }

        tree[subj]["grades"][grade]["files"].append({
            "filename": f["filename"],
            "size": f["size_formatted"],
            "modified_at": f["modified_at"],
        })
        tree[subj]["count"] += 1

    return tree
