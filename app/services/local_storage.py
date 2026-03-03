"""
Local file storage service for templates organized by subject and grade.

Structure:
uploads/
├── math/
│   ├── grade-1/
│   ├── grade-2/
│   └── ...
├── english/
│   ├── grade-1/
│   └── ...
└── ...
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Base upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# Subject mappings (key -> display name)
SUBJECTS = {
    "math": "Toán",
    "english": "Tiếng Anh",
    "science": "Khoa học",
    "physics": "Vật lý",
    "chemistry": "Hóa học",
    "biology": "Sinh học",
    "history": "Lịch sử",
    "geography": "Địa lý",
    "literature": "Ngữ văn",
    "informatics": "Tin học",
    "other": "Khác",
}

# Grade mappings
GRADES = {
    "grade-1": "Lớp 1",
    "grade-2": "Lớp 2",
    "grade-3": "Lớp 3",
    "grade-4": "Lớp 4",
    "grade-5": "Lớp 5",
    "grade-6": "Lớp 6",
    "grade-7": "Lớp 7",
    "grade-8": "Lớp 8",
    "grade-9": "Lớp 9",
    "grade-10": "Lớp 10",
    "grade-11": "Lớp 11",
    "grade-12": "Lớp 12",
    "university": "Đại học",
    "other": "Khác",
}


def ensure_upload_dirs():
    """Ensure all subject/grade directories exist."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    for subject in SUBJECTS.keys():
        subject_dir = UPLOAD_DIR / subject
        subject_dir.mkdir(exist_ok=True)
        for grade in GRADES.keys():
            grade_dir = subject_dir / grade
            grade_dir.mkdir(exist_ok=True)


def get_file_path(subject: str, grade: str, filename: str) -> Path:
    """Get the full path for a file."""
    return UPLOAD_DIR / subject / grade / filename


def save_file(
    file_content: bytes,
    filename: str,
    subject: str,
    grade: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Save a file to local storage.

    Returns dict with file info.
    """
    ensure_upload_dirs()

    # Validate subject and grade
    if subject not in SUBJECTS:
        subject = "other"
    if grade not in GRADES:
        grade = "other"

    # Create unique filename if exists
    file_path = get_file_path(subject, grade, filename)
    if file_path.exists():
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        file_path = get_file_path(subject, grade, filename)

    # Save file
    file_path.write_bytes(file_content)

    # Save metadata if provided
    if metadata:
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        metadata["uploaded_at"] = datetime.now().isoformat()
        metadata["original_filename"] = filename
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    return {
        "filename": filename,
        "subject": subject,
        "grade": grade,
        "path": str(file_path.relative_to(UPLOAD_DIR)),
        "size": file_path.stat().st_size,
        "uploaded_at": datetime.now().isoformat(),
    }


def list_files(
    subject: Optional[str] = None,
    grade: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List files in storage, optionally filtered by subject/grade.
    """
    ensure_upload_dirs()
    files = []

    subjects_to_scan = [subject] if subject and subject in SUBJECTS else SUBJECTS.keys()

    for subj in subjects_to_scan:
        subj_dir = UPLOAD_DIR / subj
        if not subj_dir.exists():
            continue

        grades_to_scan = [grade] if grade and grade in GRADES else GRADES.keys()

        for gr in grades_to_scan:
            grade_dir = subj_dir / gr
            if not grade_dir.exists():
                continue

            for file_path in grade_dir.iterdir():
                # Skip metadata files
                if file_path.suffix == ".json" and ".meta.json" in file_path.name:
                    continue
                if not file_path.is_file():
                    continue

                # Get metadata if exists
                meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
                metadata = {}
                if meta_path.exists():
                    try:
                        metadata = json.loads(meta_path.read_text())
                    except:
                        pass

                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "subject": subj,
                    "subject_label": SUBJECTS.get(subj, subj),
                    "grade": gr,
                    "grade_label": GRADES.get(gr, gr),
                    "path": str(file_path.relative_to(UPLOAD_DIR)),
                    "full_path": str(file_path),
                    "size": stat.st_size,
                    "size_formatted": format_size(stat.st_size),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "metadata": metadata,
                })

    # Sort by modified time, newest first
    files.sort(key=lambda x: x["modified_at"], reverse=True)
    return files


def get_file(subject: str, grade: str, filename: str) -> Optional[bytes]:
    """Get file content."""
    file_path = get_file_path(subject, grade, filename)
    if file_path.exists():
        return file_path.read_bytes()
    return None


def delete_file(subject: str, grade: str, filename: str) -> bool:
    """Delete a file and its metadata."""
    file_path = get_file_path(subject, grade, filename)
    if not file_path.exists():
        return False

    # Delete file
    file_path.unlink()

    # Delete metadata if exists
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
    if meta_path.exists():
        meta_path.unlink()

    return True


def move_file(
    subject: str,
    grade: str,
    filename: str,
    new_subject: str,
    new_grade: str,
) -> Optional[Dict[str, Any]]:
    """Move a file to a different subject/grade."""
    ensure_upload_dirs()

    old_path = get_file_path(subject, grade, filename)
    if not old_path.exists():
        return None

    new_path = get_file_path(new_subject, new_grade, filename)

    # Handle name collision
    if new_path.exists():
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
        new_path = get_file_path(new_subject, new_grade, new_filename)
    else:
        new_filename = filename

    # Move file
    shutil.move(str(old_path), str(new_path))

    # Move metadata if exists
    old_meta = old_path.with_suffix(old_path.suffix + ".meta.json")
    if old_meta.exists():
        new_meta = new_path.with_suffix(new_path.suffix + ".meta.json")
        shutil.move(str(old_meta), str(new_meta))

    return {
        "filename": new_filename,
        "subject": new_subject,
        "grade": new_grade,
        "path": str(new_path.relative_to(UPLOAD_DIR)),
    }


def get_stats() -> Dict[str, Any]:
    """Get storage statistics."""
    ensure_upload_dirs()

    stats = {
        "total_files": 0,
        "total_size": 0,
        "by_subject": {},
        "by_grade": {},
    }

    for subj in SUBJECTS.keys():
        subj_dir = UPLOAD_DIR / subj
        if not subj_dir.exists():
            continue

        subj_count = 0
        subj_size = 0

        for gr in GRADES.keys():
            grade_dir = subj_dir / gr
            if not grade_dir.exists():
                continue

            grade_count = 0
            grade_size = 0

            for file_path in grade_dir.iterdir():
                if file_path.is_file() and ".meta.json" not in file_path.name:
                    grade_count += 1
                    grade_size += file_path.stat().st_size

            if grade_count > 0:
                if gr not in stats["by_grade"]:
                    stats["by_grade"][gr] = {"count": 0, "size": 0, "label": GRADES[gr]}
                stats["by_grade"][gr]["count"] += grade_count
                stats["by_grade"][gr]["size"] += grade_size

            subj_count += grade_count
            subj_size += grade_size

        if subj_count > 0:
            stats["by_subject"][subj] = {
                "count": subj_count,
                "size": subj_size,
                "size_formatted": format_size(subj_size),
                "label": SUBJECTS[subj],
            }

        stats["total_files"] += subj_count
        stats["total_size"] += subj_size

    stats["total_size_formatted"] = format_size(stats["total_size"])
    return stats


def format_size(size_bytes: int) -> str:
    """Format size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Initialize directories on import
ensure_upload_dirs()
