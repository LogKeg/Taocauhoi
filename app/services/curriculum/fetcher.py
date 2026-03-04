"""
Fetcher for curriculum documents from Bộ GD&ĐT website.
"""

import httpx
from typing import Optional, Dict, List
from pathlib import Path


# Known curriculum PDF URLs from moet.gov.vn
CURRICULUM_URLS = {
    "toan": {
        "name": "Toán học",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/1.%20Toan.pdf",
        "description": "Chương trình môn Toán - GDPT 2018",
    },
    "van": {
        "name": "Ngữ văn",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/2.%20Ngu%20van.pdf",
        "description": "Chương trình môn Ngữ văn - GDPT 2018",
    },
    "ly": {
        "name": "Vật lý",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Vat%20li.pdf",
        "description": "Chương trình môn Vật lý - GDPT 2018",
    },
    "hoa": {
        "name": "Hóa học",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Hoa%20hoc.pdf",
        "description": "Chương trình môn Hóa học - GDPT 2018",
    },
    "sinh": {
        "name": "Sinh học",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Sinh%20hoc.pdf",
        "description": "Chương trình môn Sinh học - GDPT 2018",
    },
    "anh": {
        "name": "Tiếng Anh",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Tieng%20Anh.pdf",
        "description": "Chương trình môn Tiếng Anh - GDPT 2018",
    },
    "su": {
        "name": "Lịch sử",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Lich%20su.pdf",
        "description": "Chương trình môn Lịch sử - GDPT 2018",
    },
    "dia": {
        "name": "Địa lý",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Dia%20li.pdf",
        "description": "Chương trình môn Địa lý - GDPT 2018",
    },
    "gdcd": {
        "name": "Giáo dục công dân",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/GDCD.pdf",
        "description": "Chương trình môn GDCD - GDPT 2018",
    },
    "tin": {
        "name": "Tin học",
        "program_2018": "https://moet.gov.vn/content/tintuc/Lists/News/Attachments/6235/Tin%20hoc.pdf",
        "description": "Chương trình môn Tin học - GDPT 2018",
    },
}


def get_curriculum_urls() -> Dict:
    """Get all available curriculum URLs."""
    return CURRICULUM_URLS


async def fetch_curriculum_pdf(
    subject: str,
    save_dir: str = "uploads/curriculum",
    timeout: float = 60.0
) -> Optional[str]:
    """
    Fetch curriculum PDF from moet.gov.vn.

    Args:
        subject: Subject key (toan, ly, hoa, etc.)
        save_dir: Directory to save downloaded PDF
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded PDF file, or None if failed
    """
    if subject not in CURRICULUM_URLS:
        print(f"Unknown subject: {subject}")
        return None

    url = CURRICULUM_URLS[subject].get("program_2018")
    if not url:
        print(f"No URL available for subject: {subject}")
        return None

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"curriculum_{subject}_2018.pdf"
    file_path = save_path / filename

    # Check if already downloaded
    if file_path.exists():
        print(f"Using cached PDF: {file_path}")
        return str(file_path)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            print(f"Downloading curriculum PDF for {subject}...")
            response = await client.get(url)
            response.raise_for_status()

            # Save to file
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"Saved to: {file_path}")
            return str(file_path)

    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching PDF: {e.response.status_code}")
        return None
    except httpx.TimeoutException:
        print(f"Timeout fetching PDF for {subject}")
        return None
    except Exception as e:
        print(f"Error fetching PDF: {e}")
        return None


def fetch_curriculum_pdf_sync(
    subject: str,
    save_dir: str = "uploads/curriculum",
    timeout: float = 60.0
) -> Optional[str]:
    """
    Synchronous version of fetch_curriculum_pdf.

    Args:
        subject: Subject key (toan, ly, hoa, etc.)
        save_dir: Directory to save downloaded PDF
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded PDF file, or None if failed
    """
    if subject not in CURRICULUM_URLS:
        print(f"Unknown subject: {subject}")
        return None

    url = CURRICULUM_URLS[subject].get("program_2018")
    if not url:
        print(f"No URL available for subject: {subject}")
        return None

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"curriculum_{subject}_2018.pdf"
    file_path = save_path / filename

    # Check if already downloaded
    if file_path.exists():
        print(f"Using cached PDF: {file_path}")
        return str(file_path)

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            print(f"Downloading curriculum PDF for {subject}...")
            response = client.get(url)
            response.raise_for_status()

            # Save to file
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"Saved to: {file_path}")
            return str(file_path)

    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching PDF: {e.response.status_code}")
        return None
    except httpx.TimeoutException:
        print(f"Timeout fetching PDF for {subject}")
        return None
    except Exception as e:
        print(f"Error fetching PDF: {e}")
        return None


async def fetch_all_curriculum_pdfs(save_dir: str = "uploads/curriculum") -> Dict[str, Optional[str]]:
    """
    Fetch all available curriculum PDFs.

    Args:
        save_dir: Directory to save downloaded PDFs

    Returns:
        Dict mapping subject to downloaded file path
    """
    results = {}

    for subject in CURRICULUM_URLS:
        path = await fetch_curriculum_pdf(subject, save_dir)
        results[subject] = path

    return results
