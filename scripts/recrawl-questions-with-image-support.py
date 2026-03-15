#!/usr/bin/env python3
"""
Script to delete old questions from a source and re-crawl with image support.

Usage:
    python scripts/recrawl-with-images.py hoc247
    python scripts/recrawl-with-images.py tracnghiem-net
    python scripts/recrawl-with-images.py all
"""
import sys
import os
import time
import importlib.util

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, Question, QuestionCRUD
from app.services.image import save_question_image, download_image_sync


def load_module(module_name: str, file_path: str):
    """Dynamically load a Python module from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load scrapers
_crawler_dir = os.path.join(os.path.dirname(__file__), "..", "app", "services", "crawler")

hoc247_scraper = load_module("hoc247", os.path.join(_crawler_dir, "hoc247-exam-scraper.py"))
tracnghiem_scraper = load_module("tracnghiem", os.path.join(_crawler_dir, "tracnghiem-net-scraper.py"))
vietjack_scraper = load_module("vietjack", os.path.join(_crawler_dir, "vietjack-exam-and-quiz-scraper.py"))
thuvienhoclieu_mod = load_module("thuvienhoclieu", os.path.join(_crawler_dir, "thuvienhoclieu.py"))


def delete_questions_by_source(db, source_pattern: str) -> int:
    """Delete all questions matching source pattern."""
    query = db.query(Question).filter(Question.source.like(f"%{source_pattern}%"))
    count = query.count()
    if count > 0:
        query.delete(synchronize_session=False)
        db.commit()
    return count


def save_questions_with_images(db, questions: list, source_domain: str) -> dict:
    """Save questions and download images. Returns {saved, images}."""
    saved = 0
    images_downloaded = 0
    skipped = 0

    for q in questions:
        content = q.get("question", "").strip()
        if not content:
            continue

        # Check for duplicates
        if QuestionCRUD.exists_by_content(db, content):
            skipped += 1
            continue

        # Create question
        import json
        options = q.get("options", [])
        question = QuestionCRUD.create(
            db,
            content=content,
            options=json.dumps(options, ensure_ascii=False) if options else None,
            answer=q.get("answer", ""),
            subject=q.get("subject", "general"),
            grade=q.get("grade", ""),
            source=q.get("source", source_domain),
            difficulty="medium",
            question_type="mcq",
        )
        saved += 1

        # Download and save image if available
        image_source_url = q.get("image_source_url", "")
        if image_source_url:
            try:
                image_bytes = download_image_sync(image_source_url)
                if image_bytes:
                    filename = image_source_url.split('/')[-1].split('?')[0] or "image.png"
                    image_url = save_question_image(question.id, image_bytes, filename)
                    if image_url:
                        QuestionCRUD.update(db, question.id, image_url=image_url)
                        images_downloaded += 1
            except Exception as e:
                print(f"  [!] Failed to download image for Q{question.id}: {e}")

    return {"saved": saved, "images": images_downloaded, "skipped": skipped}


def recrawl_hoc247(db):
    """Delete old Hoc247 questions and re-crawl with images."""
    print("\n=== Hoc247.net ===")

    # Delete old questions
    deleted = delete_questions_by_source(db, "hoc247.net")
    print(f"Deleted {deleted} old questions")

    # Categories to crawl
    categories = [
        "https://hoc247.net/trac-nghiem-toan-12-index.html",
        "https://hoc247.net/trac-nghiem-toan-11-index.html",
        "https://hoc247.net/trac-nghiem-toan-10-index.html",
        "https://hoc247.net/trac-nghiem-vat-ly-12-index.html",
        "https://hoc247.net/trac-nghiem-vat-ly-11-index.html",
        "https://hoc247.net/trac-nghiem-hoa-hoc-12-index.html",
        "https://hoc247.net/trac-nghiem-hoa-hoc-11-index.html",
        "https://hoc247.net/trac-nghiem-sinh-12-index.html",
        "https://hoc247.net/trac-nghiem-tieng-anh-12-index.html",
        "https://hoc247.net/trac-nghiem-lich-su-12-index.html",
        "https://hoc247.net/trac-nghiem-dia-12-index.html",
    ]

    total_saved = 0
    total_images = 0

    for cat_url in categories:
        print(f"\nCrawling: {cat_url.split('/')[-1]}")
        try:
            questions, errors = hoc247_scraper.scrape_category(cat_url, max_pages=15)
            if questions:
                result = save_questions_with_images(db, questions, "hoc247.net")
                total_saved += result["saved"]
                total_images += result["images"]
                print(f"  Saved: {result['saved']}, Images: {result['images']}, Skipped: {result['skipped']}")
            if errors:
                print(f"  Errors: {len(errors)}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(2)

    print(f"\nHoc247 Total: {total_saved} questions, {total_images} images")
    return total_saved, total_images


def recrawl_tracnghiem(db):
    """Delete old TracNghiem.net questions and re-crawl with images."""
    print("\n=== TracNghiem.net ===")

    # Delete old questions
    deleted = delete_questions_by_source(db, "tracnghiem.net")
    print(f"Deleted {deleted} old questions")

    # Categories to crawl (using correct URL format)
    categories = [
        "https://tracnghiem.net/de-thi/toan-lop-12/",
        "https://tracnghiem.net/de-thi/vat-ly-lop-12/",
        "https://tracnghiem.net/de-thi/hoa-hoc-lop-12/",
        "https://tracnghiem.net/de-thi/sinh-hoc-lop-12/",
        "https://tracnghiem.net/de-thi/tieng-anh-lop-12/",
        "https://tracnghiem.net/de-thi/lich-su-lop-12/",
        "https://tracnghiem.net/de-thi/dia-ly-lop-12/",
        "https://tracnghiem.net/de-thi/toan-lop-11/",
        "https://tracnghiem.net/de-thi/vat-ly-lop-11/",
    ]

    total_saved = 0
    total_images = 0

    for cat_url in categories:
        print(f"\nCrawling: {cat_url.split('/')[-2]}")
        try:
            questions, errors = tracnghiem_scraper.scrape_category(cat_url, max_pages=10)
            if questions:
                result = save_questions_with_images(db, questions, "tracnghiem.net")
                total_saved += result["saved"]
                total_images += result["images"]
                print(f"  Saved: {result['saved']}, Images: {result['images']}, Skipped: {result['skipped']}")
            if errors:
                print(f"  Errors: {len(errors)}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(2)

    print(f"\nTracNghiem.net Total: {total_saved} questions, {total_images} images")
    return total_saved, total_images


def recrawl_thuvienhoclieu(db):
    """Delete old ThuVienHocLieu questions and re-crawl with images."""
    print("\n=== ThuVienHocLieu.com ===")

    # Delete old questions
    deleted = delete_questions_by_source(db, "thuvienhoclieu.com")
    print(f"Deleted {deleted} old questions")

    # Categories to crawl (using correct URL format)
    categories = [
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-dia-li/trac-nghiem-online-dia-li-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-dia-li/trac-nghiem-online-dia-li-on-thi-tn-thpt/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-hoa/trac-nghiem-online-hoa-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-hoa/trac-nghiem-online-mon-hoa-on-thi-tnthpt/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-sinh/trac-nghiem-online-mon-sinh-on-thi-tn-thpt/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-toan/",
        "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-lich-su/",
    ]

    total_saved = 0
    total_images = 0

    for cat_url in categories:
        print(f"\nCrawling: {cat_url.split('/')[-2]}")
        try:
            result = thuvienhoclieu_mod.scrape_category(cat_url, max_quizzes=20)
            questions = result.get("questions", [])
            if questions:
                save_result = save_questions_with_images(db, questions, "thuvienhoclieu.com")
                total_saved += save_result["saved"]
                total_images += save_result["images"]
                print(f"  Saved: {save_result['saved']}, Images: {save_result['images']}, Skipped: {save_result['skipped']}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(2)

    print(f"\nThuVienHocLieu Total: {total_saved} questions, {total_images} images")
    return total_saved, total_images


def recrawl_vietjack(db):
    """Delete old VietJack questions and re-crawl with images."""
    print("\n=== VietJack.com ===")

    # Delete old questions
    deleted = delete_questions_by_source(db, "vietjack.com")
    print(f"Deleted {deleted} old questions")

    # Categories to crawl
    categories = [
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-hoc-ki-1.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-dia-li-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-lich-su-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cong-nghe-12-giua-ki-1-ket-noi-tri-thuc.jsp",
    ]

    total_saved = 0
    total_images = 0

    for url in categories:
        print(f"\nCrawling: {url.split('/')[-1]}")
        try:
            questions, error = vietjack_scraper.scrape_quiz(url)
            if error:
                print(f"  Error: {error}")
                continue
            if questions:
                result = save_questions_with_images(db, questions, "vietjack.com")
                total_saved += result["saved"]
                total_images += result["images"]
                print(f"  Saved: {result['saved']}, Images: {result['images']}, Skipped: {result['skipped']}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(2)

    print(f"\nVietJack Total: {total_saved} questions, {total_images} images")
    return total_saved, total_images


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/recrawl-with-images.py [source|all]")
        print("Sources: hoc247, tracnghiem-net, thuvienhoclieu, vietjack, all")
        sys.exit(1)

    source = sys.argv[1].lower()

    db = SessionLocal()
    try:
        grand_total_saved = 0
        grand_total_images = 0

        if source in ["hoc247", "all"]:
            saved, images = recrawl_hoc247(db)
            grand_total_saved += saved
            grand_total_images += images

        if source in ["tracnghiem-net", "tracnghiem", "all"]:
            saved, images = recrawl_tracnghiem(db)
            grand_total_saved += saved
            grand_total_images += images

        if source in ["thuvienhoclieu", "all"]:
            saved, images = recrawl_thuvienhoclieu(db)
            grand_total_saved += saved
            grand_total_images += images

        if source in ["vietjack", "all"]:
            saved, images = recrawl_vietjack(db)
            grand_total_saved += saved
            grand_total_images += images

        print(f"\n{'='*50}")
        print(f"GRAND TOTAL: {grand_total_saved} questions, {grand_total_images} images")
        print(f"{'='*50}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
