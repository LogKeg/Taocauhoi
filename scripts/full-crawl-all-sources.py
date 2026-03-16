#!/usr/bin/env python3
"""
Full crawl from all 5 sources:
- Hoc247, TracNghiem.net, ThuVienHocLieu, VietJack (Vietnamese)
- Open Trivia DB (International - English)
Deletes existing questions first, then crawls everything.
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, QuestionCRUD, Question


def delete_all_questions():
    """Delete all questions from database."""
    db = SessionLocal()
    try:
        count = db.query(Question).count()
        db.query(Question).delete()
        db.commit()
        print(f"Deleted {count} existing questions")
        return count
    finally:
        db.close()


def load_scraper(name: str, filename: str):
    """Load scraper module with proper path setup."""
    import importlib.util

    crawler_dir = os.path.join(os.path.dirname(__file__), "..", "app", "services", "crawler")
    # Add crawler_dir to path so relative imports work
    if crawler_dir not in sys.path:
        sys.path.insert(0, crawler_dir)

    spec = importlib.util.spec_from_file_location(name, os.path.join(crawler_dir, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crawl_hoc247():
    """Crawl all Hoc247 categories."""
    hoc247 = load_scraper("hoc247", "hoc247-exam-scraper.py")

    categories = [
        # Toán
        "https://hoc247.net/trac-nghiem-toan-12-index.html",
        "https://hoc247.net/trac-nghiem-toan-11-index.html",
        "https://hoc247.net/trac-nghiem-toan-10-index.html",
        # Vật lý
        "https://hoc247.net/trac-nghiem-vat-ly-12-index.html",
        "https://hoc247.net/trac-nghiem-vat-ly-11-index.html",
        "https://hoc247.net/trac-nghiem-vat-ly-10-index.html",
        # Hóa học
        "https://hoc247.net/trac-nghiem-hoa-hoc-12-index.html",
        "https://hoc247.net/trac-nghiem-hoa-hoc-10-index.html",
        # Sinh học
        "https://hoc247.net/trac-nghiem-sinh-12-index.html",
        "https://hoc247.net/trac-nghiem-sinh-hoc-11-index.html",
        "https://hoc247.net/trac-nghiem-sinh-hoc-10-index.html",
        # Tiếng Anh
        "https://hoc247.net/trac-nghiem-tieng-anh-12-index.html",
        "https://hoc247.net/trac-nghiem-tieng-anh-11-ket-noi-tri-thuc-index.html",
        "https://hoc247.net/trac-nghiem-tieng-anh-10-ket-noi-tri-thuc-index.html",
        # Lịch sử
        "https://hoc247.net/trac-nghiem-lich-su-12-index.html",
        "https://hoc247.net/trac-nghiem-lich-su-11-index.html",
        "https://hoc247.net/trac-nghiem-lich-su-10-index.html",
        # Địa lý
        "https://hoc247.net/trac-nghiem-dia-12-index.html",
        "https://hoc247.net/trac-nghiem-dia-li-11-index.html",
        "https://hoc247.net/trac-nghiem-dia-10-index.html",
        # Công nghệ
        "https://hoc247.net/trac-nghiem-cong-nghe-12-index.html",
        "https://hoc247.net/trac-nghiem-cong-nghe-11-index.html",
    ]

    all_questions = []
    for url in categories:
        print(f"  Crawling: {url}")
        questions, errors = hoc247.scrape_category(url, max_pages=30)
        all_questions.extend(questions)
        print(f"    Found {len(questions)} questions")
        time.sleep(1)

    return all_questions


def crawl_tracnghiem_net():
    """Crawl all TracNghiem.net categories."""
    tracnghiem = load_scraper("tracnghiem", "tracnghiem-net-scraper.py")

    categories = [
        # THPT
        "https://tracnghiem.net/de-thi/toan-lop-12/",
        "https://tracnghiem.net/de-thi/vat-ly-lop-12/",
        "https://tracnghiem.net/de-thi/hoa-hoc-lop-12/",
        "https://tracnghiem.net/de-thi/sinh-hoc-lop-12/",
        "https://tracnghiem.net/de-thi/tieng-anh-lop-12/",
        "https://tracnghiem.net/de-thi/lich-su-lop-12/",
        "https://tracnghiem.net/de-thi/dia-ly-lop-12/",
        # Lớp 11
        "https://tracnghiem.net/de-thi/toan-lop-11/",
        "https://tracnghiem.net/de-thi/vat-ly-lop-11/",
        "https://tracnghiem.net/de-thi/hoa-hoc-lop-11/",
        # Lớp 10
        "https://tracnghiem.net/de-thi/toan-lop-10/",
        "https://tracnghiem.net/de-thi/vat-ly-lop-10/",
        "https://tracnghiem.net/de-thi/hoa-hoc-lop-10/",
        # THCS
        "https://tracnghiem.net/de-thi/toan-lop-9/",
        "https://tracnghiem.net/de-thi/tieng-anh-lop-9/",
    ]

    all_questions = []
    for url in categories:
        print(f"  Crawling: {url}")
        questions, errors = tracnghiem.scrape_category(url, max_pages=20)
        all_questions.extend(questions)
        print(f"    Found {len(questions)} questions")
        time.sleep(1)

    return all_questions


def crawl_thuvienhoclieu():
    """Crawl ThuVienHocLieu categories."""
    tvhl = load_scraper("thuvienhoclieu", "thuvienhoclieu.py")

    categories = [
        "https://thuvienhoclieu.com/trac-nghiem-online/toan-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/vat-li-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/hoa-hoc-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/sinh-hoc-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/tieng-anh-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/lich-su-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/dia-li-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/gdcd-12/",
        "https://thuvienhoclieu.com/trac-nghiem-online/toan-11/",
        "https://thuvienhoclieu.com/trac-nghiem-online/toan-10/",
    ]

    all_questions = []
    for url in categories:
        print(f"  Crawling: {url}")
        result = tvhl.scrape_category(url, max_quizzes=50)
        questions = result.get("questions", [])
        all_questions.extend(questions)
        print(f"    Found {len(questions)} questions")
        time.sleep(1)

    return all_questions


def crawl_opentdb():
    """Crawl Open Trivia Database (international questions)."""
    opentdb = load_scraper("opentdb", "opentdb-api-scraper.py")

    print("  Fetching all categories from Open Trivia DB...")
    questions, errors = opentdb.scrape_all_categories()

    if errors:
        for err in errors:
            print(f"    Warning: {err}")

    return questions


def crawl_vietjack():
    """Crawl VietJack categories."""
    vietjack = load_scraper("vietjack", "vietjack-exam-and-quiz-scraper.py")

    categories = [
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/bo-de-thi-toan-lop-12-giua-hoc-ki-1.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-toan-12-hoc-ki-1.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-dia-li-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-lich-su-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cong-nghe-12-giua-ki-1-ket-noi-tri-thuc.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/bo-de-thi-tieng-anh-12-bright.jsp",
        "https://vietjack.com/de-kiem-tra-lop-12/de-thi-cuoi-ki-1-lop-12.jsp",
    ]

    all_questions = []
    for url in categories:
        print(f"  Crawling: {url}")
        questions, error = vietjack.scrape_quiz(url)
        if questions:
            all_questions.extend(questions)
            print(f"    Found {len(questions)} questions")
        elif error:
            print(f"    Error: {error}")
        time.sleep(1)

    return all_questions


def save_questions(questions, source_name):
    """Save questions to database with deduplication."""
    from app.services.image import save_question_image, download_image_sync
    import json

    db = SessionLocal()
    saved = 0
    skipped = 0
    images = 0

    try:
        for q in questions:
            # Handle different field names
            content = q.get("question", q.get("content", "")).strip()
            if not content:
                continue

            # Check duplicate
            if QuestionCRUD.exists_by_content(db, content):
                skipped += 1
                continue

            # Format options
            options = q.get("options")
            if isinstance(options, list):
                options = json.dumps(options, ensure_ascii=False)

            # Create question
            question = QuestionCRUD.create(
                db,
                content=content,
                options=options,
                answer=q.get("answer", ""),
                subject=q.get("subject", "general"),
                grade=q.get("grade", ""),
                source=q.get("source", source_name),
                difficulty=q.get("difficulty", "medium"),
                question_type=q.get("question_type", "mcq"),
            )
            saved += 1

            # Download image if available
            image_source_url = q.get("image_source_url", "")
            if image_source_url:
                try:
                    image_bytes = download_image_sync(image_source_url)
                    if image_bytes:
                        filename = image_source_url.split('/')[-1].split('?')[0] or "image.png"
                        image_url = save_question_image(question.id, image_bytes, filename)
                        if image_url:
                            QuestionCRUD.update(db, question.id, image_url=image_url)
                            images += 1
                except Exception as e:
                    pass  # Skip image errors
    finally:
        db.close()

    return {"saved": saved, "skipped": skipped, "images": images}


def main():
    print("=" * 60)
    print("FULL CRAWL - ALL 5 SOURCES")
    print("=" * 60)

    # Step 1: Delete existing data
    print("\n[1/6] Deleting existing questions...")
    delete_all_questions()

    # Also clear images folder
    images_dir = os.path.join(os.path.dirname(__file__), "..", "static", "images", "questions")
    if os.path.exists(images_dir):
        import shutil
        shutil.rmtree(images_dir)
        os.makedirs(images_dir)
        print("Cleared images folder")

    total_saved = 0
    total_images = 0

    # Step 2: Hoc247
    print("\n[2/6] Crawling Hoc247...")
    try:
        questions = crawl_hoc247()
        result = save_questions(questions, "hoc247.net")
        total_saved += result["saved"]
        total_images += result["images"]
        print(f"  => Saved: {result['saved']}, Skipped: {result['skipped']}, Images: {result['images']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 3: TracNghiem.net
    print("\n[3/6] Crawling TracNghiem.net...")
    try:
        questions = crawl_tracnghiem_net()
        result = save_questions(questions, "tracnghiem.net")
        total_saved += result["saved"]
        total_images += result["images"]
        print(f"  => Saved: {result['saved']}, Skipped: {result['skipped']}, Images: {result['images']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 4: ThuVienHocLieu
    print("\n[4/6] Crawling ThuVienHocLieu...")
    try:
        questions = crawl_thuvienhoclieu()
        result = save_questions(questions, "thuvienhoclieu.com")
        total_saved += result["saved"]
        total_images += result["images"]
        print(f"  => Saved: {result['saved']}, Skipped: {result['skipped']}, Images: {result['images']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 5: VietJack
    print("\n[5/6] Crawling VietJack...")
    try:
        questions = crawl_vietjack()
        result = save_questions(questions, "vietjack.com")
        total_saved += result["saved"]
        total_images += result["images"]
        print(f"  => Saved: {result['saved']}, Skipped: {result['skipped']}, Images: {result['images']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 6: Open Trivia DB (International)
    print("\n[6/6] Crawling Open Trivia DB (International)...")
    try:
        questions = crawl_opentdb()
        result = save_questions(questions, "opentdb.com")
        total_saved += result["saved"]
        total_images += result["images"]
        print(f"  => Saved: {result['saved']}, Skipped: {result['skipped']}, Images: {result['images']}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions saved: {total_saved}")
    print(f"Total images downloaded: {total_images}")

    # Verify
    db = SessionLocal()
    try:
        count = db.query(Question).count()
        print(f"Database count: {count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
