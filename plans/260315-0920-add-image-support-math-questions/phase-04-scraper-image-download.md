---
phase: 4
title: "Scraper Image Download"
status: pending
effort: 2h
---

# Phase 4: Scraper Image Download

## Context

- Hoc247 scraper: `app/services/crawler/hoc247-exam-scraper.py`
- TracNghiem.net scraper: `app/services/crawler/tracnghiem-net-scraper.py`
- Both use BeautifulSoup for HTML parsing
- Both return question dicts with content, options, answer, subject, grade, source

## Requirements

1. Detect images in question/option elements
2. Return image source URL in question dict
3. Download images during bulk save
4. Handle failures gracefully (skip, log)

## Related Code Files

**Modify:**
- `/Users/long/Downloads/Tạo đề online/app/services/crawler/hoc247-exam-scraper.py`
- `/Users/long/Downloads/Tạo đề online/app/services/crawler/tracnghiem-net-scraper.py`
- `/Users/long/Downloads/Tạo đề online/app/api/routers/crawler.py` (bulk save handler)

## Implementation Steps

### 1. Add image extraction helper to hoc247-exam-scraper.py

After `_extract_latex()` function (~line 119):

```python
def _extract_image_url(el) -> Optional[str]:
    """Extract first image URL from element."""
    if el is None:
        return None

    img = el.find('img')
    if img and img.get('src'):
        src = img['src']
        # Handle protocol-relative URLs
        if src.startswith('//'):
            return 'https:' + src
        # Handle relative URLs
        if src.startswith('/'):
            return 'https://hoc247.net' + src
        return src
    return None
```

### 2. Update parse_quiz_page in hoc247-exam-scraper.py

In the question parsing loop (~line 206), add:

```python
# Extract image from question
image_url = None
if question_link:
    image_url = _extract_image_url(question_link)
if not image_url and h3:
    image_url = _extract_image_url(h3)

# ... existing validation ...

if question_text and len(options) >= 2:
    questions.append({
        "question": question_text,
        "options": options[:4],
        "answer": "",
        "subject": subject,
        "grade": grade,
        "source": url,
        "image_source_url": image_url,  # NEW
    })
```

### 3. Add image extraction to tracnghiem-net-scraper.py

Similar helper function:

```python
def _extract_image_url_from_html(html: str, base_url: str) -> Optional[str]:
    """Extract image URL from HTML snippet."""
    soup = BeautifulSoup(html, "html.parser")
    img = soup.find('img')
    if img and img.get('src'):
        src = img['src']
        if src.startswith('//'):
            return 'https:' + src
        if src.startswith('/'):
            return urljoin(base_url, src)
        if src.startswith('http'):
            return src
    return None
```

Note: TracNghiem.net uses text-based parsing, so image extraction is harder. May need to look at original HTML blocks instead of get_text().

### 4. Update crawler bulk save API

In `/app/api/routers/crawler.py` (or wherever bulk save happens):

```python
from app.services.image import save_question_image, download_image, get_extension_from_url

async def save_questions_to_bank(questions: List[dict], db: Session):
    saved = []
    for q in questions:
        # Download image if present
        image_url = None
        if q.get("image_source_url"):
            img_bytes, err = download_image(q["image_source_url"])
            if img_bytes and not err:
                # Will set actual path after question created
                pass  # Handle below

        # Create question first to get ID
        db_question = Question(
            content=q["question"],
            options=json.dumps(q.get("options", [])),
            answer=q.get("answer", ""),
            subject=q.get("subject", "general"),
            grade=q.get("grade", ""),
            source=q.get("source", ""),
        )
        db.add(db_question)
        db.flush()  # Get ID without commit

        # Now save image with question ID
        if q.get("image_source_url"):
            img_bytes, err = download_image(q["image_source_url"])
            if img_bytes and not err:
                ext = get_extension_from_url(q["image_source_url"])
                image_url = save_question_image(db_question.id, img_bytes, f"main{ext}")
                db_question.image_url = image_url

        saved.append(db_question)

    db.commit()
    return saved
```

### 5. Handle edge cases

- Skip image download if URL is data: URI
- Skip MathJax-rendered images (usually SVG with specific classes)
- Log download failures but continue saving question
- Timeout after 10s per image

```python
# Skip data URIs and MathJax
def should_download_image(url: str) -> bool:
    if not url:
        return False
    if url.startswith('data:'):
        return False
    if 'mathjax' in url.lower() or 'latex' in url.lower():
        return False
    return True
```

## TODO

- [ ] Add `_extract_image_url()` to hoc247 scraper
- [ ] Update hoc247 `parse_quiz_page()` to return image_source_url
- [ ] Add image extraction to tracnghiem.net scraper
- [ ] Update bulk save API to download images
- [ ] Add should_download_image filter
- [ ] Test with real URLs containing images
- [ ] Add error logging for failed downloads

## Success Criteria

- Scraper returns image_source_url when present
- Bulk save downloads images to uploads/images/
- Questions have correct image_url in DB
- Failed downloads don't break import
- MathJax/LaTeX images skipped (we have text LaTeX)
