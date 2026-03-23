---
title: "Add Image Support for Math Questions"
description: "Enable image storage, display, and scraping for math questions"
status: pending
priority: P1
effort: 6h
branch: main
tags: [feature, images, database, scrapers, frontend]
created: 2026-03-15
---

# Add Image Support for Math Questions

## Overview

Enable image support for questions: store images in `uploads/images/`, add `image_url` column to DB, update scrapers to download images, and display images in frontend UI.

## Decision: File System Storage

**Chosen approach**: Store images as files in `uploads/images/{question_id}/` directory.

| Approach | Pros | Cons |
|----------|------|------|
| File system | Simple, fast serving, CDN-ready | Needs cleanup on delete |
| Base64 in DB | Single source of truth | DB bloat, slow queries |

File system wins for performance and simplicity. Reuse existing `local_storage.py` patterns.

## Phase Summary

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Database schema migration | 1h |
| 2 | Image storage service | 1h |
| 3 | Frontend image display | 1.5h |
| 4 | Scraper image download | 2h |
| 5 | Manual upload support | 0.5h |

---

## Phase 1: Database Schema Migration

**Files to modify:**
- `app/database/models.py`

**Changes:**
1. Add `image_url` column (nullable Text) to Question model
2. Create migration script or use Alembic

```python
# In Question model
image_url = Column(Text, nullable=True)  # Relative path: "images/{id}/filename.png"
```

**Migration SQL:**
```sql
ALTER TABLE questions ADD COLUMN image_url TEXT;
```

**TODO:**
- [ ] Add `image_url` column to Question model
- [ ] Run migration on existing DB
- [ ] Test backward compatibility

---

## Phase 2: Image Storage Service

**Files to create:**
- `app/services/image/question_images.py`

**Files to modify:**
- `app/services/image/__init__.py`

**Functionality:**
1. Save image for question: `save_question_image(question_id, image_bytes, filename) -> str`
2. Get image path: `get_question_image_path(question_id, filename) -> Path`
3. Delete images: `delete_question_images(question_id)`
4. Download from URL: `download_image(url) -> bytes`

**Storage structure:**
```
uploads/
  images/
    {question_id}/
      main.png
      thumb.png (optional future)
```

**TODO:**
- [ ] Create `question_images.py` service
- [ ] Add download function with httpx
- [ ] Add cleanup on question delete
- [ ] Export in `__init__.py`

---

## Phase 3: Frontend Image Display

**Files to modify:**
- `app/templates/index.html`
- `app/static/styles.css` (if exists)

**Changes to `renderQuestionCard` function (line ~1158):**
```javascript
function renderQuestionCard(qText, options, answer, index, imageUrl = null) {
  const safeText = escapeHtml(qText).replace(/\n/g, '<br>');
  const imageHtml = imageUrl
    ? `<img src="/uploads/${imageUrl}" class="question-image" alt="Question image" />`
    : '';
  return `
    <div class="created-question" data-index="${index}">
      ${imageHtml}
      ...
    </div>`;
}
```

**Changes to bank-item display (line ~2030):**
- Add image thumbnail if `q.image_url` exists

**Static file serving:**
- Already mounted: `app.mount("/static", ...)`
- Add: `app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")`

**TODO:**
- [ ] Mount uploads directory in main.py
- [ ] Update `renderQuestionCard` to accept imageUrl
- [ ] Update bank-item template for thumbnails
- [ ] Add CSS for `.question-image` class
- [ ] Update question detail modal to show image

---

## Phase 4: Scraper Image Download

**Files to modify:**
- `app/services/crawler/hoc247-exam-scraper.py`
- `app/services/crawler/tracnghiem-net-scraper.py`

**Hoc247 scraper changes:**
1. In `parse_quiz_page()`, detect `<img>` tags in question/option elements
2. Download images before saving question
3. Return `image_url` in question dict

```python
def _extract_image(el) -> Optional[str]:
    """Extract first image URL from element."""
    img = el.find('img')
    if img and img.get('src'):
        src = img['src']
        if src.startswith('//'):
            return 'https:' + src
        return src
    return None
```

**TracNghiem.net scraper changes:**
- Similar pattern but check for MathJax-rendered images
- Some math formulas are rendered as images (skip text-based LaTeX)

**Image download flow:**
1. Scraper returns `image_source_url` (external URL)
2. Bulk save API downloads images
3. Stores locally, sets `image_url` to local path

**TODO:**
- [ ] Add `_extract_image()` helper to both scrapers
- [ ] Return `image_source_url` in question dict
- [ ] Update bulk save API to download images
- [ ] Handle download failures gracefully (log, skip)
- [ ] Add image deduplication (hash-based)

---

## Phase 5: Manual Upload Support

**Files to modify:**
- `app/templates/index.html` (question add/edit modal)
- `app/api/routers/questions.py` (if exists)

**Changes:**
1. Add file input to question modal
2. Add image preview before save
3. API endpoint accepts multipart form with image

**TODO:**
- [ ] Add image upload input to question modal
- [ ] Add preview on select
- [ ] Update save question API to handle image
- [ ] Clear image option

---

## API Changes Summary

| Endpoint | Change |
|----------|--------|
| `POST /questions` | Accept optional image file |
| `PUT /questions/{id}` | Accept optional image file |
| `DELETE /questions/{id}` | Delete associated images |
| `POST /crawler/scrape` | Download images during bulk save |

---

## Success Criteria

1. Questions with images display correctly in all views
2. Scrapers download and store images from source sites
3. Manual image upload works from question modal
4. Images cleaned up when questions deleted
5. No performance regression (lazy loading, thumbnails)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large images slow page | Medium | Lazy load, max width CSS |
| Scraper blocked for images | Low | Fallback to text-only |
| Disk space growth | Low | Optional cleanup job |

---

## Unresolved Questions

1. Should we generate thumbnails for list views?
2. Max image size limit (suggest 5MB)?
3. Support multiple images per question?
