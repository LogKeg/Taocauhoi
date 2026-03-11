---
phase: 4
status: completed
priority: medium
effort: 30m
---

# Phase 4: Dọn dẹp main.py và Verify

## Mục tiêu

Xóa code đã di chuyển, thêm imports, và verify app hoạt động.

## Các bước thực hiện

### 1. Tách utils còn lại

- [ ] Tạo `app/utils/question-rewriter.py`
- [ ] Di chuyển các hàm (dòng 133-239):
  - `_is_mcq_block`
  - `_force_variation`
  - `_rewrite_english_question`
  - `_rewrite_mcq_block`
- [ ] Cập nhật import trong `main.py`

### 2. Cập nhật main.py imports

```python
# Thêm imports mới
from app.parsers import (
    _parse_math_exam_questions,
    _parse_english_exam_questions,
    _parse_envie_questions,
    _dedup_bilingual_science,
)

from app.services.omr import (
    _detect_template_from_image,
    _grade_single_sheet,
    _grade_mixed_format_sheet,
    _parse_answer_key_for_template,
)

from app.services.image import (
    _preprocess_omr_image,
    _extract_student_info_ocr,
)

from app.utils.question_rewriter import (
    _is_mcq_block,
    _force_variation,
    _rewrite_english_question,
    _rewrite_mcq_block,
)
```

### 3. Xóa code đã di chuyển

- [ ] Xóa dòng 133-239 (question rewriter utils)
- [ ] Xóa dòng 280-3308 (exam parsers + helpers)
- [ ] Xóa dòng 3309-5384 (OMR + image processing)
- [ ] Giữ lại: imports, app setup, routes (dòng 1-132, 240-279)

### 4. Verify final structure

```
app/main.py (~150 dòng)
├── imports
├── app = FastAPI()
├── mount static
├── include routers
├── @app.get("/") index()
└── @app.post("/upload-sample") upload_sample()
```

### 5. Full test

- [ ] `python -c "import app.main; print('OK')"`
- [ ] `uvicorn app.main:app --reload --port 8000`
- [ ] Test upload đề mẫu
- [ ] Test chấm điểm OMR
- [ ] Test parse các loại đề

## Verification Commands

```bash
cd "/Users/long/Downloads/Tạo đề online"
source .venv/bin/activate

# Test imports
python -c "from app.parsers import _parse_math_exam_questions; print('Parser OK')"
python -c "from app.services.omr import _grade_single_sheet; print('OMR OK')"
python -c "from app.services.image import _preprocess_omr_image; print('Image OK')"
python -c "import app.main; print('Main OK')"

# Line count
wc -l app/main.py  # Should be ~150

# Run app
uvicorn app.main:app --reload --port 8000
```

## Kết quả mong đợi

| File | Dòng trước | Dòng sau |
|------|------------|----------|
| `app/main.py` | 5,384 | ~150 |
| `app/parsers/exam-*.py` | - | ~2,800 |
| `app/services/omr/*.py` | - | ~2,075 |
| `app/services/image/*.py` | - | ~260 |
| `app/utils/question-rewriter.py` | - | ~110 |

## Rollback

Nếu có lỗi, revert bằng:
```bash
git checkout app/main.py
git clean -fd app/parsers/exam-*.py app/services/omr/ app/services/image/ app/utils/
```
