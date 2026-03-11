---
phase: 2
status: completed
priority: high
effort: 1.5h
---

# Phase 2: Tách OMR/Grading Services

## Mục tiêu

Tách các hàm OMR và chấm điểm từ `main.py` (dòng 3309-5384, ~2,075 dòng) sang `app/services/omr/`.

## Files tạo mới

### 1. `app/services/omr/__init__.py`

```python
from .template-detector import _detect_template_from_image
from .bubble-detector import (
    _detect_bubbles,
    _detect_bubbles_grid_based,
    _detect_all_rectangles,
    _cluster_by_rows,
    _analyze_bubble_fill,
    _analyze_bubble_fill_improved,
    _group_bubbles_to_questions,
    _group_bubbles_to_questions_improved,
    _detect_seamo_bubbles_fixed_grid,
    _detect_seamo_grid_dynamic,
)
from .grader import (
    _grade_single_sheet,
    _grade_mixed_format_sheet,
)
from .answer-key-parser import (
    _parse_answer_key_for_template,
    _extract_answers_from_text,
)
```

### 2. `app/services/omr/template-detector.py` (~170 dòng)

```python
# Di chuyển từ main.py dòng 3309-3476
def _detect_template_from_image(image_bytes: bytes, num_questions_detected: int = 0) -> dict:
    ...
```

### 3. `app/services/omr/bubble-detector.py` (~600 dòng)

```python
# Di chuyển từ main.py dòng 3740-4233, 4524-4806
def _find_answer_grid_region(gray_image, binary_image): ...  # 3740-3777
def _detect_all_rectangles(binary_image, min_size=15, max_size=80): ...  # 3778-3816
def _cluster_by_rows(rectangles, tolerance=15): ...  # 3817-3844
def _detect_bubbles_grid_based(gray_image, binary_image, template_type): ...  # 3845-3930
def _analyze_bubble_fill_improved(gray_image, rect, threshold=0.4): ...  # 3931-3985
def _group_bubbles_to_questions_improved(rows, template_type): ...  # 3986-4134
def _detect_bubbles(binary_image, template_type): ...  # 4135-4160
def _analyze_bubble_fill(binary_image, bubble_contour, threshold): ...  # 4161-4183
def _group_bubbles_to_questions(bubbles, template_type): ...  # 4184-4233
def _detect_seamo_bubbles_fixed_grid(gray_image): ...  # 4524-4631
def _detect_seamo_grid_dynamic(gray_image): ...  # 4632-4806
```

### 4. `app/services/omr/grader.py` (~560 dòng)

```python
# Di chuyển từ main.py dòng 4234-4523, 4807-5081
def _grade_single_sheet(image_bytes, answer_key, template_type, extract_info): ...  # 4234-4523
def _grade_mixed_format_sheet(...): ...  # 4807-5081
```

### 5. `app/services/omr/answer-key-parser.py` (~300 dòng)

```python
# Di chuyển từ main.py dòng 5082-5384
def _extract_answers_from_text(text: str, num_questions: int) -> dict: ...  # 5082-5102
def _parse_answer_key_for_template(answer_file_content, file_ext, template_type): ...  # 5103-5384
```

## Các bước thực hiện

- [ ] Tạo thư mục `app/services/omr/`
- [ ] Tạo `app/services/omr/__init__.py`
- [ ] Tạo `app/services/omr/template-detector.py`
- [ ] Tạo `app/services/omr/bubble-detector.py`
- [ ] Tạo `app/services/omr/grader.py`
- [ ] Tạo `app/services/omr/answer-key-parser.py`
- [ ] Cập nhật import trong `main.py`
- [ ] Test import: `python -c "from app.services.omr import _grade_single_sheet"`
- [ ] Test app: `uvicorn app.main:app --reload`

## Dependencies cần import

```python
import cv2
import numpy as np
from typing import List, Tuple, Optional
```

## Verification

```bash
python -c "from app.services.omr import _grade_single_sheet, _detect_template_from_image; print('OK')"
```
