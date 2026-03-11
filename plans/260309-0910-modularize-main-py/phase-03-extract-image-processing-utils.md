---
phase: 3
status: completed
priority: medium
effort: 30m
---

# Phase 3: Tách Image Processing Utils

## Mục tiêu

Tách các hàm xử lý ảnh từ `main.py` (dòng 3477-3739, ~260 dòng) sang `app/services/image/`.

## Files tạo mới

### 1. `app/services/image/__init__.py`

```python
from .preprocessing import (
    _order_points,
    _four_point_transform,
    _deskew_image,
    _find_document_contour,
    _preprocess_omr_image,
)
from .ocr-student-info-extractor import _extract_student_info_ocr
```

### 2. `app/services/image/preprocessing.py` (~160 dòng)

```python
# Di chuyển từ main.py dòng 3579-3739
def _order_points(pts): ...  # 3579-3591
def _four_point_transform(image, pts): ...  # 3592-3621
def _deskew_image(image): ...  # 3622-3662
def _find_document_contour(image): ...  # 3663-3688
def _preprocess_omr_image(image_bytes: bytes): ...  # 3689-3739
```

### 3. `app/services/image/ocr-student-info-extractor.py` (~100 dòng)

```python
# Di chuyển từ main.py dòng 3477-3578
def _extract_student_info_ocr(image_bytes: bytes) -> dict:
    ...
```

## Các bước thực hiện

- [ ] Tạo thư mục `app/services/image/`
- [ ] Tạo `app/services/image/__init__.py`
- [ ] Tạo `app/services/image/preprocessing.py`
- [ ] Di chuyển các hàm preprocessing (dòng 3579-3739)
- [ ] Tạo `app/services/image/ocr-student-info-extractor.py`
- [ ] Di chuyển `_extract_student_info_ocr` (dòng 3477-3578)
- [ ] Cập nhật import trong `main.py`
- [ ] Cập nhật import trong `app/services/omr/` (nếu cần dùng preprocessing)
- [ ] Test import
- [ ] Test app

## Dependencies cần import

```python
import cv2
import numpy as np
from typing import Tuple, Optional
```

## Lưu ý

- `_preprocess_omr_image` được gọi bởi các hàm OMR → cần import trong `app/services/omr/`
- `_extract_student_info_ocr` dùng pytesseract → cần xử lý optional import

## Verification

```bash
python -c "from app.services.image import _preprocess_omr_image, _extract_student_info_ocr; print('OK')"
```
