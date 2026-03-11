---
phase: 3
status: completed
priority: medium
effort: 45m
---

# Phase 3: Tach OMR grader

## Muc tieu

Tach `services/omr/omr_answer_sheet_grader.py` (903 dong) thanh cac module nho.

## Ket qua

| File | Dong | Chuc nang |
|------|------|-----------|
| `omr_answer_sheet_grader.py` | 42 | Thin wrapper (backward compat) |
| `omr-single-sheet-grader.py` | 308 | _grade_single_sheet |
| `omr-mixed-format-sheet-grader.py` | 310 | _grade_mixed_format_sheet |
| `omr-answer-key-parser-and-text-extractor.py` | 348 | _parse_answer_key_for_template, _extract_answers_from_text |

## Cac buoc

- [x] Phan tich dependencies giua cac ham
- [x] Tach omr-single-sheet-grader.py
- [x] Tach omr-mixed-format-sheet-grader.py
- [x] Tach omr-answer-key-parser-and-text-extractor.py
- [x] Cap nhat omr_answer_sheet_grader.py (thin wrapper)
- [x] Test imports
- [x] Test app startup

## Verification

```bash
python -c "from app.services.omr import _grade_single_sheet; print('OK')"
# Output: OK

python -c "import app.main; print(f'App routes: {len(app.main.app.routes)}')"
# Output: App routes: 86
```
