---
phase: 1
status: completed
priority: high
effort: 1h
---

# Phase 1: Tách Exam Parsers

## Mục tiêu

Tách 3 hàm parse đề thi từ `main.py` (dòng 280-3073, ~2,800 dòng) sang `app/parsers/`.

## Files tạo mới

### 1. `app/parsers/exam-math-parser.py` (~200 dòng)

```python
# Hàm cần di chuyển từ main.py dòng 280-478
def _parse_math_exam_questions(lines: List[str]) -> List[dict]:
    ...
```

### 2. `app/parsers/exam-english-parser.py` (~400 dòng)

```python
# Hàm cần di chuyển từ main.py dòng 479-639 + 3074-3308
def _parse_english_exam_questions(doc: Document) -> List[dict]:
    ...

# Helper functions (dòng 3074-3308)
def is_matching_section(text: str) -> bool: ...
def is_matching_table_line(text: str) -> bool: ...
def is_dialogue_completion(text: str) -> bool: ...
def is_blank_only_line(text: str) -> bool: ...
def is_dialogue_blank_line(text: str) -> bool: ...
def is_dialogue_prompt_line(text: str) -> bool: ...
def is_reading_passage_start(text: str) -> bool: ...
def is_question_with_single_word_options(text: str) -> bool: ...
def extract_passage_questions(lines, start_idx) -> Tuple[List[dict], int]: ...
def is_passage_with_blanks(text: str) -> bool: ...
def extract_cloze_questions(start_idx, lines_list) -> Tuple[List[dict], int]: ...
```

### 3. `app/parsers/exam-envie-bilingual-parser.py` (~2,200 dòng)

```python
# Hàm cần di chuyển từ main.py dòng 640-2954
def _parse_envie_questions(doc: Document) -> List[dict]:
    ...

# Hàm helper từ dòng 2955-3073
def _dedup_bilingual_science(questions: List[dict]) -> List[dict]:
    ...
```

## Các bước thực hiện

- [ ] Tạo `app/parsers/exam-math-parser.py`
- [ ] Di chuyển `_parse_math_exam_questions` (dòng 280-478)
- [ ] Tạo `app/parsers/exam-english-parser.py`
- [ ] Di chuyển `_parse_english_exam_questions` (dòng 479-639)
- [ ] Di chuyển các helper functions (dòng 3074-3308)
- [ ] Tạo `app/parsers/exam-envie-bilingual-parser.py`
- [ ] Di chuyển `_parse_envie_questions` (dòng 640-2954)
- [ ] Di chuyển `_dedup_bilingual_science` (dòng 2955-3073)
- [ ] Cập nhật `app/parsers/__init__.py` export các hàm
- [ ] Cập nhật import trong `main.py`
- [ ] Test: `python -c "from app.parsers import _parse_math_exam_questions"`
- [ ] Test: chạy app `uvicorn app.main:app --reload`

## Dependencies cần import

```python
import re
from typing import List, Tuple
from docx import Document
```

## Cập nhật main.py

```python
# Thêm import
from app.parsers import (
    _parse_math_exam_questions,
    _parse_english_exam_questions,
    _parse_envie_questions,
    _dedup_bilingual_science,
)
```

## Verification

```bash
cd "/Users/long/Downloads/Tạo đề online"
source .venv/bin/activate
python -c "from app.parsers import _parse_math_exam_questions; print('OK')"
uvicorn app.main:app --reload --port 8000
```
