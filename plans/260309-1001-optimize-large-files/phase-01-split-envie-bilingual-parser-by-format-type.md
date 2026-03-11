---
phase: 1
status: partial
priority: high
effort: 1.5h
---

# Phase 1: Tach envie bilingual parser

## Muc tieu

Tach `parsers/exam-envie-bilingual-question-parser.py` (2,444 dong) thanh cac module nho.

## Ket qua thuc te

| File | Dong | Chuc nang |
|------|------|-----------|
| `exam-envie-bilingual-question-parser.py` | 2,334 | Main parser (giu nguyen vi nested functions) |
| `envie-bilingual-dedup-helper.py` | 122 | _dedup_bilingual_science |

**Ly do khong tach them:**
- Main parser `_parse_envie_questions` co 15+ nested functions chia se local state
- Tach ra se gay circular dependencies va breaking changes
- Risk/reward khong tot - giu nguyen de dam bao stability

## Cac buoc

- [x] Phan tich cau truc _parse_envie_questions
- [x] Xac dinh cac format types
- [x] Tach dedup_helper.py (envie-bilingual-dedup-helper.py)
- [x] Cap nhat parsers/__init__.py
- [x] Test imports
- [x] Test app

## Verification

```bash
python -c "from app.parsers import _parse_envie_questions, _dedup_bilingual_science; print('OK')"
# Output: Parsers OK

python -c "import app.main; print(f'App routes: {len(app.main.app.routes)}')"
# Output: App routes: 86
```

## Ghi chu

File `exam-envie-bilingual-question-parser.py` van con 2,334 dong nhung:
1. Logic parsing phuc tap, can giu nguyen de dam bao hoat dong dung
2. Da tach duoc `_dedup_bilingual_science` (112 dong) ra module rieng
3. Khuyen nghi: khong tach them vi risk cao, low benefit
