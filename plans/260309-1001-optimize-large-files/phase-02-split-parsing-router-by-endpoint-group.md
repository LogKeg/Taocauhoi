---
phase: 2
status: completed
priority: high
effort: 1h
---

# Phase 2: Tach parsing router

## Muc tieu

Tach `api/routers/parsing.py` (1,134 dong) thanh cac module nho.

## Ket qua

| File | Dong | Chuc nang |
|------|------|-----------|
| `parsing/__init__.py` | 8 | Re-export router |
| `parsing/router.py` | 35 | Main router, include sub-routers |
| `parsing/helpers.py` | 313 | Shared helper functions |
| `parsing/parse-exam-endpoints.py` | 114 | POST /api/parse-exam |
| `parsing/convert-word-to-excel-endpoint.py` | 475 | POST /convert-word-to-excel |
| `parsing/ai-exam-analysis-and-generation-endpoints.py` | 260 | POST /api/analyze-exam, /api/generate-similar-exam |
| `parsing/answer-templates-endpoint.py` | 25 | GET /api/answer-templates |
| **Tong** | **1,230** | |

## Cac buoc

- [x] Phan tich cac endpoint groups
- [x] Tao thu muc api/routers/parsing/
- [x] Tach helpers.py (313 dong)
- [x] Tach parse-exam-endpoints.py (114 dong)
- [x] Tach convert-word-to-excel-endpoint.py (475 dong)
- [x] Tach ai-exam-analysis-and-generation-endpoints.py (260 dong)
- [x] Tach answer-templates-endpoint.py (25 dong)
- [x] Tao router.py voi include logic (35 dong)
- [x] Cap nhat __init__.py
- [x] Test imports
- [x] Test app startup

## Verification

```bash
python -c "from app.api.routers.parsing import router; print(f'Router loaded, routes: {len(router.routes)}')"
# Output: Router loaded, routes: 5

python -c "import app.main; print(f'App routes: {len(app.main.app.routes)}')"
# Output: App routes: 86
```

## Ghi chu

- File `convert-word-to-excel-endpoint.py` (475 dong) van lon, nhung chua 1 endpoint duy nhat voi logic phuc tap
- Dung importlib de load kebab-case filenames trong router.py
- Dung absolute imports trong endpoint files de tranh relative import issues
