---
phase: 4
status: completed
priority: low
effort: 1.5h
---

# Phase 4: Tach cac file lon con lai

## Muc tieu

Tach cac file >500 dong con lai thanh cac module nho hon.

## Ket qua thuc te

| File goc | Dong goc | Trang thai | Ket qua |
|----------|----------|------------|---------|
| `api/routers/generation.py` | 759 | DONE | Tach thanh 5 files trong `generation/` |
| `api/routers/grading.py` | 675 | DONE | Tach thanh 4 files trong `grading/` |
| `services/math_renderer.py` | 621 | GIU NGUYEN | Tight dependencies, tach gay phuc tap |
| `database.py` | 593 | DONE | Tach thanh 5 files trong `database/` |

## 4.1 Tach generation.py (759 dong) - DONE

### Cau truc moi
```
app/api/routers/generation/
├── __init__.py                         # 8 dong - Re-exports
├── router.py                           # 33 dong - Router aggregator
├── helpers.py                          # 106 dong - Shared utilities
├── question-generation-endpoints.py    # 220 dong - /generate, /generate-topic, /auto-generate
├── export-to-file-endpoints.py         # 361 dong - /export, /api/export-exam
└── sample-file-endpoints.py            # 104 dong - /sample-folders, /sample-files, etc
```

### Verification
```bash
python -c "from app.api.routers.generation import router; print(f'Routes: {len(router.routes)}')"
# Output: Routes: 9
```

## 4.2 Tach grading.py (675 dong) - DONE

### Cau truc moi
```
app/api/routers/grading/
├── __init__.py                         # 8 dong - Re-exports
├── router.py                           # 31 dong - Router aggregator
├── helpers.py                          # 72 dong - Shared utilities
├── omr-sheet-grading-endpoints.py      # 427 dong - /api/grade-sheets, /api/grade-sheets/export
└── handwritten-grading-endpoints.py    # 195 dong - /api/grade-handwritten, /api/grade-handwritten/export
```

### Verification
```bash
python -c "from app.api.routers.grading import router; print(f'Routes: {len(router.routes)}')"
# Output: Routes: 4
```

## 4.3 math_renderer.py (621 dong) - GIU NGUYEN

### Phan tich
- File co tight dependencies giua cac function groups
- `render_text_with_math` su dung `has_latex`, `extract_latex_parts`, `latex_to_unicode`
- OMML functions chain together: `latex_to_mathml` -> `mathml_to_omml` -> `add_math_to_paragraph`
- Tach ra se tao import complexity without significant benefit

### Quyet dinh: GIU NGUYEN
- Risk cao, benefit thap
- File van hoat dong tot o 621 dong

## 4.4 Tach database.py (593 dong) - DONE

### Cau truc moi
```
app/database/
├── __init__.py                         # 64 dong - Re-exports, backward compatibility
├── models.py                           # 191 dong - All SQLAlchemy models
├── question-crud-operations.py         # 108 dong - QuestionCRUD class
├── exam-crud-operations.py             # 99 dong - ExamCRUD class
├── history-crud-operations.py          # 43 dong - HistoryCRUD class
└── curriculum-crud-operations.py       # 119 dong - CurriculumCRUD class
```

### Verification
```bash
python -c "from app.database import QuestionCRUD, ExamCRUD, HistoryCRUD, CurriculumCRUD; print('DB OK')"
# Output: DB OK

python -c "import app.main; print(f'App routes: {len(app.main.app.routes)}')"
# Output: App routes: 86
```

## Tong ket

- **3/4 files da tach thanh cong**
- **1/4 file giu nguyen** (math_renderer.py) do tight dependencies
- **Tong dong code da tach**: 759 + 675 + 593 = 2,027 dong
- **Backward compatibility**: 100% - moi import cu van hoat dong
