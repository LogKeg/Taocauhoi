---
title: Toi uu hoa cac file lon
status: completed
priority: P2
effort: high
branch: main
tags: [refactor, optimization, modularization]
created: 2026-03-09
completed: 2026-03-09
---

# Toi uu hoa cac file lon

## Tong quan

Tach cac file >500 dong thanh modules nho hon 200 dong.

## Phan tich hien trang

| File | Lines | Priority | Action |
|------|-------|----------|--------|
| `parsers/exam-envie-bilingual-parser.py` | 2,444 | P1 | Tach theo format type |
| `api/routers/parsing.py` | 1,134 | P1 | Tach theo endpoint group |
| `services/omr/omr_answer_sheet_grader.py` | 903 | P2 | Tach grader logic |
| `api/routers/generation.py` | 759 | P2 | Tach theo feature |
| `api/routers/grading.py` | 675 | P2 | Tach helper functions |
| `services/math_renderer.py` | 621 | P3 | Giu nguyen (tight deps) |
| `database.py` | 593 | P3 | Tach models vs CRUD |

## Phases

- [~] [Phase 1](phase-01-split-envie-bilingual-parser-by-format-type.md): Tach envie parser (2,444→2,334+122) - **partial** (nested funcs)
- [x] [Phase 2](phase-02-split-parsing-router-by-endpoint-group.md): Tach parsing router (1,134→5 files) - **completed**
- [x] [Phase 3](phase-03-split-omr-answer-sheet-grader-logic.md): Tach OMR grader (903→4 files) - **completed**
- [x] [Phase 4](phase-04-split-remaining-large-files.md): Tach generation, grading, database - **completed**

## Nguyen tac

1. **YAGNI** - Chi tach khi can thiet
2. **KISS** - Giu logic don gian
3. **DRY** - Khong lap lai code
4. Moi module <200 dong
5. Test sau moi phase

## Uoc tinh

| Phase | Effort | Lines |
|-------|--------|-------|
| Phase 1 | 1.5h | ~2,444 |
| Phase 2 | 1h | ~1,134 |
| Phase 3 | 45m | ~903 |
| Phase 4 | 1h | ~2,648 |
| **Tong** | **4.25h** | **~7,129** |

## Ket qua mong doi

- Tat ca files <500 dong
- App van hoat dong binh thuong
- Code de bao tri hon
