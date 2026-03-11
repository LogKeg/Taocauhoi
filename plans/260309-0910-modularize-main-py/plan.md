---
title: Modular hóa main.py
status: completed
priority: P1
effort: medium
branch: main
tags: [refactor, modularization]
created: 2026-03-09
completed: 2026-03-09
---

# Modular hóa main.py

## Tổng quan

Tách `app/main.py` (5,384 dòng) thành các module nhỏ hơn 200 dòng.

## Phân tích hiện trạng

| Nhóm chức năng | Dòng | Hàm chính |
|----------------|------|-----------|
| **Exam Parsers** | 280-3073 (~2,800) | `_parse_math_exam_questions`, `_parse_english_exam_questions`, `_parse_envie_questions` |
| **English Helpers** | 3074-3308 (~235) | `is_matching_section`, `extract_passage_questions`, `extract_cloze_questions` |
| **OMR/Grading** | 3309-5384 (~2,075) | `_detect_template_from_image`, `_grade_single_sheet`, `_grade_mixed_format_sheet` |
| **Image Processing** | 3579-3739 (~160) | `_order_points`, `_deskew_image`, `_preprocess_omr_image` |
| **Routes & Utils** | 1-279 (~280) | `index`, `upload_sample`, `_force_variation` |

## Cấu trúc mới

```
app/
├── main.py                          # ~150 dòng (routes + app setup)
├── parsers/
│   ├── __init__.py
│   ├── docx.py                      # (đã có)
│   ├── exam-math-parser.py          # NEW: parse đề Toán
│   ├── exam-english-parser.py       # NEW: parse đề Tiếng Anh
│   └── exam-envie-parser.py         # NEW: parse đề song ngữ EnVie
├── services/
│   ├── omr/
│   │   ├── __init__.py              # NEW
│   │   ├── template-detector.py     # NEW: detect template type
│   │   ├── bubble-detector.py       # NEW: detect bubbles
│   │   ├── grader.py                # NEW: grade sheets
│   │   └── answer-key-parser.py     # NEW: parse answer keys
│   └── image/
│       ├── __init__.py              # NEW
│       ├── preprocessing.py         # NEW: deskew, transform
│       └── ocr-extractor.py         # NEW: OCR student info
└── utils/
    └── question-rewriter.py         # NEW: _force_variation, _rewrite_mcq_block
```

## Phases

- [Phase 1](phase-01-extract-exam-parsers.md): Tách exam parsers (~2,800 dòng)
- [Phase 2](phase-02-extract-omr-services.md): Tách OMR services (~2,075 dòng)
- [Phase 3](phase-03-extract-image-utils.md): Tách image processing (~160 dòng)
- [Phase 4](phase-04-cleanup-main.md): Dọn dẹp main.py

## Nguyên tắc

1. **Không thay đổi logic** - chỉ di chuyển code
2. **Giữ nguyên tên hàm** - để không cần sửa nơi gọi
3. **Test sau mỗi phase** - đảm bảo app vẫn chạy
4. **Import từ module mới** - trong main.py

## Ước tính

| Phase | Effort | Dòng tách |
|-------|--------|-----------|
| Phase 1 | 1h | ~2,800 |
| Phase 2 | 1.5h | ~2,075 |
| Phase 3 | 30m | ~160 |
| Phase 4 | 30m | cleanup |
| **Tổng** | **3.5h** | **~5,035** |

## Kết quả mong đợi

- `main.py`: ~150 dòng (chỉ routes + setup)
- Mỗi module: <200 dòng
- App vẫn hoạt động bình thường
