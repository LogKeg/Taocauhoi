"""Mixed format OMR sheet grading (MCQ + fill-in answers).

Grade answer sheets with both multiple choice and fill-in sections.
Used for SEAMO Math: 20 MCQ + 5 fill-in questions.
"""

from typing import List

from app.core import ANSWER_TEMPLATES
from app.services.omr.omr_template_detector_and_student_info_ocr import _extract_student_info_ocr
from app.services.omr.omr_bubble_detection_grid_fill_analysis_and_grouping import (
    _detect_bubbles_grid_based,
    _group_bubbles_to_questions_improved,
    _analyze_bubble_fill_improved,
    _detect_seamo_bubbles_fixed_grid,
)
from app.services.image import _preprocess_omr_image


def _get_easyocr_reader():
    """Create and return an EasyOCR reader instance (English only, CPU mode)."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import easyocr
    return easyocr.Reader(['en'], gpu=False, verbose=False)


def _grade_mixed_format_sheet(
    image_bytes: bytes,
    answer_key: List[str],
    template_type: str,
    extract_info: bool = True
):
    """Chấm phiếu trả lời có format hỗn hợp (trắc nghiệm + điền đáp án)

    Dùng cho SEAMO Math: 20 câu trắc nghiệm + 5 câu điền đáp án
    """
    import cv2
    import numpy as np

    # Import from single sheet grader module directly to avoid circular dependency
    import importlib.util
    import os
    _current_dir = os.path.dirname(__file__)
    _single_sheet_path = os.path.join(_current_dir, "omr-single-sheet-grader.py")
    _single_sheet_spec = importlib.util.spec_from_file_location("omr_single_sheet_grader", _single_sheet_path)
    _single_sheet_module = importlib.util.module_from_spec(_single_sheet_spec)
    _single_sheet_spec.loader.exec_module(_single_sheet_module)
    _grade_single_sheet = _single_sheet_module._grade_single_sheet

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    mixed_format = template.get("mixed_format", None)

    if not mixed_format:
        # Không có mixed format, dùng hàm chấm thông thường
        return _grade_single_sheet(image_bytes, answer_key, template_type, extract_info)

    mcq_count = mixed_format.get("mcq", 20)
    fill_in_count = mixed_format.get("fill_in", 5)

    # Trích xuất thông tin học sinh
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # ========== PHẦN 1: Chấm 20 câu trắc nghiệm bằng OMR ==========
    # Kiểm tra nếu là SEAMO, sử dụng fixed grid detection
    is_seamo = "SEAMO" in template_type.upper()

    mcq_questions = []
    if is_seamo:
        # SEAMO có layout cố định, dùng fixed grid
        mcq_questions = _detect_seamo_bubbles_fixed_grid(gray)
    else:
        # Các template khác dùng dynamic detection
        mcq_template_type = template_type
        rows, all_rects = _detect_bubbles_grid_based(gray, binary, mcq_template_type)

        if rows and len(rows) >= 2:
            mcq_questions = _group_bubbles_to_questions_improved(rows, mcq_template_type)
            # Chỉ lấy các câu từ 1 đến mcq_count
            mcq_questions = [q for q in mcq_questions if q["index"] <= mcq_count]

    # Chấm phần trắc nghiệm
    option_labels = ["A", "B", "C", "D", "E"]
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    questions_by_index = {q["index"]: q for q in mcq_questions}

    for q_idx in range(mcq_count):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if q_num not in questions_by_index:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "mcq",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            result_fill = _analyze_bubble_fill_improved(gray, bubble)
            if len(result_fill) == 3:
                is_filled, fill_ratio, mean_val = result_fill
            else:
                is_filled, fill_ratio = result_fill

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        if selected_option is None and max_fill > 0.25:
            sorted_ratios = sorted(fill_ratios, reverse=True)
            if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                selected_option = option_labels[fill_ratios.index(max_fill)]

        if selected_option is None:
            student_answers.append(None)
            blank_count += 1
            status = "blank"
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            student_answers.append(selected_option)
            correct_count += 1
            status = "correct"
        else:
            student_answers.append(selected_option)
            wrong_count += 1
            status = "wrong"

        details.append({
            "q": q_num,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "type": "mcq",
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # ========== PHẦN 2: Chấm 5 câu điền đáp án bằng OCR ==========
    try:
        reader = _get_easyocr_reader()
    except Exception as e:
        # Nếu không load được OCR, đánh dấu các câu fill-in là not_found
        for q_idx in range(mcq_count, num_questions):
            q_num = q_idx + 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0,
                "error": f"OCR error: {str(e)}"
            })

        # Tính điểm và trả về
        score = (
            scoring.get("base", 0) +
            correct_count * scoring.get("correct", 1) +
            wrong_count * scoring.get("wrong", 0) +
            blank_count * scoring.get("blank", 0)
        )

        return {
            "answers": student_answers,
            "score": round(score, 2),
            "correct": correct_count,
            "wrong": wrong_count,
            "blank": blank_count,
            "total": num_questions,
            "details": details,
            "student_info": student_info,
            "format": "mixed",
            "mcq_count": mcq_count,
            "fill_in_count": fill_in_count
        }

    # Tìm vùng chứa đáp án điền (thường ở phía dưới phiếu)
    # Phát hiện các ô điền đáp án
    height, width = gray.shape[:2]

    # Giả sử phần điền đáp án nằm ở 1/3 dưới của ảnh
    fill_in_region = gray[int(height * 0.6):, :]

    # Sử dụng OCR để đọc toàn bộ vùng
    try:
        ocr_results = reader.readtext(fill_in_region, detail=1, paragraph=False)
    except Exception:
        ocr_results = []

    # Tìm các đáp án số/chữ
    recognized_fill_ins = []
    for (bbox, text, confidence) in ocr_results:
        text = text.strip()
        # Lọc các text có vẻ là đáp án (số hoặc chữ ngắn)
        if text and len(text) <= 10:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            recognized_fill_ins.append({
                'text': text,
                'confidence': confidence,
                'cx': cx,
                'cy': cy + int(height * 0.6)  # Offset lại vị trí
            })

    # Sắp xếp theo vị trí (trái sang phải, trên xuống dưới)
    recognized_fill_ins.sort(key=lambda r: (r['cy'], r['cx']))

    # Gán đáp án cho các câu fill-in
    for i, q_idx in enumerate(range(mcq_count, num_questions)):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if i < len(recognized_fill_ins):
            recognized = recognized_fill_ins[i]['text']
            confidence = recognized_fill_ins[i]['confidence']

            # So sánh đáp án (có thể là số hoặc chữ)
            if correct_answer:
                # Chuẩn hóa để so sánh
                student_norm = recognized.upper().strip()
                correct_norm = str(correct_answer).upper().strip()

                if student_norm == correct_norm:
                    status = "correct"
                    correct_count += 1
                else:
                    status = "wrong"
                    wrong_count += 1
            else:
                status = "unknown"

            student_answers.append(recognized)
            details.append({
                "q": q_num,
                "student": recognized,
                "correct": correct_answer,
                "status": status,
                "type": "fill_in",
                "confidence": round(confidence, 3)
            })
        else:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0
            })

    # Tính điểm
    score = (
        scoring.get("base", 0) +
        correct_count * scoring.get("correct", 1) +
        wrong_count * scoring.get("wrong", 0) +
        blank_count * scoring.get("blank", 0)
    )

    return {
        "answers": student_answers,
        "score": round(score, 2),
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details,
        "student_info": student_info,
        "format": "mixed",
        "mcq_count": mcq_count,
        "fill_in_count": fill_in_count
    }
