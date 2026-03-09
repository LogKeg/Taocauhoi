"""Single OMR sheet grading logic.

Grade individual answer sheets using improved bubble detection algorithm.
"""

from typing import List

from app.core import ANSWER_TEMPLATES
from app.services.omr.omr_template_detector_and_student_info_ocr import _extract_student_info_ocr
from app.services.omr.omr_bubble_detection_grid_fill_analysis_and_grouping import (
    _detect_bubbles_grid_based,
    _group_bubbles_to_questions_improved,
    _detect_bubbles,
    _group_bubbles_to_questions,
    _analyze_bubble_fill_improved,
    _analyze_bubble_fill,
)
from app.services.image import _preprocess_omr_image


def _grade_single_sheet(image_bytes: bytes, answer_key: List[str], template_type: str = "IKSC_BENJAMIN", extract_info: bool = True):
    """Chấm một phiếu trả lời với thuật toán OMR cải tiến"""
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    option_labels = ["A", "B", "C", "D", "E"]

    # Trích xuất thông tin học sinh bằng OCR
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh với deskew và perspective correction
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # Thử phương pháp mới trước: phát hiện dựa trên grid
    rows, all_rects = _detect_bubbles_grid_based(gray, binary, template_type)

    questions = []
    use_new_method = False

    if rows and len(rows) >= 3:
        # Sử dụng phương pháp mới nếu phát hiện đủ hàng
        questions = _group_bubbles_to_questions_improved(rows, template_type)
        use_new_method = True

    # Fallback: Sử dụng phương pháp cũ nếu phương pháp mới không hiệu quả
    if len(questions) < num_questions * 0.3:
        bubbles = _detect_bubbles(binary, template_type)

        if len(bubbles) >= num_questions * 5 * 0.3:
            questions = _group_bubbles_to_questions(bubbles, template_type)
            use_new_method = False

    if len(questions) < num_questions * 0.3:
        return {
            "error": f"Không phát hiện đủ câu hỏi. Tìm thấy: {len(questions)}, cần: {num_questions}. "
                     f"Vui lòng đảm bảo ảnh rõ nét và phiếu được căn chỉnh đúng."
        }

    # Phân tích từng câu hỏi
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    # Tạo dict để tra cứu câu hỏi theo index (thay vì dùng vị trí trong mảng)
    questions_by_index = {q["index"]: q for q in questions}

    for q_idx in range(num_questions):
        q_num = q_idx + 1  # Số thứ tự câu hỏi (1-based)
        if q_num not in questions_by_index:
            # Không tìm thấy câu hỏi này
            student_answers.append(None)
            blank_count += 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        mean_vals = []  # Lưu mean value của từng option
        is_filled_list = []  # Lưu trạng thái is_filled của từng option
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            if use_new_method:
                # Phương pháp mới: bubble là dict
                result = _analyze_bubble_fill_improved(gray, bubble)
                if len(result) == 3:
                    is_filled, fill_ratio, mean_val = result
                else:
                    is_filled, fill_ratio = result
                    mean_val = 128.0
                mean_vals.append(mean_val)
                is_filled_list.append(is_filled)
            else:
                # Phương pháp cũ: bubble là tuple với contour
                is_filled, fill_ratio = _analyze_bubble_fill(binary, bubble[4])
                mean_vals.append(128.0)
                is_filled_list.append(is_filled)

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        # Ưu tiên option có is_filled=True và mean thấp nhất (được tô đậm nhất)
        filled_options = [(i, mean_vals[i]) for i in range(len(is_filled_list)) if is_filled_list[i]]
        if filled_options and selected_option is None:
            # Có option được đánh dấu filled nhưng chưa được chọn
            # Chọn option có mean thấp nhất (tối nhất = được tô)
            darkest_filled = min(filled_options, key=lambda x: x[1])
            selected_option = option_labels[darkest_filled[0]]
        elif len(filled_options) == 1:
            # Chỉ có 1 option filled -> chọn option đó
            selected_option = option_labels[filled_options[0][0]]

        # Phát hiện vùng tối bất thường (bóng/rìa ảnh)
        # Nếu nhiều options có mean rất thấp (<50), đây có thể là vùng tối
        dark_region_count = sum(1 for m in mean_vals if m < 50)
        is_dark_region = dark_region_count >= 3

        if is_dark_region:
            # Vùng tối: chọn option có mean CAO nhất (sáng nhất = không bị bóng che)
            # vì các vùng tối là do bóng, không phải do được tô
            max_mean = max(mean_vals)
            min_mean = min(mean_vals)

            # Chỉ chọn nếu có 1 option sáng hơn hẳn (chênh lệch > 50)
            if max_mean - min_mean > 50:
                bright_option_idx = mean_vals.index(max_mean)
                # Kiểm tra option sáng này có được tô không
                if fill_ratios[bright_option_idx] > 0.1:
                    selected_option = option_labels[bright_option_idx]
                else:
                    # Option sáng nhưng không được tô -> có thể là blank hoặc tô option khác
                    # Trong vùng tối, tìm option có score cao nhất trong các option không quá tối
                    valid_options = [(i, fill_ratios[i]) for i in range(len(mean_vals))
                                    if mean_vals[i] > 100 or fill_ratios[i] > 0.3]
                    if valid_options:
                        best_idx = max(valid_options, key=lambda x: x[1])[0]
                        selected_option = option_labels[best_idx]
        else:
            # Vùng bình thường: sử dụng logic cũ với cải tiến

            # Ngưỡng động: nếu max_fill > 0.25 và vượt trội hơn các option khác
            if selected_option is None and max_fill > 0.25:
                # Kiểm tra xem có một option nào vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)
                if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Option đầu lớn hơn 30% so với option thứ 2
                    selected_option = option_labels[fill_ratios.index(max_fill)]

            # Kiểm tra nếu có nhiều đáp án được chọn
            # Sử dụng ngưỡng động dựa trên max_fill
            if max_fill > 0.5:
                # Nếu có option được tô đậm, các option khác cần đạt ít nhất 60% của max
                filled_threshold = max_fill * 0.6
            else:
                filled_threshold = 0.35

            filled_count = sum(1 for r in fill_ratios if r > filled_threshold)
            if filled_count > 1:
                # Kiểm tra xem có 1 option rõ ràng vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)

                # Tính chênh lệch mean giữa option cao nhất và thấp nhất
                max_score_idx = fill_ratios.index(sorted_ratios[0])
                max_score_mean = mean_vals[max_score_idx]

                # Nếu option có score cao nhất cũng có mean thấp nhất -> đây là bubble được tô
                if max_score_mean == min(mean_vals) or sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Có 1 option vượt trội rõ ràng
                    selected_option = option_labels[max_score_idx]
                else:
                    # Kiểm tra thêm: nếu max > 0.5 và second < 0.4, vẫn chọn max
                    if sorted_ratios[0] > 0.5 and sorted_ratios[1] < 0.4:
                        selected_option = option_labels[fill_ratios.index(sorted_ratios[0])]
                    else:
                        # Phân tích thêm bằng mean value
                        # Option được tô sẽ có mean thấp hơn các option không tô
                        mean_diff = max(mean_vals) - min(mean_vals)
                        if mean_diff > 30:
                            # Có sự khác biệt rõ ràng về độ sáng
                            darkest_idx = mean_vals.index(min(mean_vals))
                            selected_option = option_labels[darkest_idx]
                        else:
                            selected_option = "MULTI"  # Đánh dấu chọn nhiều đáp án

        student_answers.append(selected_option)

        # So sánh với đáp án
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if selected_option is None:
            status = "blank"
            blank_count += 1
        elif selected_option == "MULTI":
            status = "invalid"
            wrong_count += 1
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            status = "correct"
            correct_count += 1
        else:
            status = "wrong"
            wrong_count += 1

        details.append({
            "q": q_idx + 1,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # Tính điểm
    if scoring.get("type") == "tiered":
        # Tính điểm theo phần (tiered scoring) - dùng cho IKSC
        score = scoring.get("base", 0)
        tiers = scoring.get("tiers", [])

        for detail in details:
            q_num = detail["q"]
            status = detail["status"]

            # Tìm tier phù hợp cho câu hỏi này
            tier_points = {"correct": 1, "wrong": 0}  # Default
            for tier in tiers:
                if tier["start"] <= q_num <= tier["end"]:
                    tier_points = tier
                    break

            if status == "correct":
                score += tier_points.get("correct", 1)
            elif status in ["wrong", "invalid"]:
                score += tier_points.get("wrong", 0)
            # blank: không cộng/trừ điểm

    elif scoring.get("type") == "best_of":
        # Tính điểm chỉ lấy N câu tốt nhất - dùng cho IKLC Benjamin-Student
        # Công thức: base + (correct * points) + (wrong * penalty)
        # Chỉ tính count_best câu có điểm cao nhất
        count_best = scoring.get("count_best", 40)
        correct_pts = scoring.get("correct", 1)
        wrong_pts = scoring.get("wrong", -0.5)

        # Tính điểm từng câu
        question_scores = []
        for detail in details:
            status = detail["status"]
            if status == "correct":
                question_scores.append(correct_pts)
            elif status in ["wrong", "invalid"]:
                question_scores.append(wrong_pts)
            else:  # blank
                question_scores.append(0)

        # Sắp xếp giảm dần và lấy N câu tốt nhất
        question_scores.sort(reverse=True)
        best_scores = question_scores[:count_best]

        score = scoring.get("base", 0) + sum(best_scores)

    else:
        # Tính điểm đơn giản (flat scoring) - dùng cho IKLC Pre-Ecolier, Ecolier và các kỳ thi khác
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
        "detection_method": "grid_based" if use_new_method else "contour_based",
        "questions_detected": len(questions)
    }
