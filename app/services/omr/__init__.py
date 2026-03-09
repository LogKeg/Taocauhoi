"""OMR services package: template detection, student info extraction, bubble detection, and grading."""

from app.services.omr.omr_template_detector_and_student_info_ocr import (
    _detect_template_from_image,
    _extract_student_info_ocr,
)
from app.services.omr.omr_bubble_detection_grid_fill_analysis_and_grouping import (
    _find_answer_grid_region,
    _detect_all_rectangles,
    _cluster_by_rows,
    _detect_bubbles_grid_based,
    _analyze_bubble_fill_improved,
    _group_bubbles_to_questions_improved,
    _detect_bubbles,
    _analyze_bubble_fill,
    _group_bubbles_to_questions,
    _detect_seamo_bubbles_fixed_grid,
    _detect_seamo_grid_dynamic,
)
from app.services.omr.omr_answer_sheet_grader import (
    _grade_single_sheet,
    _grade_mixed_format_sheet,
    _extract_answers_from_text,
    _parse_answer_key_for_template,
)

__all__ = [
    "_detect_template_from_image",
    "_extract_student_info_ocr",
    "_find_answer_grid_region",
    "_detect_all_rectangles",
    "_cluster_by_rows",
    "_detect_bubbles_grid_based",
    "_analyze_bubble_fill_improved",
    "_group_bubbles_to_questions_improved",
    "_detect_bubbles",
    "_analyze_bubble_fill",
    "_group_bubbles_to_questions",
    "_detect_seamo_bubbles_fixed_grid",
    "_detect_seamo_grid_dynamic",
    "_grade_single_sheet",
    "_grade_mixed_format_sheet",
    "_extract_answers_from_text",
    "_parse_answer_key_for_template",
]
