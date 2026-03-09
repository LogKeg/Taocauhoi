"""
Helper functions for grading endpoints.

Shared utilities for OMR and handwritten answer sheet grading.
"""
from typing import List


def _grade_handwritten_sheet(
    image_bytes: bytes,
    answer_key: List[str],
    num_questions: int = 30,
    valid_answers: List[str] = None
) -> dict:
    """
    Grade a handwritten answer sheet using OCR.

    Note: This is a placeholder implementation. Full OCR-based grading
    requires external services (Google Vision API, Tesseract, etc.)
    """
    import cv2
    import numpy as np

    if valid_answers is None:
        valid_answers = ['A', 'B', 'C', 'D', 'E']

    try:
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Không thể đọc ảnh"}

        # Placeholder: Return empty result - real implementation needs OCR
        return {
            "detected_answers": [],
            "correct": 0,
            "wrong": 0,
            "blank": num_questions,
            "total": num_questions,
            "details": [],
            "note": "OCR handwritten grading not fully implemented. Please use OMR grading for multiple choice."
        }

    except Exception as e:
        return {"error": f"Lỗi xử lý ảnh: {str(e)}"}


def _get_grading_functions():
    """Lazy import of grading functions from OMR modules."""
    from app.services.omr import (
        _grade_single_sheet,
        _grade_mixed_format_sheet,
        _detect_bubbles_grid_based,
        _group_bubbles_to_questions_improved,
        _extract_student_info_ocr,
        _detect_template_from_image,
        _parse_answer_key_for_template,
    )
    from app.services.image import _preprocess_omr_image
    return {
        'grade_single_sheet': _grade_single_sheet,
        'grade_mixed_format_sheet': _grade_mixed_format_sheet,
        'preprocess_omr_image': _preprocess_omr_image,
        'detect_bubbles_grid_based': _detect_bubbles_grid_based,
        'group_bubbles_to_questions_improved': _group_bubbles_to_questions_improved,
        'extract_student_info_ocr': _extract_student_info_ocr,
        'detect_template_from_image': _detect_template_from_image,
        'parse_answer_key_for_template': _parse_answer_key_for_template,
        'grade_handwritten_sheet': _grade_handwritten_sheet,
    }
