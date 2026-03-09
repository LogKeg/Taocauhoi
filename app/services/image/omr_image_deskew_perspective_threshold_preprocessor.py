"""Image preprocessing functions for OMR (Optical Mark Recognition).

Provides perspective correction, deskewing, and binary thresholding
utilities used during answer sheet image ingestion.
"""

import cv2
import numpy as np


def _order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left có tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]  # bottom-right có tổng lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image, pts):
    """Thực hiện perspective transform với 4 điểm"""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def _deskew_image(image):
    """Tự động căn chỉnh ảnh bị nghiêng"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Tìm đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image, 0

    # Tính góc nghiêng trung bình
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Chỉ lấy các đường gần ngang (±15 độ)
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return image, 0

    # Lấy góc trung vị
    median_angle = np.median(angles)

    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, median_angle


def _find_document_contour(image):
    """Tìm contour của tài liệu (phiếu trả lời)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Làm dày cạnh
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def _preprocess_omr_image(image_bytes: bytes):
    """Tiền xử lý ảnh cho OMR với deskew và perspective correction"""
    # Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

    original = img.copy()  # noqa: F841 – kept for caller compatibility

    # Bước 1: Tìm và căn chỉnh tài liệu nếu bị méo
    # CHÚ Ý: Chỉ áp dụng perspective transform khi contour bao phủ gần như toàn bộ ảnh
    # để tránh cắt mất nội dung (ví dụ: hàng cuối của phiếu 50 câu)
    doc_contour = _find_document_contour(img)
    if doc_contour is not None:
        contour_area = cv2.contourArea(doc_contour)
        img_area = img.shape[0] * img.shape[1]
        # Chỉ transform nếu contour bao phủ > 80% diện tích ảnh
        if contour_area > 0.8 * img_area:
            img = _four_point_transform(img, doc_contour)

    # Bước 2: Deskew (căn chỉnh góc nghiêng)
    img, skew_angle = _deskew_image(img)  # noqa: F841

    # Bước 3: Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bước 4: Tăng contrast bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bước 5: Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 6: Adaptive threshold (tốt hơn cho điều kiện ánh sáng khác nhau)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Bước 7: Morphological operations để làm sạch
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return img, gray, binary
