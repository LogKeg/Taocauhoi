"""OMR bubble detection functions: grid-based detection, fill analysis, and grouping."""

from collections import Counter

import cv2
import numpy as np

from app.core.constants import ANSWER_TEMPLATES


def _find_answer_grid_region(gray_image, binary_image):
    """Tìm vùng chứa lưới đáp án trong ảnh dựa trên cấu trúc grid"""
    height, width = gray_image.shape[:2]

    # Tìm các đường ngang và dọc
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

    # Kết hợp các đường
    grid = cv2.add(horizontal_lines, vertical_lines)

    # Tìm contours của grid
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tìm bounding box lớn nhất
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Mở rộng một chút
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)

        return (x, y, w, h)

    # Fallback: Giả sử vùng đáp án nằm ở phần dưới 2/3 của ảnh
    return (0, int(height * 0.25), width, int(height * 0.75))


def _detect_all_rectangles(binary_image, min_size=15, max_size=80):
    """Phát hiện tất cả hình chữ nhật (ô đáp án) trong ảnh"""
    # Tìm contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for i, contour in enumerate(contours):
        # Lấy bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Lọc theo kích thước
        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue

        # Kiểm tra tỉ lệ gần vuông
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.6 <= aspect_ratio <= 1.4):
            continue

        # Kiểm tra diện tích contour so với bounding box (phải gần vuông/chữ nhật)
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        if bbox_area > 0:
            extent = contour_area / bbox_area
            if extent < 0.5:  # Bỏ qua các hình không đầy đặn
                continue

        rectangles.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': x + w // 2, 'cy': y + h // 2,
            'contour': contour
        })

    return rectangles


def _cluster_by_rows(rectangles, tolerance=15):
    """Nhóm các hình chữ nhật theo hàng dựa trên tọa độ y"""
    if not rectangles:
        return []

    # Sắp xếp theo y
    sorted_rects = sorted(rectangles, key=lambda r: r['cy'])

    rows = []
    current_row = [sorted_rects[0]]

    for rect in sorted_rects[1:]:
        # Nếu tọa độ y gần với hàng hiện tại
        if abs(rect['cy'] - current_row[0]['cy']) <= tolerance:
            current_row.append(rect)
        else:
            # Sắp xếp hàng theo x và thêm vào danh sách
            rows.append(sorted(current_row, key=lambda r: r['cx']))
            current_row = [rect]

    # Thêm hàng cuối cùng
    rows.append(sorted(current_row, key=lambda r: r['cx']))

    return rows


def _detect_bubbles_grid_based(gray_image, binary_image, template_type: str):
    """Phát hiện bubble dựa trên cấu trúc grid của phiếu"""
    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]  # A-E = 5
    questions_per_row = template["questions_per_row"]  # 4

    height, width = gray_image.shape[:2]

    # Phát hiện tất cả hình chữ nhật
    rectangles = _detect_all_rectangles(binary_image)

    if len(rectangles) < num_questions * num_options * 0.3:
        # Thử với ngưỡng khác
        _, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rectangles = _detect_all_rectangles(binary_otsu)

    if not rectangles:
        return [], []

    # Phân tích phân bố kích thước để tìm kích thước phổ biến nhất (ô đáp án)
    widths = [r['w'] for r in rectangles]
    heights = [r['h'] for r in rectangles]

    # Tìm mode (giá trị phổ biến nhất) cho width và height
    width_counts = Counter([int(w/5)*5 for w in widths])  # Bin by 5 (rộng hơn)
    height_counts = Counter([int(h/5)*5 for h in heights])

    # Lấy top kích thước phổ biến nhất
    common_widths = [w for w, _ in width_counts.most_common(3)]
    common_heights = [h for h, _ in height_counts.most_common(3)]

    # Lọc các ô có kích thước nằm trong nhóm phổ biến
    target_width = common_widths[0] if common_widths else np.median(widths)
    target_height = common_heights[0] if common_heights else np.median(heights)

    # Lọc với tolerance ±40% (rộng hơn để bắt các ô hơi khác kích thước)
    filtered_rects = [
        r for r in rectangles
        if 0.6 * target_width <= r['w'] <= 1.4 * target_width
        and 0.6 * target_height <= r['h'] <= 1.4 * target_height
    ]

    if not filtered_rects:
        # Fallback: dùng median
        avg_width = np.median(widths)
        avg_height = np.median(heights)
        filtered_rects = [
            r for r in rectangles
            if 0.5 * avg_width <= r['w'] <= 1.5 * avg_width
            and 0.5 * avg_height <= r['h'] <= 1.5 * avg_height
        ]

    # Nhóm theo hàng
    avg_height = np.median([r['h'] for r in filtered_rects]) if filtered_rects else 30
    rows = _cluster_by_rows(filtered_rects, tolerance=int(avg_height * 0.6))

    # Phân loại các hàng
    expected_per_row = questions_per_row * num_options
    valid_rows = []
    partial_rows = []

    for row in rows:
        if len(row) >= expected_per_row * 0.8:
            # Hàng đầy đủ (4 câu/hàng)
            valid_rows.append(row)
        elif len(row) >= num_options:
            # Hàng không đầy đủ (1-3 câu) - có thể là hàng cuối
            partial_rows.append(row)

    # Nếu không đủ hàng valid, thử relax điều kiện
    if len(valid_rows) < num_questions / questions_per_row * 0.5:
        valid_rows = [row for row in rows if len(row) >= num_options]
        partial_rows = []

    # Kết hợp valid_rows và partial_rows, sắp xếp theo y
    all_rows = valid_rows + partial_rows
    all_rows.sort(key=lambda row: row[0]['cy'] if row else 0)

    return all_rows, filtered_rects


def _analyze_bubble_fill_improved(gray_image, rect, threshold=0.4):
    """Phân tích bubble fill với phương pháp cải tiến

    Trả về tuple (is_filled, score, mean_val) để hỗ trợ so sánh tương đối
    """
    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']

    # Lấy vùng bubble với margin nhỏ bên trong
    margin = max(2, int(min(w, h) * 0.15))
    roi = gray_image[y+margin:y+h-margin, x+margin:x+w-margin]

    if roi.size == 0:
        return False, 0.0, 255.0

    # Tính các chỉ số
    mean_val = np.mean(roi)
    min_val = np.min(roi)

    # Phương pháp chính: Đếm pixel tối
    # Bubble được tô bằng bút chì 2B sẽ có nhiều pixel rất tối
    dark_pixels = np.sum(roi < 100) / roi.size  # Pixel tối (< 100)
    very_dark_pixels = np.sum(roi < 60) / roi.size  # Pixel rất tối (< 60)

    # Phương pháp phụ: Binary threshold với Otsu (cho ảnh scan tốt)
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = np.sum(binary_roi > 0) / binary_roi.size

    # Tính score dựa trên pixel tối
    # Ưu tiên pixel rất tối (có trọng số cao hơn)
    darkness_score = dark_pixels * 0.6 + very_dark_pixels * 1.5

    # Kiểm tra có được tô không
    # Tiêu chí: có nhiều pixel tối HOẶC mean thấp
    if very_dark_pixels > 0.05 or dark_pixels > 0.15:
        # Có vùng được tô rõ ràng
        is_filled = True
        score = darkness_score
    elif mean_val < 120 and dark_pixels > 0.05:
        # Mean thấp và có một ít pixel tối
        is_filled = True
        score = darkness_score + (120 - mean_val) / 200
    elif fill_ratio > threshold and mean_val < 150:
        # Fallback: Otsu + mean thấp
        is_filled = True
        score = fill_ratio * 0.5  # Giảm trọng số của Otsu
    else:
        is_filled = False
        score = darkness_score

    return is_filled, score, mean_val


def _group_bubbles_to_questions_improved(rows, template_type: str):
    """Nhóm bubble thành câu hỏi dựa trên vị trí trong grid"""
    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]
    layout = template.get("layout", "row")  # "row" hoặc "column"

    questions = []

    # Lọc các hàng có số bubble hợp lý
    expected_per_row = questions_per_row * num_options
    valid_rows = []
    partial_rows = []  # Hàng có ít bubble hơn (có thể là hàng cuối)

    for row in rows:
        # Loại bỏ các bubble trùng lặp (gap < 5 pixels)
        filtered_row = [row[0]] if row else []
        for i in range(1, len(row)):
            gap = row[i]['cx'] - filtered_row[-1]['cx']
            if gap > 10:  # Chỉ thêm nếu cách bubble trước > 10 pixels
                filtered_row.append(row[i])

        # Phân loại hàng theo số bubble
        if expected_per_row * 0.8 <= len(filtered_row) <= expected_per_row * 1.3:
            # Hàng đầy đủ (4 câu/hàng)
            valid_rows.append(filtered_row)
        elif num_options <= len(filtered_row) < expected_per_row * 0.8:
            # Hàng không đầy đủ (có thể là hàng cuối với 1-3 câu)
            partial_rows.append(filtered_row)

    # Loại bỏ các hàng partial ở đầu (trước hàng valid đầu tiên)
    # Đây thường là header hoặc phần thông tin học sinh
    if valid_rows and partial_rows:
        first_valid_y = valid_rows[0][0]['cy'] if valid_rows[0] else float('inf')
        # Chỉ giữ lại partial_rows sau hàng valid cuối cùng (hàng cuối của grid)
        last_valid_y = valid_rows[-1][0]['cy'] if valid_rows[-1] else 0
        partial_rows = [row for row in partial_rows if row and row[0]['cy'] > last_valid_y]

    # Tách mỗi hàng thành các nhóm câu hỏi (mỗi nhóm = 5 bubbles cho 1 câu)
    all_question_groups = []  # List of lists: mỗi hàng chứa các câu hỏi

    for row in valid_rows:
        if len(row) < num_options:
            continue

        row_questions = []

        # Tính khoảng cách giữa các bubble liên tiếp
        gaps = []
        for i in range(1, len(row)):
            gap = row[i]['cx'] - row[i-1]['cx']
            gaps.append((i, gap))

        if not gaps:
            continue

        # Phân tích gaps để tìm điểm phân tách câu hỏi
        gap_values = [g[1] for g in gaps]
        median_gap = np.median(gap_values)
        max_gap = max(gap_values)

        # Nếu max_gap > 1.5 * median_gap, đó là điểm phân tách câu hỏi
        if max_gap > median_gap * 1.4:
            # Có điểm phân tách rõ ràng
            large_gap_threshold = median_gap * 1.3

            current_question_bubbles = [row[0]]
            for i in range(1, len(row)):
                gap = row[i]['cx'] - row[i-1]['cx']

                # Cho phép split khi có large gap và có ít nhất 4 bubbles (thiếu 1 do không phát hiện được)
                if gap > large_gap_threshold and len(current_question_bubbles) >= num_options - 1:
                    row_questions.append(current_question_bubbles[:num_options])
                    current_question_bubbles = [row[i]]
                else:
                    current_question_bubbles.append(row[i])

            # Thêm câu hỏi cuối cùng trong hàng
            # Cho phép thiếu 1 bubble (4/5) vì có thể bubble không được phát hiện
            if len(current_question_bubbles) >= num_options - 1:
                row_questions.append(current_question_bubbles[:num_options])
        else:
            # Không có điểm phân tách rõ ràng, chia đều theo số options
            for i in range(0, len(row), num_options):
                question_bubbles = row[i:i+num_options]
                if len(question_bubbles) == num_options:
                    row_questions.append(question_bubbles)

        if row_questions:
            all_question_groups.append(row_questions)

    # Xử lý các hàng partial (hàng cuối có ít câu hơn)
    for row in partial_rows:
        if len(row) < num_options:
            continue

        row_questions = []
        # Chia hàng thành các câu hỏi
        for i in range(0, len(row), num_options):
            question_bubbles = row[i:i+num_options]
            if len(question_bubbles) == num_options:
                row_questions.append(question_bubbles)

        if row_questions:
            all_question_groups.append(row_questions)

    # Đánh số câu hỏi dựa trên layout
    if layout == "column":
        # Layout theo cột: cột 1 có câu 1,5,9..., cột 2 có câu 2,6,10...
        # Mỗi hàng có 4 câu hỏi (4 cột)
        # Câu hỏi thứ i ở cột (i-1) % 4, hàng (i-1) // 4
        num_cols = questions_per_row
        num_rows = len(all_question_groups)

        for row_idx, row_questions in enumerate(all_question_groups):
            for col_idx, bubbles in enumerate(row_questions):
                if col_idx >= num_cols:
                    break
                # Tính số thứ tự câu hỏi: hàng * 4 + cột + 1
                # Ví dụ: hàng 0, cột 0 = câu 1; hàng 0, cột 1 = câu 2
                # hàng 1, cột 0 = câu 5; hàng 1, cột 1 = câu 6
                q_num = row_idx * num_cols + col_idx + 1
                if q_num <= num_questions:
                    questions.append({
                        "index": q_num,
                        "bubbles": bubbles
                    })
    else:
        # Layout theo hàng (mặc định): đọc từ trái sang phải, trên xuống dưới
        question_idx = 0
        for row_questions in all_question_groups:
            for bubbles in row_questions:
                if question_idx >= num_questions:
                    break
                questions.append({
                    "index": question_idx + 1,
                    "bubbles": bubbles
                })
                question_idx += 1

    # Sắp xếp theo index
    questions.sort(key=lambda x: x["index"])

    return questions


def _detect_bubbles(binary_image, template_type: str = "IKSC_BENJAMIN"):
    """Phát hiện các bubble trong ảnh (legacy function for compatibility)"""
    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]

    # Tìm contours
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc các contour có dạng bubble (gần vuông/tròn)
    bubbles = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Bubble phải gần vuông và có kích thước hợp lý
        if 0.6 <= aspect_ratio <= 1.4 and 12 <= w <= 80 and 12 <= h <= 80:
            bubbles.append((x, y, w, h, contour))

    return bubbles


def _analyze_bubble_fill(binary_image, bubble_contour, threshold: float = 0.35):
    """Phân tích xem bubble có được tô hay không (legacy)"""
    # Tạo mask cho bubble
    mask = np.zeros(binary_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [bubble_contour], -1, 255, -1)

    # Đếm pixel trong mask
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return False, 0

    # Đếm pixel được tô (giao của mask và binary image)
    filled = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    filled_pixels = cv2.countNonZero(filled)

    fill_ratio = filled_pixels / total_pixels

    return fill_ratio > threshold, fill_ratio


def _group_bubbles_to_questions(bubbles, template_type: str = "IKSC_BENJAMIN"):
    """Nhóm các bubble thành câu hỏi (legacy)"""
    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]

    if not bubbles:
        return []

    # Sắp xếp bubble theo y (hàng) rồi theo x (cột)
    sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

    # Nhóm theo hàng dựa trên tọa độ y
    rows = []
    current_row = [sorted_bubbles[0]]
    y_threshold = 30  # Ngưỡng để phân biệt hàng

    for bubble in sorted_bubbles[1:]:
        if abs(bubble[1] - current_row[0][1]) < y_threshold:
            current_row.append(bubble)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [bubble]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Mỗi hàng chứa questions_per_row câu hỏi × num_options lựa chọn
    expected_bubbles_per_row = questions_per_row * num_options

    questions = []
    question_idx = 0

    for row in rows:
        # Chia hàng thành các nhóm 5 bubble (A-E) cho mỗi câu hỏi
        for i in range(0, len(row), num_options):
            if question_idx >= num_questions:
                break
            question_bubbles = row[i:i+num_options]
            if len(question_bubbles) == num_options:
                questions.append({
                    "index": question_idx + 1,
                    "bubbles": question_bubbles
                })
                question_idx += 1

    return questions


def _detect_seamo_bubbles_fixed_grid(gray_image):
    """Phát hiện bubbles trong phiếu SEAMO với dynamic grid detection

    Sử dụng kết hợp:
    1. Phát hiện đường kẻ ngang để tìm vị trí các hàng
    2. Phát hiện đường kẻ dọc để tìm vị trí các cột
    3. Fallback về tọa độ cố định nếu không detect được
    """
    h, w = gray_image.shape[:2]

    # Detect loại ảnh: PDF vector vs scan image
    # - Scan 300 DPI A4: ~2480 x 3508 (width > 2000)
    # - PDF vector render: ~1191 x 1685 (width ~1200)
    is_high_res_scan = w > 2000

    if not is_high_res_scan:
        # PDF vector render - Thử dynamic detection trước
        grid_info = _detect_seamo_grid_dynamic(gray_image)

        if grid_info is not None:
            grid_start_x = grid_info['start_x']
            grid_start_y = grid_info['start_y']
            option_spacing = grid_info['col_spacing']
            row_spacing = grid_info['row_spacing']
            bubble_w = grid_info['bubble_w']
            bubble_h = grid_info['bubble_h']
        else:
            # Fallback cho PDF vector (từ 72 DPI gốc)
            expected_w, expected_h = 1191, 1685
            scale_x = w / expected_w
            scale_y = h / expected_h
            grid_start_x = int(68 * scale_x)
            grid_start_y = int(541 * scale_y)
            option_spacing = int(49 * scale_x)
            row_spacing = int(42 * scale_y)
            bubble_w = int(30 * scale_x)
            bubble_h = int(18 * scale_y)
    else:
        # Ảnh scan - sử dụng vị trí cột tuyệt đối đã calibrate cẩn thận
        # (SEAMO có spacing không đều giữa các cột A-E)
        # Expected size: 2480 x 3508 (A4 @ 300 DPI)
        expected_scan_w, expected_scan_h = 2480, 3508
        scan_scale_x = w / expected_scan_w
        scan_scale_y = h / expected_scan_h

        # Vị trí tuyệt đối cho mỗi cột (đã calibrate từ scan thực tế)
        col_lefts_base = [238, 318, 435, 551, 663]  # A, B, C, D, E
        col_lefts = [int(c * scan_scale_x) for c in col_lefts_base]

        grid_start_y = int(1164 * scan_scale_y)
        row_spacing = int(82 * scan_scale_y)
        bubble_w = int(50 * scan_scale_x)
        bubble_h = int(35 * scan_scale_y)

        # Build questions với vị trí cột tuyệt đối
        questions = []
        for q_idx in range(20):
            row_y = grid_start_y + q_idx * row_spacing
            bubbles = []
            for opt_idx in range(5):
                bubble_x = col_lefts[opt_idx]
                bubble_cx = bubble_x + bubble_w // 2
                bubble_cy = row_y + bubble_h // 2
                bubbles.append({
                    'x': bubble_x,
                    'y': row_y,
                    'w': bubble_w,
                    'h': bubble_h,
                    'cx': bubble_cx,
                    'cy': bubble_cy
                })
            questions.append({
                'index': q_idx + 1,
                'bubbles': bubbles
            })
        return questions

    questions = []

    for q_idx in range(20):
        row_y = grid_start_y + q_idx * row_spacing

        bubbles = []
        for opt_idx in range(5):
            bubble_x = grid_start_x + opt_idx * option_spacing
            bubble_cx = bubble_x + bubble_w // 2
            bubble_cy = row_y + bubble_h // 2

            bubbles.append({
                'x': bubble_x,
                'y': row_y,
                'w': bubble_w,
                'h': bubble_h,
                'cx': bubble_cx,
                'cy': bubble_cy
            })

        questions.append({
            'index': q_idx + 1,
            'bubbles': bubbles
        })

    return questions


def _detect_seamo_grid_dynamic(gray_image):
    """Phát hiện động vị trí grid SEAMO bằng Canny edge detection + Hough Lines

    Cải thiện: Sử dụng Canny + HoughLinesP để detect đường kẻ chính xác hơn,
    hoạt động tốt với cả PDF vector và ảnh scan.

    Returns:
        dict với các key: start_x, start_y, col_spacing, row_spacing, bubble_w, bubble_h
        hoặc None nếu không detect được
    """
    h, w = gray_image.shape[:2]

    # ===== BƯỚC 1: Edge detection với Canny =====
    edges = cv2.Canny(gray_image, 50, 150)

    # Crop vùng câu hỏi (25%-90% chiều cao, 2%-40% chiều rộng)
    crop_y1, crop_y2 = int(h * 0.25), int(h * 0.9)
    crop_x1, crop_x2 = int(w * 0.02), int(w * 0.4)
    edges_crop = edges[crop_y1:crop_y2, crop_x1:crop_x2]

    # ===== BƯỚC 2: Detect đường ngang với Hough Lines =====
    h_lines = cv2.HoughLinesP(edges_crop, 1, np.pi/180,
                              threshold=80, minLineLength=80, maxLineGap=10)

    if h_lines is None or len(h_lines) < 10:
        return None

    # Lọc và nhóm đường ngang
    horizontal_y = []
    for line in h_lines:
        x1, y1, x2, y2 = line[0]
        # Đường ngang: góc < 5 độ
        if abs(y2 - y1) < 5 and abs(x2 - x1) > 50:
            y_center = (y1 + y2) // 2 + crop_y1
            horizontal_y.append(y_center)

    if len(horizontal_y) < 10:
        return None

    # Nhóm các đường gần nhau
    horizontal_y = sorted(horizontal_y)
    row_lines = []
    current_group = [horizontal_y[0]]

    for y in horizontal_y[1:]:
        if y - current_group[-1] <= 5:
            current_group.append(y)
        else:
            row_lines.append(int(np.mean(current_group)))
            current_group = [y]

    if current_group:
        row_lines.append(int(np.mean(current_group)))

    if len(row_lines) < 10:
        return None

    # ===== BƯỚC 3: Tính row_spacing =====
    row_gaps = []
    for i in range(1, len(row_lines)):
        gap = row_lines[i] - row_lines[i-1]
        if 25 < gap < 55:  # Điều chỉnh range phù hợp hơn
            row_gaps.append(gap)

    if len(row_gaps) < 5:
        return None

    row_spacing = int(np.median(row_gaps))

    # ===== BƯỚC 4: Detect đường dọc =====
    v_lines = cv2.HoughLinesP(edges_crop, 1, np.pi/180,
                              threshold=50, minLineLength=50, maxLineGap=10)

    if v_lines is None:
        return None

    # Lọc đường dọc
    vertical_x = []
    for line in v_lines:
        x1, y1, x2, y2 = line[0]
        # Đường dọc: góc > 85 độ
        if abs(x2 - x1) < 5 and abs(y2 - y1) > 30:
            x_center = (x1 + x2) // 2 + crop_x1
            vertical_x.append(x_center)

    if len(vertical_x) < 5:
        return None

    # Nhóm các đường gần nhau
    vertical_x = sorted(vertical_x)
    col_lines = []
    current_group = [vertical_x[0]]

    for x in vertical_x[1:]:
        if x - current_group[-1] <= 8:
            current_group.append(x)
        else:
            col_lines.append(int(np.mean(current_group)))
            current_group = [x]

    if current_group:
        col_lines.append(int(np.mean(current_group)))

    if len(col_lines) < 5:
        return None

    # ===== BƯỚC 5: Tính col_spacing =====
    col_gaps = []
    for i in range(1, len(col_lines)):
        gap = col_lines[i] - col_lines[i-1]
        if 30 < gap < 60:
            col_gaps.append(gap)

    if len(col_gaps) < 3:
        return None

    col_spacing = int(np.median(col_gaps))

    # ===== BƯỚC 6: Xác định vị trí bắt đầu =====
    # Tìm header - thường là 2 đường liên tiếp gần nhau ở đầu (header đen + viền)
    # Sau đó các row có khoảng cách đều (row_spacing)

    # Tìm vị trí bắt đầu của grid thực sự (sau header)
    # Header thường có khoảng cách < row_spacing * 0.8
    content_start_idx = 0
    for i in range(1, len(row_lines)):
        gap = row_lines[i] - row_lines[i-1]
        if gap < row_spacing * 0.7:
            # Đây vẫn là header area
            content_start_idx = i
        elif gap >= row_spacing * 0.85:
            # Đây là row đầu tiên của content
            content_start_idx = i
            break

    # Row đầu tiên của content (câu 1)
    first_content_row = row_lines[content_start_idx] if content_start_idx < len(row_lines) else row_lines[-1]

    # Tìm cột bubble A (sau cột số thứ tự)
    first_bubble_x = None
    for i in range(1, len(col_lines)):
        gap = col_lines[i] - col_lines[i-1]
        if gap >= col_spacing * 0.8:
            first_bubble_x = col_lines[i-1] + 5  # Offset nhỏ sau viền
            break

    if first_bubble_x is None and len(col_lines) > 1:
        first_bubble_x = col_lines[1] + 5

    if first_bubble_x is None:
        return None

    # start_y: vùng tô của câu 1 (sau đường kẻ, bỏ qua label A,B,C,D,E)
    # Offset ~45% row_spacing để vào vùng tô
    start_y = first_content_row + int(row_spacing * 0.5)

    # Bubble size
    bubble_w = int(col_spacing * 0.6)
    bubble_h = int(row_spacing * 0.35)

    return {
        'start_x': first_bubble_x,
        'start_y': start_y,
        'col_spacing': col_spacing,
        'row_spacing': row_spacing,
        'bubble_w': bubble_w,
        'bubble_h': bubble_h,
        'row_lines': row_lines[:25],
        'col_lines': col_lines
    }
