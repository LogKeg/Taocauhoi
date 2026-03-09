"""OMR Template Detection and Student Info Extraction via OCR.

Provides functions to detect contest template type from answer sheet images
and extract student information using OCR (EasyOCR / Pytesseract).
"""

import re


def _detect_template_from_image(image_bytes: bytes, num_questions_detected: int = 0) -> dict:
    """Nhận diện loại đề và cấp độ từ phiếu bằng OCR

    Trả về dict với keys:
    - detected_template: template_type đầy đủ (ví dụ: "IKSC_BENJAMIN")
    - detected_contest: IKSC hoặc IKLC
    - detected_level: PRE_ECOLIER, ECOLIER, BENJAMIN, CADET, JUNIOR, STUDENT

    Sử dụng kết hợp:
    1. OCR để đọc text từ header
    2. Số câu hỏi được phát hiện để xác định level
    """
    import cv2
    import numpy as np

    result = {
        "detected_template": "",
        "detected_contest": "",
        "detected_level": ""
    }

    # Đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return result

    height, width = img.shape[:2]
    text = ""

    # Thử các OCR engines theo thứ tự ưu tiên
    ocr_success = False

    # 1. Thử EasyOCR
    try:
        # Fix SSL certificate issue
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        import easyocr
        # Lấy phần trên của ảnh (chứa thông tin loại đề) - khoảng 15% trên
        top_region = img[0:int(height * 0.15), :]
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        ocr_results = reader.readtext(top_region)
        text = ' '.join([r[1] for r in ocr_results])
        ocr_success = True
    except:
        pass

    # 2. Thử Pytesseract
    if not ocr_success:
        try:
            import pytesseract
            top_region = img[0:int(height * 0.15), :]
            # Chuyển sang grayscale và tăng contrast
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='eng')
            ocr_success = True
        except:
            pass

    # 3. Nếu OCR không thành công, sử dụng số câu hỏi để đoán
    if not ocr_success and num_questions_detected > 0:
        # Dựa vào số câu hỏi để đoán template
        if num_questions_detected <= 24:
            # 24 câu -> Pre-Ecolier hoặc Ecolier (IKSC) hoặc Pre-Ecolier (IKLC)
            result["detected_level"] = "PRE_ECOLIER"  # Mặc định
        elif num_questions_detected <= 30:
            # 30 câu -> Benjamin/Cadet/Junior/Student (IKSC) hoặc Ecolier (IKLC)
            result["detected_level"] = "BENJAMIN"  # Mặc định cho IKSC
        elif num_questions_detected <= 50:
            # 50 câu -> Benjamin/Cadet/Junior/Student (IKLC)
            result["detected_contest"] = "IKLC"
            result["detected_level"] = "BENJAMIN"  # Mặc định

        return result

    text_lower = text.lower()

    # === NHẬN DIỆN LOẠI CUỘC THI (IKSC hoặc IKLC) ===
    if 'science' in text_lower or 'iksc' in text_lower:
        result["detected_contest"] = "IKSC"
    elif 'linguistic' in text_lower or 'iklc' in text_lower or 'english' in text_lower:
        result["detected_contest"] = "IKLC"

    # === NHẬN DIỆN CẤP ĐỘ (LEVEL) ===
    level_detected = ""

    # Tìm theo CLASS pattern (ví dụ: "CLASS 5 & 6", "CLASS 5&6", "5 & 6")
    class_match = re.search(r'class\s*(\d+)\s*[&]\s*(\d+)', text_lower)
    if not class_match:
        # Thử pattern không có "class"
        class_match = re.search(r'(\d+)\s*[&]\s*(\d+)', text_lower)

    if class_match:
        class1 = int(class_match.group(1))
        class2 = int(class_match.group(2))
        if class1 == 1 and class2 == 2:
            level_detected = "PRE_ECOLIER"
        elif class1 == 3 and class2 == 4:
            level_detected = "ECOLIER"
        elif class1 == 5 and class2 == 6:
            level_detected = "BENJAMIN"
        elif class1 == 7 and class2 == 8:
            level_detected = "CADET"
        elif class1 == 9 and class2 == 10:
            level_detected = "JUNIOR"
        elif class1 == 11 and class2 == 12:
            level_detected = "STUDENT"

    # Nếu không tìm thấy theo class, thử tìm theo tên level
    if not level_detected:
        if 'pre-ecolier' in text_lower or 'preecolier' in text_lower or 'pre_ecolier' in text_lower:
            level_detected = "PRE_ECOLIER"
        elif 'benjamin' in text_lower:
            level_detected = "BENJAMIN"
        elif 'cadet' in text_lower:
            level_detected = "CADET"
        elif 'junior' in text_lower:
            level_detected = "JUNIOR"
        elif 'student' in text_lower:
            level_detected = "STUDENT"
        elif 'ecolier' in text_lower:
            level_detected = "ECOLIER"
        # IKLC specific names
        elif 'start' in text_lower:
            level_detected = "PRE_ECOLIER"
        elif 'story' in text_lower:
            level_detected = "ECOLIER"
        elif 'joey' in text_lower:
            level_detected = "BENJAMIN"
        elif 'wallaby' in text_lower:
            level_detected = "CADET"
        elif 'grey' in text_lower:
            level_detected = "JUNIOR"
        elif 'red k' in text_lower:
            level_detected = "STUDENT"

    # Nếu vẫn không tìm được level nhưng có số câu hỏi
    if not level_detected and num_questions_detected > 0:
        if num_questions_detected <= 24:
            level_detected = "PRE_ECOLIER"
        elif num_questions_detected <= 30:
            level_detected = "BENJAMIN"
        elif num_questions_detected <= 50:
            level_detected = "BENJAMIN"

    result["detected_level"] = level_detected

    # Tạo template_type đầy đủ
    if result["detected_contest"] and level_detected:
        result["detected_template"] = f"{result['detected_contest']}_{level_detected}"
    elif level_detected:
        # Nếu chỉ có level, thử đoán contest từ số câu
        if num_questions_detected == 50:
            result["detected_contest"] = "IKLC"
        elif num_questions_detected == 30:
            result["detected_contest"] = "IKSC"
        elif num_questions_detected == 24:
            result["detected_contest"] = "IKSC"

        if result["detected_contest"]:
            result["detected_template"] = f"{result['detected_contest']}_{level_detected}"

    return result


def _extract_student_info_ocr(image_bytes: bytes) -> dict:
    """Trích xuất thông tin học sinh và loại đề từ phiếu bằng OCR (EasyOCR)"""
    import cv2
    import numpy as np

    # Bắt đầu với việc nhận diện template
    template_info = _detect_template_from_image(image_bytes)

    # Parse thông tin từ text
    info = {
        "full_name": "",
        "class": "",
        "dob": "",
        "id_no": "",
        "school_name": "",
        "detected_template": template_info.get("detected_template", ""),
        "detected_contest": template_info.get("detected_contest", ""),
        "detected_level": template_info.get("detected_level", "")
    }

    try:
        import easyocr
    except ImportError:
        return info

    # Đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return info

    # Lấy phần trên của ảnh (chứa thông tin học sinh) - khoảng 25% trên
    height, width = img.shape[:2]
    top_region = img[0:int(height * 0.25), :]

    # Khởi tạo EasyOCR reader
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(top_region)
    except:
        return info

    # Ghép kết quả OCR thành text
    text = '\n'.join([result[1] for result in results])

    # === PARSE THÔNG TIN HỌC SINH ===
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower().strip()

        # Tìm Full Name
        if 'full name' in line_lower or 'họ tên' in line_lower or 'name:' in line_lower:
            # Lấy phần sau dấu :
            parts = line.split(':')
            if len(parts) > 1:
                info["full_name"] = parts[1].strip()
            else:
                # Tìm trên cùng dòng sau label
                match = re.search(r'(?:full name|họ tên|name)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["full_name"] = match.group(1).strip()

        # Tìm Class (thông tin lớp học của học sinh, không phải level)
        elif 'class:' in line_lower and 'school' not in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["class"] = parts[1].strip()

        # Tìm DOB
        elif 'dob' in line_lower or 'date of birth' in line_lower or 'ngày sinh' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["dob"] = parts[1].strip()
            else:
                match = re.search(r'(?:dob|date of birth|ngày sinh)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["dob"] = match.group(1).strip()

        # Tìm ID NO
        elif 'id no' in line_lower or 'id:' in line_lower or 'số báo danh' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["id_no"] = parts[1].strip()
            else:
                match = re.search(r'(?:id no|id|số báo danh)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["id_no"] = match.group(1).strip()

        # Tìm School Name
        elif 'school' in line_lower or 'trường' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["school_name"] = parts[1].strip()
            else:
                match = re.search(r'(?:school name|school|trường)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["school_name"] = match.group(1).strip()

    return info
