"""
Constants and configuration data for the exam generation system.
"""
import re

# Topic definitions with labels, keywords, and context replacements
TOPICS = {
    "toan_hoc": {
        "label": "Toán học",
        "keywords": ["hình chữ nhật", "tam giác", "chu vi", "diện tích", "tỉ lệ"],
        "context": {
            "quả táo": "viên bi",
            "cửa hàng": "siêu thị",
            "học sinh": "bạn An",
        },
    },
    "vat_ly": {
        "label": "Vật lý",
        "keywords": ["vận tốc", "quãng đường", "thời gian", "lực", "gia tốc"],
        "context": {
            "chiếc xe": "tàu hỏa",
            "đi bộ": "chạy",
            "bạn An": "bạn Bình",
        },
    },
    "sinh_hoc": {
        "label": "Sinh học",
        "keywords": ["tế bào", "quang hợp", "gen", "nhiễm sắc thể", "tiến hóa"],
        "context": {
            "cây": "thực vật",
            "động vật": "sinh vật",
            "môi trường": "hệ sinh thái",
        },
    },
    "lich_su": {
        "label": "Lịch sử",
        "keywords": ["triều đại", "chiến dịch", "hiệp định", "năm", "sự kiện"],
        "context": {
            "vua": "chúa",
            "triều đình": "chính quyền",
            "chiến tranh": "xung đột",
        },
    },
    "dia_ly": {
        "label": "Địa lý",
        "keywords": ["khí hậu", "địa hình", "sông", "dân cư", "tài nguyên"],
        "context": {
            "vùng": "khu vực",
            "núi": "cao nguyên",
            "đồng bằng": "bồn trũng",
        },
    },
}

# Question templates by subject
TEMPLATES = {
    "toan_hoc": [
        "Một hình chữ nhật có chiều dài ...cm và chiều rộng ...cm. Tính diện tích.",
        "Một tam giác có đáy ...cm và chiều cao ...cm. Tính diện tích.",
        "Chu vi hình vuông là ...cm. Tính cạnh hình vuông.",
        "Tỉ lệ bản đồ là 1:.... Tính khoảng cách thực tế.",
        "Tính tổng các số ...; ...; ... .",
        "Tìm x sao cho ...x + ... = ... .",
        "Tính giá trị biểu thức: (... + ...) × ... .",
        "Một bể nước chứa ... lít, mỗi ngày dùng ... lít. Hỏi sau ... ngày còn bao nhiêu lít?",
        "Một lớp có ... học sinh, tỉ lệ nữ là ...%. Tính số học sinh nữ.",
        "Quãng đường AB dài ... km. Người đi bộ với vận tốc ... km/h. Tính thời gian đi.",
        "Tìm số lớn nhất/nhỏ nhất trong các số: ..., ..., ... .",
        "Giải bài toán về tỉ số: ... và ... có tổng là ... .",
    ],
    "vat_ly": [
        "Một chiếc xe đi với vận tốc ... km/h trong ... giờ. Tính quãng đường đi được.",
        "Một vật chuyển động nhanh dần đều với gia tốc ... m/s^2 trong ... s. Tính vận tốc cuối.",
        "Một lực ... N tác dụng lên vật có khối lượng ... kg. Tính gia tốc.",
        "Một vật rơi tự do từ độ cao ... m. Tính thời gian rơi (g = ... m/s^2).",
        "Dòng điện có cường độ ... A chạy qua điện trở ... Ω. Tính hiệu điện thế.",
        "Một bóng đèn công suất ... W hoạt động trong ... giờ. Tính điện năng tiêu thụ.",
        "Một vật chuyển động thẳng đều đi được ... m trong ... s. Tính vận tốc.",
        "Một vật chịu lực kéo ... N, hệ số ma sát ... . Tính lực ma sát.",
    ],
    "sinh_hoc": [
        "Quá trình quang hợp cần những điều kiện nào? Nêu vai trò của ...",
        "Ở người, gen quy định ... nằm trên nhiễm sắc thể số ... .",
        "Trình bày cơ chế ... trong tế bào.",
        "So sánh nguyên phân và giảm phân về ... .",
        "Nêu vai trò của enzim ... trong quá trình ... .",
        "Mô tả cấu trúc của tế bào ... .",
        "Giải thích vì sao ... ảnh hưởng tới quang hợp.",
        "Trình bày cơ chế di truyền của tính trạng ... .",
    ],
    "lich_su": [
        "Trình bày ý nghĩa của sự kiện ... năm ... .",
        "Nguyên nhân dẫn tới ... là gì?",
        "So sánh ... và ... trong giai đoạn ... .",
        "Nêu bối cảnh lịch sử dẫn tới ... .",
        "Phân tích vai trò của nhân vật ... trong sự kiện ... .",
        "Trình bày diễn biến chính của chiến dịch ... .",
        "Đánh giá kết quả và ý nghĩa của hiệp định ... .",
    ],
    "dia_ly": [
        "Phân tích đặc điểm khí hậu của ... .",
        "Nêu vai trò của sông ... đối với phát triển kinh tế vùng ... .",
        "Trình bày đặc điểm địa hình của ... .",
        "So sánh khí hậu vùng ... và vùng ... .",
        "Nêu các nhân tố ảnh hưởng tới phân bố dân cư ở ... .",
        "Trình bày tiềm năng tài nguyên thiên nhiên của ... .",
        "Giải thích sự phân hóa tự nhiên theo vĩ độ ở ... .",
    ],
}

# AI generation guidance by topic
TOPIC_AI_GUIDE = {
    "toan_hoc": (
        "Tập trung vào bài toán định lượng, công thức cơ bản; "
        "đảm bảo có dữ kiện đủ để tính toán. Có thể thay số liệu hợp lý."
    ),
    "vat_ly": (
        "Ưu tiên các đại lượng vật lý, đơn vị rõ ràng; "
        "nêu điều kiện (g, bỏ qua ma sát) nếu cần."
    ),
    "sinh_hoc": (
        "Ưu tiên câu hỏi giải thích, so sánh, mô tả quá trình sinh học; "
        "tránh dùng số liệu phức tạp."
    ),
    "lich_su": (
        "Ưu tiên bối cảnh, nguyên nhân, diễn biến, ý nghĩa; "
        "giữ mốc thời gian hợp lý."
    ),
    "dia_ly": (
        "Ưu tiên đặc điểm tự nhiên - kinh tế, phân bố, so sánh vùng; "
        "dùng thuật ngữ địa lý phổ biến."
    ),
}

# Subject definitions
SUBJECTS = {
    "toan_hoc": {"label": "Toán học", "lang": "vi"},
    "tieng_anh": {"label": "Tiếng Anh", "lang": "en"},
    "khoa_hoc": {"label": "Khoa học", "lang": "vi"},
}

# Subject topics mapping
SUBJECT_TOPICS: dict[str, list[dict[str, str]]] = {
    "toan_hoc": [
        {"key": "so_hoc", "label": "Số học"},
        {"key": "dai_so", "label": "Đại số"},
        {"key": "hinh_hoc", "label": "Hình học"},
        {"key": "do_luong", "label": "Đo lường"},
        {"key": "thong_ke", "label": "Thống kê và xác suất"},
        {"key": "phan_so", "label": "Phân số và thập phân"},
        {"key": "phuong_trinh", "label": "Phương trình và bất phương trình"},
        {"key": "ham_so", "label": "Hàm số"},
        {"key": "luong_giac", "label": "Lượng giác"},
        {"key": "tich_phan", "label": "Tích phân và đạo hàm"},
    ],
    "tieng_anh": [
        {"key": "grammar", "label": "Grammar"},
        {"key": "vocabulary", "label": "Vocabulary"},
        {"key": "reading", "label": "Reading Comprehension"},
        {"key": "tenses", "label": "Tenses"},
        {"key": "prepositions", "label": "Prepositions"},
        {"key": "articles", "label": "Articles & Determiners"},
        {"key": "conditionals", "label": "Conditionals"},
        {"key": "passive_voice", "label": "Passive Voice"},
        {"key": "reported_speech", "label": "Reported Speech"},
        {"key": "word_formation", "label": "Word Formation"},
    ],
    "khoa_hoc": [
        {"key": "vat_ly", "label": "Vật lý"},
        {"key": "hoa_hoc", "label": "Hóa học"},
        {"key": "sinh_hoc", "label": "Sinh học"},
        {"key": "trai_dat", "label": "Trái Đất và bầu trời"},
        {"key": "moi_truong", "label": "Môi trường và sinh thái"},
        {"key": "co_the_nguoi", "label": "Cơ thể người"},
        {"key": "dong_vat", "label": "Động vật"},
        {"key": "thuc_vat", "label": "Thực vật"},
        {"key": "nang_luong", "label": "Năng lượng và lực"},
        {"key": "nuoc_khi_tuong", "label": "Nước và khí tượng"},
    ],
}

# Question type definitions
QUESTION_TYPES = {
    "mcq": "Trắc nghiệm",
    "blank": "Điền khuyết",
    "essay": "Tự luận",
}

# Vietnamese synonyms for text variation
SYNONYMS = {
    "tính": ["xác định", "tìm"],
    "hãy": ["vui lòng", "mời"],
    "bao nhiêu": ["bao nhiêu là", "là bao nhiêu"],
    "số": ["con số", "giá trị"],
    "tổng": ["cộng", "tổng số"],
    "hiệu": ["chênh lệch", "hiệu số"],
    "lớn hơn": ["cao hơn", "nhiều hơn"],
    "nhỏ hơn": ["ít hơn", "thấp hơn"],
}

# Compiled regex patterns
NUMBER_RE = re.compile(r"\b\d+\b")
LEADING_NUM_RE = re.compile(r"^\s*\d{1,3}[\).\-:]\s+")
MCQ_OPTION_RE = re.compile(r"^[A-H][\).\-:]\s+", re.IGNORECASE)

# OMR Answer Templates for different contests
ANSWER_TEMPLATES = {
    # IKSC - Khoa học (Science Contest)
    "IKSC_PRE_ECOLIER": {
        "name": "Khoa học - Pre-Ecolier (Lớp 1-2)",
        "questions": 24,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 8, "correct": 3, "wrong": -1},
                {"start": 9, "end": 16, "correct": 4, "wrong": -1},
                {"start": 17, "end": 24, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 24
        }
    },
    "IKSC_ECOLIER": {
        "name": "Khoa học - Ecolier (Lớp 3-4)",
        "questions": 24,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 8, "correct": 3, "wrong": -1},
                {"start": 9, "end": 16, "correct": 4, "wrong": -1},
                {"start": 17, "end": 24, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 24
        }
    },
    "IKSC_BENJAMIN": {
        "name": "Khoa học - Benjamin (Lớp 5-6)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 10, "correct": 3, "wrong": -1},
                {"start": 11, "end": 20, "correct": 4, "wrong": -1},
                {"start": 21, "end": 30, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 30
        }
    },
    "IKSC_CADET": {
        "name": "Khoa học - Cadet (Lớp 7-8)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 10, "correct": 3, "wrong": -1},
                {"start": 11, "end": 20, "correct": 4, "wrong": -1},
                {"start": 21, "end": 30, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 30
        }
    },
    "IKSC_JUNIOR": {
        "name": "Khoa học - Junior (Lớp 9-10)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 10, "correct": 3, "wrong": -1},
                {"start": 11, "end": 20, "correct": 4, "wrong": -1},
                {"start": 21, "end": 30, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 30
        }
    },
    "IKSC_STUDENT": {
        "name": "Khoa học - Student (Lớp 11-12)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",
            "tiers": [
                {"start": 1, "end": 10, "correct": 3, "wrong": -1},
                {"start": 11, "end": 20, "correct": 4, "wrong": -1},
                {"start": 21, "end": 30, "correct": 5, "wrong": -1},
            ],
            "blank": 0,
            "base": 30
        }
    },
    # IKLC - Tiếng Anh (Linguistic Contest)
    "IKLC_PRE_ECOLIER": {
        "name": "Tiếng Anh - Pre-Ecolier (Lớp 1-2)",
        "questions": 25,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 2, "wrong": 0, "blank": 0, "base": 0}
    },
    "IKLC_ECOLIER": {
        "name": "Tiếng Anh - Ecolier (Lớp 3-4)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 2, "wrong": 0, "blank": 0, "base": 0}
    },
    "IKLC_BENJAMIN": {
        "name": "Tiếng Anh - Benjamin (Lớp 5-6)",
        "questions": 50,
        "options": 5,
        "questions_per_row": 4,
        "layout": "column",
        "scoring": {
            "type": "best_of",
            "count_best": 40,
            "correct": 1.0,
            "wrong": -0.5,
            "blank": 0,
            "base": 10
        }
    },
    "IKLC_CADET": {
        "name": "Tiếng Anh - Cadet (Lớp 7-8)",
        "questions": 50,
        "options": 5,
        "questions_per_row": 4,
        "layout": "column",
        "scoring": {
            "type": "best_of",
            "count_best": 40,
            "correct": 1.25,
            "wrong": -0.5,
            "blank": 0,
            "base": 10
        }
    },
    "IKLC_JUNIOR": {
        "name": "Tiếng Anh - Junior (Lớp 9-10)",
        "questions": 50,
        "options": 5,
        "questions_per_row": 4,
        "layout": "column",
        "scoring": {
            "type": "best_of",
            "count_best": 40,
            "correct": 1.5,
            "wrong": -0.5,
            "blank": 0,
            "base": 10
        }
    },
    "IKLC_STUDENT": {
        "name": "Tiếng Anh - Student (Lớp 11-12)",
        "questions": 50,
        "options": 5,
        "questions_per_row": 4,
        "layout": "column",
        "scoring": {
            "type": "best_of",
            "count_best": 40,
            "correct": 1.75,
            "wrong": -0.5,
            "blank": 0,
            "base": 10
        }
    },
    # ASMO - Math
    "ASMO_MATH_GRADE_1": {"name": "ASMO Toán - Lớp 1", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_2": {"name": "ASMO Toán - Lớp 2", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_3": {"name": "ASMO Toán - Lớp 3", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_4": {"name": "ASMO Toán - Lớp 4", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_5": {"name": "ASMO Toán - Lớp 5", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_6": {"name": "ASMO Toán - Lớp 6", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_7": {"name": "ASMO Toán - Lớp 7", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_8": {"name": "ASMO Toán - Lớp 8", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_9": {"name": "ASMO Toán - Lớp 9", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_10": {"name": "ASMO Toán - Lớp 10", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_11": {"name": "ASMO Toán - Lớp 11", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_MATH_GRADE_12": {"name": "ASMO Toán - Lớp 12", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    # ASMO - Science
    "ASMO_SCIENCE_LEVEL_1": {"name": "ASMO Khoa học - Level 1 (Lớp 1-2)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_2": {"name": "ASMO Khoa học - Level 2 (Lớp 3-4)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_3": {"name": "ASMO Khoa học - Level 3 (Lớp 5-6)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_4": {"name": "ASMO Khoa học - Level 4 (Lớp 7-8)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_5": {"name": "ASMO Khoa học - Level 5 (Lớp 9-10)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_6": {"name": "ASMO Khoa học - Level 6 (Lớp 11-12)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    # ASMO - English
    "ASMO_ENGLISH_LEVEL_1": {"name": "ASMO Tiếng Anh - Level 1 (Lớp 1-2)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_2": {"name": "ASMO Tiếng Anh - Level 2 (Lớp 3-4)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_3": {"name": "ASMO Tiếng Anh - Level 3 (Lớp 5-6)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_4": {"name": "ASMO Tiếng Anh - Level 4 (Lớp 7-8)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_5": {"name": "ASMO Tiếng Anh - Level 5 (Lớp 9-10)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_6": {"name": "ASMO Tiếng Anh - Level 6 (Lớp 11-12)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    # SEAMO - Math
    "SEAMO_MATH_PAPER_A": {"name": "SEAMO Toán - Paper A (Lớp 1-2)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_B": {"name": "SEAMO Toán - Paper B (Lớp 3-4)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_C": {"name": "SEAMO Toán - Paper C (Lớp 5-6)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_D": {"name": "SEAMO Toán - Paper D (Lớp 7-8)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_E": {"name": "SEAMO Toán - Paper E (Lớp 9-10)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_F": {"name": "SEAMO Toán - Paper F (Lớp 11-12)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    # Custom
    "CUSTOM": {
        "name": "Tùy chỉnh",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}
    }
}
