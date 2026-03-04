"""
Sample curriculum data for Vietnamese education system.
Based on Chương trình Giáo dục Phổ thông 2018.
"""

# Sample curriculum data - can be expanded or imported from external source
SAMPLE_CURRICULUM_DATA = {
    "toan": {
        "name": "Toán học",
        "grades": {
            10: [
                {
                    "chapter": "Chương 1: Mệnh đề và tập hợp",
                    "topics": [
                        {"topic": "Mệnh đề", "lessons": ["Mệnh đề", "Mệnh đề chứa biến"], "periods": 3},
                        {"topic": "Tập hợp", "lessons": ["Tập hợp và các phép toán", "Các tập hợp số"], "periods": 4},
                    ],
                    "knowledge": "Hiểu khái niệm mệnh đề, tập hợp, các phép toán tập hợp",
                    "skills": "Xác định mệnh đề đúng/sai, thực hiện phép toán tập hợp",
                },
                {
                    "chapter": "Chương 2: Bất phương trình và hệ bất phương trình bậc nhất hai ẩn",
                    "topics": [
                        {"topic": "Bất phương trình bậc nhất hai ẩn", "lessons": ["Bất phương trình bậc nhất hai ẩn"], "periods": 2},
                        {"topic": "Hệ bất phương trình bậc nhất hai ẩn", "lessons": ["Hệ bất phương trình bậc nhất hai ẩn"], "periods": 3},
                    ],
                    "knowledge": "Hiểu khái niệm bất phương trình và hệ bất phương trình bậc nhất hai ẩn",
                    "skills": "Giải và biểu diễn miền nghiệm của bất phương trình, hệ bất phương trình",
                },
                {
                    "chapter": "Chương 3: Hàm số bậc hai và đồ thị",
                    "topics": [
                        {"topic": "Hàm số và đồ thị", "lessons": ["Hàm số", "Đồ thị hàm số"], "periods": 3},
                        {"topic": "Hàm số bậc hai", "lessons": ["Hàm số bậc hai", "Phương trình bậc hai"], "periods": 5},
                    ],
                    "knowledge": "Hiểu khái niệm hàm số, đồ thị, hàm số bậc hai",
                    "skills": "Vẽ đồ thị, tìm giá trị lớn nhất, nhỏ nhất",
                },
                {
                    "chapter": "Chương 4: Hệ thức lượng trong tam giác",
                    "topics": [
                        {"topic": "Giá trị lượng giác của góc", "lessons": ["Giá trị lượng giác của góc từ 0° đến 180°"], "periods": 3},
                        {"topic": "Định lý cosin và định lý sin", "lessons": ["Định lý cosin", "Định lý sin"], "periods": 4},
                    ],
                    "knowledge": "Hiểu giá trị lượng giác, định lý cosin, định lý sin",
                    "skills": "Tính các yếu tố trong tam giác, giải tam giác",
                },
                {
                    "chapter": "Chương 5: Vectơ",
                    "topics": [
                        {"topic": "Khái niệm vectơ", "lessons": ["Vectơ và các phép toán"], "periods": 4},
                        {"topic": "Tích vô hướng", "lessons": ["Tích vô hướng của hai vectơ"], "periods": 3},
                    ],
                    "knowledge": "Hiểu khái niệm vectơ, các phép toán vectơ",
                    "skills": "Thực hiện phép toán vectơ, tính tích vô hướng",
                },
                {
                    "chapter": "Chương 6: Thống kê",
                    "topics": [
                        {"topic": "Các số đặc trưng đo xu thế trung tâm", "lessons": ["Số trung bình, trung vị, mốt"], "periods": 3},
                        {"topic": "Các số đặc trưng đo độ phân tán", "lessons": ["Phương sai, độ lệch chuẩn"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các số đặc trưng thống kê",
                    "skills": "Tính toán và phân tích dữ liệu thống kê",
                },
                {
                    "chapter": "Chương 7: Xác suất",
                    "topics": [
                        {"topic": "Xác suất của biến cố", "lessons": ["Không gian mẫu, biến cố", "Xác suất của biến cố"], "periods": 5},
                    ],
                    "knowledge": "Hiểu khái niệm xác suất",
                    "skills": "Tính xác suất của các biến cố",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Hàm số lượng giác và phương trình lượng giác",
                    "topics": [
                        {"topic": "Góc lượng giác và công thức lượng giác", "lessons": ["Góc lượng giác", "Công thức lượng giác"], "periods": 5},
                        {"topic": "Hàm số lượng giác", "lessons": ["Hàm số lượng giác", "Đồ thị hàm số lượng giác"], "periods": 4},
                        {"topic": "Phương trình lượng giác", "lessons": ["Phương trình lượng giác cơ bản"], "periods": 5},
                    ],
                    "knowledge": "Hiểu góc lượng giác, hàm số lượng giác, phương trình lượng giác",
                    "skills": "Tính giá trị lượng giác, vẽ đồ thị, giải phương trình",
                },
                {
                    "chapter": "Chương 2: Dãy số. Cấp số cộng và cấp số nhân",
                    "topics": [
                        {"topic": "Dãy số", "lessons": ["Dãy số và cách cho dãy số"], "periods": 2},
                        {"topic": "Cấp số cộng", "lessons": ["Cấp số cộng"], "periods": 3},
                        {"topic": "Cấp số nhân", "lessons": ["Cấp số nhân"], "periods": 3},
                    ],
                    "knowledge": "Hiểu dãy số, cấp số cộng, cấp số nhân",
                    "skills": "Tìm số hạng tổng quát, tính tổng n số hạng đầu",
                },
                {
                    "chapter": "Chương 3: Giới hạn. Hàm số liên tục",
                    "topics": [
                        {"topic": "Giới hạn của dãy số", "lessons": ["Giới hạn của dãy số"], "periods": 3},
                        {"topic": "Giới hạn của hàm số", "lessons": ["Giới hạn của hàm số"], "periods": 4},
                        {"topic": "Hàm số liên tục", "lessons": ["Hàm số liên tục"], "periods": 2},
                    ],
                    "knowledge": "Hiểu giới hạn, hàm số liên tục",
                    "skills": "Tính giới hạn, xét tính liên tục",
                },
                {
                    "chapter": "Chương 4: Đường thẳng và mặt phẳng trong không gian",
                    "topics": [
                        {"topic": "Đường thẳng và mặt phẳng", "lessons": ["Đại cương về đường thẳng và mặt phẳng"], "periods": 3},
                        {"topic": "Hai đường thẳng song song", "lessons": ["Hai đường thẳng song song"], "periods": 2},
                        {"topic": "Đường thẳng và mặt phẳng song song", "lessons": ["Đường thẳng và mặt phẳng song song"], "periods": 3},
                        {"topic": "Hai mặt phẳng song song", "lessons": ["Hai mặt phẳng song song"], "periods": 2},
                    ],
                    "knowledge": "Hiểu quan hệ song song trong không gian",
                    "skills": "Chứng minh song song, xác định giao tuyến",
                },
                {
                    "chapter": "Chương 5: Vectơ trong không gian",
                    "topics": [
                        {"topic": "Vectơ trong không gian", "lessons": ["Vectơ trong không gian"], "periods": 3},
                        {"topic": "Hai đường thẳng vuông góc", "lessons": ["Hai đường thẳng vuông góc"], "periods": 2},
                        {"topic": "Đường thẳng vuông góc với mặt phẳng", "lessons": ["Đường thẳng vuông góc với mặt phẳng"], "periods": 3},
                        {"topic": "Hai mặt phẳng vuông góc", "lessons": ["Hai mặt phẳng vuông góc"], "periods": 3},
                    ],
                    "knowledge": "Hiểu quan hệ vuông góc trong không gian",
                    "skills": "Chứng minh vuông góc, tính góc và khoảng cách",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Đạo hàm và ứng dụng",
                    "topics": [
                        {"topic": "Đạo hàm", "lessons": ["Đạo hàm", "Các quy tắc tính đạo hàm"], "periods": 5},
                        {"topic": "Ứng dụng của đạo hàm", "lessons": ["Sự đồng biến, nghịch biến", "Cực trị", "Giá trị lớn nhất, nhỏ nhất"], "periods": 7},
                    ],
                    "knowledge": "Hiểu đạo hàm, các quy tắc tính đạo hàm, ứng dụng",
                    "skills": "Tính đạo hàm, khảo sát hàm số",
                },
                {
                    "chapter": "Chương 2: Hàm số mũ và hàm số logarit",
                    "topics": [
                        {"topic": "Hàm số mũ", "lessons": ["Lũy thừa", "Hàm số mũ"], "periods": 4},
                        {"topic": "Hàm số logarit", "lessons": ["Logarit", "Hàm số logarit"], "periods": 4},
                        {"topic": "Phương trình và bất phương trình mũ, logarit", "lessons": ["Phương trình mũ và logarit", "Bất phương trình mũ và logarit"], "periods": 5},
                    ],
                    "knowledge": "Hiểu hàm số mũ, logarit",
                    "skills": "Tính logarit, giải phương trình và bất phương trình",
                },
                {
                    "chapter": "Chương 3: Nguyên hàm và tích phân",
                    "topics": [
                        {"topic": "Nguyên hàm", "lessons": ["Nguyên hàm"], "periods": 3},
                        {"topic": "Tích phân", "lessons": ["Tích phân", "Ứng dụng tích phân"], "periods": 6},
                    ],
                    "knowledge": "Hiểu nguyên hàm, tích phân",
                    "skills": "Tính nguyên hàm, tích phân, tính diện tích và thể tích",
                },
                {
                    "chapter": "Chương 4: Số phức",
                    "topics": [
                        {"topic": "Số phức", "lessons": ["Số phức", "Các phép toán số phức"], "periods": 4},
                        {"topic": "Phương trình bậc hai với hệ số thực", "lessons": ["Phương trình bậc hai với hệ số thực"], "periods": 2},
                    ],
                    "knowledge": "Hiểu số phức, các phép toán",
                    "skills": "Thực hiện phép toán số phức, giải phương trình",
                },
                {
                    "chapter": "Chương 5: Hình học tọa độ trong không gian",
                    "topics": [
                        {"topic": "Phương trình mặt phẳng", "lessons": ["Hệ tọa độ trong không gian", "Phương trình mặt phẳng"], "periods": 4},
                        {"topic": "Phương trình đường thẳng", "lessons": ["Phương trình đường thẳng"], "periods": 4},
                        {"topic": "Phương trình mặt cầu", "lessons": ["Phương trình mặt cầu"], "periods": 2},
                    ],
                    "knowledge": "Hiểu hệ tọa độ không gian, phương trình mặt phẳng, đường thẳng, mặt cầu",
                    "skills": "Viết phương trình, tính khoảng cách, góc",
                },
            ],
        },
    },
    "ly": {
        "name": "Vật lý",
        "grades": {
            10: [
                {
                    "chapter": "Chương 1: Động học chất điểm",
                    "topics": [
                        {"topic": "Mô tả chuyển động", "lessons": ["Chuyển động cơ", "Vận tốc và tốc độ"], "periods": 4},
                        {"topic": "Chuyển động thẳng", "lessons": ["Chuyển động thẳng đều", "Chuyển động thẳng biến đổi đều"], "periods": 5},
                        {"topic": "Sự rơi tự do", "lessons": ["Sự rơi tự do"], "periods": 2},
                    ],
                    "knowledge": "Hiểu các khái niệm động học cơ bản",
                    "skills": "Giải bài toán chuyển động thẳng",
                },
                {
                    "chapter": "Chương 2: Động lực học chất điểm",
                    "topics": [
                        {"topic": "Các định luật Newton", "lessons": ["Định luật I Newton", "Định luật II Newton", "Định luật III Newton"], "periods": 6},
                        {"topic": "Các lực cơ học", "lessons": ["Lực hấp dẫn", "Lực đàn hồi", "Lực ma sát"], "periods": 5},
                    ],
                    "knowledge": "Hiểu các định luật Newton và các lực cơ học",
                    "skills": "Giải bài toán động lực học",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Điện tích. Điện trường",
                    "topics": [
                        {"topic": "Điện tích. Định luật Coulomb", "lessons": ["Điện tích", "Định luật Coulomb"], "periods": 3},
                        {"topic": "Điện trường", "lessons": ["Điện trường", "Đường sức điện"], "periods": 4},
                    ],
                    "knowledge": "Hiểu điện tích, điện trường",
                    "skills": "Tính lực điện, cường độ điện trường",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Dao động cơ",
                    "topics": [
                        {"topic": "Dao động điều hòa", "lessons": ["Dao động điều hòa", "Con lắc đơn, con lắc lò xo"], "periods": 6},
                        {"topic": "Dao động tắt dần và dao động cưỡng bức", "lessons": ["Dao động tắt dần", "Dao động cưỡng bức"], "periods": 3},
                    ],
                    "knowledge": "Hiểu dao động cơ",
                    "skills": "Giải bài toán dao động",
                },
            ],
        },
    },
    "hoa": {
        "name": "Hóa học",
        "grades": {
            10: [
                {
                    "chapter": "Chương 1: Cấu tạo nguyên tử",
                    "topics": [
                        {"topic": "Thành phần nguyên tử", "lessons": ["Thành phần nguyên tử"], "periods": 2},
                        {"topic": "Cấu hình electron", "lessons": ["Nguyên tố hóa học", "Cấu hình electron"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cấu tạo nguyên tử",
                    "skills": "Viết cấu hình electron",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Cân bằng hóa học",
                    "topics": [
                        {"topic": "Cân bằng hóa học", "lessons": ["Cân bằng hóa học", "Hằng số cân bằng"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cân bằng hóa học",
                    "skills": "Tính hằng số cân bằng",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Este - Lipit",
                    "topics": [
                        {"topic": "Este", "lessons": ["Este"], "periods": 3},
                        {"topic": "Lipit", "lessons": ["Chất béo"], "periods": 2},
                    ],
                    "knowledge": "Hiểu este, lipit",
                    "skills": "Viết phương trình phản ứng",
                },
            ],
        },
    },
    "sinh": {
        "name": "Sinh học",
        "grades": {
            10: [
                {
                    "chapter": "Chương 1: Thành phần hóa học của tế bào",
                    "topics": [
                        {"topic": "Các nguyên tố và nước", "lessons": ["Các nguyên tố hóa học và nước"], "periods": 2},
                        {"topic": "Cacbohydrat và Lipit", "lessons": ["Cacbohydrat", "Lipit"], "periods": 3},
                        {"topic": "Protein và Axit nucleic", "lessons": ["Protein", "Axit nucleic"], "periods": 4},
                    ],
                    "knowledge": "Hiểu thành phần hóa học của tế bào",
                    "skills": "Phân biệt các đại phân tử sinh học",
                },
            ],
        },
    },
    "van": {
        "name": "Ngữ văn",
        "grades": {
            10: [
                {
                    "chapter": "Phần 1: Văn học dân gian Việt Nam",
                    "topics": [
                        {"topic": "Sử thi", "lessons": ["Đăm Săn"], "periods": 4},
                        {"topic": "Truyền thuyết", "lessons": ["An Dương Vương và Mị Châu - Trọng Thủy"], "periods": 3},
                        {"topic": "Truyện cổ tích", "lessons": ["Tấm Cám"], "periods": 3},
                    ],
                    "knowledge": "Hiểu đặc trưng văn học dân gian",
                    "skills": "Phân tích tác phẩm văn học dân gian",
                },
            ],
        },
    },
    "anh": {
        "name": "Tiếng Anh",
        "grades": {
            10: [
                {
                    "chapter": "Unit 1: Family life",
                    "topics": [
                        {"topic": "Vocabulary: Family life", "lessons": ["Household chores", "Family routines"], "periods": 3},
                        {"topic": "Grammar: Present tenses", "lessons": ["Present simple", "Present continuous"], "periods": 2},
                    ],
                    "knowledge": "Vocabulary about family, present tenses",
                    "skills": "Talking about family life and routines",
                },
            ],
        },
    },
}


def get_sample_curriculum(subject: str = None, grade: int = None) -> list:
    """
    Convert sample curriculum data to list of dict for database import.

    Args:
        subject: Subject key (toan, ly, hoa, etc.)
        grade: Grade level (10, 11, 12)

    Returns:
        List of curriculum items ready for database insertion
    """
    items = []

    subjects_to_process = [subject] if subject else SAMPLE_CURRICULUM_DATA.keys()

    for subj in subjects_to_process:
        if subj not in SAMPLE_CURRICULUM_DATA:
            continue

        subj_data = SAMPLE_CURRICULUM_DATA[subj]
        grades_to_process = [grade] if grade else subj_data.get("grades", {}).keys()

        for g in grades_to_process:
            chapters = subj_data.get("grades", {}).get(g, [])

            for chapter_data in chapters:
                chapter_name = chapter_data.get("chapter", "")
                knowledge = chapter_data.get("knowledge", "")
                skills = chapter_data.get("skills", "")

                for topic_data in chapter_data.get("topics", []):
                    topic_name = topic_data.get("topic", "")
                    lessons = topic_data.get("lessons", [])
                    periods = topic_data.get("periods", 0)

                    for lesson in lessons:
                        items.append({
                            "subject": subj,
                            "grade": g,
                            "chapter": chapter_name,
                            "topic": topic_name,
                            "lesson": lesson,
                            "knowledge": knowledge,
                            "skills": skills,
                            "periods": periods // len(lessons) if lessons else periods,
                            "source_document": "Chương trình Giáo dục Phổ thông 2018",
                        })

    return items
