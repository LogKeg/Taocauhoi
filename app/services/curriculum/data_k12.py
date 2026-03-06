"""
Chương trình Giáo dục Phổ thông 2018 - Lớp 1 đến Lớp 9.
Bao gồm: Toán, Tiếng Việt/Ngữ văn, Tiếng Anh, KHTN, Lịch sử & Địa lý.
"""

CURRICULUM_1_TO_9 = {
    # =========================================================================
    # TOÁN LỚP 1-9
    # =========================================================================
    "toan": {
        "name": "Toán học",
        "grades": {
            1: [
                {
                    "chapter": "Các số đến 10",
                    "topics": [
                        {"topic": "Các số đến 10", "lessons": ["Đếm, đọc, viết các số đến 10", "So sánh các số đến 10"], "periods": 8},
                        {"topic": "Phép cộng, phép trừ trong phạm vi 10", "lessons": ["Phép cộng trong phạm vi 10", "Phép trừ trong phạm vi 10"], "periods": 10},
                    ],
                    "knowledge": "Đếm, đọc, viết, so sánh các số đến 10; phép cộng, trừ trong phạm vi 10",
                    "skills": "Đếm, so sánh, thực hiện phép cộng trừ đơn giản",
                },
                {
                    "chapter": "Các số đến 20",
                    "topics": [
                        {"topic": "Các số đến 20", "lessons": ["Đếm, đọc, viết các số từ 11 đến 20"], "periods": 4},
                        {"topic": "Phép cộng, phép trừ trong phạm vi 20", "lessons": ["Phép cộng không nhớ", "Phép cộng có nhớ", "Phép trừ không nhớ", "Phép trừ có nhớ"], "periods": 12},
                    ],
                    "knowledge": "Đếm, đọc, viết, so sánh các số đến 20; phép cộng, trừ trong phạm vi 20",
                    "skills": "Cộng trừ trong phạm vi 20, giải bài toán đơn giản",
                },
                {
                    "chapter": "Các số đến 100",
                    "topics": [
                        {"topic": "Các số đến 100", "lessons": ["Đếm, đọc, viết các số đến 100", "So sánh các số đến 100"], "periods": 6},
                        {"topic": "Phép cộng, phép trừ trong phạm vi 100", "lessons": ["Phép cộng không nhớ", "Phép trừ không nhớ"], "periods": 8},
                    ],
                    "knowledge": "Đếm, đọc, viết, so sánh các số đến 100",
                    "skills": "Cộng trừ không nhớ trong phạm vi 100",
                },
                {
                    "chapter": "Hình học và Đo lường",
                    "topics": [
                        {"topic": "Hình học trực quan", "lessons": ["Hình vuông, hình tròn, hình tam giác, hình chữ nhật"], "periods": 3},
                        {"topic": "Đo lường", "lessons": ["Đo độ dài: xăng-ti-mét", "Đọc giờ đúng trên đồng hồ"], "periods": 3},
                    ],
                    "knowledge": "Nhận biết các hình cơ bản, đo độ dài bằng cm, đọc giờ",
                    "skills": "Nhận dạng hình, đo độ dài, đọc giờ",
                },
            ],
            2: [
                {
                    "chapter": "Ôn tập và bổ sung phép cộng, trừ trong phạm vi 100",
                    "topics": [
                        {"topic": "Phép cộng, trừ có nhớ trong phạm vi 100", "lessons": ["Phép cộng có nhớ", "Phép trừ có nhớ"], "periods": 10},
                    ],
                    "knowledge": "Phép cộng, trừ có nhớ trong phạm vi 100",
                    "skills": "Cộng trừ thành thạo trong phạm vi 100",
                },
                {
                    "chapter": "Phép nhân, phép chia",
                    "topics": [
                        {"topic": "Phép nhân", "lessons": ["Khái niệm phép nhân", "Bảng nhân 2, 3, 4, 5"], "periods": 10},
                        {"topic": "Phép chia", "lessons": ["Khái niệm phép chia", "Bảng chia 2, 3, 4, 5"], "periods": 10},
                    ],
                    "knowledge": "Khái niệm phép nhân, phép chia; bảng nhân chia 2, 3, 4, 5",
                    "skills": "Thực hiện phép nhân chia đơn giản, giải bài toán",
                },
                {
                    "chapter": "Các số đến 1000",
                    "topics": [
                        {"topic": "Các số đến 1000", "lessons": ["Đọc, viết các số có ba chữ số", "So sánh các số có ba chữ số"], "periods": 6},
                        {"topic": "Phép cộng, trừ trong phạm vi 1000", "lessons": ["Cộng trừ không nhớ", "Cộng trừ có nhớ"], "periods": 8},
                    ],
                    "knowledge": "Đọc, viết, so sánh các số đến 1000; cộng trừ trong phạm vi 1000",
                    "skills": "Cộng trừ trong phạm vi 1000",
                },
                {
                    "chapter": "Hình học và Đo lường",
                    "topics": [
                        {"topic": "Hình học trực quan", "lessons": ["Đường thẳng, đoạn thẳng, đường gấp khúc", "Hình tứ giác"], "periods": 4},
                        {"topic": "Đo lường", "lessons": ["Đo độ dài: m, km", "Đo khối lượng: kg", "Đo thời gian: giờ, phút"], "periods": 5},
                    ],
                    "knowledge": "Nhận biết hình học cơ bản, đơn vị đo lường",
                    "skills": "Đo và tính toán với các đơn vị đo",
                },
            ],
            3: [
                {
                    "chapter": "Phép nhân và phép chia trong phạm vi 1000",
                    "topics": [
                        {"topic": "Bảng nhân, chia 6, 7, 8, 9", "lessons": ["Bảng nhân 6, 7", "Bảng nhân 8, 9", "Bảng chia 6, 7, 8, 9"], "periods": 12},
                        {"topic": "Nhân, chia số có hai, ba chữ số", "lessons": ["Nhân số có hai chữ số với số có một chữ số", "Chia số có hai, ba chữ số cho số có một chữ số"], "periods": 10},
                    ],
                    "knowledge": "Bảng nhân chia đến 9; nhân chia số có nhiều chữ số",
                    "skills": "Thực hiện phép nhân chia, giải bài toán có lời văn",
                },
                {
                    "chapter": "Các số đến 10 000",
                    "topics": [
                        {"topic": "Các số đến 10 000", "lessons": ["Đọc, viết, so sánh các số đến 10 000"], "periods": 5},
                        {"topic": "Phép cộng, trừ trong phạm vi 10 000", "lessons": ["Cộng trừ các số trong phạm vi 10 000"], "periods": 6},
                    ],
                    "knowledge": "Các số đến 10 000; cộng trừ trong phạm vi 10 000",
                    "skills": "Đọc, viết, so sánh, cộng trừ các số đến 10 000",
                },
                {
                    "chapter": "Phân số (bước đầu)",
                    "topics": [
                        {"topic": "Làm quen với phân số", "lessons": ["Khái niệm phân số đơn giản", "So sánh phân số cùng mẫu"], "periods": 4},
                    ],
                    "knowledge": "Bước đầu làm quen với phân số",
                    "skills": "Đọc, viết, so sánh phân số đơn giản",
                },
                {
                    "chapter": "Hình học và Đo lường",
                    "topics": [
                        {"topic": "Hình học", "lessons": ["Góc vuông, góc không vuông", "Hình chữ nhật, hình vuông, chu vi"], "periods": 5},
                        {"topic": "Đo lường", "lessons": ["Đo diện tích: cm², dm²", "Đo thời gian: ngày, tháng, năm"], "periods": 4},
                    ],
                    "knowledge": "Nhận biết góc, tính chu vi, diện tích; đơn vị đo",
                    "skills": "Tính chu vi hình chữ nhật, hình vuông; đổi đơn vị đo",
                },
            ],
            4: [
                {
                    "chapter": "Số tự nhiên",
                    "topics": [
                        {"topic": "Các số đến lớp triệu", "lessons": ["Đọc, viết, so sánh các số đến lớp triệu"], "periods": 5},
                        {"topic": "Phép tính với số tự nhiên", "lessons": ["Cộng, trừ, nhân, chia số tự nhiên", "Tính chất của phép tính"], "periods": 10},
                        {"topic": "Dấu hiệu chia hết", "lessons": ["Dấu hiệu chia hết cho 2, 5, 3, 9"], "periods": 4},
                    ],
                    "knowledge": "Số tự nhiên đến lớp triệu; bốn phép tính; dấu hiệu chia hết",
                    "skills": "Tính toán với số lớn, vận dụng dấu hiệu chia hết",
                },
                {
                    "chapter": "Phân số",
                    "topics": [
                        {"topic": "Phân số", "lessons": ["Phân số và phân số bằng nhau", "Rút gọn, quy đồng mẫu số"], "periods": 5},
                        {"topic": "Phép tính phân số", "lessons": ["Cộng, trừ phân số", "Nhân, chia phân số"], "periods": 8},
                    ],
                    "knowledge": "Phân số: khái niệm, rút gọn, quy đồng; bốn phép tính với phân số",
                    "skills": "Tính toán với phân số, giải bài toán có phân số",
                },
                {
                    "chapter": "Hình học và Đo lường",
                    "topics": [
                        {"topic": "Hình học", "lessons": ["Góc nhọn, góc tù, góc bẹt", "Hai đường thẳng vuông góc, song song", "Hình bình hành, hình thoi"], "periods": 6},
                        {"topic": "Diện tích", "lessons": ["Diện tích hình bình hành, hình thoi"], "periods": 3},
                    ],
                    "knowledge": "Các loại góc, đường vuông góc/song song, hình bình hành, hình thoi",
                    "skills": "Nhận dạng hình, tính diện tích",
                },
                {
                    "chapter": "Thống kê và Xác suất",
                    "topics": [
                        {"topic": "Thống kê", "lessons": ["Biểu đồ cột, biểu đồ cột kép"], "periods": 3},
                    ],
                    "knowledge": "Đọc và phân tích biểu đồ cột",
                    "skills": "Đọc, vẽ biểu đồ cột đơn giản",
                },
            ],
            5: [
                {
                    "chapter": "Số thập phân",
                    "topics": [
                        {"topic": "Khái niệm số thập phân", "lessons": ["Khái niệm số thập phân", "So sánh số thập phân"], "periods": 5},
                        {"topic": "Phép tính số thập phân", "lessons": ["Cộng, trừ số thập phân", "Nhân, chia số thập phân"], "periods": 10},
                    ],
                    "knowledge": "Số thập phân: khái niệm, so sánh, bốn phép tính",
                    "skills": "Tính toán với số thập phân, giải bài toán thực tế",
                },
                {
                    "chapter": "Tỷ số phần trăm",
                    "topics": [
                        {"topic": "Tỷ số phần trăm", "lessons": ["Khái niệm tỷ số phần trăm", "Giải toán về tỷ số phần trăm"], "periods": 6},
                    ],
                    "knowledge": "Tỷ số phần trăm và ứng dụng",
                    "skills": "Tính tỷ số phần trăm, giải bài toán thực tế",
                },
                {
                    "chapter": "Hình học",
                    "topics": [
                        {"topic": "Hình tam giác, hình thang", "lessons": ["Diện tích hình tam giác", "Hình thang, diện tích hình thang"], "periods": 5},
                        {"topic": "Hình tròn", "lessons": ["Hình tròn, đường tròn", "Chu vi và diện tích hình tròn"], "periods": 4},
                        {"topic": "Hình hộp chữ nhật, hình lập phương", "lessons": ["Hình hộp chữ nhật", "Hình lập phương", "Thể tích hình hộp, hình lập phương"], "periods": 5},
                    ],
                    "knowledge": "Diện tích tam giác, hình thang, hình tròn; thể tích hình hộp",
                    "skills": "Tính diện tích, thể tích các hình cơ bản",
                },
                {
                    "chapter": "Thống kê và Xác suất",
                    "topics": [
                        {"topic": "Thống kê", "lessons": ["Biểu đồ hình quạt", "Số trung bình cộng"], "periods": 3},
                        {"topic": "Xác suất thực nghiệm", "lessons": ["Khả năng xảy ra của một sự kiện"], "periods": 2},
                    ],
                    "knowledge": "Biểu đồ hình quạt, trung bình cộng, xác suất thực nghiệm",
                    "skills": "Đọc biểu đồ, tính trung bình cộng",
                },
            ],
            6: [
                {
                    "chapter": "Chương 1: Số tự nhiên",
                    "topics": [
                        {"topic": "Tập hợp, phần tử", "lessons": ["Tập hợp", "Tập hợp số tự nhiên"], "periods": 3},
                        {"topic": "Chia hết và số nguyên tố", "lessons": ["Phép chia hết", "Ước và bội", "Số nguyên tố", "ƯCLN và BCNN"], "periods": 8},
                    ],
                    "knowledge": "Tập hợp, phép chia hết, ước bội, số nguyên tố, ƯCLN, BCNN",
                    "skills": "Tìm ước bội, ƯCLN, BCNN; phân tích thừa số nguyên tố",
                },
                {
                    "chapter": "Chương 2: Số nguyên",
                    "topics": [
                        {"topic": "Số nguyên", "lessons": ["Số nguyên âm", "Tập hợp số nguyên", "Thứ tự trên tập số nguyên"], "periods": 4},
                        {"topic": "Phép tính với số nguyên", "lessons": ["Phép cộng, trừ số nguyên", "Phép nhân, chia số nguyên"], "periods": 6},
                    ],
                    "knowledge": "Số nguyên: khái niệm, so sánh, bốn phép tính",
                    "skills": "Tính toán với số nguyên, giải bài toán",
                },
                {
                    "chapter": "Chương 3: Phân số",
                    "topics": [
                        {"topic": "Phân số", "lessons": ["Phân số với tử và mẫu là số nguyên", "So sánh phân số"], "periods": 3},
                        {"topic": "Phép tính phân số", "lessons": ["Cộng, trừ phân số", "Nhân, chia phân số", "Hỗn số, số thập phân"], "periods": 7},
                    ],
                    "knowledge": "Phân số với số nguyên; bốn phép tính; hỗn số, số thập phân",
                    "skills": "Tính toán phân số, hỗn số, số thập phân",
                },
                {
                    "chapter": "Chương 4: Hình học trực quan",
                    "topics": [
                        {"topic": "Hình phẳng", "lessons": ["Tam giác đều, hình vuông, lục giác đều", "Hình chữ nhật, hình thoi, hình bình hành, hình thang cân"], "periods": 5},
                        {"topic": "Hình khối", "lessons": ["Hình hộp chữ nhật, hình lập phương", "Hình lăng trụ đứng, hình chóp đều"], "periods": 4},
                    ],
                    "knowledge": "Nhận biết các hình phẳng và hình khối trong thực tế",
                    "skills": "Tính diện tích, thể tích các hình cơ bản",
                },
                {
                    "chapter": "Chương 5: Thống kê và Xác suất",
                    "topics": [
                        {"topic": "Thu thập, phân loại dữ liệu", "lessons": ["Thu thập, phân loại dữ liệu", "Biểu đồ tranh, biểu đồ cột"], "periods": 4},
                        {"topic": "Xác suất thực nghiệm", "lessons": ["Phép thử nghiệm, xác suất thực nghiệm"], "periods": 2},
                    ],
                    "knowledge": "Thu thập dữ liệu, biểu đồ, xác suất thực nghiệm",
                    "skills": "Lập bảng, vẽ biểu đồ, tính xác suất thực nghiệm",
                },
            ],
            7: [
                {
                    "chapter": "Chương 1: Số hữu tỷ",
                    "topics": [
                        {"topic": "Số hữu tỷ", "lessons": ["Tập hợp số hữu tỷ", "Cộng, trừ, nhân, chia số hữu tỷ"], "periods": 6},
                        {"topic": "Lũy thừa với số mũ tự nhiên", "lessons": ["Lũy thừa với số mũ tự nhiên"], "periods": 3},
                        {"topic": "Tỷ lệ thức", "lessons": ["Tỷ lệ thức", "Tính chất dãy tỷ số bằng nhau"], "periods": 4},
                    ],
                    "knowledge": "Số hữu tỷ, lũy thừa, tỷ lệ thức",
                    "skills": "Tính toán với số hữu tỷ, giải bài toán tỷ lệ",
                },
                {
                    "chapter": "Chương 2: Số thực",
                    "topics": [
                        {"topic": "Số vô tỷ và số thực", "lessons": ["Số vô tỷ", "Số thực", "Căn bậc hai"], "periods": 5},
                    ],
                    "knowledge": "Số vô tỷ, số thực, căn bậc hai",
                    "skills": "Tính căn bậc hai, so sánh số thực",
                },
                {
                    "chapter": "Chương 3: Góc và đường thẳng song song",
                    "topics": [
                        {"topic": "Góc", "lessons": ["Các góc ở vị trí đặc biệt", "Tia phân giác"], "periods": 3},
                        {"topic": "Đường thẳng song song", "lessons": ["Hai đường thẳng song song", "Tiên đề Euclid"], "periods": 4},
                    ],
                    "knowledge": "Các loại góc, đường thẳng song song, tiên đề Euclid",
                    "skills": "Tính góc, chứng minh song song",
                },
                {
                    "chapter": "Chương 4: Tam giác",
                    "topics": [
                        {"topic": "Tam giác bằng nhau", "lessons": ["Khái niệm tam giác bằng nhau", "Trường hợp bằng nhau c.c.c, c.g.c, g.c.g"], "periods": 8},
                        {"topic": "Tam giác cân, tam giác đều", "lessons": ["Tam giác cân", "Tam giác đều"], "periods": 3},
                        {"topic": "Quan hệ giữa các yếu tố trong tam giác", "lessons": ["Quan hệ góc - cạnh", "Bất đẳng thức tam giác"], "periods": 4},
                    ],
                    "knowledge": "Tam giác bằng nhau, tam giác cân/đều, bất đẳng thức tam giác",
                    "skills": "Chứng minh tam giác bằng nhau, tính toán trong tam giác",
                },
                {
                    "chapter": "Chương 5: Thống kê",
                    "topics": [
                        {"topic": "Bảng tần số, tần số tương đối", "lessons": ["Bảng tần số", "Biểu đồ đoạn thẳng", "Số trung bình cộng, mốt"], "periods": 5},
                    ],
                    "knowledge": "Bảng tần số, biểu đồ, số trung bình cộng, mốt",
                    "skills": "Lập bảng tần số, tính trung bình cộng, mốt",
                },
            ],
            8: [
                {
                    "chapter": "Chương 1: Biểu thức đại số",
                    "topics": [
                        {"topic": "Đơn thức, đa thức", "lessons": ["Đơn thức", "Đa thức", "Cộng, trừ đa thức"], "periods": 5},
                        {"topic": "Hằng đẳng thức đáng nhớ", "lessons": ["Hằng đẳng thức đáng nhớ", "Phân tích đa thức thành nhân tử"], "periods": 7},
                    ],
                    "knowledge": "Đơn thức, đa thức, hằng đẳng thức, phân tích nhân tử",
                    "skills": "Tính toán đa thức, phân tích đa thức thành nhân tử",
                },
                {
                    "chapter": "Chương 2: Phân thức đại số",
                    "topics": [
                        {"topic": "Phân thức đại số", "lessons": ["Phân thức đại số", "Tính chất cơ bản"], "periods": 3},
                        {"topic": "Phép tính phân thức", "lessons": ["Cộng, trừ phân thức", "Nhân, chia phân thức"], "periods": 5},
                    ],
                    "knowledge": "Phân thức đại số, bốn phép tính",
                    "skills": "Rút gọn, tính toán phân thức đại số",
                },
                {
                    "chapter": "Chương 3: Phương trình bậc nhất một ẩn",
                    "topics": [
                        {"topic": "Phương trình bậc nhất", "lessons": ["Phương trình bậc nhất một ẩn", "Phương trình tích, phương trình chứa ẩn ở mẫu"], "periods": 6},
                        {"topic": "Giải bài toán bằng cách lập phương trình", "lessons": ["Giải bài toán bằng cách lập phương trình"], "periods": 4},
                    ],
                    "knowledge": "Phương trình bậc nhất một ẩn và ứng dụng",
                    "skills": "Giải phương trình, lập phương trình giải bài toán",
                },
                {
                    "chapter": "Chương 4: Tứ giác",
                    "topics": [
                        {"topic": "Tứ giác", "lessons": ["Tứ giác", "Hình thang, hình thang cân"], "periods": 4},
                        {"topic": "Hình bình hành và đặc biệt", "lessons": ["Hình bình hành", "Hình chữ nhật", "Hình thoi", "Hình vuông"], "periods": 7},
                    ],
                    "knowledge": "Tứ giác, hình thang, hình bình hành, hình đặc biệt",
                    "skills": "Chứng minh tính chất tứ giác, tính diện tích",
                },
                {
                    "chapter": "Chương 5: Định lý Pythagore và các dạng tam giác đặc biệt",
                    "topics": [
                        {"topic": "Định lý Pythagore", "lessons": ["Định lý Pythagore", "Định lý Pythagore đảo"], "periods": 4},
                        {"topic": "Tam giác đồng dạng", "lessons": ["Tam giác đồng dạng", "Trường hợp đồng dạng"], "periods": 6},
                    ],
                    "knowledge": "Định lý Pythagore, tam giác đồng dạng",
                    "skills": "Áp dụng Pythagore, chứng minh tam giác đồng dạng",
                },
                {
                    "chapter": "Chương 6: Thống kê",
                    "topics": [
                        {"topic": "Thống kê", "lessons": ["Bảng tần số ghép nhóm", "Biểu đồ", "Trung bình, trung vị"], "periods": 4},
                    ],
                    "knowledge": "Bảng tần số ghép nhóm, biểu đồ, số đặc trưng",
                    "skills": "Xử lý dữ liệu, vẽ biểu đồ",
                },
            ],
            9: [
                {
                    "chapter": "Chương 1: Căn bậc hai, căn bậc ba",
                    "topics": [
                        {"topic": "Căn bậc hai", "lessons": ["Căn bậc hai", "Biến đổi đơn giản biểu thức chứa căn"], "periods": 6},
                        {"topic": "Căn bậc ba", "lessons": ["Căn bậc ba"], "periods": 2},
                    ],
                    "knowledge": "Căn bậc hai, căn bậc ba; biến đổi biểu thức chứa căn",
                    "skills": "Tính toán và biến đổi biểu thức chứa căn",
                },
                {
                    "chapter": "Chương 2: Hàm số bậc nhất",
                    "topics": [
                        {"topic": "Hàm số bậc nhất", "lessons": ["Nhắc lại hàm số", "Hàm số bậc nhất y = ax + b"], "periods": 4},
                        {"topic": "Đồ thị hàm số bậc nhất", "lessons": ["Đồ thị hàm số y = ax + b", "Hệ số góc của đường thẳng"], "periods": 4},
                    ],
                    "knowledge": "Hàm số bậc nhất, đồ thị, hệ số góc",
                    "skills": "Vẽ đồ thị, xác định hệ số góc",
                },
                {
                    "chapter": "Chương 3: Hệ phương trình bậc nhất hai ẩn",
                    "topics": [
                        {"topic": "Hệ phương trình", "lessons": ["Hệ hai phương trình bậc nhất hai ẩn", "Giải hệ bằng phương pháp thế, cộng đại số"], "periods": 6},
                        {"topic": "Giải bài toán bằng hệ phương trình", "lessons": ["Giải bài toán bằng cách lập hệ phương trình"], "periods": 4},
                    ],
                    "knowledge": "Hệ phương trình bậc nhất hai ẩn; phương pháp giải",
                    "skills": "Giải hệ phương trình, lập hệ phương trình giải bài toán",
                },
                {
                    "chapter": "Chương 4: Hàm số bậc hai y = ax²",
                    "topics": [
                        {"topic": "Hàm số y = ax²", "lessons": ["Hàm số y = ax²", "Đồ thị hàm số y = ax²"], "periods": 3},
                        {"topic": "Phương trình bậc hai một ẩn", "lessons": ["Phương trình bậc hai", "Công thức nghiệm", "Hệ thức Viète"], "periods": 7},
                    ],
                    "knowledge": "Hàm số y = ax²; phương trình bậc hai; hệ thức Viète",
                    "skills": "Vẽ đồ thị, giải phương trình bậc hai",
                },
                {
                    "chapter": "Chương 5: Hệ thức lượng trong tam giác vuông",
                    "topics": [
                        {"topic": "Hệ thức lượng trong tam giác vuông", "lessons": ["Tỷ số lượng giác góc nhọn", "Hệ thức lượng trong tam giác vuông"], "periods": 6},
                        {"topic": "Đường tròn", "lessons": ["Đường tròn", "Vị trí tương đối đường thẳng và đường tròn", "Góc nội tiếp, góc tâm"], "periods": 8},
                    ],
                    "knowledge": "Tỷ số lượng giác, hệ thức lượng; đường tròn",
                    "skills": "Tính tỷ số lượng giác, giải tam giác vuông; tính toán với đường tròn",
                },
                {
                    "chapter": "Chương 6: Thống kê và Xác suất",
                    "topics": [
                        {"topic": "Thống kê", "lessons": ["Phương sai và độ lệch chuẩn"], "periods": 3},
                        {"topic": "Xác suất", "lessons": ["Không gian mẫu và biến cố", "Xác suất của biến cố"], "periods": 4},
                    ],
                    "knowledge": "Phương sai, độ lệch chuẩn; xác suất",
                    "skills": "Tính phương sai, xác suất",
                },
            ],
        },
    },
    # =========================================================================
    # TIẾNG VIỆT LỚP 1-5, NGỮ VĂN LỚP 6-9
    # =========================================================================
    "van": {
        "name": "Tiếng Việt / Ngữ văn",
        "grades": {
            1: [
                {
                    "chapter": "Học vần",
                    "topics": [
                        {"topic": "Âm và chữ cái", "lessons": ["Các chữ cái", "Các vần đơn giản"], "periods": 20},
                        {"topic": "Ghép vần, đọc từ", "lessons": ["Ghép vần", "Đọc từ ngữ, câu ngắn"], "periods": 15},
                    ],
                    "knowledge": "Nhận biết chữ cái, ghép vần, đọc từ và câu ngắn",
                    "skills": "Đọc, viết chữ cái và vần; đọc từ ngữ đơn giản",
                },
                {
                    "chapter": "Tập đọc và Kể chuyện",
                    "topics": [
                        {"topic": "Tập đọc", "lessons": ["Đọc bài văn ngắn"], "periods": 10},
                        {"topic": "Tập viết", "lessons": ["Viết chữ hoa, chữ thường", "Chính tả nghe viết"], "periods": 10},
                    ],
                    "knowledge": "Đọc trơn, viết đúng chính tả",
                    "skills": "Đọc hiểu bài ngắn, viết đúng nét chữ",
                },
            ],
            2: [
                {
                    "chapter": "Tập đọc",
                    "topics": [
                        {"topic": "Đọc hiểu văn bản", "lessons": ["Đọc văn bản truyện, thơ", "Trả lời câu hỏi về nội dung"], "periods": 15},
                    ],
                    "knowledge": "Đọc hiểu bài văn, thơ ngắn",
                    "skills": "Đọc trôi chảy, hiểu nội dung chính",
                },
                {
                    "chapter": "Luyện từ và câu",
                    "topics": [
                        {"topic": "Từ ngữ", "lessons": ["Mở rộng vốn từ theo chủ đề"], "periods": 6},
                        {"topic": "Câu", "lessons": ["Câu giới thiệu, câu nêu hoạt động", "Dấu chấm, dấu chấm hỏi, dấu chấm than"], "periods": 6},
                    ],
                    "knowledge": "Mở rộng vốn từ, nhận biết các kiểu câu và dấu câu",
                    "skills": "Dùng từ đúng, đặt câu đúng ngữ pháp",
                },
                {
                    "chapter": "Tập làm văn và Chính tả",
                    "topics": [
                        {"topic": "Tập làm văn", "lessons": ["Viết đoạn văn ngắn theo chủ đề"], "periods": 6},
                        {"topic": "Chính tả", "lessons": ["Nghe viết, nhớ viết"], "periods": 6},
                    ],
                    "knowledge": "Viết đoạn văn ngắn, viết đúng chính tả",
                    "skills": "Viết đoạn 3-5 câu, chính tả đúng",
                },
            ],
            3: [
                {
                    "chapter": "Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Văn bản văn học", "lessons": ["Truyện, thơ, văn miêu tả"], "periods": 12},
                        {"topic": "Văn bản thông tin", "lessons": ["Bài đọc thông tin khoa học, đời sống"], "periods": 6},
                    ],
                    "knowledge": "Đọc hiểu truyện, thơ, văn bản thông tin",
                    "skills": "Đọc diễn cảm, tóm tắt nội dung, nhận biết ý chính",
                },
                {
                    "chapter": "Luyện từ và câu",
                    "topics": [
                        {"topic": "Từ loại", "lessons": ["Danh từ, động từ, tính từ"], "periods": 5},
                        {"topic": "Câu", "lessons": ["Câu đơn, mở rộng câu", "So sánh, nhân hóa"], "periods": 5},
                    ],
                    "knowledge": "Từ loại cơ bản, câu đơn, biện pháp tu từ đơn giản",
                    "skills": "Phân loại từ, đặt câu, sử dụng so sánh/nhân hóa",
                },
                {
                    "chapter": "Tập làm văn",
                    "topics": [
                        {"topic": "Văn kể chuyện", "lessons": ["Kể lại câu chuyện đã đọc/nghe"], "periods": 5},
                        {"topic": "Văn miêu tả", "lessons": ["Tả đồ vật, con vật, cây cối"], "periods": 5},
                    ],
                    "knowledge": "Viết văn kể chuyện, miêu tả đơn giản",
                    "skills": "Viết bài văn kể chuyện, miêu tả ngắn",
                },
            ],
            4: [
                {
                    "chapter": "Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Văn bản văn học", "lessons": ["Truyện, thơ Việt Nam và nước ngoài", "Kịch bản ngắn"], "periods": 12},
                        {"topic": "Văn bản thông tin", "lessons": ["Văn bản giới thiệu, hướng dẫn, báo cáo"], "periods": 6},
                    ],
                    "knowledge": "Đọc hiểu các thể loại văn học, văn bản thông tin",
                    "skills": "Phân tích nội dung, nhân vật, ý nghĩa",
                },
                {
                    "chapter": "Luyện từ và câu",
                    "topics": [
                        {"topic": "Cấu tạo từ", "lessons": ["Từ ghép, từ láy", "Thành ngữ, tục ngữ"], "periods": 5},
                        {"topic": "Câu ghép đơn giản", "lessons": ["Câu ghép", "Trạng ngữ, định ngữ"], "periods": 5},
                    ],
                    "knowledge": "Cấu tạo từ, từ ghép/láy, câu ghép, trạng ngữ",
                    "skills": "Phân tích cấu tạo từ câu, đặt câu ghép",
                },
                {
                    "chapter": "Tập làm văn",
                    "topics": [
                        {"topic": "Văn miêu tả", "lessons": ["Tả người, tả cảnh"], "periods": 6},
                        {"topic": "Văn kể chuyện", "lessons": ["Kể chuyện sáng tạo"], "periods": 4},
                        {"topic": "Viết thư, báo cáo", "lessons": ["Viết thư", "Viết báo cáo ngắn"], "periods": 3},
                    ],
                    "knowledge": "Viết văn miêu tả, kể chuyện, thư, báo cáo",
                    "skills": "Viết bài văn có bố cục rõ ràng",
                },
            ],
            5: [
                {
                    "chapter": "Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Văn bản văn học", "lessons": ["Truyện ngắn, thơ, ký", "Tác phẩm về quê hương, đất nước"], "periods": 12},
                        {"topic": "Văn bản thông tin", "lessons": ["Văn bản thuyết minh, nghị luận đơn giản"], "periods": 6},
                    ],
                    "knowledge": "Đọc hiểu truyện, thơ, ký; văn bản thuyết minh, nghị luận",
                    "skills": "Nhận biết đặc điểm thể loại, phân tích sâu nội dung",
                },
                {
                    "chapter": "Luyện từ và câu",
                    "topics": [
                        {"topic": "Từ đồng nghĩa, trái nghĩa, đồng âm", "lessons": ["Từ đồng nghĩa, trái nghĩa", "Từ đồng âm, từ nhiều nghĩa"], "periods": 5},
                        {"topic": "Câu phức, liên kết câu", "lessons": ["Nối câu bằng quan hệ từ", "Liên kết câu trong đoạn văn"], "periods": 5},
                    ],
                    "knowledge": "Quan hệ nghĩa của từ, câu phức, liên kết câu",
                    "skills": "Sử dụng từ chính xác, viết đoạn văn liên kết",
                },
                {
                    "chapter": "Tập làm văn",
                    "topics": [
                        {"topic": "Văn miêu tả", "lessons": ["Tả người, tả cảnh sinh hoạt"], "periods": 6},
                        {"topic": "Văn nghị luận đơn giản", "lessons": ["Trình bày ý kiến, bảo vệ quan điểm"], "periods": 4},
                    ],
                    "knowledge": "Viết văn miêu tả, nghị luận đơn giản",
                    "skills": "Viết bài văn hoàn chỉnh, trình bày quan điểm",
                },
            ],
            6: [
                {
                    "chapter": "Truyện và tiểu thuyết",
                    "topics": [
                        {"topic": "Truyền thuyết, cổ tích", "lessons": ["Con Rồng cháu Tiên", "Thánh Gióng", "Sọ Dừa"], "periods": 6},
                        {"topic": "Truyện ngắn", "lessons": ["Bức tranh của em gái tôi", "Bài học đường đời đầu tiên"], "periods": 5},
                    ],
                    "knowledge": "Đặc trưng truyện dân gian và truyện ngắn",
                    "skills": "Phân tích nhân vật, cốt truyện, ý nghĩa",
                },
                {
                    "chapter": "Thơ",
                    "topics": [
                        {"topic": "Thơ lục bát", "lessons": ["Thể thơ lục bát", "Ca dao, dân ca"], "periods": 4},
                        {"topic": "Thơ hiện đại", "lessons": ["Đêm nay Bác không ngủ", "Lượm"], "periods": 4},
                    ],
                    "knowledge": "Thơ lục bát, ca dao; thơ hiện đại",
                    "skills": "Phân tích thơ, nhận biết thể thơ",
                },
                {
                    "chapter": "Văn bản thông tin và nghị luận",
                    "topics": [
                        {"topic": "Văn bản thông tin", "lessons": ["Đọc hiểu văn bản thông tin"], "periods": 3},
                        {"topic": "Nghị luận đơn giản", "lessons": ["Trình bày ý kiến về hiện tượng đời sống"], "periods": 3},
                    ],
                    "knowledge": "Đọc hiểu văn bản thông tin, viết nghị luận đơn giản",
                    "skills": "Tóm tắt thông tin, nêu ý kiến cá nhân",
                },
                {
                    "chapter": "Tiếng Việt",
                    "topics": [
                        {"topic": "Từ vựng", "lessons": ["Từ mượn, nghĩa của từ", "Từ Hán Việt"], "periods": 3},
                        {"topic": "Ngữ pháp", "lessons": ["Các thành phần câu", "Câu trần thuật đơn"], "periods": 4},
                    ],
                    "knowledge": "Từ mượn, từ Hán Việt, thành phần câu",
                    "skills": "Sử dụng từ chính xác, phân tích câu",
                },
            ],
            7: [
                {
                    "chapter": "Văn học dân gian",
                    "topics": [
                        {"topic": "Tục ngữ, ca dao", "lessons": ["Tục ngữ về thiên nhiên, lao động", "Ca dao về tình cảm gia đình, quê hương"], "periods": 5},
                    ],
                    "knowledge": "Đặc trưng tục ngữ, ca dao",
                    "skills": "Phân tích ý nghĩa tục ngữ, ca dao",
                },
                {
                    "chapter": "Truyện và ký",
                    "topics": [
                        {"topic": "Truyện ngắn", "lessons": ["Cuộc chia tay của những con búp bê", "Sống chết mặc bay"], "periods": 5},
                        {"topic": "Ký, tùy bút", "lessons": ["Một thứ quà của lúa non: Cốm", "Mùa xuân của tôi"], "periods": 4},
                    ],
                    "knowledge": "Truyện ngắn, ký, tùy bút",
                    "skills": "Phân tích tác phẩm tự sự, biểu cảm",
                },
                {
                    "chapter": "Văn bản nghị luận",
                    "topics": [
                        {"topic": "Nghị luận xã hội", "lessons": ["Tinh thần yêu nước của nhân dân ta", "Sự giàu đẹp của tiếng Việt"], "periods": 5},
                        {"topic": "Viết văn nghị luận", "lessons": ["Nghị luận chứng minh", "Nghị luận giải thích"], "periods": 5},
                    ],
                    "knowledge": "Văn bản nghị luận: chứng minh, giải thích",
                    "skills": "Viết bài nghị luận chứng minh, giải thích",
                },
                {
                    "chapter": "Tiếng Việt",
                    "topics": [
                        {"topic": "Từ vựng", "lessons": ["Từ ghép, từ láy", "Từ đồng nghĩa, trái nghĩa, đồng âm"], "periods": 4},
                        {"topic": "Ngữ pháp", "lessons": ["Rút gọn câu, câu đặc biệt", "Dùng cụm chủ vị để mở rộng câu"], "periods": 4},
                    ],
                    "knowledge": "Từ ghép/láy, quan hệ nghĩa; câu rút gọn, mở rộng câu",
                    "skills": "Sử dụng từ linh hoạt, viết câu đa dạng",
                },
            ],
            8: [
                {
                    "chapter": "Văn học trung đại",
                    "topics": [
                        {"topic": "Văn xuôi trung đại", "lessons": ["Tôi đi học", "Trong lòng mẹ", "Lão Hạc"], "periods": 6},
                        {"topic": "Thơ trung đại", "lessons": ["Vào nhà ngục Quảng Đông", "Đập đá ở Côn Lôn", "Muốn làm thằng Cuội"], "periods": 5},
                    ],
                    "knowledge": "Văn học trung đại Việt Nam: văn xuôi, thơ",
                    "skills": "Phân tích tác phẩm, nhân vật, bối cảnh lịch sử",
                },
                {
                    "chapter": "Văn học nước ngoài",
                    "topics": [
                        {"topic": "Truyện nước ngoài", "lessons": ["Cô bé bán diêm", "Chiếc lá cuối cùng"], "periods": 4},
                    ],
                    "knowledge": "Văn học nước ngoài tiêu biểu",
                    "skills": "So sánh, đối chiếu văn hóa qua tác phẩm",
                },
                {
                    "chapter": "Văn bản nghị luận",
                    "topics": [
                        {"topic": "Nghị luận văn học", "lessons": ["Phân tích tác phẩm văn học"], "periods": 5},
                        {"topic": "Nghị luận xã hội", "lessons": ["Nghị luận về vấn đề xã hội, đạo đức"], "periods": 5},
                    ],
                    "knowledge": "Nghị luận văn học và nghị luận xã hội",
                    "skills": "Viết bài nghị luận phân tích, bình luận",
                },
                {
                    "chapter": "Tiếng Việt",
                    "topics": [
                        {"topic": "Từ vựng", "lessons": ["Trường từ vựng", "Từ tượng hình, từ tượng thanh"], "periods": 3},
                        {"topic": "Ngữ pháp", "lessons": ["Câu ghép", "Câu nghi vấn, câu cầu khiến, câu cảm thán"], "periods": 5},
                    ],
                    "knowledge": "Trường từ vựng, từ tượng hình/thanh; câu ghép, kiểu câu",
                    "skills": "Phân tích và sử dụng đa dạng kiểu câu",
                },
            ],
            9: [
                {
                    "chapter": "Văn học trung đại Việt Nam",
                    "topics": [
                        {"topic": "Truyện Kiều", "lessons": ["Chị em Thúy Kiều", "Kiều ở lầu Ngưng Bích", "Thúy Kiều báo ân báo oán"], "periods": 8},
                        {"topic": "Thơ và văn nghị luận trung đại", "lessons": ["Hoàng Lê nhất thống chí", "Chuyện người con gái Nam Xương"], "periods": 5},
                    ],
                    "knowledge": "Truyện Kiều, văn học trung đại tiêu biểu",
                    "skills": "Phân tích sâu tác phẩm, nghệ thuật, tư tưởng",
                },
                {
                    "chapter": "Văn học hiện đại Việt Nam",
                    "topics": [
                        {"topic": "Truyện", "lessons": ["Làng", "Lặng lẽ Sa Pa", "Chiếc lược ngà"], "periods": 6},
                        {"topic": "Thơ", "lessons": ["Đồng chí", "Bài thơ về tiểu đội xe không kính", "Bếp lửa", "Ánh trăng"], "periods": 6},
                    ],
                    "knowledge": "Văn học hiện đại: truyện ngắn, thơ kháng chiến và đời sống",
                    "skills": "Phân tích tác phẩm trong bối cảnh lịch sử",
                },
                {
                    "chapter": "Nghị luận",
                    "topics": [
                        {"topic": "Nghị luận văn học", "lessons": ["Phân tích thơ", "Phân tích truyện ngắn"], "periods": 6},
                        {"topic": "Nghị luận xã hội", "lessons": ["Nghị luận về tư tưởng đạo lý", "Nghị luận về hiện tượng đời sống"], "periods": 5},
                    ],
                    "knowledge": "Nghị luận văn học và xã hội nâng cao",
                    "skills": "Viết bài nghị luận hoàn chỉnh, sâu sắc",
                },
                {
                    "chapter": "Tiếng Việt",
                    "topics": [
                        {"topic": "Từ vựng nâng cao", "lessons": ["Thuật ngữ", "Sự phát triển từ vựng"], "periods": 3},
                        {"topic": "Ngữ pháp nâng cao", "lessons": ["Nghĩa tường minh, hàm ý", "Liên kết câu và liên kết đoạn"], "periods": 4},
                    ],
                    "knowledge": "Thuật ngữ, nghĩa hàm ý, liên kết văn bản",
                    "skills": "Sử dụng ngôn ngữ tinh tế, liên kết chặt chẽ",
                },
            ],
        },
    },
    # =========================================================================
    # TIẾNG ANH LỚP 3-9
    # =========================================================================
    "anh": {
        "name": "Tiếng Anh",
        "grades": {
            3: [
                {
                    "chapter": "Unit 1-5: Getting started",
                    "topics": [
                        {"topic": "Greetings and Introduction", "lessons": ["Hello", "What's your name?", "How are you?"], "periods": 6},
                        {"topic": "School", "lessons": ["This is my school", "My classroom"], "periods": 4},
                        {"topic": "Family and Friends", "lessons": ["My family", "My friends"], "periods": 4},
                    ],
                    "knowledge": "Greeting, introducing; school objects; family members",
                    "skills": "Saying hello, introducing self and family, naming school items",
                },
                {
                    "chapter": "Unit 6-10: Daily life",
                    "topics": [
                        {"topic": "Body and Health", "lessons": ["Parts of the body", "Are you OK?"], "periods": 4},
                        {"topic": "Animals and Toys", "lessons": ["I like animals", "My toys"], "periods": 4},
                        {"topic": "Numbers and Colors", "lessons": ["How many?", "What color is it?"], "periods": 4},
                    ],
                    "knowledge": "Body parts, animals, toys, numbers 1-20, colors",
                    "skills": "Describing things, counting, identifying colors",
                },
            ],
            4: [
                {
                    "chapter": "Unit 1-5: Me and my world",
                    "topics": [
                        {"topic": "Daily routines", "lessons": ["What time is it?", "My daily routine"], "periods": 5},
                        {"topic": "My town", "lessons": ["Where is the school?", "Directions"], "periods": 4},
                        {"topic": "Abilities", "lessons": ["Can you swim?", "What can you do?"], "periods": 4},
                    ],
                    "knowledge": "Telling time, daily routines, directions, abilities",
                    "skills": "Talking about time, giving directions, expressing abilities",
                },
                {
                    "chapter": "Unit 6-10: Fun and hobbies",
                    "topics": [
                        {"topic": "Hobbies and Sports", "lessons": ["What's your hobby?", "Do you like sports?"], "periods": 4},
                        {"topic": "Food and Drinks", "lessons": ["What do you want to eat?", "At the restaurant"], "periods": 4},
                        {"topic": "Weather and Seasons", "lessons": ["What's the weather like?", "My favorite season"], "periods": 4},
                    ],
                    "knowledge": "Hobbies, sports, food, weather, seasons",
                    "skills": "Expressing preferences, ordering food, describing weather",
                },
            ],
            5: [
                {
                    "chapter": "Unit 1-5: Growing up",
                    "topics": [
                        {"topic": "Personal information", "lessons": ["Where are you from?", "Nationality and language"], "periods": 4},
                        {"topic": "Daily life", "lessons": ["What do you do in the morning?", "Housework"], "periods": 4},
                        {"topic": "Health and Illness", "lessons": ["What's the matter?", "You should see a doctor"], "periods": 4},
                    ],
                    "knowledge": "Nationality, daily activities, health problems and advice",
                    "skills": "Describing routines, giving health advice, asking about origin",
                },
                {
                    "chapter": "Unit 6-10: Exploring",
                    "topics": [
                        {"topic": "Festivals and Celebrations", "lessons": ["Vietnamese festivals", "Christmas, New Year"], "periods": 4},
                        {"topic": "Travel and Vacation", "lessons": ["Where did you go?", "Past simple tense"], "periods": 5},
                        {"topic": "The environment", "lessons": ["Save the environment", "Reduce, reuse, recycle"], "periods": 4},
                    ],
                    "knowledge": "Festivals, past tense, environment vocabulary",
                    "skills": "Talking about past events, festivals, environment",
                },
            ],
            6: [
                {
                    "chapter": "Unit 1-6: My world",
                    "topics": [
                        {"topic": "My new school", "lessons": ["School activities", "Present simple and continuous"], "periods": 5},
                        {"topic": "My home", "lessons": ["Types of houses", "Prepositions of place", "There is / There are"], "periods": 5},
                        {"topic": "My friends", "lessons": ["Describing people", "Adjectives of personality"], "periods": 4},
                    ],
                    "knowledge": "School, home, friends; present simple/continuous, prepositions",
                    "skills": "Describing school, home, friends; using basic tenses",
                },
                {
                    "chapter": "Unit 7-12: The world around me",
                    "topics": [
                        {"topic": "Our Tet holiday", "lessons": ["Tet traditions", "Should / Shouldn't"], "periods": 4},
                        {"topic": "Sports and games", "lessons": ["Sports vocabulary", "Imperatives"], "periods": 4},
                        {"topic": "Cities of the world", "lessons": ["Describing places", "Comparative adjectives"], "periods": 5},
                    ],
                    "knowledge": "Holidays, sports, cities; comparatives, should/shouldn't",
                    "skills": "Comparing, giving advice, describing places",
                },
            ],
            7: [
                {
                    "chapter": "Unit 1-6: Community",
                    "topics": [
                        {"topic": "Hobbies", "lessons": ["Hobbies and interests", "Verbs of liking + gerund"], "periods": 4},
                        {"topic": "Health", "lessons": ["Health problems", "Compound sentences"], "periods": 4},
                        {"topic": "Community service", "lessons": ["Volunteering", "Past simple for experiences"], "periods": 4},
                    ],
                    "knowledge": "Hobbies, health, community; gerunds, compound sentences, past simple",
                    "skills": "Discussing hobbies, health, community activities",
                },
                {
                    "chapter": "Unit 7-12: Environment and culture",
                    "topics": [
                        {"topic": "Traffic", "lessons": ["Means of transport", "Used to"], "periods": 4},
                        {"topic": "Films", "lessons": ["Types of films", "Adjectives ending -ed/-ing"], "periods": 4},
                        {"topic": "Energy sources", "lessons": ["Renewable energy", "Future simple: will"], "periods": 5},
                    ],
                    "knowledge": "Transport, films, energy; used to, will, -ed/-ing adjectives",
                    "skills": "Talking about past habits, future plans, expressing opinions",
                },
            ],
            8: [
                {
                    "chapter": "Unit 1-6: Our world",
                    "topics": [
                        {"topic": "Leisure activities", "lessons": ["Free time activities", "Verbs of liking + to-inf/gerund"], "periods": 4},
                        {"topic": "Life in the countryside", "lessons": ["Country vs city life", "Comparative and superlative"], "periods": 4},
                        {"topic": "Peoples of Viet Nam", "lessons": ["Ethnic groups", "Articles, possessive pronouns"], "periods": 4},
                    ],
                    "knowledge": "Leisure, countryside, ethnic groups; comparatives/superlatives, articles",
                    "skills": "Comparing lifestyles, describing cultures",
                },
                {
                    "chapter": "Unit 7-12: Science and Technology",
                    "topics": [
                        {"topic": "Science and Technology", "lessons": ["Inventions", "Passive voice (present simple)"], "periods": 5},
                        {"topic": "English speaking countries", "lessons": ["Countries and cultures", "Present perfect"], "periods": 5},
                        {"topic": "Communication", "lessons": ["Technology in communication", "Reported speech"], "periods": 5},
                    ],
                    "knowledge": "Science, countries, communication; passive voice, present perfect, reported speech",
                    "skills": "Discussing technology, cultures; using advanced grammar",
                },
            ],
            9: [
                {
                    "chapter": "Unit 1-6: Local and global",
                    "topics": [
                        {"topic": "Local environment", "lessons": ["Environmental problems", "Conditional sentences type 1, 2"], "periods": 5},
                        {"topic": "City life", "lessons": ["Life in the city", "Too/enough, phrasal verbs"], "periods": 4},
                        {"topic": "Teen stress and pressure", "lessons": ["Stress and solutions", "Wish sentences"], "periods": 5},
                    ],
                    "knowledge": "Environment, city life, teen issues; conditionals, wish, phrasal verbs",
                    "skills": "Discussing social issues, expressing wishes, giving advice",
                },
                {
                    "chapter": "Unit 7-12: International culture",
                    "topics": [
                        {"topic": "Tourism", "lessons": ["Travel and tourism", "Relative clauses"], "periods": 5},
                        {"topic": "Changing roles in society", "lessons": ["Gender equality", "Passive voice (various tenses)"], "periods": 4},
                        {"topic": "Space exploration", "lessons": ["Space and planets", "Reported speech (advanced)"], "periods": 5},
                    ],
                    "knowledge": "Tourism, society, space; relative clauses, passive voice, reported speech",
                    "skills": "Debating issues, writing formal texts, discussing global topics",
                },
            ],
        },
    },
    # =========================================================================
    # KHOA HỌC TỰ NHIÊN LỚP 6-9 (Tích hợp Lý, Hóa, Sinh)
    # =========================================================================
    "khtn": {
        "name": "Khoa học Tự nhiên",
        "grades": {
            6: [
                {
                    "chapter": "Chương 1: Mở đầu về KHTN",
                    "topics": [
                        {"topic": "Giới thiệu về KHTN", "lessons": ["Khoa học tự nhiên là gì", "An toàn trong phòng thí nghiệm"], "periods": 3},
                    ],
                    "knowledge": "Khái niệm KHTN, quy tắc an toàn phòng thí nghiệm",
                    "skills": "Thực hiện an toàn thí nghiệm",
                },
                {
                    "chapter": "Chương 2: Chất và sự biến đổi",
                    "topics": [
                        {"topic": "Chất quanh ta", "lessons": ["Sự đa dạng của chất", "Tính chất của chất"], "periods": 4},
                        {"topic": "Nguyên tử, phân tử", "lessons": ["Nguyên tử", "Nguyên tố hóa học", "Phân tử"], "periods": 5},
                        {"topic": "Hỗn hợp và tách chất", "lessons": ["Hỗn hợp, dung dịch", "Tách chất khỏi hỗn hợp"], "periods": 4},
                    ],
                    "knowledge": "Chất, nguyên tử, phân tử, hỗn hợp, dung dịch",
                    "skills": "Phân biệt chất, tách chất khỏi hỗn hợp",
                },
                {
                    "chapter": "Chương 3: Vật sống",
                    "topics": [
                        {"topic": "Tế bào", "lessons": ["Tế bào là đơn vị cơ bản của sự sống", "Tế bào nhân sơ, nhân thực"], "periods": 4},
                        {"topic": "Đa dạng thế giới sống", "lessons": ["Vi khuẩn, nguyên sinh vật", "Nấm", "Thực vật", "Động vật"], "periods": 8},
                    ],
                    "knowledge": "Tế bào, phân loại sinh vật: vi khuẩn, nấm, thực vật, động vật",
                    "skills": "Quan sát tế bào, phân loại sinh vật",
                },
                {
                    "chapter": "Chương 4: Năng lượng và sự biến đổi",
                    "topics": [
                        {"topic": "Lực và chuyển động", "lessons": ["Lực và tác dụng của lực", "Lực ma sát, lực cản"], "periods": 4},
                        {"topic": "Năng lượng", "lessons": ["Các dạng năng lượng", "Sự chuyển hóa năng lượng"], "periods": 3},
                    ],
                    "knowledge": "Lực, các dạng năng lượng, sự chuyển hóa năng lượng",
                    "skills": "Nhận biết lực, mô tả chuyển hóa năng lượng",
                },
                {
                    "chapter": "Chương 5: Trái Đất và bầu trời",
                    "topics": [
                        {"topic": "Hệ Mặt Trời, Ngân Hà", "lessons": ["Chuyển động nhìn thấy của Mặt Trời", "Mặt Trăng", "Hệ Mặt Trời"], "periods": 4},
                    ],
                    "knowledge": "Hệ Mặt Trời, chuyển động Trái Đất, Mặt Trăng",
                    "skills": "Giải thích hiện tượng ngày đêm, mùa",
                },
            ],
            7: [
                {
                    "chapter": "Chương 1: Nguyên tử. Bảng tuần hoàn",
                    "topics": [
                        {"topic": "Nguyên tử", "lessons": ["Mô hình nguyên tử", "Nguyên tố hóa học"], "periods": 4},
                        {"topic": "Bảng tuần hoàn", "lessons": ["Bảng tuần hoàn các nguyên tố hóa học", "Chu kỳ, nhóm"], "periods": 3},
                    ],
                    "knowledge": "Cấu tạo nguyên tử, bảng tuần hoàn",
                    "skills": "Sử dụng bảng tuần hoàn, xác định vị trí nguyên tố",
                },
                {
                    "chapter": "Chương 2: Phân tử. Liên kết hóa học",
                    "topics": [
                        {"topic": "Phân tử", "lessons": ["Phân tử đơn chất, hợp chất", "Công thức hóa học"], "periods": 4},
                        {"topic": "Liên kết hóa học", "lessons": ["Liên kết ion", "Liên kết cộng hóa trị"], "periods": 3},
                    ],
                    "knowledge": "Phân tử, công thức hóa học, liên kết hóa học",
                    "skills": "Viết công thức hóa học, phân biệt liên kết",
                },
                {
                    "chapter": "Chương 3: Tốc độ",
                    "topics": [
                        {"topic": "Tốc độ chuyển động", "lessons": ["Tốc độ", "Đo tốc độ", "Đồ thị quãng đường - thời gian"], "periods": 5},
                    ],
                    "knowledge": "Tốc độ, đo tốc độ, đồ thị chuyển động",
                    "skills": "Tính tốc độ, vẽ và đọc đồ thị",
                },
                {
                    "chapter": "Chương 4: Âm thanh và Ánh sáng",
                    "topics": [
                        {"topic": "Âm thanh", "lessons": ["Sóng âm", "Độ to, độ cao của âm"], "periods": 3},
                        {"topic": "Ánh sáng", "lessons": ["Sự truyền ánh sáng", "Sự phản xạ ánh sáng", "Ảnh qua gương phẳng"], "periods": 5},
                    ],
                    "knowledge": "Sóng âm, tính chất âm thanh; truyền và phản xạ ánh sáng",
                    "skills": "Giải thích hiện tượng âm thanh, ánh sáng",
                },
                {
                    "chapter": "Chương 5: Trao đổi chất và chuyển hóa năng lượng",
                    "topics": [
                        {"topic": "Trao đổi chất ở sinh vật", "lessons": ["Quang hợp", "Hô hấp tế bào", "Trao đổi nước và khoáng ở thực vật"], "periods": 6},
                        {"topic": "Cảm ứng ở sinh vật", "lessons": ["Cảm ứng ở thực vật", "Cảm ứng ở động vật"], "periods": 3},
                    ],
                    "knowledge": "Quang hợp, hô hấp, trao đổi chất; cảm ứng sinh vật",
                    "skills": "Mô tả quá trình quang hợp, hô hấp, cảm ứng",
                },
                {
                    "chapter": "Chương 6: Sinh trưởng và sinh sản",
                    "topics": [
                        {"topic": "Sinh trưởng và phát triển", "lessons": ["Sinh trưởng ở thực vật", "Sinh trưởng ở động vật"], "periods": 3},
                        {"topic": "Sinh sản ở sinh vật", "lessons": ["Sinh sản vô tính", "Sinh sản hữu tính"], "periods": 4},
                    ],
                    "knowledge": "Sinh trưởng, phát triển, sinh sản sinh vật",
                    "skills": "Phân biệt sinh sản vô tính, hữu tính",
                },
            ],
            8: [
                {
                    "chapter": "Chương 1: Phản ứng hóa học",
                    "topics": [
                        {"topic": "Biến đổi chất", "lessons": ["Biến đổi vật lý, hóa học", "Phản ứng hóa học", "Mol và tỷ khối"], "periods": 6},
                        {"topic": "Phương trình hóa học", "lessons": ["Lập phương trình hóa học", "Tính theo phương trình"], "periods": 5},
                    ],
                    "knowledge": "Phản ứng hóa học, mol, phương trình hóa học",
                    "skills": "Cân bằng phương trình, tính theo phương trình",
                },
                {
                    "chapter": "Chương 2: Acid - Base - pH - Oxide - Muối",
                    "topics": [
                        {"topic": "Acid - Base", "lessons": ["Acid", "Base", "Thang pH"], "periods": 5},
                        {"topic": "Oxide và Muối", "lessons": ["Oxide", "Muối"], "periods": 4},
                    ],
                    "knowledge": "Acid, base, pH, oxide, muối; tính chất hóa học",
                    "skills": "Phân biệt acid/base, tính pH, viết phương trình",
                },
                {
                    "chapter": "Chương 3: Khối lượng riêng và áp suất",
                    "topics": [
                        {"topic": "Khối lượng riêng", "lessons": ["Khối lượng riêng", "Đo khối lượng riêng"], "periods": 3},
                        {"topic": "Áp suất", "lessons": ["Áp suất trên bề mặt", "Áp suất chất lỏng, chất khí"], "periods": 4},
                    ],
                    "knowledge": "Khối lượng riêng, áp suất chất rắn/lỏng/khí",
                    "skills": "Tính khối lượng riêng, áp suất",
                },
                {
                    "chapter": "Chương 4: Tác dụng của lực",
                    "topics": [
                        {"topic": "Lực và chuyển động", "lessons": ["Lực ma sát", "Lực cản trong chất lỏng", "Moment lực"], "periods": 5},
                    ],
                    "knowledge": "Lực ma sát, lực cản, moment lực",
                    "skills": "Tính lực ma sát, moment lực",
                },
                {
                    "chapter": "Chương 5: Điện",
                    "topics": [
                        {"topic": "Điện tích và dòng điện", "lessons": ["Sự nhiễm điện", "Dòng điện, mạch điện"], "periods": 4},
                        {"topic": "Cường độ dòng điện và hiệu điện thế", "lessons": ["Cường độ dòng điện", "Hiệu điện thế", "Định luật Ohm"], "periods": 6},
                    ],
                    "knowledge": "Điện tích, dòng điện, mạch điện, định luật Ohm",
                    "skills": "Mắc mạch điện, tính cường độ/hiệu điện thế",
                },
                {
                    "chapter": "Chương 6: Cơ thể người",
                    "topics": [
                        {"topic": "Hệ vận động, tiêu hóa, tuần hoàn", "lessons": ["Hệ xương và cơ", "Hệ tiêu hóa", "Hệ tuần hoàn"], "periods": 6},
                        {"topic": "Hệ hô hấp, bài tiết, thần kinh", "lessons": ["Hệ hô hấp", "Hệ bài tiết", "Hệ thần kinh"], "periods": 6},
                    ],
                    "knowledge": "Các hệ cơ quan trong cơ thể người",
                    "skills": "Mô tả cấu tạo, chức năng các hệ cơ quan",
                },
            ],
            9: [
                {
                    "chapter": "Chương 1: Kim loại. Phi kim",
                    "topics": [
                        {"topic": "Kim loại", "lessons": ["Tính chất kim loại", "Dãy hoạt động hóa học", "Hợp kim"], "periods": 5},
                        {"topic": "Phi kim", "lessons": ["Tính chất phi kim", "Cacbon, silic", "Clo, brom"], "periods": 4},
                    ],
                    "knowledge": "Tính chất kim loại, phi kim; dãy hoạt động hóa học",
                    "skills": "So sánh tính chất kim loại/phi kim, viết phương trình",
                },
                {
                    "chapter": "Chương 2: Hợp chất hữu cơ",
                    "topics": [
                        {"topic": "Đại cương hữu cơ", "lessons": ["Hợp chất hữu cơ", "Hydrocarbon: methane, ethylene, acetylene"], "periods": 5},
                        {"topic": "Dẫn xuất hydrocarbon", "lessons": ["Ethanol, acetic acid", "Chất béo, glucose, protein"], "periods": 5},
                    ],
                    "knowledge": "Hóa học hữu cơ cơ bản: hydrocarbon, dẫn xuất",
                    "skills": "Nhận biết, viết phương trình phản ứng hữu cơ",
                },
                {
                    "chapter": "Chương 3: Năng lượng cơ học",
                    "topics": [
                        {"topic": "Công và công suất", "lessons": ["Công cơ học", "Công suất"], "periods": 3},
                        {"topic": "Năng lượng", "lessons": ["Động năng, thế năng", "Cơ năng, định luật bảo toàn cơ năng"], "periods": 4},
                    ],
                    "knowledge": "Công, công suất, động năng, thế năng, cơ năng",
                    "skills": "Tính công, công suất, năng lượng",
                },
                {
                    "chapter": "Chương 4: Điện từ",
                    "topics": [
                        {"topic": "Nam châm và từ trường", "lessons": ["Nam châm", "Từ trường", "Lực từ tác dụng lên dây dẫn có dòng điện"], "periods": 5},
                        {"topic": "Cảm ứng điện từ", "lessons": ["Hiện tượng cảm ứng điện từ", "Máy phát điện, động cơ điện"], "periods": 4},
                    ],
                    "knowledge": "Nam châm, từ trường, cảm ứng điện từ",
                    "skills": "Mô tả từ trường, giải thích cảm ứng điện từ",
                },
                {
                    "chapter": "Chương 5: Di truyền và Tiến hóa",
                    "topics": [
                        {"topic": "Di truyền", "lessons": ["Gen, DNA, nhiễm sắc thể", "Các quy luật di truyền Mendel", "Di truyền người"], "periods": 8},
                        {"topic": "Tiến hóa", "lessons": ["Bằng chứng tiến hóa", "Chọn lọc tự nhiên"], "periods": 3},
                    ],
                    "knowledge": "Gen, DNA, di truyền Mendel, tiến hóa",
                    "skills": "Giải bài tập di truyền, phân tích quy luật",
                },
                {
                    "chapter": "Chương 6: Sinh thái",
                    "topics": [
                        {"topic": "Sinh thái học", "lessons": ["Quần thể, quần xã", "Hệ sinh thái", "Bảo vệ môi trường"], "periods": 4},
                    ],
                    "knowledge": "Quần thể, quần xã, hệ sinh thái, bảo vệ môi trường",
                    "skills": "Phân tích hệ sinh thái, đề xuất bảo vệ môi trường",
                },
            ],
        },
    },
    # =========================================================================
    # LỊCH SỬ & ĐỊA LÝ LỚP 4-9
    # =========================================================================
    "su_dia": {
        "name": "Lịch sử & Địa lý",
        "grades": {
            4: [
                {
                    "chapter": "Lịch sử: Buổi đầu dựng nước",
                    "topics": [
                        {"topic": "Nước Văn Lang - Âu Lạc", "lessons": ["Nước Văn Lang", "Nước Âu Lạc"], "periods": 3},
                        {"topic": "Hơn 1000 năm Bắc thuộc", "lessons": ["Khởi nghĩa Hai Bà Trưng", "Chiến thắng Bạch Đằng"], "periods": 3},
                    ],
                    "knowledge": "Buổi đầu dựng nước, 1000 năm Bắc thuộc",
                    "skills": "Kể lại sự kiện lịch sử, nhận biết trên bản đồ",
                },
                {
                    "chapter": "Lịch sử: Buổi đầu độc lập",
                    "topics": [
                        {"topic": "Các triều đại phong kiến", "lessons": ["Nhà Lý dời đô", "Nhà Trần đánh giặc Mông Nguyên"], "periods": 4},
                    ],
                    "knowledge": "Các triều đại Lý, Trần; sự kiện tiêu biểu",
                    "skills": "Sắp xếp sự kiện theo trình tự thời gian",
                },
                {
                    "chapter": "Địa lý: Thiên nhiên và con người Việt Nam",
                    "topics": [
                        {"topic": "Bản đồ và bảng số liệu", "lessons": ["Cách đọc bản đồ", "Bảng số liệu đơn giản"], "periods": 2},
                        {"topic": "Thiên nhiên Việt Nam", "lessons": ["Địa hình, sông ngòi", "Khí hậu, thời tiết"], "periods": 4},
                        {"topic": "Dân cư, kinh tế", "lessons": ["Dân cư Việt Nam", "Hoạt động sản xuất"], "periods": 3},
                    ],
                    "knowledge": "Bản đồ, thiên nhiên, dân cư, kinh tế Việt Nam",
                    "skills": "Đọc bản đồ, mô tả đặc điểm thiên nhiên",
                },
            ],
            5: [
                {
                    "chapter": "Lịch sử: 80 năm chống Pháp",
                    "topics": [
                        {"topic": "Phong trào chống Pháp", "lessons": ["Pháp đánh chiếm nước ta", "Phong trào Cần Vương"], "periods": 3},
                        {"topic": "Cách mạng tháng Tám", "lessons": ["Đảng Cộng sản Việt Nam ra đời", "Cách mạng tháng Tám 1945"], "periods": 3},
                    ],
                    "knowledge": "Kháng chiến chống Pháp, Cách mạng tháng Tám",
                    "skills": "Trình bày diễn biến sự kiện lịch sử",
                },
                {
                    "chapter": "Lịch sử: Kháng chiến và thống nhất",
                    "topics": [
                        {"topic": "Kháng chiến chống Pháp", "lessons": ["Chiến thắng Điện Biên Phủ"], "periods": 2},
                        {"topic": "Kháng chiến chống Mỹ", "lessons": ["Chiến dịch Hồ Chí Minh", "Thống nhất đất nước"], "periods": 3},
                    ],
                    "knowledge": "Kháng chiến chống Mỹ, thống nhất đất nước",
                    "skills": "Phân tích nguyên nhân, ý nghĩa sự kiện",
                },
                {
                    "chapter": "Địa lý: Châu Á, Châu Âu",
                    "topics": [
                        {"topic": "Châu Á", "lessons": ["Vị trí, đặc điểm tự nhiên", "Các nước Đông Nam Á"], "periods": 4},
                        {"topic": "Châu Âu", "lessons": ["Vị trí, đặc điểm tự nhiên", "Một số nước châu Âu"], "periods": 3},
                    ],
                    "knowledge": "Đặc điểm tự nhiên, xã hội Châu Á, Châu Âu",
                    "skills": "Đọc bản đồ châu lục, so sánh đặc điểm",
                },
            ],
            6: [
                {
                    "chapter": "Lịch sử: Xã hội nguyên thủy",
                    "topics": [
                        {"topic": "Sự xuất hiện con người", "lessons": ["Nguồn gốc loài người", "Xã hội nguyên thủy"], "periods": 3},
                        {"topic": "Các nền văn minh cổ đại", "lessons": ["Ai Cập cổ đại", "Hy Lạp và La Mã cổ đại"], "periods": 4},
                    ],
                    "knowledge": "Xã hội nguyên thủy, văn minh cổ đại",
                    "skills": "Mô tả đời sống xã hội cổ đại",
                },
                {
                    "chapter": "Lịch sử: Buổi đầu lịch sử Việt Nam",
                    "topics": [
                        {"topic": "Việt Nam thời nguyên thủy", "lessons": ["Các nền văn hóa tiêu biểu"], "periods": 2},
                        {"topic": "Thời kỳ Văn Lang - Âu Lạc", "lessons": ["Nhà nước Văn Lang", "Nhà nước Âu Lạc"], "periods": 3},
                    ],
                    "knowledge": "Lịch sử Việt Nam thời cổ đại",
                    "skills": "Phân tích tư liệu lịch sử",
                },
                {
                    "chapter": "Địa lý: Trái Đất",
                    "topics": [
                        {"topic": "Bản đồ", "lessons": ["Bản đồ, tỷ lệ bản đồ, kinh vĩ tuyến"], "periods": 3},
                        {"topic": "Trái Đất", "lessons": ["Trái Đất trong hệ Mặt Trời", "Cấu tạo bên trong Trái Đất"], "periods": 3},
                        {"topic": "Các thành phần tự nhiên", "lessons": ["Khí quyển", "Thủy quyển", "Thổ nhưỡng quyển, sinh quyển"], "periods": 5},
                    ],
                    "knowledge": "Bản đồ, cấu tạo Trái Đất, các quyển",
                    "skills": "Sử dụng bản đồ, mô tả các thành phần tự nhiên",
                },
            ],
            7: [
                {
                    "chapter": "Lịch sử: Thế giới trung đại",
                    "topics": [
                        {"topic": "Phong kiến phương Tây", "lessons": ["Xã hội phong kiến Tây Âu", "Thành thị trung đại"], "periods": 3},
                        {"topic": "Phong kiến phương Đông", "lessons": ["Trung Quốc phong kiến", "Ấn Độ phong kiến"], "periods": 3},
                    ],
                    "knowledge": "Xã hội phong kiến phương Đông và phương Tây",
                    "skills": "So sánh đặc điểm phong kiến Đông - Tây",
                },
                {
                    "chapter": "Lịch sử: Việt Nam từ thế kỷ X đến XIX",
                    "topics": [
                        {"topic": "Các triều đại độc lập", "lessons": ["Nhà Lý, Trần, Lê"], "periods": 4},
                        {"topic": "Đại Việt thời Nguyễn", "lessons": ["Nhà Nguyễn thành lập", "Kinh tế, văn hóa thời Nguyễn"], "periods": 3},
                    ],
                    "knowledge": "Lịch sử Việt Nam phong kiến: Lý, Trần, Lê, Nguyễn",
                    "skills": "Trình bày và phân tích các triều đại",
                },
                {
                    "chapter": "Địa lý: Châu Âu, Châu Á",
                    "topics": [
                        {"topic": "Dân cư và xã hội thế giới", "lessons": ["Dân số thế giới", "Đô thị hóa", "Các chủng tộc"], "periods": 4},
                        {"topic": "Châu Âu", "lessons": ["Vị trí, tự nhiên châu Âu", "EU và một số quốc gia"], "periods": 4},
                        {"topic": "Châu Á", "lessons": ["Tự nhiên châu Á", "Kinh tế châu Á", "Khu vực Đông Nam Á"], "periods": 5},
                    ],
                    "knowledge": "Dân cư thế giới, địa lý châu Âu, châu Á",
                    "skills": "Phân tích bản đồ, số liệu dân cư, kinh tế",
                },
            ],
            8: [
                {
                    "chapter": "Lịch sử: Thế giới cận đại",
                    "topics": [
                        {"topic": "Cách mạng tư sản", "lessons": ["Cách mạng Anh, Pháp, Mỹ"], "periods": 4},
                        {"topic": "Cách mạng công nghiệp", "lessons": ["Cách mạng công nghiệp", "Chủ nghĩa tư bản"], "periods": 3},
                    ],
                    "knowledge": "Các cuộc cách mạng tư sản, cách mạng công nghiệp",
                    "skills": "Phân tích nguyên nhân, ý nghĩa cách mạng",
                },
                {
                    "chapter": "Lịch sử: Việt Nam từ 1858 đến 1918",
                    "topics": [
                        {"topic": "Pháp xâm lược Việt Nam", "lessons": ["Pháp xâm lược", "Phong trào chống Pháp"], "periods": 4},
                        {"topic": "Phong trào yêu nước đầu thế kỷ XX", "lessons": ["Phan Bội Châu, Phan Châu Trinh", "Nguyễn Ái Quốc"], "periods": 4},
                    ],
                    "knowledge": "Lịch sử Việt Nam thời cận đại",
                    "skills": "Phân tích phong trào yêu nước, so sánh xu hướng",
                },
                {
                    "chapter": "Địa lý: Châu Phi, Châu Mỹ, Châu Đại Dương",
                    "topics": [
                        {"topic": "Châu Phi", "lessons": ["Tự nhiên châu Phi", "Kinh tế xã hội châu Phi"], "periods": 3},
                        {"topic": "Châu Mỹ", "lessons": ["Bắc Mỹ", "Trung và Nam Mỹ"], "periods": 4},
                        {"topic": "Châu Đại Dương", "lessons": ["Ôxtrâylia, Niu Di-lân"], "periods": 2},
                    ],
                    "knowledge": "Địa lý châu Phi, châu Mỹ, châu Đại Dương",
                    "skills": "So sánh đặc điểm các châu lục",
                },
            ],
            9: [
                {
                    "chapter": "Lịch sử: Thế giới hiện đại",
                    "topics": [
                        {"topic": "Chiến tranh thế giới", "lessons": ["Chiến tranh thế giới thứ nhất", "Chiến tranh thế giới thứ hai"], "periods": 4},
                        {"topic": "Thế giới sau 1945", "lessons": ["Trật tự thế giới mới", "Phong trào giải phóng dân tộc"], "periods": 3},
                    ],
                    "knowledge": "Hai cuộc chiến tranh thế giới, thế giới sau 1945",
                    "skills": "Phân tích nguyên nhân, hệ quả chiến tranh",
                },
                {
                    "chapter": "Lịch sử: Việt Nam 1919-1975",
                    "topics": [
                        {"topic": "Cách mạng tháng Tám", "lessons": ["Phong trào 1930-1931", "Cách mạng tháng Tám 1945"], "periods": 4},
                        {"topic": "Kháng chiến chống Pháp và Mỹ", "lessons": ["Kháng chiến chống Pháp", "Kháng chiến chống Mỹ"], "periods": 5},
                        {"topic": "Đổi mới đất nước", "lessons": ["Đổi mới 1986", "Việt Nam hội nhập quốc tế"], "periods": 3},
                    ],
                    "knowledge": "Lịch sử Việt Nam hiện đại: cách mạng, kháng chiến, đổi mới",
                    "skills": "Phân tích đường lối cách mạng, ý nghĩa lịch sử",
                },
                {
                    "chapter": "Địa lý: Việt Nam",
                    "topics": [
                        {"topic": "Tự nhiên Việt Nam", "lessons": ["Vị trí, lãnh thổ", "Địa hình, khí hậu, sông ngòi", "Tài nguyên thiên nhiên"], "periods": 5},
                        {"topic": "Dân cư, kinh tế Việt Nam", "lessons": ["Dân số, lao động", "Kinh tế: nông nghiệp, công nghiệp, dịch vụ"], "periods": 5},
                        {"topic": "Các vùng kinh tế", "lessons": ["Vùng Bắc Bộ", "Vùng Trung Bộ", "Vùng Nam Bộ"], "periods": 4},
                    ],
                    "knowledge": "Địa lý tự nhiên, dân cư, kinh tế, các vùng Việt Nam",
                    "skills": "Phân tích bản đồ, số liệu; đánh giá tiềm năng vùng",
                },
            ],
        },
    },
}
