"""
Sample curriculum data for Vietnamese education system.
Based on Chương trình Giáo dục Phổ thông 2018.
"""

# Curriculum data for grades 10-12
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
                        {"topic": "Định lý cosin và định lý sin", "lessons": ["Định lý cosin", "Định lý sin", "Giải tam giác và ứng dụng"], "periods": 5},
                    ],
                    "knowledge": "Hiểu giá trị lượng giác, định lý cosin, định lý sin",
                    "skills": "Tính các yếu tố trong tam giác, giải tam giác",
                },
                {
                    "chapter": "Chương 5: Vectơ",
                    "topics": [
                        {"topic": "Khái niệm vectơ", "lessons": ["Vectơ", "Tổng và hiệu của hai vectơ", "Tích của một số với một vectơ"], "periods": 5},
                        {"topic": "Tích vô hướng", "lessons": ["Tích vô hướng của hai vectơ"], "periods": 3},
                    ],
                    "knowledge": "Hiểu khái niệm vectơ, các phép toán vectơ",
                    "skills": "Thực hiện phép toán vectơ, tính tích vô hướng",
                },
                {
                    "chapter": "Chương 6: Thống kê",
                    "topics": [
                        {"topic": "Các số đặc trưng đo xu thế trung tâm", "lessons": ["Số trung bình cộng", "Trung vị", "Mốt"], "periods": 3},
                        {"topic": "Các số đặc trưng đo độ phân tán", "lessons": ["Phương sai", "Độ lệch chuẩn"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các số đặc trưng thống kê",
                    "skills": "Tính toán và phân tích dữ liệu thống kê",
                },
                {
                    "chapter": "Chương 7: Xác suất",
                    "topics": [
                        {"topic": "Xác suất của biến cố", "lessons": ["Không gian mẫu và biến cố", "Xác suất của biến cố"], "periods": 5},
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
                        {"topic": "Phương trình lượng giác", "lessons": ["Phương trình lượng giác cơ bản", "Một số dạng phương trình lượng giác thường gặp"], "periods": 5},
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
                    "chapter": "Chương 4: Đường thẳng và mặt phẳng trong không gian. Quan hệ song song",
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
                    "chapter": "Chương 5: Quan hệ vuông góc trong không gian",
                    "topics": [
                        {"topic": "Vectơ trong không gian", "lessons": ["Vectơ trong không gian"], "periods": 3},
                        {"topic": "Hai đường thẳng vuông góc", "lessons": ["Hai đường thẳng vuông góc"], "periods": 2},
                        {"topic": "Đường thẳng vuông góc với mặt phẳng", "lessons": ["Đường thẳng vuông góc với mặt phẳng"], "periods": 3},
                        {"topic": "Hai mặt phẳng vuông góc", "lessons": ["Hai mặt phẳng vuông góc"], "periods": 3},
                        {"topic": "Khoảng cách", "lessons": ["Khoảng cách trong không gian"], "periods": 3},
                    ],
                    "knowledge": "Hiểu quan hệ vuông góc trong không gian",
                    "skills": "Chứng minh vuông góc, tính góc và khoảng cách",
                },
                {
                    "chapter": "Chương 6: Thống kê",
                    "topics": [
                        {"topic": "Các số đặc trưng của mẫu số liệu ghép nhóm", "lessons": ["Bảng tần số ghép nhóm", "Số trung bình, trung vị, mốt của mẫu số liệu ghép nhóm"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cách xử lý dữ liệu ghép nhóm",
                    "skills": "Tính các đặc trưng thống kê cho dữ liệu ghép nhóm",
                },
                {
                    "chapter": "Chương 7: Xác suất có điều kiện",
                    "topics": [
                        {"topic": "Xác suất có điều kiện", "lessons": ["Xác suất có điều kiện", "Công thức Bayes"], "periods": 4},
                    ],
                    "knowledge": "Hiểu xác suất có điều kiện, công thức Bayes",
                    "skills": "Tính xác suất có điều kiện",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Đạo hàm và ứng dụng",
                    "topics": [
                        {"topic": "Đạo hàm", "lessons": ["Đạo hàm", "Các quy tắc tính đạo hàm", "Đạo hàm cấp hai"], "periods": 6},
                        {"topic": "Ứng dụng của đạo hàm", "lessons": ["Sự đồng biến, nghịch biến", "Cực trị của hàm số", "Giá trị lớn nhất, nhỏ nhất", "Đường tiệm cận", "Khảo sát và vẽ đồ thị hàm số"], "periods": 10},
                    ],
                    "knowledge": "Hiểu đạo hàm, các quy tắc tính đạo hàm, ứng dụng",
                    "skills": "Tính đạo hàm, khảo sát hàm số, tìm GTLN-GTNN",
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
                        {"topic": "Nguyên hàm", "lessons": ["Nguyên hàm", "Phương pháp tính nguyên hàm"], "periods": 4},
                        {"topic": "Tích phân", "lessons": ["Tích phân", "Phương pháp tính tích phân", "Ứng dụng tích phân trong hình học"], "periods": 7},
                    ],
                    "knowledge": "Hiểu nguyên hàm, tích phân",
                    "skills": "Tính nguyên hàm, tích phân, tính diện tích và thể tích",
                },
                {
                    "chapter": "Chương 4: Số phức",
                    "topics": [
                        {"topic": "Số phức", "lessons": ["Số phức", "Các phép toán số phức", "Biểu diễn hình học của số phức"], "periods": 5},
                        {"topic": "Phương trình bậc hai với hệ số thực", "lessons": ["Phương trình bậc hai với hệ số thực"], "periods": 2},
                    ],
                    "knowledge": "Hiểu số phức, các phép toán",
                    "skills": "Thực hiện phép toán số phức, giải phương trình",
                },
                {
                    "chapter": "Chương 5: Hình học tọa độ trong không gian",
                    "topics": [
                        {"topic": "Hệ tọa độ trong không gian", "lessons": ["Hệ tọa độ trong không gian"], "periods": 2},
                        {"topic": "Phương trình mặt phẳng", "lessons": ["Phương trình mặt phẳng"], "periods": 4},
                        {"topic": "Phương trình đường thẳng", "lessons": ["Phương trình đường thẳng trong không gian"], "periods": 4},
                        {"topic": "Phương trình mặt cầu", "lessons": ["Phương trình mặt cầu"], "periods": 2},
                    ],
                    "knowledge": "Hiểu hệ tọa độ không gian, phương trình mặt phẳng, đường thẳng, mặt cầu",
                    "skills": "Viết phương trình, tính khoảng cách, góc",
                },
                {
                    "chapter": "Chương 6: Xác suất và thống kê",
                    "topics": [
                        {"topic": "Biến ngẫu nhiên rời rạc", "lessons": ["Biến ngẫu nhiên rời rạc", "Kỳ vọng, phương sai, độ lệch chuẩn"], "periods": 4},
                        {"topic": "Phân bố nhị thức", "lessons": ["Phép thử Bernoulli", "Phân bố nhị thức"], "periods": 3},
                    ],
                    "knowledge": "Hiểu biến ngẫu nhiên, phân bố nhị thức",
                    "skills": "Tính kỳ vọng, phương sai, xác suất nhị thức",
                },
            ],
        },
    },
    "ly": {
        "name": "Vật lý",
        "grades": {
            10: [
                {
                    "chapter": "Chương 1: Động học",
                    "topics": [
                        {"topic": "Mô tả chuyển động", "lessons": ["Chuyển động cơ", "Vận tốc và tốc độ", "Đồ thị chuyển động"], "periods": 5},
                        {"topic": "Chuyển động thẳng", "lessons": ["Chuyển động thẳng đều", "Chuyển động thẳng biến đổi đều"], "periods": 5},
                        {"topic": "Sự rơi tự do", "lessons": ["Sự rơi tự do", "Chuyển động ném"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các khái niệm động học cơ bản",
                    "skills": "Giải bài toán chuyển động thẳng, vẽ đồ thị",
                },
                {
                    "chapter": "Chương 2: Động lực học",
                    "topics": [
                        {"topic": "Các định luật Newton", "lessons": ["Định luật I Newton", "Định luật II Newton", "Định luật III Newton"], "periods": 6},
                        {"topic": "Các lực cơ học", "lessons": ["Lực hấp dẫn", "Lực đàn hồi", "Lực ma sát"], "periods": 5},
                        {"topic": "Ứng dụng các định luật Newton", "lessons": ["Bài toán về hệ vật", "Chuyển động trên mặt phẳng nghiêng"], "periods": 4},
                    ],
                    "knowledge": "Hiểu các định luật Newton và các lực cơ học",
                    "skills": "Giải bài toán động lực học",
                },
                {
                    "chapter": "Chương 3: Năng lượng",
                    "topics": [
                        {"topic": "Công và năng lượng", "lessons": ["Công", "Công suất"], "periods": 3},
                        {"topic": "Động năng và thế năng", "lessons": ["Động năng", "Thế năng"], "periods": 4},
                        {"topic": "Cơ năng", "lessons": ["Định luật bảo toàn cơ năng"], "periods": 3},
                    ],
                    "knowledge": "Hiểu công, năng lượng, định luật bảo toàn cơ năng",
                    "skills": "Tính công, năng lượng, áp dụng định luật bảo toàn",
                },
                {
                    "chapter": "Chương 4: Động lượng",
                    "topics": [
                        {"topic": "Động lượng", "lessons": ["Động lượng", "Xung lượng của lực"], "periods": 3},
                        {"topic": "Định luật bảo toàn động lượng", "lessons": ["Định luật bảo toàn động lượng", "Bài toán va chạm"], "periods": 4},
                    ],
                    "knowledge": "Hiểu động lượng, định luật bảo toàn động lượng",
                    "skills": "Giải bài toán va chạm, chuyển động bằng phản lực",
                },
                {
                    "chapter": "Chương 5: Chuyển động tròn và biến dạng",
                    "topics": [
                        {"topic": "Chuyển động tròn đều", "lessons": ["Chuyển động tròn đều", "Lực hướng tâm"], "periods": 4},
                        {"topic": "Biến dạng của vật rắn", "lessons": ["Biến dạng kéo, nén", "Biến dạng cắt"], "periods": 3},
                    ],
                    "knowledge": "Hiểu chuyển động tròn, lực hướng tâm, biến dạng",
                    "skills": "Tính gia tốc hướng tâm, lực hướng tâm",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Dao động",
                    "topics": [
                        {"topic": "Dao động điều hòa", "lessons": ["Dao động điều hòa", "Con lắc lò xo", "Con lắc đơn"], "periods": 7},
                        {"topic": "Dao động tắt dần và dao động cưỡng bức", "lessons": ["Dao động tắt dần", "Dao động cưỡng bức", "Hiện tượng cộng hưởng"], "periods": 4},
                        {"topic": "Tổng hợp dao động", "lessons": ["Tổng hợp hai dao động cùng phương, cùng tần số"], "periods": 3},
                    ],
                    "knowledge": "Hiểu dao động điều hòa, con lắc, cộng hưởng",
                    "skills": "Viết phương trình dao động, tính chu kỳ, tần số, năng lượng",
                },
                {
                    "chapter": "Chương 2: Sóng",
                    "topics": [
                        {"topic": "Sóng cơ", "lessons": ["Sóng cơ và sự truyền sóng", "Các đặc trưng của sóng"], "periods": 4},
                        {"topic": "Giao thoa sóng", "lessons": ["Giao thoa sóng", "Sóng dừng"], "periods": 4},
                        {"topic": "Sóng âm", "lessons": ["Sóng âm", "Các đặc trưng vật lý và sinh lý của âm"], "periods": 3},
                    ],
                    "knowledge": "Hiểu sóng cơ, giao thoa, sóng dừng, sóng âm",
                    "skills": "Tính bước sóng, tần số, xác định vân giao thoa",
                },
                {
                    "chapter": "Chương 3: Điện trường",
                    "topics": [
                        {"topic": "Điện tích và định luật Coulomb", "lessons": ["Điện tích", "Định luật Coulomb"], "periods": 3},
                        {"topic": "Điện trường", "lessons": ["Cường độ điện trường", "Đường sức điện", "Điện thế, hiệu điện thế"], "periods": 5},
                        {"topic": "Tụ điện", "lessons": ["Tụ điện", "Năng lượng điện trường"], "periods": 3},
                    ],
                    "knowledge": "Hiểu điện tích, điện trường, tụ điện",
                    "skills": "Tính lực điện, cường độ điện trường, điện dung",
                },
                {
                    "chapter": "Chương 4: Dòng điện. Mạch điện",
                    "topics": [
                        {"topic": "Dòng điện và cường độ dòng điện", "lessons": ["Dòng điện", "Cường độ dòng điện"], "periods": 2},
                        {"topic": "Điện trở. Định luật Ohm", "lessons": ["Điện trở", "Định luật Ohm cho đoạn mạch"], "periods": 4},
                        {"topic": "Năng lượng điện", "lessons": ["Công và công suất điện", "Định luật Joule-Lenz"], "periods": 3},
                        {"topic": "Mạch điện", "lessons": ["Ghép điện trở", "Định luật Ohm cho toàn mạch"], "periods": 4},
                    ],
                    "knowledge": "Hiểu dòng điện, điện trở, mạch điện",
                    "skills": "Giải mạch điện, tính công suất, hiệu suất",
                },
                {
                    "chapter": "Chương 5: Từ trường",
                    "topics": [
                        {"topic": "Từ trường", "lessons": ["Từ trường", "Đường sức từ", "Từ trường của dòng điện"], "periods": 4},
                        {"topic": "Lực từ", "lessons": ["Lực từ tác dụng lên đoạn dây dẫn mang dòng điện", "Lực Lorentz"], "periods": 4},
                    ],
                    "knowledge": "Hiểu từ trường, lực từ, lực Lorentz",
                    "skills": "Xác định từ trường, tính lực từ",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Cảm ứng điện từ",
                    "topics": [
                        {"topic": "Từ thông và cảm ứng điện từ", "lessons": ["Từ thông", "Cảm ứng điện từ", "Suất điện động cảm ứng"], "periods": 5},
                        {"topic": "Tự cảm", "lessons": ["Hiện tượng tự cảm", "Năng lượng từ trường"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cảm ứng điện từ, tự cảm",
                    "skills": "Tính suất điện động cảm ứng, độ tự cảm",
                },
                {
                    "chapter": "Chương 2: Dòng điện xoay chiều",
                    "topics": [
                        {"topic": "Đại cương về dòng điện xoay chiều", "lessons": ["Dòng điện xoay chiều", "Giá trị hiệu dụng"], "periods": 3},
                        {"topic": "Mạch điện xoay chiều", "lessons": ["Mạch điện xoay chiều chỉ có R, L, C", "Mạch RLC nối tiếp", "Cộng hưởng điện"], "periods": 6},
                        {"topic": "Máy biến áp và truyền tải điện năng", "lessons": ["Máy biến áp", "Truyền tải điện năng"], "periods": 3},
                    ],
                    "knowledge": "Hiểu dòng điện xoay chiều, mạch RLC, máy biến áp",
                    "skills": "Giải bài toán mạch xoay chiều, tính công suất",
                },
                {
                    "chapter": "Chương 3: Sóng điện từ. Truyền thông",
                    "topics": [
                        {"topic": "Sóng điện từ", "lessons": ["Điện từ trường", "Sóng điện từ", "Thang sóng điện từ"], "periods": 4},
                        {"topic": "Truyền thông bằng sóng điện từ", "lessons": ["Nguyên tắc truyền thông bằng sóng điện từ"], "periods": 2},
                    ],
                    "knowledge": "Hiểu sóng điện từ, nguyên tắc truyền thông",
                    "skills": "Phân loại sóng điện từ, giải thích ứng dụng",
                },
                {
                    "chapter": "Chương 4: Sóng ánh sáng",
                    "topics": [
                        {"topic": "Tán sắc ánh sáng", "lessons": ["Tán sắc ánh sáng", "Phổ ánh sáng"], "periods": 3},
                        {"topic": "Giao thoa ánh sáng", "lessons": ["Giao thoa ánh sáng", "Thí nghiệm Young"], "periods": 4},
                    ],
                    "knowledge": "Hiểu tán sắc, giao thoa ánh sáng",
                    "skills": "Tính bước sóng, khoảng vân giao thoa",
                },
                {
                    "chapter": "Chương 5: Lượng tử ánh sáng",
                    "topics": [
                        {"topic": "Hiện tượng quang điện", "lessons": ["Hiện tượng quang điện", "Thuyết lượng tử ánh sáng"], "periods": 4},
                        {"topic": "Mẫu nguyên tử Bohr", "lessons": ["Quang phổ vạch của nguyên tử hidro", "Mẫu nguyên tử Bohr"], "periods": 3},
                    ],
                    "knowledge": "Hiểu hiện tượng quang điện, mẫu Bohr",
                    "skills": "Tính năng lượng photon, giới hạn quang điện",
                },
                {
                    "chapter": "Chương 6: Vật lý hạt nhân",
                    "topics": [
                        {"topic": "Cấu tạo hạt nhân", "lessons": ["Cấu tạo hạt nhân", "Năng lượng liên kết hạt nhân"], "periods": 3},
                        {"topic": "Phóng xạ", "lessons": ["Hiện tượng phóng xạ", "Định luật phóng xạ"], "periods": 4},
                        {"topic": "Phản ứng hạt nhân", "lessons": ["Phản ứng hạt nhân", "Phản ứng phân hạch và nhiệt hạch"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cấu tạo hạt nhân, phóng xạ, phản ứng hạt nhân",
                    "skills": "Tính năng lượng liên kết, chu kỳ bán rã",
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
                        {"topic": "Thành phần nguyên tử", "lessons": ["Thành phần nguyên tử", "Hạt nhân nguyên tử"], "periods": 3},
                        {"topic": "Cấu hình electron", "lessons": ["Lớp và phân lớp electron", "Cấu hình electron nguyên tử"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cấu tạo nguyên tử, cấu hình electron",
                    "skills": "Viết cấu hình electron, xác định vị trí nguyên tố",
                },
                {
                    "chapter": "Chương 2: Bảng tuần hoàn các nguyên tố hóa học",
                    "topics": [
                        {"topic": "Bảng tuần hoàn", "lessons": ["Bảng tuần hoàn các nguyên tố hóa học", "Xu hướng biến đổi tính chất"], "periods": 4},
                        {"topic": "Định luật tuần hoàn", "lessons": ["Định luật tuần hoàn", "Ý nghĩa của bảng tuần hoàn"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cấu trúc bảng tuần hoàn, quy luật biến đổi",
                    "skills": "Xác định vị trí, dự đoán tính chất nguyên tố",
                },
                {
                    "chapter": "Chương 3: Liên kết hóa học",
                    "topics": [
                        {"topic": "Liên kết ion", "lessons": ["Liên kết ion"], "periods": 2},
                        {"topic": "Liên kết cộng hóa trị", "lessons": ["Liên kết cộng hóa trị", "Liên kết cho – nhận"], "periods": 3},
                        {"topic": "Liên kết hydrogen và tương tác van der Waals", "lessons": ["Liên kết hydrogen", "Tương tác van der Waals"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các loại liên kết hóa học",
                    "skills": "Xác định loại liên kết, viết công thức Lewis",
                },
                {
                    "chapter": "Chương 4: Phản ứng oxi hóa – khử",
                    "topics": [
                        {"topic": "Số oxi hóa", "lessons": ["Số oxi hóa", "Quy tắc xác định số oxi hóa"], "periods": 2},
                        {"topic": "Phản ứng oxi hóa – khử", "lessons": ["Phản ứng oxi hóa – khử", "Cân bằng phản ứng oxi hóa – khử"], "periods": 4},
                    ],
                    "knowledge": "Hiểu phản ứng oxi hóa – khử",
                    "skills": "Cân bằng phương trình oxi hóa – khử",
                },
                {
                    "chapter": "Chương 5: Năng lượng hóa học",
                    "topics": [
                        {"topic": "Biến thiên enthalpy", "lessons": ["Phản ứng tỏa nhiệt, thu nhiệt", "Biến thiên enthalpy chuẩn"], "periods": 4},
                    ],
                    "knowledge": "Hiểu năng lượng hóa học, enthalpy",
                    "skills": "Tính biến thiên enthalpy, dự đoán chiều phản ứng",
                },
                {
                    "chapter": "Chương 6: Tốc độ phản ứng hóa học",
                    "topics": [
                        {"topic": "Tốc độ phản ứng", "lessons": ["Tốc độ phản ứng hóa học", "Các yếu tố ảnh hưởng đến tốc độ phản ứng"], "periods": 4},
                    ],
                    "knowledge": "Hiểu tốc độ phản ứng và các yếu tố ảnh hưởng",
                    "skills": "Phân tích yếu tố ảnh hưởng, tính tốc độ phản ứng",
                },
                {
                    "chapter": "Chương 7: Nguyên tố nhóm VIIA (Halogen)",
                    "topics": [
                        {"topic": "Đơn chất halogen", "lessons": ["Khái quát nhóm halogen", "Chlorine", "Hydrogen chloride và muối chloride"], "periods": 5},
                    ],
                    "knowledge": "Hiểu tính chất nhóm halogen",
                    "skills": "Viết phương trình phản ứng, nhận biết halogen",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Cân bằng hóa học",
                    "topics": [
                        {"topic": "Cân bằng hóa học", "lessons": ["Cân bằng trong phản ứng hóa học", "Hằng số cân bằng"], "periods": 4},
                        {"topic": "Sự dịch chuyển cân bằng", "lessons": ["Nguyên lý Le Chatelier", "Các yếu tố ảnh hưởng đến cân bằng"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cân bằng hóa học, nguyên lý Le Chatelier",
                    "skills": "Tính hằng số cân bằng, dự đoán chiều dịch chuyển",
                },
                {
                    "chapter": "Chương 2: Nitrogen và sulfur",
                    "topics": [
                        {"topic": "Nitrogen và hợp chất", "lessons": ["Đơn chất nitrogen", "Ammonia và muối ammonium", "Axit nitric và muối nitrate"], "periods": 6},
                        {"topic": "Sulfur và hợp chất", "lessons": ["Đơn chất sulfur", "Sulfur dioxide và sulfur trioxide", "Axit sulfuric và muối sulfate"], "periods": 5},
                    ],
                    "knowledge": "Hiểu tính chất nitrogen, sulfur và hợp chất",
                    "skills": "Viết phương trình, giải bài tập định lượng",
                },
                {
                    "chapter": "Chương 3: Đại cương về hóa học hữu cơ",
                    "topics": [
                        {"topic": "Hợp chất hữu cơ", "lessons": ["Hợp chất hữu cơ và hóa học hữu cơ", "Phân loại hợp chất hữu cơ"], "periods": 2},
                        {"topic": "Công thức phân tử hợp chất hữu cơ", "lessons": ["Phương pháp phân tích nguyên tố", "Công thức phân tử"], "periods": 3},
                        {"topic": "Cấu trúc phân tử hợp chất hữu cơ", "lessons": ["Công thức cấu tạo", "Đồng đẳng, đồng phân"], "periods": 3},
                    ],
                    "knowledge": "Hiểu đặc điểm hợp chất hữu cơ, đồng đẳng, đồng phân",
                    "skills": "Viết công thức cấu tạo, xác định đồng phân",
                },
                {
                    "chapter": "Chương 4: Hydrocarbon",
                    "topics": [
                        {"topic": "Alkane", "lessons": ["Alkane", "Phản ứng của alkane"], "periods": 3},
                        {"topic": "Alkene", "lessons": ["Alkene", "Phản ứng của alkene"], "periods": 3},
                        {"topic": "Alkyne", "lessons": ["Alkyne", "Phản ứng của alkyne"], "periods": 3},
                        {"topic": "Arene (Hydrocarbon thơm)", "lessons": ["Benzene và đồng đẳng", "Tính chất hóa học của arene"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cấu trúc, tính chất các loại hydrocarbon",
                    "skills": "Viết phương trình, phân biệt hydrocarbon",
                },
                {
                    "chapter": "Chương 5: Dẫn xuất halogen. Alcohol. Phenol",
                    "topics": [
                        {"topic": "Dẫn xuất halogen", "lessons": ["Dẫn xuất halogen của hydrocarbon"], "periods": 2},
                        {"topic": "Alcohol", "lessons": ["Alcohol", "Tính chất hóa học của alcohol"], "periods": 4},
                        {"topic": "Phenol", "lessons": ["Phenol"], "periods": 2},
                    ],
                    "knowledge": "Hiểu dẫn xuất halogen, alcohol, phenol",
                    "skills": "Viết phương trình, so sánh tính chất",
                },
                {
                    "chapter": "Chương 6: Hợp chất carbonyl. Carboxylic acid",
                    "topics": [
                        {"topic": "Aldehyde và ketone", "lessons": ["Aldehyde", "Ketone", "Phản ứng đặc trưng"], "periods": 4},
                        {"topic": "Carboxylic acid", "lessons": ["Carboxylic acid", "Tính chất hóa học"], "periods": 3},
                    ],
                    "knowledge": "Hiểu hợp chất carbonyl, carboxylic acid",
                    "skills": "Viết phương trình, nhận biết hợp chất",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Ester – Lipid",
                    "topics": [
                        {"topic": "Ester", "lessons": ["Ester", "Phản ứng este hóa", "Phản ứng thủy phân"], "periods": 4},
                        {"topic": "Lipid", "lessons": ["Chất béo", "Xà phòng và chất giặt rửa"], "periods": 3},
                    ],
                    "knowledge": "Hiểu ester, lipid, phản ứng este hóa, thủy phân",
                    "skills": "Viết phương trình phản ứng, tính toán",
                },
                {
                    "chapter": "Chương 2: Carbohydrate",
                    "topics": [
                        {"topic": "Glucose và fructose", "lessons": ["Glucose", "Fructose"], "periods": 3},
                        {"topic": "Saccharose và maltose", "lessons": ["Saccharose", "Maltose"], "periods": 2},
                        {"topic": "Tinh bột và cellulose", "lessons": ["Tinh bột", "Cellulose"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cấu trúc, tính chất các loại carbohydrate",
                    "skills": "Phân biệt, viết phương trình phản ứng",
                },
                {
                    "chapter": "Chương 3: Hợp chất chứa nitrogen",
                    "topics": [
                        {"topic": "Amine", "lessons": ["Amine", "Tính chất hóa học của amine"], "periods": 3},
                        {"topic": "Amino acid", "lessons": ["Amino acid", "Peptide"], "periods": 3},
                        {"topic": "Protein", "lessons": ["Protein", "Enzyme"], "periods": 3},
                    ],
                    "knowledge": "Hiểu amine, amino acid, protein",
                    "skills": "Viết phương trình, nhận biết hợp chất nitrogen",
                },
                {
                    "chapter": "Chương 4: Polymer",
                    "topics": [
                        {"topic": "Đại cương về polymer", "lessons": ["Khái niệm polymer", "Phân loại polymer"], "periods": 2},
                        {"topic": "Vật liệu polymer", "lessons": ["Chất dẻo", "Tơ", "Cao su"], "periods": 4},
                    ],
                    "knowledge": "Hiểu polymer, vật liệu polymer",
                    "skills": "Phân loại polymer, viết phản ứng trùng hợp",
                },
                {
                    "chapter": "Chương 5: Đại cương về kim loại",
                    "topics": [
                        {"topic": "Kim loại", "lessons": ["Tính chất vật lý của kim loại", "Tính chất hóa học của kim loại"], "periods": 4},
                        {"topic": "Dãy hoạt động hóa học", "lessons": ["Dãy hoạt động hóa học của kim loại"], "periods": 2},
                        {"topic": "Hợp kim", "lessons": ["Hợp kim"], "periods": 1},
                        {"topic": "Ăn mòn kim loại", "lessons": ["Ăn mòn hóa học", "Ăn mòn điện hóa", "Chống ăn mòn kim loại"], "periods": 3},
                        {"topic": "Điều chế kim loại", "lessons": ["Nguyên tắc và phương pháp điều chế kim loại"], "periods": 3},
                    ],
                    "knowledge": "Hiểu tính chất kim loại, ăn mòn, điều chế",
                    "skills": "Giải bài toán kim loại, phân biệt phương pháp điều chế",
                },
                {
                    "chapter": "Chương 6: Kim loại kiềm, kiềm thổ, nhôm",
                    "topics": [
                        {"topic": "Kim loại kiềm", "lessons": ["Kim loại kiềm và hợp chất"], "periods": 3},
                        {"topic": "Kim loại kiềm thổ", "lessons": ["Kim loại kiềm thổ và hợp chất", "Nước cứng"], "periods": 3},
                        {"topic": "Nhôm", "lessons": ["Nhôm và hợp chất của nhôm"], "periods": 3},
                    ],
                    "knowledge": "Hiểu tính chất kim loại kiềm, kiềm thổ, nhôm",
                    "skills": "Viết phương trình, giải bài tập",
                },
                {
                    "chapter": "Chương 7: Sắt và một số kim loại quan trọng",
                    "topics": [
                        {"topic": "Sắt", "lessons": ["Sắt", "Hợp chất của sắt"], "periods": 4},
                        {"topic": "Chromium và đồng", "lessons": ["Chromium và hợp chất", "Đồng và hợp chất"], "periods": 3},
                        {"topic": "Nhận biết ion kim loại", "lessons": ["Nhận biết một số ion trong dung dịch"], "periods": 2},
                    ],
                    "knowledge": "Hiểu tính chất sắt, chromium, đồng",
                    "skills": "Nhận biết ion kim loại, giải bài tập",
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
                {
                    "chapter": "Chương 2: Cấu trúc tế bào",
                    "topics": [
                        {"topic": "Tế bào nhân sơ và nhân thực", "lessons": ["Tế bào nhân sơ", "Tế bào nhân thực"], "periods": 4},
                        {"topic": "Các bào quan", "lessons": ["Nhân tế bào", "Lưới nội chất và ribosome", "Bộ máy Golgi và lysosome", "Ti thể và lục lạp"], "periods": 6},
                        {"topic": "Màng sinh chất", "lessons": ["Cấu trúc màng sinh chất", "Vận chuyển các chất qua màng"], "periods": 4},
                    ],
                    "knowledge": "Hiểu cấu trúc và chức năng các thành phần tế bào",
                    "skills": "So sánh tế bào nhân sơ và nhân thực, mô tả bào quan",
                },
                {
                    "chapter": "Chương 3: Trao đổi chất và chuyển hóa năng lượng ở tế bào",
                    "topics": [
                        {"topic": "Enzyme", "lessons": ["Enzyme và vai trò của enzyme", "Các yếu tố ảnh hưởng đến hoạt tính enzyme"], "periods": 3},
                        {"topic": "Hô hấp tế bào", "lessons": ["Tổng quan hô hấp tế bào", "Đường phân", "Chu trình Krebs", "Chuỗi truyền electron"], "periods": 5},
                        {"topic": "Quang hợp", "lessons": ["Tổng quan quang hợp", "Pha sáng", "Pha tối"], "periods": 4},
                    ],
                    "knowledge": "Hiểu quá trình chuyển hóa năng lượng trong tế bào",
                    "skills": "Mô tả hô hấp tế bào, quang hợp",
                },
                {
                    "chapter": "Chương 4: Phân bào",
                    "topics": [
                        {"topic": "Chu kỳ tế bào và nguyên phân", "lessons": ["Chu kỳ tế bào", "Nguyên phân"], "periods": 3},
                        {"topic": "Giảm phân", "lessons": ["Giảm phân I", "Giảm phân II"], "periods": 3},
                    ],
                    "knowledge": "Hiểu chu kỳ tế bào, nguyên phân, giảm phân",
                    "skills": "So sánh nguyên phân và giảm phân",
                },
            ],
            11: [
                {
                    "chapter": "Chương 1: Trao đổi chất và chuyển hóa năng lượng ở sinh vật",
                    "topics": [
                        {"topic": "Trao đổi nước và khoáng ở thực vật", "lessons": ["Hấp thụ nước và muối khoáng", "Vận chuyển các chất trong cây", "Thoát hơi nước"], "periods": 5},
                        {"topic": "Quang hợp ở thực vật", "lessons": ["Quang hợp ở các nhóm thực vật C3, C4, CAM"], "periods": 3},
                        {"topic": "Hô hấp ở thực vật", "lessons": ["Hô hấp ở thực vật", "Mối quan hệ giữa quang hợp và hô hấp"], "periods": 3},
                        {"topic": "Dinh dưỡng và tiêu hóa ở động vật", "lessons": ["Tiêu hóa ở các nhóm động vật", "Hấp thụ chất dinh dưỡng"], "periods": 4},
                        {"topic": "Hô hấp ở động vật", "lessons": ["Hô hấp ở các nhóm động vật"], "periods": 2},
                        {"topic": "Tuần hoàn ở động vật", "lessons": ["Cấu tạo và chức năng hệ tuần hoàn", "Hoạt động của tim và hệ mạch"], "periods": 4},
                    ],
                    "knowledge": "Hiểu trao đổi chất ở thực vật và động vật",
                    "skills": "So sánh quá trình trao đổi chất ở các nhóm sinh vật",
                },
                {
                    "chapter": "Chương 2: Cảm ứng ở sinh vật",
                    "topics": [
                        {"topic": "Cảm ứng ở thực vật", "lessons": ["Hướng động", "Ứng động"], "periods": 3},
                        {"topic": "Cảm ứng ở động vật", "lessons": ["Cảm ứng ở động vật có hệ thần kinh dạng lưới, chuỗi hạch", "Cảm ứng ở động vật có hệ thần kinh dạng ống"], "periods": 4},
                        {"topic": "Truyền tin qua synapse", "lessons": ["Điện thế nghỉ và điện thế hoạt động", "Truyền tin qua synapse"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cảm ứng ở thực vật và động vật",
                    "skills": "Phân tích cơ chế cảm ứng, truyền xung thần kinh",
                },
                {
                    "chapter": "Chương 3: Sinh trưởng và phát triển ở sinh vật",
                    "topics": [
                        {"topic": "Sinh trưởng và phát triển ở thực vật", "lessons": ["Sinh trưởng sơ cấp và thứ cấp", "Hormone thực vật"], "periods": 4},
                        {"topic": "Sinh trưởng và phát triển ở động vật", "lessons": ["Sinh trưởng và phát triển qua biến thái và không qua biến thái", "Hormone ở động vật"], "periods": 4},
                    ],
                    "knowledge": "Hiểu sinh trưởng, phát triển, vai trò hormone",
                    "skills": "So sánh sinh trưởng ở thực vật và động vật",
                },
                {
                    "chapter": "Chương 4: Sinh sản ở sinh vật",
                    "topics": [
                        {"topic": "Sinh sản ở thực vật", "lessons": ["Sinh sản vô tính ở thực vật", "Sinh sản hữu tính ở thực vật có hoa"], "periods": 4},
                        {"topic": "Sinh sản ở động vật", "lessons": ["Sinh sản vô tính ở động vật", "Sinh sản hữu tính ở động vật", "Cơ chế điều hòa sinh sản"], "periods": 4},
                    ],
                    "knowledge": "Hiểu các hình thức sinh sản ở sinh vật",
                    "skills": "So sánh sinh sản vô tính và hữu tính",
                },
            ],
            12: [
                {
                    "chapter": "Chương 1: Di truyền phân tử",
                    "topics": [
                        {"topic": "Gen và mã di truyền", "lessons": ["Gen", "Mã di truyền"], "periods": 2},
                        {"topic": "Nhân đôi ADN", "lessons": ["Cơ chế nhân đôi ADN"], "periods": 2},
                        {"topic": "Phiên mã", "lessons": ["Phiên mã"], "periods": 2},
                        {"topic": "Dịch mã", "lessons": ["Dịch mã", "Mối quan hệ ADN - ARN - Protein"], "periods": 3},
                        {"topic": "Điều hòa biểu hiện gen", "lessons": ["Điều hòa biểu hiện gen ở sinh vật nhân sơ", "Điều hòa biểu hiện gen ở sinh vật nhân thực"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cơ chế di truyền ở mức phân tử",
                    "skills": "Giải bài tập nhân đôi, phiên mã, dịch mã",
                },
                {
                    "chapter": "Chương 2: Đột biến",
                    "topics": [
                        {"topic": "Đột biến gen", "lessons": ["Đột biến gen", "Nguyên nhân và hậu quả đột biến gen"], "periods": 3},
                        {"topic": "Đột biến nhiễm sắc thể", "lessons": ["Đột biến cấu trúc nhiễm sắc thể", "Đột biến số lượng nhiễm sắc thể"], "periods": 4},
                    ],
                    "knowledge": "Hiểu đột biến gen và đột biến nhiễm sắc thể",
                    "skills": "Phân loại đột biến, giải thích hậu quả",
                },
                {
                    "chapter": "Chương 3: Quy luật di truyền",
                    "topics": [
                        {"topic": "Các quy luật Mendel", "lessons": ["Quy luật phân li", "Quy luật phân li độc lập"], "periods": 4},
                        {"topic": "Tương tác gen và tác động đa hiệu", "lessons": ["Tương tác gen", "Gen đa hiệu"], "periods": 3},
                        {"topic": "Liên kết gen và hoán vị gen", "lessons": ["Liên kết gen", "Hoán vị gen"], "periods": 3},
                        {"topic": "Di truyền liên kết giới tính", "lessons": ["Di truyền liên kết với giới tính", "Di truyền ngoài nhân"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các quy luật di truyền",
                    "skills": "Giải bài tập di truyền, xác định kiểu gen",
                },
                {
                    "chapter": "Chương 4: Di truyền học quần thể",
                    "topics": [
                        {"topic": "Cấu trúc di truyền quần thể", "lessons": ["Quần thể tự phối", "Quần thể giao phối ngẫu nhiên"], "periods": 3},
                        {"topic": "Định luật Hardy-Weinberg", "lessons": ["Định luật Hardy-Weinberg", "Các nhân tố tiến hóa"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cấu trúc di truyền quần thể",
                    "skills": "Tính tần số alen, kiểu gen trong quần thể",
                },
                {
                    "chapter": "Chương 5: Tiến hóa",
                    "topics": [
                        {"topic": "Bằng chứng tiến hóa", "lessons": ["Bằng chứng giải phẫu so sánh", "Bằng chứng phôi sinh học", "Bằng chứng sinh học phân tử"], "periods": 3},
                        {"topic": "Cơ chế tiến hóa", "lessons": ["Thuyết tiến hóa của Darwin", "Thuyết tiến hóa tổng hợp hiện đại", "Các nhân tố tiến hóa"], "periods": 5},
                        {"topic": "Loài và sự hình thành loài", "lessons": ["Khái niệm loài", "Các hình thức hình thành loài"], "periods": 3},
                    ],
                    "knowledge": "Hiểu bằng chứng và cơ chế tiến hóa",
                    "skills": "Phân tích bằng chứng tiến hóa, giải thích hình thành loài",
                },
                {
                    "chapter": "Chương 6: Sinh thái học",
                    "topics": [
                        {"topic": "Cá thể và quần thể", "lessons": ["Các nhân tố sinh thái", "Quần thể sinh vật", "Các đặc trưng của quần thể"], "periods": 4},
                        {"topic": "Quần xã và hệ sinh thái", "lessons": ["Quần xã sinh vật", "Hệ sinh thái", "Chuỗi và lưới thức ăn"], "periods": 5},
                        {"topic": "Sinh quyển", "lessons": ["Sinh quyển và các khu sinh học", "Con người và môi trường"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các cấp tổ chức sống: quần thể, quần xã, hệ sinh thái",
                    "skills": "Phân tích mối quan hệ sinh thái, giải thích hiện tượng",
                },
            ],
        },
    },
    "van": {
        "name": "Ngữ văn",
        "grades": {
            10: [
                {
                    "chapter": "Phần 1: Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Thần thoại và sử thi", "lessons": ["Thần thoại Việt Nam", "Sử thi Đăm Săn", "Sử thi Odyssey (trích)"], "periods": 6},
                        {"topic": "Truyện ngắn hiện đại", "lessons": ["Chí Phèo (Nam Cao)", "Hai đứa trẻ (Thạch Lam)"], "periods": 5},
                        {"topic": "Thơ hiện đại", "lessons": ["Vội vàng (Xuân Diệu)", "Tràng giang (Huy Cận)"], "periods": 4},
                        {"topic": "Văn bản nghị luận", "lessons": ["Đọc hiểu văn bản nghị luận xã hội", "Đọc hiểu văn bản nghị luận văn học"], "periods": 4},
                        {"topic": "Văn bản thông tin", "lessons": ["Đọc hiểu văn bản thông tin"], "periods": 3},
                    ],
                    "knowledge": "Đọc hiểu các thể loại văn bản",
                    "skills": "Phân tích, đánh giá nội dung và nghệ thuật văn bản",
                },
                {
                    "chapter": "Phần 2: Viết",
                    "topics": [
                        {"topic": "Nghị luận xã hội", "lessons": ["Viết bài nghị luận về một vấn đề xã hội"], "periods": 4},
                        {"topic": "Nghị luận văn học", "lessons": ["Viết bài nghị luận phân tích một tác phẩm văn học"], "periods": 4},
                        {"topic": "Văn bản thông tin", "lessons": ["Viết văn bản thuyết minh"], "periods": 3},
                    ],
                    "knowledge": "Hiểu cách viết các kiểu bài",
                    "skills": "Viết bài nghị luận, thuyết minh",
                },
                {
                    "chapter": "Phần 3: Nói và nghe",
                    "topics": [
                        {"topic": "Thuyết trình", "lessons": ["Thuyết trình về một vấn đề"], "periods": 2},
                        {"topic": "Tranh luận", "lessons": ["Tranh luận về một vấn đề trong đời sống"], "periods": 2},
                    ],
                    "knowledge": "Kỹ năng nói và nghe",
                    "skills": "Thuyết trình, tranh luận, phản biện",
                },
            ],
            11: [
                {
                    "chapter": "Phần 1: Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Truyện trung đại Việt Nam", "lessons": ["Truyện Kiều (Nguyễn Du) - trích", "Chuyện người con gái Nam Xương (Nguyễn Dữ)"], "periods": 6},
                        {"topic": "Thơ trung đại", "lessons": ["Cảnh ngày hè (Nguyễn Trãi)", "Đọc Tiểu Thanh ký (Nguyễn Du)"], "periods": 4},
                        {"topic": "Truyện ngắn hiện đại", "lessons": ["Chữ người tử tù (Nguyễn Tuân)", "Vợ nhặt (Kim Lân)"], "periods": 5},
                        {"topic": "Thơ hiện đại", "lessons": ["Đây thôn Vĩ Dạ (Hàn Mặc Tử)", "Sóng (Xuân Quỳnh)"], "periods": 4},
                        {"topic": "Kịch", "lessons": ["Vũ Như Tô (Nguyễn Huy Tưởng) - trích", "Hồn Trương Ba, da hàng thịt (Lưu Quang Vũ) - trích"], "periods": 4},
                        {"topic": "Văn bản nghị luận", "lessons": ["Tuyên ngôn độc lập (Hồ Chí Minh)"], "periods": 3},
                    ],
                    "knowledge": "Đọc hiểu văn học trung đại và hiện đại Việt Nam",
                    "skills": "Phân tích chuyên sâu tác phẩm văn học",
                },
                {
                    "chapter": "Phần 2: Viết",
                    "topics": [
                        {"topic": "Nghị luận văn học nâng cao", "lessons": ["Viết bài nghị luận so sánh hai tác phẩm", "Viết bài nghị luận về một ý kiến bàn về văn học"], "periods": 5},
                        {"topic": "Nghị luận xã hội nâng cao", "lessons": ["Viết bài nghị luận về một vấn đề xã hội đặt ra trong tác phẩm văn học"], "periods": 3},
                    ],
                    "knowledge": "Kỹ năng viết nghị luận nâng cao",
                    "skills": "Viết bài phân tích, so sánh, đánh giá",
                },
                {
                    "chapter": "Phần 3: Nói và nghe",
                    "topics": [
                        {"topic": "Thảo luận nhóm", "lessons": ["Thảo luận về một vấn đề văn học"], "periods": 2},
                        {"topic": "Thuyết trình nâng cao", "lessons": ["Thuyết trình kết quả nghiên cứu về một vấn đề"], "periods": 2},
                    ],
                    "knowledge": "Kỹ năng thảo luận, thuyết trình học thuật",
                    "skills": "Trao đổi, thuyết trình có luận điểm rõ ràng",
                },
            ],
            12: [
                {
                    "chapter": "Phần 1: Đọc hiểu văn bản",
                    "topics": [
                        {"topic": "Văn học Việt Nam 1945-1975", "lessons": ["Tuyên ngôn độc lập (Hồ Chí Minh)", "Việt Bắc (Tố Hữu)", "Đất nước (Nguyễn Khoa Điềm)"], "periods": 6},
                        {"topic": "Truyện hiện đại Việt Nam", "lessons": ["Rừng xà nu (Nguyễn Trung Thành)", "Chiếc thuyền ngoài xa (Nguyễn Minh Châu)", "Vợ chồng A Phủ (Tô Hoài)"], "periods": 6},
                        {"topic": "Văn học nước ngoài", "lessons": ["Ông già và biển cả (Hemingway) - trích", "Số phận con người (Sholokhov) - trích"], "periods": 4},
                        {"topic": "Văn bản nghị luận", "lessons": ["Nghị luận xã hội tổng hợp", "Nghị luận văn học tổng hợp"], "periods": 4},
                    ],
                    "knowledge": "Đọc hiểu văn học Việt Nam và thế giới",
                    "skills": "Phân tích, tổng hợp, đánh giá tác phẩm văn học",
                },
                {
                    "chapter": "Phần 2: Viết",
                    "topics": [
                        {"topic": "Nghị luận tổng hợp", "lessons": ["Viết bài nghị luận tổng hợp về một vấn đề văn học", "Viết bài nghị luận tổng hợp về một vấn đề xã hội"], "periods": 5},
                        {"topic": "Viết sáng tạo", "lessons": ["Viết truyện ngắn", "Viết thơ"], "periods": 3},
                    ],
                    "knowledge": "Kỹ năng viết nghị luận tổng hợp và sáng tạo",
                    "skills": "Viết bài hoàn chỉnh, sáng tạo",
                },
                {
                    "chapter": "Phần 3: Nói và nghe",
                    "topics": [
                        {"topic": "Diễn thuyết", "lessons": ["Diễn thuyết về một vấn đề xã hội hoặc văn học"], "periods": 2},
                    ],
                    "knowledge": "Kỹ năng diễn thuyết trước đám đông",
                    "skills": "Trình bày ý kiến thuyết phục, logic",
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
                {
                    "chapter": "Unit 2: Humans and the environment",
                    "topics": [
                        {"topic": "Vocabulary: Environment", "lessons": ["Environmental problems", "Green living"], "periods": 3},
                        {"topic": "Grammar: Past tenses", "lessons": ["Past simple", "Past continuous"], "periods": 2},
                    ],
                    "knowledge": "Vocabulary about environment, past tenses",
                    "skills": "Discussing environmental issues",
                },
                {
                    "chapter": "Unit 3: Music",
                    "topics": [
                        {"topic": "Vocabulary: Music", "lessons": ["Types of music", "Musical instruments"], "periods": 3},
                        {"topic": "Grammar: Comparisons", "lessons": ["Comparative and superlative adjectives"], "periods": 2},
                    ],
                    "knowledge": "Music vocabulary, comparisons",
                    "skills": "Expressing preferences about music",
                },
                {
                    "chapter": "Unit 4: For a better community",
                    "topics": [
                        {"topic": "Vocabulary: Community", "lessons": ["Volunteer work", "Community service"], "periods": 3},
                        {"topic": "Grammar: Past perfect", "lessons": ["Past perfect simple", "Past perfect vs past simple"], "periods": 2},
                    ],
                    "knowledge": "Community vocabulary, past perfect",
                    "skills": "Talking about community activities",
                },
                {
                    "chapter": "Unit 5: Inventions",
                    "topics": [
                        {"topic": "Vocabulary: Technology", "lessons": ["Modern inventions", "Impact of inventions"], "periods": 3},
                        {"topic": "Grammar: Passive voice", "lessons": ["Passive voice in different tenses"], "periods": 2},
                    ],
                    "knowledge": "Technology vocabulary, passive voice",
                    "skills": "Describing inventions and their impact",
                },
            ],
            11: [
                {
                    "chapter": "Unit 1: A long and healthy life",
                    "topics": [
                        {"topic": "Vocabulary: Health", "lessons": ["Healthy lifestyle", "Common health problems"], "periods": 3},
                        {"topic": "Grammar: Gerunds and infinitives", "lessons": ["Gerunds after certain verbs", "Infinitives after certain verbs"], "periods": 2},
                    ],
                    "knowledge": "Health vocabulary, gerunds and infinitives",
                    "skills": "Discussing health issues and giving advice",
                },
                {
                    "chapter": "Unit 2: The generation gap",
                    "topics": [
                        {"topic": "Vocabulary: Family conflicts", "lessons": ["Generation gap issues", "Family values"], "periods": 3},
                        {"topic": "Grammar: Modals", "lessons": ["Should, ought to", "Must, have to"], "periods": 2},
                    ],
                    "knowledge": "Vocabulary about family conflicts, modal verbs",
                    "skills": "Discussing generation gap and giving opinions",
                },
                {
                    "chapter": "Unit 3: Cities of the future",
                    "topics": [
                        {"topic": "Vocabulary: Urban life", "lessons": ["Smart cities", "Urban problems and solutions"], "periods": 3},
                        {"topic": "Grammar: Conditionals", "lessons": ["First conditional", "Second conditional"], "periods": 3},
                    ],
                    "knowledge": "Urban vocabulary, conditional sentences",
                    "skills": "Predicting future trends, discussing urban issues",
                },
                {
                    "chapter": "Unit 4: ASEAN and Viet Nam",
                    "topics": [
                        {"topic": "Vocabulary: International organizations", "lessons": ["ASEAN countries", "Cultural diversity"], "periods": 3},
                        {"topic": "Grammar: Relative clauses", "lessons": ["Defining relative clauses", "Non-defining relative clauses"], "periods": 2},
                    ],
                    "knowledge": "Vocabulary about ASEAN, relative clauses",
                    "skills": "Presenting information about countries and cultures",
                },
                {
                    "chapter": "Unit 5: Global warming",
                    "topics": [
                        {"topic": "Vocabulary: Climate change", "lessons": ["Causes and effects of global warming", "Solutions to climate change"], "periods": 3},
                        {"topic": "Grammar: Reported speech", "lessons": ["Reported statements", "Reported questions"], "periods": 3},
                    ],
                    "knowledge": "Climate change vocabulary, reported speech",
                    "skills": "Discussing environmental issues, reporting information",
                },
            ],
            12: [
                {
                    "chapter": "Unit 1: Life stories",
                    "topics": [
                        {"topic": "Vocabulary: Biographies", "lessons": ["Famous people", "Achievements and milestones"], "periods": 3},
                        {"topic": "Grammar: Cleft sentences", "lessons": ["It is/was... that/who...", "What... is/was..."], "periods": 2},
                    ],
                    "knowledge": "Biography vocabulary, cleft sentences",
                    "skills": "Telling life stories, emphasizing information",
                },
                {
                    "chapter": "Unit 2: Urbanisation",
                    "topics": [
                        {"topic": "Vocabulary: Urbanisation", "lessons": ["Rural-urban migration", "Urbanisation problems"], "periods": 3},
                        {"topic": "Grammar: Adverbial clauses", "lessons": ["Clauses of reason", "Clauses of concession"], "periods": 2},
                    ],
                    "knowledge": "Urbanisation vocabulary, adverbial clauses",
                    "skills": "Discussing urbanisation trends and impacts",
                },
                {
                    "chapter": "Unit 3: The green movement",
                    "topics": [
                        {"topic": "Vocabulary: Sustainability", "lessons": ["Green products", "Sustainable development"], "periods": 3},
                        {"topic": "Grammar: Review of tenses", "lessons": ["Mixed tenses in context"], "periods": 2},
                    ],
                    "knowledge": "Sustainability vocabulary, tense review",
                    "skills": "Discussing green initiatives",
                },
                {
                    "chapter": "Unit 4: The mass media",
                    "topics": [
                        {"topic": "Vocabulary: Media", "lessons": ["Types of mass media", "Social media influence"], "periods": 3},
                        {"topic": "Grammar: Participle and to-infinitive clauses", "lessons": ["Present participle clauses", "Past participle clauses"], "periods": 2},
                    ],
                    "knowledge": "Media vocabulary, participle clauses",
                    "skills": "Analyzing media influence, expressing opinions",
                },
                {
                    "chapter": "Unit 5: Artificial intelligence",
                    "topics": [
                        {"topic": "Vocabulary: AI and technology", "lessons": ["AI applications", "Future of AI"], "periods": 3},
                        {"topic": "Grammar: Mixed conditionals", "lessons": ["Third conditional", "Mixed conditionals"], "periods": 3},
                    ],
                    "knowledge": "AI vocabulary, mixed conditionals",
                    "skills": "Discussing AI benefits and risks",
                },
            ],
        },
    },
    "su_dia": {
        "name": "Lịch sử & Địa lý",
        "grades": {
            10: [
                {
                    "chapter": "Phần Lịch sử: Một số nền văn minh thế giới thời cổ - trung đại",
                    "topics": [
                        {"topic": "Văn minh Ai Cập và Lưỡng Hà", "lessons": ["Văn minh Ai Cập cổ đại", "Văn minh Lưỡng Hà cổ đại"], "periods": 4},
                        {"topic": "Văn minh Ấn Độ và Trung Hoa", "lessons": ["Văn minh Ấn Độ cổ đại", "Văn minh Trung Hoa cổ đại"], "periods": 4},
                        {"topic": "Văn minh Hi Lạp và La Mã", "lessons": ["Văn minh Hi Lạp cổ đại", "Văn minh La Mã cổ đại"], "periods": 4},
                        {"topic": "Văn minh Đông Nam Á", "lessons": ["Các nền văn minh cổ Đông Nam Á"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các nền văn minh cổ đại thế giới",
                    "skills": "So sánh, đánh giá đặc trưng các nền văn minh",
                },
                {
                    "chapter": "Phần Lịch sử: Các cuộc cách mạng công nghiệp",
                    "topics": [
                        {"topic": "Cách mạng công nghiệp", "lessons": ["Cách mạng công nghiệp lần thứ nhất và thứ hai", "Cách mạng công nghiệp lần thứ ba và thứ tư"], "periods": 4},
                    ],
                    "knowledge": "Hiểu các cuộc cách mạng công nghiệp",
                    "skills": "Phân tích tác động của cách mạng công nghiệp",
                },
                {
                    "chapter": "Phần Địa lý: Một số vấn đề về khoa học Địa lý",
                    "topics": [
                        {"topic": "Bản đồ", "lessons": ["Một số phép chiếu bản đồ", "Sử dụng bản đồ trong học tập và đời sống"], "periods": 3},
                        {"topic": "Hệ thống thông tin địa lý (GIS)", "lessons": ["Khái niệm GIS", "Ứng dụng GIS"], "periods": 2},
                    ],
                    "knowledge": "Hiểu phương pháp nghiên cứu địa lý, GIS",
                    "skills": "Đọc bản đồ, sử dụng GIS cơ bản",
                },
                {
                    "chapter": "Phần Địa lý: Địa lý tự nhiên",
                    "topics": [
                        {"topic": "Vũ trụ và Trái Đất", "lessons": ["Hệ Mặt Trời", "Trái Đất và các chuyển động"], "periods": 4},
                        {"topic": "Thạch quyển", "lessons": ["Cấu trúc Trái Đất", "Nội lực và ngoại lực"], "periods": 4},
                        {"topic": "Khí quyển", "lessons": ["Khí quyển", "Các nhân tố ảnh hưởng đến khí hậu"], "periods": 3},
                        {"topic": "Thủy quyển", "lessons": ["Thủy quyển và vòng tuần hoàn nước"], "periods": 2},
                        {"topic": "Sinh quyển và thổ nhưỡng quyển", "lessons": ["Sinh quyển", "Thổ nhưỡng quyển"], "periods": 3},
                    ],
                    "knowledge": "Hiểu các thành phần tự nhiên của Trái Đất",
                    "skills": "Phân tích mối quan hệ giữa các thành phần tự nhiên",
                },
            ],
            11: [
                {
                    "chapter": "Phần Lịch sử: Chiến tranh và hòa bình",
                    "topics": [
                        {"topic": "Chiến tranh thế giới thứ nhất", "lessons": ["Nguyên nhân", "Diễn biến và kết quả"], "periods": 3},
                        {"topic": "Chiến tranh thế giới thứ hai", "lessons": ["Nguyên nhân", "Diễn biến chính", "Kết quả và bài học"], "periods": 5},
                        {"topic": "Trật tự thế giới sau 1945", "lessons": ["Trật tự hai cực Yalta", "Chiến tranh lạnh"], "periods": 3},
                        {"topic": "Liên hợp quốc", "lessons": ["Sự ra đời và vai trò của Liên hợp quốc"], "periods": 2},
                    ],
                    "knowledge": "Hiểu chiến tranh thế giới, trật tự thế giới",
                    "skills": "Phân tích nguyên nhân, đánh giá hậu quả chiến tranh",
                },
                {
                    "chapter": "Phần Lịch sử: Cách mạng Việt Nam",
                    "topics": [
                        {"topic": "Phong trào giải phóng dân tộc", "lessons": ["Phong trào dân tộc dân chủ đầu thế kỷ XX", "Nguyễn Ái Quốc và con đường cứu nước"], "periods": 4},
                        {"topic": "Cách mạng tháng Tám 1945", "lessons": ["Cách mạng tháng Tám 1945"], "periods": 3},
                    ],
                    "knowledge": "Hiểu lịch sử cách mạng Việt Nam",
                    "skills": "Phân tích sự kiện lịch sử, đánh giá nhân vật",
                },
                {
                    "chapter": "Phần Địa lý: Địa lý kinh tế - xã hội thế giới",
                    "topics": [
                        {"topic": "Dân số thế giới", "lessons": ["Dân số và phân bố dân cư", "Đô thị hóa"], "periods": 3},
                        {"topic": "Các nền kinh tế lớn", "lessons": ["Nền kinh tế Mỹ", "Liên minh châu Âu (EU)", "Nhật Bản", "Trung Quốc"], "periods": 6},
                        {"topic": "Đông Nam Á", "lessons": ["Tự nhiên Đông Nam Á", "Kinh tế - xã hội Đông Nam Á", "Hiệp hội ASEAN"], "periods": 4},
                    ],
                    "knowledge": "Hiểu địa lý kinh tế - xã hội thế giới",
                    "skills": "Phân tích số liệu, đánh giá tiềm năng kinh tế",
                },
            ],
            12: [
                {
                    "chapter": "Phần Lịch sử: Việt Nam 1945-1975",
                    "topics": [
                        {"topic": "Kháng chiến chống Pháp", "lessons": ["Tình hình sau Cách mạng tháng Tám", "Chiến dịch Điện Biên Phủ", "Hiệp định Geneva"], "periods": 5},
                        {"topic": "Kháng chiến chống Mỹ", "lessons": ["Xây dựng CNXH ở miền Bắc", "Phong trào cách mạng miền Nam", "Chiến dịch Hồ Chí Minh 1975"], "periods": 6},
                    ],
                    "knowledge": "Hiểu lịch sử kháng chiến chống Pháp và Mỹ",
                    "skills": "Phân tích sự kiện, đánh giá ý nghĩa lịch sử",
                },
                {
                    "chapter": "Phần Lịch sử: Việt Nam từ 1975 đến nay",
                    "topics": [
                        {"topic": "Thống nhất và đổi mới", "lessons": ["Thống nhất đất nước", "Công cuộc đổi mới từ 1986", "Thành tựu và thách thức"], "periods": 4},
                    ],
                    "knowledge": "Hiểu quá trình đổi mới đất nước",
                    "skills": "Đánh giá thành tựu đổi mới",
                },
                {
                    "chapter": "Phần Địa lý: Địa lý Việt Nam",
                    "topics": [
                        {"topic": "Vị trí địa lý và tài nguyên", "lessons": ["Vị trí địa lý", "Tài nguyên thiên nhiên", "Đất, nước, sinh vật", "Biển Đông và chủ quyền biển đảo"], "periods": 5},
                        {"topic": "Dân cư và lao động", "lessons": ["Đặc điểm dân số", "Lao động và việc làm", "Đô thị hóa ở Việt Nam"], "periods": 4},
                        {"topic": "Kinh tế Việt Nam", "lessons": ["Chuyển dịch cơ cấu kinh tế", "Nông nghiệp", "Công nghiệp", "Dịch vụ", "Giao thông vận tải"], "periods": 6},
                        {"topic": "Các vùng kinh tế", "lessons": ["Đồng bằng sông Hồng", "Trung du miền núi Bắc Bộ", "Bắc Trung Bộ và Duyên hải miền Trung", "Tây Nguyên", "Đông Nam Bộ", "Đồng bằng sông Cửu Long"], "periods": 8},
                    ],
                    "knowledge": "Hiểu địa lý tự nhiên và kinh tế - xã hội Việt Nam",
                    "skills": "Phân tích tiềm năng, thách thức phát triển các vùng",
                },
            ],
        },
    },
}


def _merge_curriculum_sources() -> dict:
    """Merge all curriculum data sources into one dict."""
    from .data_k12 import CURRICULUM_1_TO_9

    merged = {}

    # Add grades 1-9 data first
    for subj, subj_data in CURRICULUM_1_TO_9.items():
        merged[subj] = {
            "name": subj_data["name"],
            "grades": dict(subj_data.get("grades", {})),
        }

    # Merge grades 10-12 data (SAMPLE_CURRICULUM_DATA)
    for subj, subj_data in SAMPLE_CURRICULUM_DATA.items():
        if subj in merged:
            merged[subj]["grades"].update(subj_data.get("grades", {}))
        else:
            merged[subj] = {
                "name": subj_data["name"],
                "grades": dict(subj_data.get("grades", {})),
            }

    return merged


def get_sample_curriculum(subject: str = None, grade: int = None) -> list:
    """
    Convert sample curriculum data to list of dict for database import.

    Args:
        subject: Subject key (toan, ly, hoa, sinh, van, anh, khtn, su_dia, etc.)
        grade: Grade level (1-12)

    Returns:
        List of curriculum items ready for database insertion
    """
    all_data = _merge_curriculum_sources()
    items = []

    subjects_to_process = [subject] if subject else all_data.keys()

    for subj in subjects_to_process:
        if subj not in all_data:
            continue

        subj_data = all_data[subj]
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
