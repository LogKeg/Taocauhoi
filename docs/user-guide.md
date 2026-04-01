# Hướng dẫn sử dụng - Tạo đề online

## Mục lục

- [1. Cài đặt và khởi chạy](#1-cài-đặt-và-khởi-chạy)
- [2. Cấu hình AI Engine](#2-cấu-hình-ai-engine)
- [3. Tạo đề từ đề mẫu](#3-tạo-đề-từ-đề-mẫu)
- [4. Tạo theo chủ đề (AI)](#4-tạo-theo-chủ-đề-ai)
- [5. Ma trận đề thi (AI)](#5-ma-trận-đề-thi-ai)
- [6. Lấy từ ngân hàng](#6-lấy-từ-ngân-hàng)
- [7. Xuất đề chuyên nghiệp](#7-xuất-đề-chuyên-nghiệp)
- [8. Chuyển đổi Word → Excel](#8-chuyển-đổi-word--excel)
- [9. Ngân hàng câu hỏi](#9-ngân-hàng-câu-hỏi)
- [10. Bổ sung đáp án bằng AI](#10-bổ-sung-đáp-án-bằng-ai)
- [11. Import dữ liệu](#11-import-dữ-liệu)
- [12. Kho lưu trữ](#12-kho-lưu-trữ)
- [13. Khung chương trình](#13-khung-chương-trình)
- [14. Dashboard thống kê](#14-dashboard-thống-kê)
- [15. Chấm bài OMR Scanner](#15-chấm-bài-omr-scanner)
- [16. Chấm bài viết tay](#16-chấm-bài-viết-tay)
- [17. Lịch sử](#17-lịch-sử)
- [18. Cài đặt](#18-cài-đặt)

---

## 1. Cài đặt và khởi chạy

### Yêu cầu hệ thống
- **Python:** 3.10 trở lên
- **RAM:** tối thiểu 2GB
- **Dung lượng ổ cứng:** 500MB (chưa tính models AI)

### Bước 1: Cài đặt Python dependencies

```bash
cd "Tạo đề online"
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# hoặc: .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Bước 2: Cấu hình API keys (tùy chọn)

Tạo file `.env` tại thư mục gốc:

```env
# OpenAI (nếu dùng GPT)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Google Gemini (nếu dùng Gemini)
GEMINI_API_KEY=your-gemini-key

# Anthropic Claude (nếu dùng Claude)
ANTHROPIC_API_KEY=your-claude-key

# API cho import câu hỏi quốc tế
QUIZAPI_KEY=your-quizapi-key
API_NINJAS_KEY=your-api-ninjas-key
```

### Bước 3: Cài Ollama (AI miễn phí, chạy local)

Nếu không có API key OpenAI/Gemini, có thể dùng Ollama miễn phí:

```bash
# Cài Ollama (macOS)
brew install ollama

# Hoặc tải từ https://ollama.com

# Tải model
ollama pull qwen3.5:4b

# Chạy Ollama server
ollama serve
```

### Bước 4: Khởi chạy ứng dụng

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://localhost:8000**

---

## 2. Cấu hình AI Engine

Vào tab **Cài đặt** (biểu tượng bánh răng) để cấu hình AI:

| Engine | Mô tả | API Key | Model mặc định |
|--------|--------|---------|-----------------|
| **Ollama** | Miễn phí, chạy local | Không cần | qwen3.5:4b |
| **OpenAI** | GPT-4o, chất lượng cao | Cần | gpt-4o-mini |
| **Google Gemini** | Nhanh, miễn phí tier | Cần | gemini-2.0-flash |
| **Anthropic Claude** | Chất lượng cao | Cần | claude-sonnet |

**Các bước cấu hình:**
1. Vào tab **Cài đặt**
2. Nhập API Key cho engine muốn dùng
3. Chọn Model phù hợp
4. Bấm **Lưu cài đặt**
5. Kiểm tra trạng thái engine (dấu chấm xanh = hoạt động)

---

## 3. Tạo đề từ đề mẫu

**Mục đích:** Upload file Word (.docx) chứa đề mẫu, AI tạo đề mới tương tự.

### Các bước:

1. **Upload file đề mẫu:** Bấm chọn file .docx chứa câu hỏi
2. **Bấm "Phân tích đề":** Hệ thống đếm số câu hỏi trong file
3. **Cấu hình:**
   - **Môn học:** Tự động nhận dạng hoặc chọn thủ công
   - **Độ khó:** Tương đương / Dễ hơn / Khó hơn so với đề mẫu
   - **Ngôn ngữ:** Tự động / Song ngữ / Tiếng Anh / Tiếng Việt
   - **AI Engine:** Chọn engine đã cấu hình
4. **Bấm "Tạo đề mới"**
5. **Xuất kết quả:** DOCX hoặc PDF

---

## 4. Tạo theo chủ đề (AI)

**Mục đích:** AI tự tạo câu hỏi theo môn học, chủ đề, lớp, và mức độ khó.

### Các bước:

1. **Chọn Môn học:** Toán, Lý, Hóa, Sinh, Anh, Sử, Địa...
2. **Chọn Chủ đề:** Danh sách chủ đề theo môn (VD: Đại số, Hình học...)
3. **Chọn Lớp:** Lớp 1 đến 12
4. **Chọn Dạng câu hỏi:**
   - Trắc nghiệm (4 đáp án A/B/C/D)
   - Điền khuyết (điền vào chỗ trống)
   - Tự luận (trả lời ngắn)
5. **Nhập Số câu:** 1 đến 50
6. **Chọn Độ khó:** Dễ / Trung bình / Khó
7. **Chọn Ngôn ngữ:** Tiếng Việt / Tiếng Anh / Song ngữ
8. **Bật/tắt RAG:** Khi bật, AI sẽ tham khảo ngân hàng câu hỏi để tạo câu phù hợp hơn
9. **Bấm "Tạo theo chủ đề"**

### Kết quả:
- Danh sách câu hỏi được tạo
- **Hiện đáp án:** Bấm để xem đáp án
- **Hiện lời giải:** Bấm để xem lời giải chi tiết
- **Xuất file:** TXT, CSV, DOCX, PDF

---

## 5. Ma trận đề thi (AI)

**Mục đích:** Tạo đề thi phân phối câu hỏi theo nhiều chủ đề và mức độ khó bằng AI, phù hợp với chuẩn ma trận đề thi Việt Nam.

### Ví dụ ma trận:

| Chủ đề | Dễ | Trung bình | Khó | Tổng |
|--------|-----|------------|-----|------|
| Đại số | 3 | 2 | 1 | 6 |
| Hình học | 2 | 2 | 1 | 5 |
| Xác suất | 2 | 1 | 1 | 4 |
| **Tổng** | **7** | **5** | **3** | **15** |

### Các bước:

1. **Chọn Môn học, Lớp, Dạng câu hỏi, Ngôn ngữ, AI Engine**
2. **Bấm "+ Thêm chủ đề"** để thêm dòng vào ma trận
3. **Mỗi dòng:**
   - Chọn chủ đề từ dropdown
   - Nhập số câu cho cột **Dễ**, **Trung bình**, **Khó**
4. Bảng tự tính **Tổng** theo dòng và cột
5. **Bấm "Tạo đề theo ma trận"**
6. AI tạo từng ô tuần tự (có skeleton loading + progress)

### Kết quả:
- Câu hỏi nhóm theo chủ đề + mức độ khó
- Badge màu hiển thị mức độ (xanh = Dễ, vàng = TB, đỏ = Khó)
- Đáp án + Lời giải
- Xuất DOCX, PDF, hoặc **Xuất đề chuyên nghiệp**

### Giới hạn:
- Tối đa **50 câu** mỗi đề
- Tối đa **15 câu** mỗi ô

---

## 6. Lấy từ ngân hàng

**Mục đích:** Tạo đề thi bằng cách chọn ngẫu nhiên câu hỏi từ ngân hàng theo ma trận. Không cần AI, nhanh và chính xác.

### Đặc điểm:
- **Ưu tiên câu ít dùng nhất** (times_used thấp) → phân phối đều
- **Theo dõi số lần sử dụng** cho mỗi câu hỏi
- **Lọc theo Lớp** ngoài môn học và độ khó
- **Xáo trộn** câu hỏi và/hoặc đáp án

### Các bước:

1. **Chọn Môn học, Lớp, Dạng câu hỏi**
2. Xem **thông tin có sẵn** (số câu Dễ/TB/Khó trong ngân hàng)
3. **Bấm "+ Thêm chủ đề"**, chọn chủ đề, nhập số câu
   - Mỗi ô hiển thị hint `/N` (số câu có sẵn)
4. Bật/tắt **Xáo trộn câu hỏi** và **Xáo trộn đáp án**
5. **Bấm "Tạo đề từ ngân hàng"**

### Kết quả:
- Câu hỏi nhóm theo chủ đề + mức độ
- Mỗi câu hiển thị: badge **"Chưa dùng"** (xanh) hoặc **"Đã dùng N lần"** (vàng)
- Đáp án đúng highlight xanh
- ID câu hỏi để truy xuất
- Nút **"Xuất đề chuyên nghiệp"** → PDF/DOCX hoàn chỉnh

### Lưu ý:
- Chỉ chọn câu hỏi **đã có đáp án** từ ngân hàng
- Nếu thiếu câu, hệ thống cảnh báo và chọn tối đa có thể

---

## 7. Xuất đề chuyên nghiệp

**Mục đích:** Xuất file PDF/DOCX đề thi hoàn chỉnh, sẵn sàng in.

### Nội dung file xuất:
- **Header:** Tên trường + Tổ bộ môn
- **Mã đề:** Góc phải
- **Tiêu đề:** VD: "ĐỀ KIỂM TRA GIỮA KỲ 1"
- **Thông tin:** Môn | Lớp | Thời gian
- **Năm học / Ngày thi**
- **Họ tên / Lớp / SBD:** Dòng kẻ chấm
- **Phiếu trả lời:** Bảng bubble A B C D (5 cột, tùy chọn)
- **Câu hỏi:** Đánh số, có options
- **Trang đáp án:** Cuối file (tùy chọn)

### Các bước:
1. Tạo câu hỏi (từ tab Chủ đề, Ma trận, hoặc Ngân hàng)
2. Bấm nút xanh **"Xuất đề chuyên nghiệp"**
3. Điền thông tin trong modal:
   - Tên trường, Tổ bộ môn
   - Tiêu đề đề thi
   - Môn học, Lớp, Thời gian, Mã đề
   - Năm học / Ngày thi
4. Bật/tắt **Phiếu trả lời** và **Đáp án trang cuối**
5. Chọn **Xuất PDF** hoặc **Xuất DOCX**

### Hỗ trợ:
- Font Times New Roman (DOCX)
- Font Vietnamese (PDF)
- Section headers cho đề theo ma trận

---

## 8. Chuyển đổi Word → Excel

**Mục đích:** Chuyển file Word chứa câu hỏi sang định dạng Excel.

### Các bước:
1. Upload file .docx chứa câu hỏi
2. (Tùy chọn) Upload file đáp án (.docx, .pdf, .xlsx)
3. Chọn môn học
4. Bấm **"Chuyển đổi"**
5. Tải file Excel kết quả

---

## 9. Ngân hàng câu hỏi

**Mục đích:** Quản lý kho câu hỏi, lọc, tìm kiếm, thêm/xóa.

### Tính năng:
- **Lọc:** Theo môn học, lớp, mức độ khó
- **Tìm kiếm:** Theo nội dung hoặc #ID
- **Thêm câu hỏi:** Nhập thủ công
- **Xóa:** Xóa từng câu, theo bộ lọc, hoặc tất cả
- **Thống kê:** Số lượng theo môn, theo mức độ khó
- **Phân trang:** Duyệt qua danh sách câu hỏi

### Mỗi câu hỏi hiển thị:
- **#ID** + **Môn học** + **(Lớp X)** hoặc **(Quốc tế)**
- Badge mức độ khó (Dễ/TB/Khó)
- Badge **"Dùng Nx"** nếu đã sử dụng (theo dõi số lần dùng)
- Icon 📷 nếu có hình ảnh

### Mỗi câu hỏi lưu trữ:
- Nội dung, đáp án, lời giải
- Môn học, chủ đề, lớp
- Mức độ khó, loại câu hỏi
- Hình ảnh minh họa (nếu có)
- Nguồn gốc, tags, số lần sử dụng
- Điểm chất lượng (AI đánh giá)

---

## 10. Bổ sung đáp án bằng AI

**Mục đích:** Tự động bổ sung đáp án cho các câu hỏi chưa có đáp án trong ngân hàng.

**Vị trí:** Tab Ngân hàng câu hỏi → Panel vàng phía trên danh sách.

### Hiển thị:
- Số câu chưa có đáp án (phân tích theo môn)
- Khi tất cả đã có đáp án → panel chuyển xanh

### Các bước:
1. Chọn **AI Engine** (Ollama/Gemini/OpenAI/Claude)
2. Chọn **Môn học** (hoặc tất cả)
3. Chọn **Batch size** (10/25/50/100 câu)
4. Bấm **"Bổ sung đáp án (AI)"**
5. Thanh progress hiển thị tiến trình
6. Toast notification khi hoàn tất

### Cơ chế:
- Chia thành batch nhỏ (10 câu/batch) gọi AI tuần tự
- AI phân tích câu hỏi + options → trả về A/B/C/D
- Cập nhật trực tiếp vào database
- Refresh thống kê sau khi xong

### Chạy bằng script (số lượng lớn):
```bash
python scripts/fill-missing-answers-with-ai.py --engine ollama --all --batch 20 --max 1000
```

---

## 11. Import dữ liệu (Scrape)

**Mục đích:** Tải câu hỏi từ các nguồn bên ngoài vào ngân hàng.

### Nguồn Việt Nam:
| Nguồn | Mô tả |
|-------|--------|
| **ThuVienHocLieu.com** | Đề thi, trắc nghiệm lớp 10-12 |
| **VietJack.com** | Đề thi các môn lớp 1-12 |
| **Hoc247.net** | Đề kiểm tra các cấp |
| **TracNghiem.net** | 2M+ câu hỏi trắc nghiệm |

### Nguồn quốc tế:
| Nguồn | API Key | Mô tả |
|-------|---------|--------|
| **Open Trivia DB** | Không cần | 4000+ câu hỏi Science, Math, History |
| **The Trivia API** | Không cần | 10 categories, đa dạng chủ đề |
| **QuizAPI** | Cần (miễn phí) | IT/Tech: Linux, Docker, SQL, DevOps |
| **API Ninjas** | Cần (miễn phí) | 100K+ câu hỏi, dạng trả lời ngắn |

### Các bước:
1. Chọn **Nguồn** từ dropdown
2. Chọn **Số lượng tối đa**
3. Chọn **Danh mục** hoặc nhập URL cụ thể
4. Bấm **"Bắt đầu Scrape"**
5. Hệ thống tự động tải và lưu vào ngân hàng, bỏ qua câu trùng lặp

### Upload file:
Ngoài scrape, có thể upload trực tiếp file .docx, .pdf, .txt chứa câu hỏi.

---

## 12. Kho lưu trữ

**Mục đích:** Quản lý file đề mẫu theo môn học và lớp.

### Tính năng:
- Upload file: .docx, .doc, .pdf, .xlsx, .xls, .txt
- Phân loại theo môn học và lớp
- Thêm mô tả cho file
- Thống kê: Tổng file, dung lượng
- Lọc theo môn và lớp

---

## 13. Khung chương trình

**Mục đích:** Tra cứu khung chương trình giáo dục theo môn và lớp.

### Các môn hỗ trợ:
Toán, Ngữ văn, Tiếng Anh, KHTN, Vật lý, Hóa học, Sinh học, Lịch sử, Địa lý

### Các bước:
1. Chọn **Môn học**
2. Chọn **Lớp** (1-12)
3. Xem danh sách chủ đề, kiến thức, kỹ năng cần đạt

---

## 14. Dashboard thống kê

**Mục đích:** Xem tổng quan ngân hàng câu hỏi qua biểu đồ trực quan.

### 6 thẻ tổng quan:
- **Tổng câu hỏi** - Số lượng trong ngân hàng
- **Có đáp án** - Số câu đã có đáp án (và %)
- **Chưa có đáp án** - Cần bổ sung
- **Có lời giải** - Số câu có explanation
- **Có hình ảnh** - Số câu có ảnh minh họa
- **Đã sử dụng** - Số câu đã dùng trong đề thi

### 6 biểu đồ (Chart.js):

| Biểu đồ | Loại | Nội dung |
|----------|------|----------|
| **Phân bố theo môn** | Bar chart | Số câu mỗi môn |
| **Độ khó** | Doughnut | Tỷ lệ Dễ/TB/Khó |
| **Tỷ lệ có đáp án** | Stacked bar | Có/chưa có đáp án theo môn |
| **Hoạt động 14 ngày** | Line chart | Câu hỏi thêm mới mỗi ngày |
| **Phân bố theo lớp** | Horizontal bar | Lớp 1-12 + Quốc tế |
| **Nguồn câu hỏi** | Horizontal bar | Top 10 nguồn |

### Cách dùng:
- Click tab **Dashboard** (biểu tượng bar chart) ở sidebar
- Charts tự load khi chuyển sang tab (lazy loading)

---

## 15. Chấm bài OMR Scanner

**Mục đích:** Chấm bài trắc nghiệm tự động từ ảnh phiếu trả lời (answer sheet).

### Hỗ trợ các mẫu thi:
- **KANGAROO** (Toán Kangaroo quốc tế)
- **ASMO** (Asian Science and Mathematics Olympiad)
- **SEAMO** (Southeast Asian Mathematical Olympiad)
- Và nhiều mẫu khác với hệ thống tính điểm riêng

### Các bước:
1. **Chọn mẫu đề thi** (template)
2. **Nhập đáp án đúng:**
   - Thủ công: chọn A-E cho từng câu
   - Hoặc upload file đáp án (Excel, Word, PDF)
3. **Upload ảnh phiếu trả lời** (JPG, PNG hoặc PDF)
4. **Bấm "Chấm bài"**

### Kết quả:
- Bảng kết quả: Tên học sinh, Số câu đúng/sai/bỏ trống, Điểm số
- Chi tiết từng câu
- Xuất kết quả ra Excel

### Hệ thống tính điểm:
- Cấu hình điểm cộng/trừ cho đúng/sai/bỏ trống
- Hỗ trợ tính điểm theo bậc (VD: Câu 1-8: +3/-1, Câu 9-16: +4/-1, Câu 17-24: +5/-1)

---

## 16. Chấm bài viết tay

**Mục đích:** Nhận diện và chấm điểm đáp án viết tay từ ảnh.

### Các bước:
1. **Cấu hình:**
   - Số lượng câu hỏi
   - Đáp án hợp lệ (A-E)
   - Điểm mỗi câu đúng/sai/bỏ trống
2. **Nhập đáp án đúng:**
   - Dạng grid (chọn từng câu)
   - Hoặc nhập text (VD: A,B,C,D,A,B,...)
3. **Upload ảnh** chứa đáp án viết tay
4. **Bấm "Chấm bài viết tay"**

### Kết quả:
- Bảng điểm với chi tiết từng câu
- Xuất Excel

---

## 17. Lịch sử

**Mục đích:** Xem lại các đề thi đã tạo trước đó.

### Tính năng:
- Danh sách đề đã tạo (sắp xếp theo thời gian)
- Thông tin: Môn học, số câu, ngày tạo
- Xem lại chi tiết đề
- Xuất lại file (DOCX, PDF)
- Xóa lịch sử

---

## 18. Cài đặt

### AI Engine:
Xem [mục 2](#2-cấu-hình-ai-engine).

### Dark Mode:
- Bấm biểu tượng **mặt trăng/mặt trời** ở header phải
- Tự lưu preference, tự phát hiện system theme

### Giao diện:
- **Lucide SVG icons** - Nhất quán, sắc nét
- **Toast notifications** - Thông báo góc phải (success/error/info/warning)
- **Skeleton loading** - Khung xương khi đang tải
- **Hamburger menu** - Sidebar trượt trên mobile (< 768px)
- **Responsive** - Tự động chỉnh layout cho mobile

---

## Phụ lục

### Định dạng file hỗ trợ

| Loại | Input | Output |
|------|-------|--------|
| Word | .docx, .doc | .docx |
| PDF | .pdf | .pdf |
| Excel | .xlsx, .xls | .xlsx |
| Text | .txt | .txt |
| CSV | - | .csv |
| Ảnh | .jpg, .png, .bmp | - |

### Cơ sở dữ liệu
- **File:** `question_bank.db` (SQLite, tự tạo khi chạy lần đầu)
- **Vị trí:** Thư mục gốc của ứng dụng

### Hỗ trợ công thức toán
- Sử dụng cú pháp LaTeX: `$x^2$`, `$$\frac{a}{b}$$`
- Hiển thị bằng KaTeX trong trình duyệt
- Hỗ trợ xuất LaTeX trong file DOCX và PDF

### Lấy API key miễn phí
- **QuizAPI:** https://quizapi.io/register
- **API Ninjas:** https://api-ninjas.com/register
- **OpenAI:** https://platform.openai.com/api-keys
- **Gemini:** https://aistudio.google.com/apikey
- **Anthropic:** https://console.anthropic.com/
