# Hướng dẫn sử dụng - Tạo đề online

## Mục lục

- [1. Cài đặt và khởi chạy](#1-cài-đặt-và-khởi-chạy)
- [2. Cấu hình AI Engine](#2-cấu-hình-ai-engine)
- [3. Tạo đề từ đề mẫu](#3-tạo-đề-từ-đề-mẫu)
- [4. Tạo theo chủ đề (AI)](#4-tạo-theo-chủ-đề-ai)
- [5. Ma trận đề thi](#5-ma-trận-đề-thi)
- [6. Chuyển đổi Word → Excel](#6-chuyển-đổi-word--excel)
- [7. Ngân hàng câu hỏi](#7-ngân-hàng-câu-hỏi)
- [8. Import dữ liệu](#8-import-dữ-liệu)
- [9. Kho lưu trữ](#9-kho-lưu-trữ)
- [10. Khung chương trình](#10-khung-chương-trình)
- [11. Chấm bài OMR Scanner](#11-chấm-bài-omr-scanner)
- [12. Chấm bài viết tay](#12-chấm-bài-viết-tay)
- [13. Lịch sử](#13-lịch-sử)
- [14. Cài đặt](#14-cài-đặt)

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

## 5. Ma trận đề thi

**Mục đích:** Tạo đề thi phân phối câu hỏi theo nhiều chủ đề và mức độ khó, phù hợp với chuẩn ma trận đề thi Việt Nam.

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
6. AI sẽ tạo từng ô tuần tự (có hiển thị tiến trình)

### Kết quả:
- Câu hỏi nhóm theo chủ đề + mức độ khó
- Badge màu hiển thị mức độ (xanh = Dễ, vàng = TB, đỏ = Khó)
- Đáp án + Lời giải
- Xuất DOCX, PDF

### Giới hạn:
- Tối đa **50 câu** mỗi đề
- Tối đa **15 câu** mỗi ô

---

## 6. Chuyển đổi Word → Excel

**Mục đích:** Chuyển file Word chứa câu hỏi sang định dạng Excel.

### Các bước:
1. Upload file .docx chứa câu hỏi
2. (Tùy chọn) Upload file đáp án (.docx, .pdf, .xlsx)
3. Chọn môn học
4. Bấm **"Chuyển đổi"**
5. Tải file Excel kết quả

---

## 7. Ngân hàng câu hỏi

**Mục đích:** Quản lý kho câu hỏi, lọc, tìm kiếm, thêm/xóa.

### Tính năng:
- **Lọc:** Theo môn học, lớp, mức độ khó
- **Tìm kiếm:** Theo nội dung hoặc #ID
- **Thêm câu hỏi:** Nhập thủ công
- **Xóa:** Xóa từng câu, theo bộ lọc, hoặc tất cả
- **Thống kê:** Số lượng theo môn, theo mức độ khó
- **Phân trang:** Duyệt qua danh sách câu hỏi

### Mỗi câu hỏi lưu trữ:
- Nội dung, đáp án, lời giải
- Môn học, chủ đề, lớp
- Mức độ khó, loại câu hỏi
- Hình ảnh minh họa (nếu có)
- Nguồn gốc, tags
- Điểm chất lượng (AI đánh giá)

---

## 8. Import dữ liệu

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

## 9. Kho lưu trữ

**Mục đích:** Quản lý file đề mẫu theo môn học và lớp.

### Tính năng:
- Upload file: .docx, .doc, .pdf, .xlsx, .xls, .txt
- Phân loại theo môn học và lớp
- Thêm mô tả cho file
- Thống kê: Tổng file, dung lượng
- Lọc theo môn và lớp

---

## 10. Khung chương trình

**Mục đích:** Tra cứu khung chương trình giáo dục theo môn và lớp.

### Các môn hỗ trợ:
Toán, Ngữ văn, Tiếng Anh, KHTN, Vật lý, Hóa học, Sinh học, Lịch sử, Địa lý

### Các bước:
1. Chọn **Môn học**
2. Chọn **Lớp** (1-12)
3. Xem danh sách chủ đề, kiến thức, kỹ năng cần đạt

---

## 11. Chấm bài OMR Scanner

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

## 12. Chấm bài viết tay

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

## 13. Lịch sử

**Mục đích:** Xem lại các đề thi đã tạo trước đó.

### Tính năng:
- Danh sách đề đã tạo (sắp xếp theo thời gian)
- Thông tin: Môn học, số câu, ngày tạo
- Xem lại chi tiết đề
- Xuất lại file (DOCX, PDF)
- Xóa lịch sử

---

## 14. Cài đặt

### AI Engine:
Xem [mục 2](#2-cấu-hình-ai-engine).

### Dark Mode:
- Bấm biểu tượng **mặt trăng/mặt trời** ở header phải
- Tự lưu preference, tự phát hiện system theme

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
