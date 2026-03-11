# Changelog - 2026-03-02

## Tính năng mới

### 1. Hỗ trợ format Word Numbered List (Kangaroo Start R2)
- Parser EN-VIE mới hỗ trợ file Word có định dạng List Paragraph
- Câu hỏi ở level 0 (numbered list), options ở level 1 (sub-list)
- Số câu hỏi được lấy từ Word numbering, không cần nhúng trong text
- Tự động clean "StartStart" watermark artifacts

### 2. Auto-detect Word Numbered List Format
- Tự động nhận diện file có định dạng Word numbered list khi convert
- Áp dụng cho file như "Kangaroo Start R2 demo.docx" (không cần "EN-VIE" trong tên)
- Điều kiện: tỉ lệ 1:1 giữa questions và options, ít nhất 10 câu

### 3. Chọn môn học khi chuyển Word sang Excel
- Thêm dropdown chọn môn học: Tiếng Anh, Toán, Khoa học, Khác
- Không còn tự động đoán môn từ tên file
- Phù hợp với Kangaroo (English/Science), ASMO (Math/Science/English)

### 4. Sửa lỗi modal xem chi tiết câu hỏi
- Hỗ trợ nhiều format đáp án: A/B/C/D, 1-4 (1-indexed), 0-3 (0-indexed)
- Thêm error handling và debug logging cho API fetch
- Hiển thị đúng nội dung câu hỏi và đáp án

## Commits

| Commit | Mô tả |
|--------|-------|
| `ee96f26` | Add subject selection dropdown for Word to Excel conversion |
| `d093a52` | Improve subject detection: add kangaroo, ikmc, iklc, asmo keywords |
| `9def185` | Fix question modal: handle multiple answer formats and add debug logging |
| `0402d26` | Auto-detect Word numbered list format in convert-word-to-excel |
| `a8eb2c8` | Add support for Word numbered list format in EN-VIE parser |

## Files thay đổi

- `app/main.py` - Parser EN-VIE, endpoint convert-word-to-excel
- `app/templates/index.html` - UI dropdown môn học, fix modal câu hỏi
- `đề mẫu/Kangaroo Start R2 demo.docx` - File test mới

## Test

File "Kangaroo Start R2 demo.docx":
- ✅ 25 câu hỏi
- ✅ Mỗi câu 4 options
- ✅ Đáp án highlighted được detect
- ✅ Xuất Excel thành công
