import io
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")


class GenerateRequest(BaseModel):
    samples: List[str]
    topic: str
    custom_keywords: List[str]
    paraphrase: bool
    change_numbers: bool
    change_context: bool
    variants_per_question: int
    use_ai: bool = False
    ai_engine: str = "openai"


class ParseSamplesRequest(BaseModel):
    urls: List[str]

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

SUBJECTS = {
    "toan_hoc": {"label": "Toán học", "lang": "vi"},
    "tieng_anh": {"label": "Tiếng Anh", "lang": "en"},
    "khoa_hoc": {"label": "Khoa học", "lang": "vi"},
}

QUESTION_TYPES = {
    "mcq": "Trắc nghiệm",
    "blank": "Điền khuyết",
    "essay": "Tự luận",
}

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

NUMBER_RE = re.compile(r"\b\d+\b")
LEADING_NUM_RE = re.compile(r"^\s*\d{1,3}[\).\-:]\s+")
MCQ_OPTION_RE = re.compile(r"^[A-H][\).\-:]\s+", re.IGNORECASE)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
BASE_DIR = Path(__file__).resolve().parent.parent


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFD", value)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\\s+", " ", text).strip().lower()


def _resolve_sample_dir() -> Optional[Path]:
    env_dir = os.getenv("SAMPLE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    target = _normalize_name("đề mẫu")
    for entry in BASE_DIR.iterdir():
        if entry.is_dir() and _normalize_name(entry.name) == target:
            return entry
    return None


SAMPLE_DIR = _resolve_sample_dir()


def _replace_numbers(text: str) -> str:
    def _swap(match: re.Match) -> str:
        value = int(match.group(0))
        if value == 0:
            return "0"
        delta = max(1, int(round(value * 0.15)))
        new_value = max(1, value + random.choice([-delta, delta]))
        return str(new_value)

    return NUMBER_RE.sub(_swap, text)


def _apply_synonyms(text: str) -> str:
    out = text
    for src, options in SYNONYMS.items():
        if src in out:
            out = out.replace(src, random.choice(options))
    return out


def _apply_context(text: str, topic_key: str, custom_keywords: List[str]) -> str:
    out = text
    topic = TOPICS.get(topic_key, {})
    for src, dst in topic.get("context", {}).items():
        out = out.replace(src, dst)
    if custom_keywords:
        for kw in custom_keywords:
            if kw.strip():
                out = out.replace("...", kw.strip(), 1)
    return out


def _strip_leading_numbering(text: str) -> str:
    return LEADING_NUM_RE.sub("", text).strip()


def _normalize_question(text: str) -> str:
    return _strip_leading_numbering(text).strip().lower()


def _is_mcq_block(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) > 1 and any(MCQ_OPTION_RE.match(ln) for ln in lines[1:])


def _force_variation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if _is_mcq_block(stripped):
        return _rewrite_mcq_block(stripped)
    # Heuristic: if mostly ASCII, use English prefix; otherwise Vietnamese.
    ascii_ratio = sum(1 for ch in stripped if ord(ch) < 128) / max(1, len(stripped))
    if ascii_ratio > 0.9:
        return f"Choose the correct option: {stripped}"
    return f"Hãy cho biết: {stripped}"


def _rewrite_english_question(question: str) -> str:
    q = question.strip()
    if not q:
        return q
    pattern = re.compile(
        r"^The\s+\.\.\.\s+you\s+try,\s+the\s+more\s+likely\s+you\s+are\s+to\s+be\s+successful\.?$",
        re.IGNORECASE,
    )
    if pattern.match(q):
        return "The more ... you attempt, the higher your chance of success."

    pattern2 = re.compile(
        r"^All the players did their best apart from Johnson\. Johnson was \.\.\. his best\.?$",
        re.IGNORECASE,
    )
    if pattern2.match(q):
        variants = [
            "Only Johnson failed to give maximum effort. Complete the sentence: Johnson did not ... his best.",
            "Everyone except Johnson gave their best; Johnson did not ... his best.",
            "Johnson was the only player who didn't perform to his best. Choose the correct completion.",
            "All the other players performed at their peak. Johnson, however, did not ... his best.",
            "Unlike the rest of the team, Johnson didn't ... his best. Select the best completion.",
            "The entire squad tried their hardest, but Johnson did not ... his best. Choose the correct option.",
        ]
        return random.choice(variants)

    candidates = [
        (r"\bapart from\b", "except for"),
        (r"\bdid their best\b", "performed to the best of their ability"),
        (r"\bwas \.\.\. his best\b", "did not ... his best"),
        (r"\bmore likely\b", "more probable"),
        (r"\byou are to be successful\b", "you will succeed"),
        (r"\bthe more\b", "the greater"),
        (r"\btry\b", "attempt"),
    ]
    rewritten = q
    for src, dst in candidates:
        rewritten = re.sub(src, dst, rewritten, flags=re.IGNORECASE)
    if _normalize_question(rewritten) != _normalize_question(q):
        return rewritten
    return q


def _rewrite_mcq_block(block: str) -> str:
    lines = block.splitlines()
    if len(lines) < 2:
        return block
    if not MCQ_OPTION_RE.match(lines[1]):
        return block
    question = lines[0]
    prefix = ""
    if ":" in question:
        head, tail = question.split(":", 1)
        if len(head.split()) <= 5:
            prefix = head.strip()
            question = tail.strip()
    ascii_ratio = sum(1 for ch in question if ord(ch) < 128) / max(1, len(question))
    if ascii_ratio > 0.9:
        rewritten = _rewrite_english_question(question)
    else:
        rewritten = question
    if _normalize_question(rewritten) == _normalize_question(question):
        rewritten = f"Complete the sentence: {question}"
    # Context swap for English MCQ to force stronger change.
    if ascii_ratio > 0.9 and re.search(r"players|team|squad", rewritten, re.IGNORECASE):
        rewritten = re.sub(r"players", "athletes", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"team|squad", "group", rewritten, flags=re.IGNORECASE)
    new_prefix = "Select the best completion"
    if re.match(r"^(Choose|Select|Complete)\\b", rewritten, re.IGNORECASE):
        qline = rewritten
    else:
        qline = f"{new_prefix}: {rewritten}"
    return "\n".join([qline] + lines[1:])


def _extract_text_from_response(payload: dict) -> str:
    if isinstance(payload, dict):
        if payload.get("output_text"):
            return payload["output_text"]
        output = payload.get("output", [])
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                parts = []
                for c in content:
                    if c.get("type") in {"output_text", "text"} and c.get("text"):
                        parts.append(c.get("text", ""))
                if parts:
                    return "\n".join(parts).strip()
    return ""


def _build_ai_prompt(
    sample: Optional[str],
    topic_key: str,
    custom_keywords: List[str],
    paraphrase: bool,
    change_numbers: bool,
    change_context: bool,
    variants: int,
) -> str:
    topic = TOPICS.get(topic_key, {})
    keywords = topic.get("keywords", [])
    guide = TOPIC_AI_GUIDE.get(topic_key, "")
    mode = []
    if paraphrase:
        mode.append("paraphrase")
    if change_numbers:
        mode.append("thay số liệu")
    if change_context:
        mode.append("đổi ngữ cảnh")
    mode_text = ", ".join(mode) if mode else "giữ nguyên ngữ nghĩa"
    kw_text = ", ".join([*keywords, *custom_keywords]) or "không có"

    if sample:
        return (
            "Bạn là trợ lý tạo câu hỏi học tập.\n"
            f"Hãy tạo {variants} câu hỏi tương tự câu mẫu sau.\n"
            f"- Chế độ: {mode_text}\n"
            f"- Chủ đề: {topic.get('label', topic_key)}\n"
            f"- Từ khóa gợi ý: {kw_text}\n"
            f"- Hướng dẫn: {guide}\n"
            "Yêu cầu bắt buộc: viết khác câu gốc (không lặp nguyên văn), đổi cấu trúc câu, dùng từ đồng nghĩa.\n"
            "Nếu là trắc nghiệm, giữ định dạng đáp án A/B/C nhưng có thể thay nội dung câu hỏi.\n"
            "Không đánh số đầu dòng.\n"
            "Ví dụ KHÔNG được làm:\n"
            "- The ... you try, the more likely you are to be successful.\n"
            "Ví dụ ĐƯỢC làm:\n"
            "- The more ... you attempt, the higher your chance of success becomes.\n"
            "Trả về mỗi câu trên một dòng, không thêm giải thích.\n"
            f"Câu mẫu: {sample}\n"
        )
    return (
        "Bạn là trợ lý tạo câu hỏi học tập.\n"
        f"Hãy tạo {variants} câu hỏi theo chủ đề.\n"
        f"- Chủ đề: {topic.get('label', topic_key)}\n"
        f"- Từ khóa gợi ý: {kw_text}\n"
        f"- Hướng dẫn: {guide}\n"
        "Yêu cầu bắt buộc: không đánh số đầu dòng.\n"
        "Trả về mỗi câu trên một dòng, không thêm giải thích.\n"
    )


def _call_openai(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if not OPENAI_API_KEY:
        return None, "missing_api_key"
    url = f"{OPENAI_API_BASE}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
        "text": {"format": {"type": "text"}},
        "max_output_tokens": 400,
        "temperature": 0.8,
    }
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            try:
                data = response.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = response.text[:200]
            return None, f"{response.status_code}: {msg}"
        return _extract_text_from_response(response.json()), None


def _call_gemini(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    if not GEMINI_API_KEY:
        return None, "missing_gemini_api_key"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.8,
        },
    }
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload)
        if response.status_code != 200:
            try:
                data = response.json()
                msg = data.get("error", {}).get("message", "")
            except Exception:
                msg = response.text[:200]
            return None, f"{response.status_code}: {msg}"
        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip(), None
        except (KeyError, IndexError):
            return None, "Không parse được phản hồi Gemini"


def _call_ollama(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 400,
            "temperature": 0.8,
        },
    }
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload)
            if response.status_code != 200:
                try:
                    data = response.json()
                    msg = data.get("error", response.text[:200])
                except Exception:
                    msg = response.text[:200]
                return None, f"{response.status_code}: {msg}"
            data = response.json()
            text = data.get("response", "")
            return text.strip() if text else None, None
    except httpx.ConnectError:
        return None, "Không kết nối được Ollama. Hãy chắc chắn Ollama đang chạy (ollama serve)."


def _call_ai(prompt: str, engine: str = "openai") -> Tuple[Optional[str], Optional[str]]:
    if engine == "gemini":
        return _call_gemini(prompt)
    if engine == "ollama":
        return _call_ollama(prompt)
    return _call_openai(prompt)


def _normalize_ai_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^\\s*\\d+[\\).\\-\\:]\\s*", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _normalize_ai_blocks(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    cleaned: List[str] = []
    for block in blocks:
        lines = block.splitlines()
        if lines:
            lines[0] = _strip_leading_numbering(lines[0])
        cleaned.append("\n".join(lines).strip())
    return cleaned


def _build_topic_prompt(subject_key: str, grade: int, qtype: str, count: int) -> str:
    subject = SUBJECTS.get(subject_key, {"label": subject_key, "lang": "vi"})
    label = subject.get("label", subject_key)
    lang = subject.get("lang", "vi")
    if lang == "en":
        grade_text = f"Grade {grade}"
        type_text = {
            "mcq": "multiple-choice",
            "blank": "fill-in-the-blank",
            "essay": "short-answer",
        }.get(qtype, "multiple-choice")
        return (
            "You are an education content writer.\n"
            f"Create {count} {type_text} questions for {label} ({grade_text}).\n"
            "Rules:\n"
            "- Do not number questions.\n"
            "- Separate questions with a blank line.\n"
            "- If multiple-choice, provide 4 options labeled A) B) C) D) on separate lines.\n"
            "- If fill-in-the-blank, use \"...\" for the blank.\n"
            "- Do not include answers.\n"
        )
    grade_text = f"Lớp {grade}"
    type_text = QUESTION_TYPES.get(qtype, "Trắc nghiệm")
    return (
        "Bạn là người biên soạn câu hỏi học tập.\n"
        f"Hãy tạo {count} câu hỏi dạng {type_text} cho môn {label} ({grade_text}).\n"
        "Yêu cầu:\n"
        "- Không đánh số đầu câu.\n"
        "- Mỗi câu cách nhau bằng một dòng trống.\n"
        "- Nếu trắc nghiệm, đưa 4 đáp án A) B) C) D) mỗi đáp án một dòng.\n"
        "- Nếu điền khuyết, dùng \"...\" để thể hiện chỗ trống.\n"
        "- Không kèm đáp án.\n"
    )


def _get_pdf_font() -> Tuple[str, Optional[str]]:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_name = "VietnameseFont"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name, path
    return "Helvetica", None


def _wrap_text(text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _read_sample_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        doc = Document(str(path))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(lines)
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_sample_url(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix.lower()
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    if suffix == ".docx":
        doc = Document(io.BytesIO(data))
        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(lines)
    return data.decode("utf-8", errors="ignore")


def _split_questions(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\\s*\n", normalized) if b.strip()]
    questions: List[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # Keep multiple-choice options attached to the preceding question line.
        merged_lines: List[str] = []
        option_re = re.compile(r"^[A-H][\\).\\-\\:]\\s+", re.IGNORECASE)
        has_options = False
        for line in lines:
            if option_re.match(line) and merged_lines:
                merged_lines[-1] = f"{merged_lines[-1]}\n{line}"
                has_options = True
            else:
                merged_lines.append(line)
        lines = merged_lines

        # Remove consecutive duplicate lines caused by copy/paste or formatting issues.
        deduped_lines: List[str] = []
        for line in lines:
            if not deduped_lines or deduped_lines[-1] != line:
                deduped_lines.append(line)
        lines = deduped_lines

        if has_options:
            questions.append("\n".join(lines).strip())
            continue

        numbered_lines = [ln for ln in lines if re.match(r"^\\d{1,3}[\\).\\-\\:]\\s+", ln)]
        if numbered_lines:
            buffer: List[str] = []
            for line in lines:
                if re.match(r"^\\d{1,3}[\\).\\-\\:]\\s+", line):
                    if buffer:
                        questions.append(" ".join(buffer).strip())
                        buffer = []
                    line = re.sub(r"^\\d{1,3}[\\).\\-\\:]\\s+", "", line)
                buffer.append(line)
            if buffer:
                questions.append(" ".join(buffer).strip())
            continue

        joined = " ".join(lines).strip()
        parts = [p.strip() for p in re.split(r"(?<=[\\?？])\\s+", joined) if p.strip()]
        if len(parts) > 1:
            questions.extend(parts)
        else:
            questions.append(joined)

    # Final de-duplication while preserving order.
    seen = set()
    unique_questions: List[str] = []
    for q in questions:
        key = q.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_questions.append(q)
    return unique_questions


def _load_questions_from_subject(subject: str) -> List[str]:
    if SAMPLE_DIR is None:
        return []
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if not SAMPLE_DIR.exists() or SAMPLE_DIR not in subject_dir.parents:
        return []
    if not subject_dir.exists() or not subject_dir.is_dir():
        return []
    questions: List[str] = []
    for path in subject_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".txt", ".docx", ".md"}:
            content = _read_sample_file(path)
            questions.extend(_split_questions(content))
    return [q for q in questions if q]


def _is_engine_available(engine: str) -> bool:
    if engine == "gemini":
        return bool(GEMINI_API_KEY)
    if engine == "ollama":
        return True  # Ollama runs locally, availability checked at call time
    return bool(OPENAI_API_KEY)


def generate_variants(req: GenerateRequest) -> List[str]:
    if req.use_ai and _is_engine_available(req.ai_engine):
        generated: List[str] = []
        samples = req.samples or [None]
        for sample in samples:
            if sample:
                sample = _strip_leading_numbering(sample)
            prompt = _build_ai_prompt(
                sample,
                req.topic,
                req.custom_keywords,
                req.paraphrase,
                req.change_numbers,
                req.change_context,
                req.variants_per_question,
            )
            attempts = 0
            while attempts < 2:
                attempts += 1
                text, err = _call_ai(prompt, req.ai_engine)
                if text:
                    lines = [_strip_leading_numbering(line) for line in _normalize_ai_lines(text)]
                    if sample:
                        lines = [ln for ln in lines if ln.strip().lower() != sample.strip().lower()]
                    if lines:
                        for ln in lines[: req.variants_per_question]:
                            generated.append(_rewrite_mcq_block(ln))
                        break
                prompt = prompt + "\nNếu câu trả về trùng câu gốc, hãy viết lại hoàn toàn khác.\n"
        if generated:
            # De-duplicate while preserving order
            seen = set()
            unique = []
            for q in generated:
                if q and q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique

    results: List[str] = []
    for sample in req.samples:
        sample = sample.strip()
        if not sample:
            continue
        sample = _strip_leading_numbering(sample)
        for _ in range(req.variants_per_question):
            variant = sample
            if req.paraphrase:
                variant = _apply_synonyms(variant)
            if req.change_numbers:
                variant = _replace_numbers(variant)
            if req.change_context:
                variant = _apply_context(variant, req.topic, req.custom_keywords)
            variant = _strip_leading_numbering(variant)
            variant = _rewrite_mcq_block(variant)
            if _normalize_question(variant) == _normalize_question(sample):
                variant = _force_variation(variant)
            results.append(variant)
    return results


@app.get("/")
def index() -> HTMLResponse:
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    engine = payload.ai_engine
    if payload.use_ai and not _is_engine_available(engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {
            "questions": generate_variants(payload),
            "message": f"Chưa cấu hình {engine_names.get(engine, engine)} nên AI không được dùng.",
        }
    questions = generate_variants(payload)
    if payload.use_ai and _is_engine_available(engine):
        src = {_normalize_question(s) for s in payload.samples if s.strip()}
        out = {_normalize_question(q) for q in questions if q.strip()}
        if out and out.issubset(src):
            return {
                "questions": questions,
                "message": "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản.",
            }
    return {"questions": questions}


@app.post("/generate-topic")
def generate_topic(
    subject: str = Form(...),
    grade: int = Form(1),
    qtype: str = Form("mcq"),
    count: int = Form(10),
    ai_engine: str = Form("openai"),
) -> dict:
    if not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {"questions": [], "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng."}
    count = max(1, min(50, count))
    grade = max(1, min(12, grade))
    prompt = _build_topic_prompt(subject, grade, qtype, count)
    text, err = _call_ai(prompt, ai_engine)
    if not text:
        msg = f"Không nhận được phản hồi từ AI. {err}" if err else "Không nhận được phản hồi từ AI."
        return {"questions": [], "message": msg}
    questions = _normalize_ai_blocks(text)
    questions = [q for q in questions if q.strip()]
    return {"questions": questions[:count]}


@app.post("/auto-generate")
def auto_generate(
    subject: str = Form(...),
    count: int = Form(10),
    ai_ratio: int = Form(30),
    topic: str = Form(""),
    custom_keywords: str = Form(""),
    paraphrase: bool = Form(True),
    change_numbers: bool = Form(True),
    change_context: bool = Form(True),
    use_ai: bool = Form(False),
    samples_text: str = Form(""),
    ai_engine: str = Form("openai"),
) -> dict:
    if samples_text.strip():
        samples = _split_questions(samples_text)
    else:
        samples = _load_questions_from_subject(subject)
    if not samples:
        return {"questions": [], "message": "Không tìm thấy câu hỏi mẫu."}
    count = max(1, min(200, count))
    ai_ratio = max(0, min(100, ai_ratio))
    ai_count = int(round(count * (ai_ratio / 100)))
    rule_count = count - ai_count

    random.shuffle(samples)
    selected = samples[: min(len(samples), max(1, rule_count))]
    if len(selected) < rule_count:
        selected = selected * (rule_count // max(1, len(selected)) + 1)
        selected = selected[:rule_count]

    req = GenerateRequest(
        samples=selected,
        topic=topic or subject,
        custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
        paraphrase=paraphrase,
        change_numbers=change_numbers,
        change_context=change_context,
        variants_per_question=1,
        use_ai=False,
    )
    questions = generate_variants(req)

    if use_ai and ai_count > 0 and not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {
            "questions": questions[:count],
            "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng.",
        }

    if use_ai and ai_count > 0:
        ai_req = GenerateRequest(
            samples=[random.choice(samples)] if samples else [],
            topic=topic or subject,
            custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
            paraphrase=paraphrase,
            change_numbers=change_numbers,
            change_context=change_context,
            variants_per_question=max(1, ai_count),
            use_ai=True,
            ai_engine=ai_engine,
        )
        ai_questions = generate_variants(ai_req)
        if ai_questions:
            questions.extend(ai_questions[:ai_count])

    random.shuffle(questions)
    result = {"questions": questions[:count]}
    if use_ai and _is_engine_available(ai_engine):
        src = {_normalize_question(s) for s in samples if s.strip()}
        out = {_normalize_question(q) for q in result["questions"] if q.strip()}
        if out and out.issubset(src):
            result["message"] = "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản."
    return result


@app.post("/export")
def export(
    samples: str = Form(...),
    topic: str = Form(...),
    custom_keywords: str = Form(""),
    paraphrase: bool = Form(False),
    change_numbers: bool = Form(False),
    change_context: bool = Form(False),
    variants_per_question: int = Form(1),
    fmt: str = Form("txt"),
    use_ai: bool = Form(False),
    ai_engine: str = Form("openai"),
):
    req = GenerateRequest(
        samples=[s for s in samples.split("\n") if s.strip()],
        topic=topic,
        custom_keywords=[s for s in custom_keywords.split(",") if s.strip()],
        paraphrase=paraphrase,
        change_numbers=change_numbers,
        change_context=change_context,
        variants_per_question=variants_per_question,
        use_ai=use_ai,
        ai_engine=ai_engine,
    )
    questions = generate_variants(req)

    if fmt == "csv":
        content = "id,question\n" + "\n".join(
            f"{i+1},\"{q.replace('"', '""')}\"" for i, q in enumerate(questions)
        )
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=questions.csv"},
        )

    if fmt == "docx":
        doc = Document()
        doc.add_heading("Danh sách câu hỏi", level=1)
        for i, q in enumerate(questions, 1):
            doc.add_paragraph(f"{i}. {q}")
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=questions.docx"},
        )

    if fmt == "pdf":
        font_name, _ = _get_pdf_font()
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 60
        font_size = 12
        c.setFont(font_name, font_size)
        c.drawString(50, y, "Danh sách câu hỏi")
        y -= 30
        for i, q in enumerate(questions, 1):
            line = f"{i}. {q}"
            wrapped = _wrap_text(line, width - 100, font_name, font_size)
            for part in wrapped:
                if y < 60:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - 60
                c.drawString(50, y, part)
                y -= 18
        c.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=questions.pdf"},
        )

    content = "\n".join(questions)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=questions.txt"},
    )


@app.get("/ai-engines")
def list_ai_engines() -> dict:
    engines = []
    if OPENAI_API_KEY:
        engines.append({"key": "openai", "label": f"OpenAI ({OPENAI_MODEL})", "available": True})
    else:
        engines.append({"key": "openai", "label": "OpenAI (chưa có API key)", "available": False})
    if GEMINI_API_KEY:
        engines.append({"key": "gemini", "label": f"Gemini ({GEMINI_MODEL})", "available": True})
    else:
        engines.append({"key": "gemini", "label": "Gemini (chưa có API key)", "available": False})
    # Ollama: try a quick check
    ollama_ok = False
    try:
        with httpx.Client(timeout=2) as client:
            r = client.get(f"{OLLAMA_BASE}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    if ollama_ok:
        engines.append({"key": "ollama", "label": f"Ollama ({OLLAMA_MODEL})", "available": True})
    else:
        engines.append({"key": "ollama", "label": "Ollama (không kết nối được)", "available": False})
    return {"engines": engines}


@app.get("/topics")
def list_topics() -> dict:
    return {
        "topics": [
            {"key": key, "label": value["label"], "keywords": value["keywords"]}
            for key, value in TOPICS.items()
        ]
    }


@app.get("/subjects")
def list_subjects() -> dict:
    return {
        "subjects": [
            {"key": key, "label": value["label"], "lang": value.get("lang", "vi")}
            for key, value in SUBJECTS.items()
        ],
        "question_types": [
            {"key": key, "label": value} for key, value in QUESTION_TYPES.items()
        ],
        "grades": list(range(1, 13)),
    }


@app.get("/templates")
def list_templates() -> dict:
    return {
        "templates": [
            {
                "key": key,
                "label": TOPICS.get(key, {}).get("label", key),
                "items": TEMPLATES.get(key, []),
            }
            for key in TEMPLATES.keys()
        ]
    }


@app.get("/version")
def version() -> dict:
    return {
        "commit": os.getenv("VERCEL_GIT_COMMIT_SHA", ""),
        "time": os.getenv("VERCEL_GIT_COMMIT_MESSAGE", ""),
    }


@app.get("/sample-folders")
def list_sample_folders() -> dict:
    if SAMPLE_DIR is None or not SAMPLE_DIR.exists():
        return {"folders": []}
    folders = [p.name for p in SAMPLE_DIR.iterdir() if p.is_dir()]
    folders.sort()
    return {"folders": folders}




@app.get("/sample-files")
def list_sample_files(subject: str) -> dict:
    if SAMPLE_DIR is None or not subject or not SAMPLE_DIR.exists():
        return {"files": []}
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if SAMPLE_DIR not in subject_dir.parents or not subject_dir.exists():
        return {"files": []}
    files = [
        p.name
        for p in subject_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".docx", ".md"}
    ]
    files.sort()
    return {"files": files}


@app.get("/sample-content")
def sample_content(subject: str, filename: str) -> dict:
    if not subject or not filename:
        return {"content": ""}
    if SAMPLE_DIR is None:
        return {"content": ""}
    subject_dir = (SAMPLE_DIR / subject).resolve()
    if not SAMPLE_DIR.exists() or SAMPLE_DIR not in subject_dir.parents:
        return {"content": ""}
    target = (subject_dir / filename).resolve()
    if subject_dir not in target.parents or not target.exists() or not target.is_file():
        return {"content": ""}
    content = _read_sample_file(target)
    return {"content": content}


@app.post("/parse-sample-urls")
def parse_sample_urls(payload: ParseSamplesRequest) -> dict:
    contents: List[str] = []
    for url in payload.urls:
        if not url:
            continue
        try:
            contents.append(_read_sample_url(url))
        except Exception:
            continue
    merged = "\n".join(contents)
    return {"content": merged, "samples": _split_questions(merged)}


@app.post("/upload-sample")
def upload_sample(subject: str = Form(...), file: UploadFile = Form(...)) -> dict:
    if not subject.strip():
        return {"ok": False, "message": "Thiếu tên môn"}
    if SAMPLE_DIR is None:
        return {"ok": False, "message": "Không tìm thấy thư mục đề mẫu"}
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    subject_dir = (SAMPLE_DIR / subject.strip()).resolve()
    if SAMPLE_DIR not in subject_dir.parents:
        return {"ok": False, "message": "Đường dẫn không hợp lệ"}
    subject_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "").name
    if not filename:
        return {"ok": False, "message": "Tên file không hợp lệ"}
    if Path(filename).suffix.lower() not in {".txt", ".docx", ".md"}:
        return {"ok": False, "message": "Chỉ hỗ trợ .txt, .docx, .md"}
    target = subject_dir / filename
    with target.open("wb") as f:
        f.write(file.file.read())
    return {"ok": True, "message": "Đã tải lên", "filename": filename, "subject": subject.strip()}
