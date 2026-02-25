import io
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, Form, Request, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
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


# Pydantic models for Question Bank
class QuestionCreate(BaseModel):
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    subject: str
    topic: Optional[str] = None
    grade: Optional[str] = None
    question_type: str = "mcq"
    difficulty: str = "medium"
    tags: Optional[str] = None
    source: Optional[str] = None


class QuestionUpdate(BaseModel):
    content: Optional[str] = None
    options: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    grade: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    tags: Optional[str] = None


class ExamCreate(BaseModel):
    title: str
    description: Optional[str] = None
    subject: str
    grade: Optional[str] = None
    duration_minutes: Optional[int] = None


class ExamUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    duration_minutes: Optional[int] = None


class BulkSaveRequest(BaseModel):
    questions: List[QuestionCreate]


class AIAnalyzeRequest(BaseModel):
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    subject: Optional[str] = None
    ai_engine: str = "openai"


class AISuggestRequest(BaseModel):
    content: str
    subject: Optional[str] = None
    count: int = 3
    ai_engine: str = "openai"


class AIReviewRequest(BaseModel):
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    subject: Optional[str] = None
    ai_engine: str = "openai"


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

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "ai_settings.json"


def _load_saved_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_settings_to_file(data: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


_saved = _load_saved_settings()

OPENAI_API_KEY = _saved.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = _saved.get("openai_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_BASE = _saved.get("openai_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

GEMINI_API_KEY = _saved.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = _saved.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OLLAMA_BASE = _saved.get("ollama_base") or os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = _saved.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3.2:latest")


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


_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(Câu(\s+hỏi)?\s*\d*\s*[:.)]\s*|Question\s*\d*\s*[:.)]\s*)",
    re.IGNORECASE,
)


def _strip_leading_numbering(text: str) -> str:
    text = _QUESTION_PREFIX_RE.sub("", text)
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
            "num_predict": 2048,
            "temperature": 0.8,
        },
    }
    try:
        with httpx.Client(timeout=180) as client:
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


_LABEL_RE = re.compile(r"^(Câu(\s+hỏi)?|Question)\s*\d*\s*:?\s*$", re.IGNORECASE)


def _normalize_ai_blocks(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    # First pass: merge blocks that are MCQ options or question labels
    merged: List[str] = []
    for block in blocks:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        is_option_block = bool(MCQ_OPTION_RE.match(first_line))
        is_label = bool(_LABEL_RE.match(block.strip()))
        if is_option_block and merged:
            # Attach options to previous question
            merged[-1] = merged[-1] + "\n" + block
        elif is_label:
            # "Câu 1:", "Câu hỏi 2:" — start a new entry that will absorb
            # the next block as the actual question content
            merged.append(block)
        elif merged and _LABEL_RE.match(merged[-1].splitlines()[0].strip()):
            # Previous block was just a label; merge this content into it
            merged[-1] = merged[-1] + "\n" + block
        else:
            merged.append(block)
    # Second pass: clean up numbering and drop empty / separator blocks
    cleaned: List[str] = []
    for block in merged:
        lines = block.splitlines()
        # Remove leading label lines like "Câu 1:"
        while lines and _LABEL_RE.match(lines[0].strip()):
            lines.pop(0)
        if lines:
            lines[0] = _strip_leading_numbering(lines[0])
        result = "\n".join(lines).strip()
        if not result or re.match(r"^-{2,}$", result):
            continue
        # Skip intro/filler lines that aren't actual questions
        # (no "?" and no MCQ options means it's likely just a preamble)
        has_question_mark = "?" in result
        has_options = any(MCQ_OPTION_RE.match(ln.strip()) for ln in result.splitlines())
        has_blank = "..." in result
        if not has_question_mark and not has_options and not has_blank:
            continue
        cleaned.append(result)
    return cleaned


def _build_topic_prompt(subject_key: str, grade: int, qtype: str, count: int, topic: str = "", difficulty: str = "medium") -> str:
    subject = SUBJECTS.get(subject_key, {"label": subject_key, "lang": "vi"})
    label = subject.get("label", subject_key)
    lang = subject.get("lang", "vi")
    # Resolve topic label
    topic_label = ""
    if topic:
        topics = SUBJECT_TOPICS.get(subject_key, [])
        for t in topics:
            if t["key"] == topic:
                topic_label = t["label"]
                break
        if not topic_label:
            topic_label = topic
    # Difficulty text
    difficulty_map_en = {"easy": "easy", "medium": "medium", "hard": "hard/challenging"}
    difficulty_map_vi = {"easy": "dễ", "medium": "trung bình", "hard": "khó"}
    if lang == "en":
        grade_text = f"Grade {grade}"
        type_text = {
            "mcq": "multiple-choice",
            "blank": "fill-in-the-blank",
            "essay": "short-answer",
        }.get(qtype, "multiple-choice")
        topic_text = f", topic: {topic_label}" if topic_label else ""
        diff_text = difficulty_map_en.get(difficulty, "medium")
        example = ""
        if qtype == "mcq":
            example = (
                "\n\nExample format:\n"
                "What is 2 + 2?\n"
                "A) 3\nB) 4\nC) 5\nD) 6\n"
                "\nWhat color is the sky?\n"
                "A) Red\nB) Green\nC) Blue\nD) Yellow\n"
                "\n---ANSWERS---\n1. B\n2. C\n"
            )
        return (
            "You are an education content writer.\n"
            f"Create {count} {type_text} questions for {label} ({grade_text}{topic_text}).\n"
            f"Difficulty level: {diff_text}.\n"
            "Rules:\n"
            "- Do NOT number questions. Do NOT write 'Question 1', 'Q1', etc.\n"
            "- Separate questions with a blank line.\n"
            "- If multiple-choice, provide 4 options labeled A) B) C) D) on separate lines.\n"
            "- If fill-in-the-blank, use \"...\" for the blank.\n"
            "- After all questions, add a line \"---ANSWERS---\" then list the correct answer for each question (e.g. 1. B, 2. A, ...).\n"
            f"{example}"
        )
    grade_text = f"Lớp {grade}"
    type_text = QUESTION_TYPES.get(qtype, "Trắc nghiệm")
    topic_text = f" về {topic_label}" if topic_label else ""
    diff_text = difficulty_map_vi.get(difficulty, "trung bình")
    mcq_rule = ""
    if qtype == "mcq":
        mcq_rule = "Mỗi câu có 4 đáp án A) B) C) D).\n"
    elif qtype == "blank":
        mcq_rule = "Dùng \"...\" cho chỗ trống.\n"
    return (
        f"Tạo {count} câu hỏi {type_text} môn {label} {grade_text}{topic_text}, độ khó: {diff_text}.\n"
        f"{mcq_rule}"
        "Mỗi câu cách nhau bằng một dòng trống.\n"
        "Cuối cùng viết ---ĐÁP ÁN--- rồi liệt kê đáp án.\n"
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


# ============== OMML to LaTeX Functions ==============

def _omml_to_latex(element, ns: dict) -> str:
    """
    Convert an OMML math element to LaTeX notation.

    Handles common OMML elements:
    - m:f (fraction) -> \\frac{num}{den}
    - m:sSup (superscript) -> base^{exp}
    - m:sSub (subscript) -> base_{idx}
    - m:sSubSup (sub+superscript) -> base_{sub}^{sup}
    - m:rad (radical/sqrt) -> \\sqrt{content} or \\sqrt[n]{content}
    - m:nary (summation, product, integral) -> \\sum, \\prod, \\int
    - m:d (delimiter/parentheses) -> \\left( ... \\right)
    - m:t (text) -> plain text
    """
    if element is None:
        return ''
    if element.tag is None:
        return ''

    ns_prefix = ns.get('m', '')
    tag = element.tag.replace(ns_prefix, '').replace('{', '').replace('}', '')

    # Text element
    if tag == 't':
        return element.text or ''

    # Fraction
    if tag == 'f':
        num = element.find('m:num', ns)
        den = element.find('m:den', ns)
        num_latex = _omml_children_to_latex(num, ns) if num is not None else ''
        den_latex = _omml_children_to_latex(den, ns) if den is not None else ''
        return f'\\frac{{{num_latex}}}{{{den_latex}}}'

    # Superscript
    if tag == 'sSup':
        base = element.find('m:e', ns)
        sup = element.find('m:sup', ns)
        base_latex = _omml_children_to_latex(base, ns) if base is not None else ''
        sup_latex = _omml_children_to_latex(sup, ns) if sup is not None else ''
        return f'{base_latex}^{{{sup_latex}}}'

    # Subscript
    if tag == 'sSub':
        base = element.find('m:e', ns)
        sub = element.find('m:sub', ns)
        base_latex = _omml_children_to_latex(base, ns) if base is not None else ''
        sub_latex = _omml_children_to_latex(sub, ns) if sub is not None else ''
        return f'{base_latex}_{{{sub_latex}}}'

    # Subscript + Superscript
    if tag == 'sSubSup':
        base = element.find('m:e', ns)
        sub = element.find('m:sub', ns)
        sup = element.find('m:sup', ns)
        base_latex = _omml_children_to_latex(base, ns) if base is not None else ''
        sub_latex = _omml_children_to_latex(sub, ns) if sub is not None else ''
        sup_latex = _omml_children_to_latex(sup, ns) if sup is not None else ''
        return f'{base_latex}_{{{sub_latex}}}^{{{sup_latex}}}'

    # Radical (square root)
    if tag == 'rad':
        deg = element.find('m:deg', ns)
        content = element.find('m:e', ns)
        content_latex = _omml_children_to_latex(content, ns) if content is not None else ''
        deg_latex = _omml_children_to_latex(deg, ns) if deg is not None else ''
        if deg_latex and deg_latex.strip():
            return f'\\sqrt[{deg_latex}]{{{content_latex}}}'
        return f'\\sqrt{{{content_latex}}}'

    # N-ary (sum, product, integral)
    if tag == 'nary':
        chr_elem = element.find('.//m:chr', ns)
        sub = element.find('m:sub', ns)
        sup = element.find('m:sup', ns)
        content = element.find('m:e', ns)

        # Get the operator character safely
        op_char = '∑'  # default
        if chr_elem is not None:
            ns_m = ns.get('m', '')
            op_char = chr_elem.get(f'{{{ns_m}}}val') or chr_elem.get('val') or '∑'
        op_map = {'∑': '\\sum', '∏': '\\prod', '∫': '\\int', '⋃': '\\bigcup', '⋂': '\\bigcap'}
        op_latex = op_map.get(op_char, op_char) or '\\sum'

        sub_latex = _omml_children_to_latex(sub, ns) if sub is not None else ''
        sup_latex = _omml_children_to_latex(sup, ns) if sup is not None else ''
        content_latex = _omml_children_to_latex(content, ns) if content is not None else ''

        result = op_latex
        if sub_latex:
            result += f'_{{{sub_latex}}}'
        if sup_latex:
            result += f'^{{{sup_latex}}}'
        result += f' {content_latex}'
        return result

    # Delimiter (parentheses, brackets)
    if tag == 'd':
        content = element.find('m:e', ns)
        content_latex = _omml_children_to_latex(content, ns) if content is not None else ''
        # Get delimiter characters safely
        ns_m = ns.get('m', '')
        beg_chr = element.find('.//m:begChr', ns)
        end_chr = element.find('.//m:endChr', ns)
        beg = '('
        end = ')'
        if beg_chr is not None:
            beg = beg_chr.get(f'{{{ns_m}}}val') or beg_chr.get('val') or '('
        if end_chr is not None:
            end = end_chr.get(f'{{{ns_m}}}val') or end_chr.get('val') or ')'
        return f'\\left{beg}{content_latex}\\right{end}'

    # Matrix
    if tag == 'm':
        rows = element.findall('m:mr', ns)
        row_latexes = []
        for row in rows:
            cells = row.findall('m:e', ns)
            cell_latexes = [_omml_children_to_latex(c, ns) for c in cells]
            row_latexes.append(' & '.join(cell_latexes))
        return '\\begin{matrix}' + ' \\\\ '.join(row_latexes) + '\\end{matrix}'

    # Run element (container for text)
    if tag == 'r':
        return _omml_children_to_latex(element, ns)

    # Default: recursively process children
    return _omml_children_to_latex(element, ns)


def _omml_children_to_latex(element, ns: dict) -> str:
    """Process all children of an OMML element and return LaTeX."""
    if element is None:
        return ''
    result = []
    for child in element:
        result.append(_omml_to_latex(child, ns))
    return ''.join(result)


def _extract_paragraph_with_math(para, use_latex: bool = False) -> str:
    """
    Extract text from a paragraph, including OMML math formulas.

    python-docx's para.text doesn't include math formulas (OMML).
    This function extracts both regular text and math formula text.

    Args:
        para: A python-docx paragraph object
        use_latex: If True, convert math formulas to LaTeX notation
    """
    # Namespace for OMML (Office Math Markup Language)
    MATH_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
    WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    ns = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}

    result_parts = []

    # Iterate through all child elements in order
    for child in para._element:
        tag = child.tag

        # Regular text run
        if tag == f'{WORD_NS}r':
            # Get text from <w:t> elements
            for t_elem in child.iter(f'{WORD_NS}t'):
                if t_elem.text:
                    result_parts.append(t_elem.text)

        # Math formula (OMML)
        elif tag == f'{MATH_NS}oMath' or tag == f'{MATH_NS}oMathPara':
            if use_latex:
                # Convert OMML to LaTeX
                latex = _omml_children_to_latex(child, ns)
                if latex:
                    result_parts.append(f'${latex}$')
            else:
                # Extract plain text from math formula
                math_texts = []
                for m_t in child.iter(f'{MATH_NS}t'):
                    if m_t.text:
                        math_texts.append(m_t.text)
                if math_texts:
                    result_parts.append(''.join(math_texts))

    return ''.join(result_parts).strip()


def _extract_cell_with_math(cell, use_latex: bool = False) -> str:
    """
    Extract text from a table cell, including OMML math formulas.

    Args:
        cell: A python-docx table cell object
        use_latex: If True, convert math formulas to LaTeX notation
    """
    cell_parts = []
    for para in cell.paragraphs:
        para_text = _extract_paragraph_with_math(para, use_latex=use_latex)
        if para_text:
            cell_parts.append(para_text)
    return '\n'.join(cell_parts)


def _extract_docx_content(doc: Document, include_textboxes: bool = True, use_latex: bool = False) -> str:
    """
    Extract all text content from a Word document.
    Handles paragraphs, tables, and optionally text boxes.

    This is the unified function for reading Word documents across all features.

    Args:
        doc: A python-docx Document object
        include_textboxes: If True, extract text from text boxes
        use_latex: If True, convert math formulas to LaTeX notation
    """
    all_text = []
    seen_text = set()  # To avoid duplicates

    def add_text(text: str):
        """Add text if not duplicate"""
        text = text.strip()
        if text and text not in seen_text:
            seen_text.add(text)
            all_text.append(text)

    # 1. Read paragraphs (including text boxes if enabled)
    for para in doc.paragraphs:
        # Get paragraph text including math formulas
        text = _extract_paragraph_with_math(para, use_latex=use_latex)
        if text:
            add_text(text)

        # Extract text from text boxes (embedded in paragraphs)
        if include_textboxes:
            try:
                xml = para._element.xml
                if 'w:txbxContent' in xml:
                    import re as re_mod
                    txbx_matches = re_mod.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
                    for txbx in txbx_matches:
                        texts = re_mod.findall(r'<w:t[^>]*>([^<]+)</w:t>', txbx)
                        content = ''.join(texts).strip()
                        if content:
                            add_text(content)
            except Exception:
                pass  # Ignore textbox extraction errors

    # 2. Read tables (important for Math exams which store questions in tables)
    for table in doc.tables:
        for row in table.rows:
            row_texts = []
            for cell in row.cells:
                cell_text = _extract_cell_with_math(cell, use_latex=use_latex)
                if cell_text:
                    row_texts.append(cell_text)

            if row_texts:
                # Join cells with newline to separate different content in the same row
                row_content = "\n".join(row_texts)
                add_text(row_content)

    # Join with double newline to separate questions/sections
    return "\n\n".join(all_text)


def _parse_cell_based_questions(doc: Document) -> List[dict]:
    """
    Parse questions from documents where each table cell contains a complete question.
    This format is used in ASMO Science exams where:
    - Each row has 1 cell
    - Each cell contains: Question EN, Question VN, Options A-E (or fill-in-blank)

    Options may be:
    - Prefixed: A. option / B. option
    - Non-prefixed bilingual: lines with "/" separator (EN/VN)
    - Non-prefixed simple: short answer options like "20 m", "40 m"

    Returns list of parsed questions, or empty list if format doesn't match.
    """
    from docx.oxml.ns import qn

    def get_cell_text_from_row(row) -> str:
        """Get text from first cell, handling edge cases where row.cells is empty.
        Also extracts options from nested tables (2x2 grid format)."""
        # First try normal cell access
        if row.cells:
            cell = row.cells[0]
            # Check for nested tables with options
            nested_tables = cell._element.findall('.//' + qn('w:tbl'))
            if nested_tables:
                # Get main paragraphs (before nested table)
                main_paras = []
                for p in cell.paragraphs:
                    p_text = p.text.strip()
                    if p_text:
                        main_paras.append(p_text)
                # Get options from nested table
                options = []
                for nt in nested_tables:
                    for tr in nt.findall('.//' + qn('w:tr')):
                        for tc in tr.findall('.//' + qn('w:tc')):
                            t_elements = tc.findall('.//' + qn('w:t'))
                            cell_text = ''.join([t.text or '' for t in t_elements]).strip()
                            if cell_text:
                                options.append(cell_text)
                # Format: question lines + option lines
                if main_paras and options:
                    # Format options as separate lines
                    return '\n'.join(main_paras + options)
            return cell.text.strip()

        # Fallback: extract directly from XML, preserving paragraph breaks
        tc_elements = row._tr.findall(qn('w:tc'))
        if tc_elements:
            tc = tc_elements[0]
            # Check for nested tables
            nested_tbls = tc.findall('.//' + qn('w:tbl'))
            if nested_tbls:
                # Get text from main paragraphs (not in nested table)
                main_paras = []
                for p in tc.findall(qn('w:p')):  # Direct children only
                    t_elements = p.findall('.//' + qn('w:t'))
                    p_text = ''.join([t.text or '' for t in t_elements]).strip()
                    if p_text:
                        main_paras.append(p_text)
                # Get options from nested table
                options = []
                for nt in nested_tbls:
                    for tr in nt.findall('.//' + qn('w:tr')):
                        for tc_inner in tr.findall('.//' + qn('w:tc')):
                            t_elements = tc_inner.findall('.//' + qn('w:t'))
                            cell_text = ''.join([t.text or '' for t in t_elements]).strip()
                            if cell_text:
                                options.append(cell_text)
                if main_paras and options:
                    return '\n'.join(main_paras + options)

            # No nested table - get all paragraphs
            p_elements = tc.findall('.//' + qn('w:p'))
            paragraphs = []
            for p in p_elements:
                t_elements = p.findall('.//' + qn('w:t'))
                p_text = ''.join([t.text or '' for t in t_elements]).strip()
                if p_text:
                    paragraphs.append(p_text)
            return '\n'.join(paragraphs)
        return ''

    questions = []

    if not doc.tables:
        return []

    # Check if this is the expected format: table with multiple rows
    # Accept 1-2 columns (some exams have 2 columns but we only use the first)
    table = doc.tables[0]
    if len(table.columns) > 2 or len(table.rows) < 5:
        return []

    # Check first few cells to see if it matches expected format
    # Must have options (A. style) OR bilingual separator "/" with multiple lines
    has_valid_format = False
    for row in table.rows[:3]:
        cell_text = get_cell_text_from_row(row)
        if not cell_text:
            continue
        lines = [l.strip() for l in cell_text.split('\n') if l.strip()]
        # Valid if: has A. style options
        if re.search(r'\n[A-E]\.\s+', cell_text):
            has_valid_format = True
            break
        # Check for bilingual lines (with / separator - may or may not have spaces)
        bilingual_lines = [l for l in lines if '/' in l and len(l) > 5]
        if len(bilingual_lines) >= 3:
            has_valid_format = True
            break
        # Check for question structure: 2+ lines ending with ?, then 3+ short answer lines
        question_ends = [i for i, l in enumerate(lines) if l.endswith('?')]
        if question_ends and len(lines) - question_ends[-1] - 1 >= 3:
            has_valid_format = True
            break
    if not has_valid_format:
        return []

    def is_bilingual_separator(line: str) -> bool:
        """Check if line contains bilingual separator (/ with or without spaces)."""
        return '/' in line and len(line) > 5

    def is_question_line(line: str) -> bool:
        """Check if line is part of question (ends with ? or is long enough)."""
        return line.endswith('?') or len(line) > 80

    # Parse each cell as a complete question
    for row in table.rows:
        cell_text = get_cell_text_from_row(row)
        if not cell_text:
            continue

        lines = [l.strip() for l in cell_text.split('\n') if l.strip()]
        if len(lines) < 2:  # Need at least 2 lines (EN + VN for fill-blank)
            continue

        question_lines = []
        options = []

        # First pass: try to find A., B., C. style options
        has_prefixed_options = any(re.match(r'^[A-E]\.\s+', line) for line in lines)

        if has_prefixed_options:
            for line in lines:
                opt_match = re.match(r'^([A-E])\.\s+(.+)$', line)
                if opt_match:
                    options.append(opt_match.group(2))
                else:
                    question_lines.append(line)
        else:
            # Non-prefixed format: find where question ends and options begin
            # Strategy 1: question lines end with "?" or are very long
            # Strategy 2: options are shorter lines with "/" separator

            # Check if we have bilingual options (lines with "/" that are shorter)
            bilingual_option_lines = []
            for i, line in enumerate(lines):
                # Options typically: shorter than question, contain "/" for bilingual
                if '/' in line and len(line) < 120 and i >= 2:
                    bilingual_option_lines.append(i)

            if len(bilingual_option_lines) >= 3:
                # Found bilingual options - split at first option
                first_opt_idx = bilingual_option_lines[0]
                question_lines = lines[:first_opt_idx]
                options = lines[first_opt_idx:]
            else:
                # Find the last line that looks like a question
                last_question_idx = -1
                for i, line in enumerate(lines):
                    if line.endswith('?') or (len(line) > 100 and i < len(lines) - 3):
                        last_question_idx = i

                if last_question_idx >= 0 and last_question_idx < len(lines) - 2:
                    # Everything up to last_question_idx is question
                    question_lines = lines[:last_question_idx + 1]
                    # Everything after is options
                    options = lines[last_question_idx + 1:]
                else:
                    # Fallback: check for bilingual options with "/"
                    for line in lines:
                        if options:
                            # Already collecting options
                            if is_bilingual_separator(line) or (len(line) < 50 and not line.endswith('?')):
                                options.append(line)
                            else:
                                question_lines.append(line)
                        elif is_bilingual_separator(line) and not line.endswith('?'):
                            # This is an option line (bilingual with EN/VN)
                            options.append(line)
                        else:
                            question_lines.append(line)

        # Accept questions with options OR fill-in-blank questions (contain ___ or ...)
        # For fill-blank without options: include all lines as question
        all_text = '\n'.join(lines)
        is_fill_blank = '___' in all_text or '________' in all_text

        if question_lines and options:
            questions.append({
                "question": '\n'.join(question_lines),
                "options": options
            })
        elif is_fill_blank and len(lines) >= 2:
            # Fill-in-blank question (may not have separate options)
            questions.append({
                "question": all_text,
                "options": []  # No options for fill-in-blank
            })

    return questions


def _extract_docx_lines(doc: Document, include_textboxes: bool = True, use_latex: bool = False) -> tuple:
    """
    Extract all text lines from a Word document as a list.
    Similar to _extract_docx_content but returns list instead of joined string.
    For Math exams, preserves line breaks within cells to keep options separate.

    Args:
        doc: A python-docx Document object
        include_textboxes: If True, extract text from text boxes
        use_latex: If True, convert math formulas to LaTeX notation

    Returns:
        tuple: (lines, table_options) where table_options is a list of option lists from standalone tables
    """
    all_lines = []
    seen_text = set()
    table_options = []  # Store options from standalone option tables
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    def add_line(text: str):
        text = text.strip()
        if not text:
            return
        # Allow duplicate option lines - they may belong to different questions
        # Check for inline options (A) B) C) D)) or single option lines (A. xxx, B. yyy)
        is_option_line = ('A)' in text and 'B)' in text) or re.match(r'^[A-E]\s*[.\)]\s*\S', text)
        if is_option_line or text not in seen_text:
            if not is_option_line:
                seen_text.add(text)
            all_lines.append(text)

    def add_multiline_text(text: str):
        """Add text that may contain multiple lines (from table cells)."""
        text = text.strip()
        if not text:
            return
        # Split by newlines and add each line
        for line in text.split('\n'):
            line = line.strip()
            if line:
                add_line(line)

    def extract_nested_table_options(cell) -> str:
        """Extract options from nested tables within a cell (2x2 grid format)."""
        nested_tables = cell._element.findall('.//w:tbl', ns)
        options = []
        for nt in nested_tables:
            rows = nt.findall('.//w:tr', ns)
            for row_elem in rows:
                cells = row_elem.findall('.//w:tc', ns)
                for cell_elem in cells:
                    t_elems = cell_elem.findall('.//w:t', ns)
                    cell_text = ''.join([t.text or '' for t in t_elems]).strip()
                    if cell_text:
                        options.append(cell_text)
        if len(options) >= 2:
            # Format as A) B) C) D) options
            labels = ['A', 'B', 'C', 'D', 'E']
            formatted = '\t'.join([f"{labels[i]}) {opt}" for i, opt in enumerate(options[:5])])
            return formatted
        return ''

    # Read paragraphs
    for para in doc.paragraphs:
        text = _extract_paragraph_with_math(para, use_latex=use_latex)
        if text:
            add_line(text)

        # Text boxes
        if include_textboxes:
            try:
                xml = para._element.xml
                if 'w:txbxContent' in xml:
                    import re as re_mod
                    txbx_matches = re_mod.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
                    for txbx in txbx_matches:
                        texts = re_mod.findall(r'<w:t[^>]*>([^<]+)</w:t>', txbx)
                        content = ''.join(texts).strip()
                        if content:
                            add_line(content)
            except Exception:
                pass

    # Read tables - including nested tables for options
    # For Math exams, each cell may contain a full question with multiple lines
    for table in doc.tables:
        # Check if this table is a standalone options table (1 row with A. B. C. D. E. cells)
        is_options_table = False
        if len(table.rows) == 1 and len(table.columns) >= 4:
            first_row_texts = [cell.text.strip() for cell in table.rows[0].cells]
            # Check if cells start with A., B., C., D., E.
            if (first_row_texts and
                first_row_texts[0].startswith('A.') and
                len(first_row_texts) >= 2 and first_row_texts[1].startswith('B.')):
                is_options_table = True
                # Extract options without the A./B./C./D./E. prefix
                opts = []
                for cell_text in first_row_texts:
                    # Remove "A. ", "B. ", etc. prefix
                    opt_text = re.sub(r'^[A-E]\.\s*', '', cell_text).strip()
                    if opt_text:
                        opts.append(opt_text)
                if opts:
                    table_options.append(opts)

        if not is_options_table:
            for row in table.rows:
                for cell in row.cells:
                    # Get cell text with math formulas
                    cell_text = _extract_cell_with_math(cell, use_latex=use_latex)
                    if cell_text:
                        # Use multiline handler to split and add each line
                        add_multiline_text(cell_text)

                    # Check for nested tables (options in 2x2 grid)
                    nested_opts = extract_nested_table_options(cell)
                    if nested_opts:
                        add_line(nested_opts)

    return all_lines, table_options


def _parse_bilingual_questions(lines: List[str], table_options: List[List[str]] = None) -> List[dict]:
    """
    Parse questions from Word document lines.
    Returns list of dicts with 'question' and 'options' keys.

    Args:
        lines: List of text lines from document
        table_options: Optional list of option lists from standalone tables
                      (for Science exams where options are in separate tables)
    """
    questions = []
    table_options = table_options or []
    table_option_idx = 0  # Index to track which table options to use next

    # Patterns
    question_num_pattern = re.compile(r'^\s*(\d+)\s*[.\)]\s*(.*)$')
    section_header_pattern = re.compile(r'^Section\s+[A-Z]\s*:', re.IGNORECASE)

    def extract_options_from_line(line: str) -> List[str]:
        """Extract options from a line with A) B) C) D) markers or EN-VIE format."""
        opts = []
        # Match uppercase A-E followed by . or ) and then actual content (not just dots)
        # Must be at start, after space/tab, and followed by real text
        marker_pattern = re.compile(r'(?:^|(?<=\s)|(?<=\t))([A-E])\s*[.\)]\s*(?=\S)', re.IGNORECASE)
        markers = list(marker_pattern.finditer(line))
        # Only process if we find at least 2 options (A and B minimum)
        if markers and len(markers) >= 2:
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                # Skip if the option text is just dots or empty
                if opt_text and opt_text not in ['...', '..', '.', '…']:
                    opts.append(opt_text)
        # EN-VIE format: option A has no marker "text\tB) text\tC) text"
        elif '\t' in line and re.search(r'B\s*\)', line, re.IGNORECASE):
            b_markers = list(re.finditer(r'(?:^|(?<=\t))([B-E])\s*\)\s*', line, re.IGNORECASE))
            if b_markers:
                # Option A is the text before B)
                first_b = b_markers[0]
                opt_a = line[:first_b.start()].strip().rstrip('\t ')
                if opt_a and opt_a not in ['...', '..', '.', '…']:
                    opts.append(opt_a)
                # Extract B), C), D), E) options
                for idx, m in enumerate(b_markers):
                    start = m.end()
                    if idx + 1 < len(b_markers):
                        end = b_markers[idx + 1].start()
                    else:
                        end = len(line)
                    opt_text = line[start:end].strip().rstrip('\t ')
                    if opt_text and opt_text not in ['...', '..', '.', '…']:
                        opts.append(opt_text)
        return opts

    def is_option_line(line: str) -> bool:
        """Check if a line looks like options (starts with A) or has tab-separated B) C) D)."""
        if re.match(r'^A\s*[.\)]', line, re.IGNORECASE):
            return True
        if '\t' in line and re.search(r'A\s*[.\)].*B\s*[.\)]', line, re.IGNORECASE):
            return True
        # EN-VIE format: option A has no marker, just "text\tB) text\tC) text"
        if '\t' in line and re.search(r'B\s*\)', line, re.IGNORECASE):
            markers = re.findall(r'[B-E]\s*\)', line, re.IGNORECASE)
            if len(markers) >= 2:  # At least B) and C)
                return True
        return False

    def has_fill_blank(text: str) -> bool:
        """Check if text contains fill-in-blank marker."""
        return '…' in text or '...' in text or '___' in text

    def is_reading_passage(text: str) -> bool:
        """Check if text is a long reading passage."""
        return len(text) > 150 and not is_option_line(text) and not has_fill_blank(text)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Skip section headers
        if section_header_pattern.match(line):
            i += 1
            continue

        # Skip standalone option lines (they belong to previous question)
        if is_option_line(line):
            i += 1
            continue

        # Get next non-empty lines for look-ahead
        next_nonempty_lines = []
        for k in range(i + 1, min(i + 10, len(lines))):
            if lines[k].strip():
                next_nonempty_lines.append(lines[k].strip())
                if len(next_nonempty_lines) >= 5:
                    break

        # Skip lines that are just single option markers (e.g., "E. 7" or "A. answer")
        single_option_only = re.match(r'^[A-E]\s*[.\)]\s*\S+$', line, re.IGNORECASE) and len(line) < 30
        if single_option_only:
            i += 1
            continue

        # Skip lines that are just standalone numbers (e.g., "1", "2", "3" - image labels)
        standalone_number = re.match(r'^\d+$', line)
        if standalone_number:
            i += 1
            continue

        # Check for question detection
        q_match = question_num_pattern.match(line)
        # Only consider fill_blank as standalone question if it starts with a number
        is_fill_blank = has_fill_blank(line) and not is_option_line(line) and q_match
        is_direct_question = line.endswith('?') and len(line) > 15

        # Check if line looks like a question stem (followed by options)
        # But NOT if it looks like a translation line (Vietnamese text following English)
        is_question_stem = False
        if len(line) < 200 and line and line[0].isupper():
            if next_nonempty_lines and is_option_line(next_nonempty_lines[0]):
                # Skip if this line looks like a Vietnamese translation (no question number)
                # Vietnamese translations typically follow a numbered English line
                if not q_match and not is_direct_question:
                    # Check if previous line was a numbered question with same structure
                    pass  # Don't treat as question stem - likely a translation
                else:
                    is_question_stem = True

        if q_match or is_fill_blank or is_question_stem or is_direct_question:
            question_text_parts = []
            options = []

            # Check for cloze passage with numbered blanks like (31), (32), etc.
            # Must also have fill-in markers (___/...) to be considered a cloze passage
            # This avoids false positives like "Look at pictures (2) and (3)"
            numbered_blank_pattern = re.compile(r'\((\d+)\)')
            numbered_blanks = numbered_blank_pattern.findall(line)
            has_blank_markers = '___' in line or ('...' in line and not line.strip().endswith('...'))
            is_cloze_passage = len(numbered_blanks) >= 2 and has_blank_markers

            if q_match:
                q_content = q_match.group(2).strip()
                if q_content:
                    line_opts = extract_options_from_line(q_content)
                    if line_opts:
                        question_text_parts.append(f"{q_match.group(1)}. ...")
                        options.extend(line_opts)
                    else:
                        question_text_parts.append(q_content)
            else:
                question_text_parts.append(line)

            # For cloze passages, collect all option lines and create separate questions
            if is_cloze_passage:
                j = i + 1
                all_options = []
                while j < len(lines) and is_option_line(lines[j].strip()):
                    line_opts = extract_options_from_line(lines[j].strip())
                    if line_opts:
                        all_options.append(line_opts)
                    j += 1

                # Create a question for each numbered blank with its options
                passage_text = line
                for idx, blank_num in enumerate(numbered_blanks):
                    if idx < len(all_options):
                        questions.append({
                            "question": f"Cloze ({blank_num}): {passage_text[:100]}...",
                            "options": all_options[idx]
                        })
                i = j
                continue

            # Collect subsequent lines until next question
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                # Stop at section header
                if section_header_pattern.match(next_line):
                    break

                # Stop at numbered question
                if question_num_pattern.match(next_line):
                    break

                # Extract options from line (inline A) B) C) D) format)
                if is_option_line(next_line):
                    line_opts = extract_options_from_line(next_line)
                    if line_opts:
                        options.extend(line_opts)
                    j += 1
                    # Stop after collecting 5 options (some exams have A-E)
                    if len(options) >= 5:
                        break
                    continue

                # Check for single option line (A. xxx or B. yyy on separate lines)
                single_opt_match = re.match(r'^([A-E])\s*[.\)]\s*(.+)$', next_line, re.IGNORECASE)
                if single_opt_match:
                    opt_text = single_opt_match.group(2).strip()
                    if opt_text:
                        options.append(opt_text)
                    j += 1
                    # Stop after collecting 5 options
                    if len(options) >= 5:
                        break
                    continue

                # Stop at another fill-in-blank or reading passage (new question)
                if (has_fill_blank(next_line) or is_reading_passage(next_line)) and options:
                    break

                # Stop at another long passage if we have options
                if len(next_line) > 100 and options:
                    break

                # Otherwise it's question text (only if no options yet)
                if not options:
                    question_text_parts.append(next_line)
                j += 1

            question_text = "\n".join(question_text_parts)

            # If no options found but we have table options available, use them
            if not options and table_option_idx < len(table_options):
                options = table_options[table_option_idx]
                table_option_idx += 1

            # Save question even without options (open-ended questions)
            if question_text.strip():
                questions.append({
                    "question": question_text.strip(),
                    "options": options,
                    "number": int(q_match.group(1)) if q_match else None
                })

            i = j
        else:
            i += 1

    return questions


def _read_sample_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        doc = Document(str(path))
        return _extract_docx_content(doc)
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
    ai_engine: str = Form("ollama"),
    topic: str = Form(""),
    difficulty: str = Form("medium"),
) -> dict:
    if not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {"questions": [], "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng."}
    count = max(1, min(50, count))
    grade = max(1, min(12, grade))
    prompt = _build_topic_prompt(subject, grade, qtype, count, topic, difficulty)
    text, err = _call_ai(prompt, ai_engine)
    if not text:
        msg = f"Không nhận được phản hồi từ AI. {err}" if err else "Không nhận được phản hồi từ AI."
        return {"questions": [], "answers": "", "message": msg}
    # Split answers section from questions using regex
    answers = ""
    answer_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:ĐÁP ÁN|Đáp án|đáp án|ANSWERS|Answers|Answer Key|answer key)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = answer_pattern.search(text)
    if match:
        answers = text[match.end():].strip()
        text = text[:match.start()]
    questions = _normalize_ai_blocks(text)
    questions = [q for q in questions if q.strip()]
    return {"questions": questions[:count], "answers": answers}


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


@app.post("/api/export-exam")
async def export_exam(request: Request):
    """Export exam questions with full content and options."""
    data = await request.json()
    questions = data.get("questions", [])
    fmt = data.get("format", "docx")
    title = data.get("title", "Đề thi")

    if not questions:
        raise HTTPException(status_code=400, detail="Không có câu hỏi để xuất")

    if fmt == "docx":
        doc = Document()
        doc.add_heading(title, level=1)

        for i, q in enumerate(questions, 1):
            content = q.get("content", "")
            options = q.get("options", [])
            correct = q.get("correct_answer", "")

            # Add question
            p = doc.add_paragraph()
            p.add_run(f"Câu {i}. ").bold = True
            p.add_run(content)

            # Add options
            if options:
                labels = ['A', 'B', 'C', 'D', 'E']
                for j, opt in enumerate(options[:5]):
                    # Strip existing label if present (e.g., "A) text" -> "text")
                    opt_clean = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', str(opt).strip())
                    doc.add_paragraph(f"    {labels[j]}) {opt_clean}")

            doc.add_paragraph("")  # Blank line

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=de_thi.docx"},
        )

    if fmt == "pdf":
        font_name, _ = _get_pdf_font()
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 60
        font_size = 12

        c.setFont(font_name, 16)
        c.drawString(50, y, title)
        y -= 40
        c.setFont(font_name, font_size)

        for i, q in enumerate(questions, 1):
            content = q.get("content", "")
            options = q.get("options", [])

            # Question
            line = f"Câu {i}. {content}"
            wrapped = _wrap_text(line, width - 100, font_name, font_size)
            for part in wrapped:
                if y < 80:
                    c.showPage()
                    c.setFont(font_name, font_size)
                    y = height - 60
                c.drawString(50, y, part)
                y -= 18

            # Options
            if options:
                labels = ['A', 'B', 'C', 'D', 'E']
                for j, opt in enumerate(options[:5]):
                    # Strip existing label if present (e.g., "A) text" -> "text")
                    opt_clean = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', str(opt).strip())
                    opt_line = f"    {labels[j]}) {opt_clean}"
                    opt_wrapped = _wrap_text(opt_line, width - 120, font_name, font_size)
                    for part in opt_wrapped:
                        if y < 80:
                            c.showPage()
                            c.setFont(font_name, font_size)
                            y = height - 60
                        c.drawString(70, y, part)
                        y -= 16

            y -= 10  # Space between questions

        c.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=de_thi.pdf"},
        )

    raise HTTPException(status_code=400, detail="Format không hỗ trợ")


@app.post("/ai-settings")
async def update_ai_settings(request: Request) -> dict:
    global OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE
    global GEMINI_API_KEY, GEMINI_MODEL
    global OLLAMA_BASE, OLLAMA_MODEL
    data = await request.json()
    if "openai_key" in data:
        OPENAI_API_KEY = data["openai_key"].strip()
    if "openai_model" in data:
        OPENAI_MODEL = data["openai_model"].strip() or "gpt-4o-mini"
    if "openai_base" in data:
        OPENAI_API_BASE = data["openai_base"].strip() or "https://api.openai.com/v1"
    if "gemini_key" in data:
        GEMINI_API_KEY = data["gemini_key"].strip()
    if "gemini_model" in data:
        GEMINI_MODEL = data["gemini_model"].strip() or "gemini-2.0-flash"
    if "ollama_base" in data:
        OLLAMA_BASE = data["ollama_base"].strip() or "http://localhost:11434"
    if "ollama_model" in data:
        OLLAMA_MODEL = data["ollama_model"].strip() or "llama3.2:latest"
    _save_settings_to_file({
        "openai_key": OPENAI_API_KEY,
        "openai_model": OPENAI_MODEL,
        "openai_base": OPENAI_API_BASE,
        "gemini_key": GEMINI_API_KEY,
        "gemini_model": GEMINI_MODEL,
        "ollama_base": OLLAMA_BASE,
        "ollama_model": OLLAMA_MODEL,
    })
    return {"ok": True}


@app.get("/ai-settings")
def get_ai_settings() -> dict:
    return {
        "openai_key": OPENAI_API_KEY[:4] + "****" if len(OPENAI_API_KEY) > 4 else "",
        "openai_model": OPENAI_MODEL,
        "openai_base": OPENAI_API_BASE,
        "gemini_key": GEMINI_API_KEY[:4] + "****" if len(GEMINI_API_KEY) > 4 else "",
        "gemini_model": GEMINI_MODEL,
        "ollama_base": OLLAMA_BASE,
        "ollama_model": OLLAMA_MODEL,
    }


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
        "topics": {key: topics for key, topics in SUBJECT_TOPICS.items()},
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


# ============================================================================
# QUESTION BANK APIs
# ============================================================================

from app.database import (
    get_db, QuestionCRUD, ExamCRUD, HistoryCRUD, Question, Exam, ExamQuestion, ExamVariant, UsageHistory, init_db
)

# Ensure database is initialized
init_db()


@app.get("/api/questions")
def get_questions(
    skip: int = 0,
    limit: int = 100,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_type: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get all questions with optional filters"""
    questions = QuestionCRUD.get_all(
        db,
        skip=skip,
        limit=limit,
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        search=search,
    )
    return {
        "questions": [
            {
                "id": q.id,
                "content": q.content,
                "options": q.options,
                "answer": q.answer,
                "explanation": q.explanation,
                "subject": q.subject,
                "topic": q.topic,
                "grade": q.grade,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "difficulty_score": q.difficulty_score,
                "tags": q.tags,
                "source": q.source,
                "quality_score": q.quality_score,
                "quality_issues": q.quality_issues,
                "times_used": q.times_used,
                "created_at": q.created_at.isoformat() if q.created_at else None,
                "updated_at": q.updated_at.isoformat() if q.updated_at else None,
            }
            for q in questions
        ],
        "total": QuestionCRUD.count(db, subject=subject, topic=topic, difficulty=difficulty),
    }


@app.get("/api/questions/stats")
def get_questions_stats(db: Session = Depends(get_db)):
    """Get statistics about questions in the bank"""
    stats = QuestionCRUD.get_by_subject_stats(db)
    total = sum(s["total"] for s in stats.values())
    return {"stats": stats, "total": total}


@app.get("/api/questions/{question_id}")
def get_question(question_id: int, db: Session = Depends(get_db)):
    """Get a single question by ID"""
    question = QuestionCRUD.get(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")
    return {
        "id": question.id,
        "content": question.content,
        "options": question.options,
        "answer": question.answer,
        "explanation": question.explanation,
        "subject": question.subject,
        "topic": question.topic,
        "grade": question.grade,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "difficulty_score": question.difficulty_score,
        "tags": question.tags,
        "source": question.source,
        "quality_score": question.quality_score,
        "quality_issues": question.quality_issues,
        "times_used": question.times_used,
        "created_at": question.created_at.isoformat() if question.created_at else None,
    }


@app.post("/api/questions")
def create_question(data: QuestionCreate, db: Session = Depends(get_db)):
    """Create a new question"""
    question = QuestionCRUD.create(
        db,
        content=data.content,
        options=data.options,
        answer=data.answer,
        explanation=data.explanation,
        subject=data.subject,
        topic=data.topic,
        grade=data.grade,
        question_type=data.question_type,
        difficulty=data.difficulty,
        tags=data.tags,
        source=data.source,
    )
    return {"ok": True, "id": question.id, "message": "Đã lưu câu hỏi"}


@app.post("/api/questions/bulk")
def bulk_create_questions(data: BulkSaveRequest, db: Session = Depends(get_db)):
    """Create multiple questions at once"""
    questions_data = [
        {
            "content": q.content,
            "options": q.options,
            "answer": q.answer,
            "explanation": q.explanation,
            "subject": q.subject,
            "topic": q.topic,
            "grade": q.grade,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "tags": q.tags,
            "source": q.source,
        }
        for q in data.questions
    ]
    questions = QuestionCRUD.bulk_create(db, questions_data)
    return {"ok": True, "count": len(questions), "message": f"Đã lưu {len(questions)} câu hỏi"}


@app.put("/api/questions/{question_id}")
def update_question(question_id: int, data: QuestionUpdate, db: Session = Depends(get_db)):
    """Update an existing question"""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    question = QuestionCRUD.update(db, question_id, **update_data)
    if not question:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")
    return {"ok": True, "message": "Đã cập nhật câu hỏi"}


@app.delete("/api/questions/{question_id}")
def delete_question(question_id: int, db: Session = Depends(get_db)):
    """Delete a question"""
    if QuestionCRUD.delete(db, question_id):
        return {"ok": True, "message": "Đã xóa câu hỏi"}
    raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi")


# ============================================================================
# USAGE HISTORY APIs
# ============================================================================

@app.get("/api/history")
def get_history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Get usage history"""
    history_items = HistoryCRUD.get_all(db, skip=skip, limit=limit)
    return {
        "history": [
            {
                "timestamp": h.timestamp,
                "type": h.history_type,
                "filename": h.filename,
                "count": h.count,
                "difficulty": h.difficulty,
                "questions": json.loads(h.questions_json) if h.questions_json else [],
            }
            for h in history_items
        ]
    }


@app.post("/api/history")
def save_history(request: Request, db: Session = Depends(get_db)):
    """Save a history item"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    body = loop.run_until_complete(request.json())
    loop.close()

    timestamp = body.get("timestamp", "")
    history_type = body.get("type", "unknown")
    filename = body.get("filename")
    count = body.get("count", 0)
    difficulty = body.get("difficulty")
    questions = body.get("questions", [])
    questions_json = json.dumps(questions, ensure_ascii=False) if questions else None

    HistoryCRUD.create(
        db,
        timestamp=timestamp,
        history_type=history_type,
        filename=filename,
        count=count,
        difficulty=difficulty,
        questions_json=questions_json
    )
    return {"ok": True, "message": "Đã lưu lịch sử"}


@app.delete("/api/history/{timestamp}")
def delete_history_item(timestamp: str, db: Session = Depends(get_db)):
    """Delete a history item by timestamp"""
    if HistoryCRUD.delete_by_timestamp(db, timestamp):
        return {"ok": True, "message": "Đã xóa lịch sử"}
    raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử")


@app.delete("/api/history")
def clear_history(db: Session = Depends(get_db)):
    """Clear all history"""
    count = HistoryCRUD.delete_all(db)
    return {"ok": True, "message": f"Đã xóa {count} mục lịch sử"}


# ============================================================================
# EXAM HISTORY APIs
# ============================================================================

@app.get("/api/exams")
def get_exams(
    skip: int = 0,
    limit: int = 100,
    subject: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get all exams with optional filters"""
    exams = ExamCRUD.get_all(db, skip=skip, limit=limit, subject=subject)
    return {
        "exams": [
            {
                "id": e.id,
                "title": e.title,
                "description": e.description,
                "subject": e.subject,
                "grade": e.grade,
                "total_questions": e.total_questions,
                "duration_minutes": e.duration_minutes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "variants_count": len(e.variants),
            }
            for e in exams
        ]
    }


@app.get("/api/exams/{exam_id}")
def get_exam(exam_id: int, db: Session = Depends(get_db)):
    """Get a single exam with its questions"""
    exam = ExamCRUD.get(db, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")

    questions = []
    for eq in sorted(exam.exam_questions, key=lambda x: x.order):
        q = eq.question
        questions.append({
            "id": q.id,
            "content": q.content,
            "options": q.options,
            "answer": q.answer,
            "order": eq.order,
            "points": eq.points,
            "difficulty": q.difficulty,
        })

    return {
        "id": exam.id,
        "title": exam.title,
        "description": exam.description,
        "subject": exam.subject,
        "grade": exam.grade,
        "total_questions": exam.total_questions,
        "duration_minutes": exam.duration_minutes,
        "created_at": exam.created_at.isoformat() if exam.created_at else None,
        "questions": questions,
        "variants": [
            {"id": v.id, "variant_code": v.variant_code, "created_at": v.created_at.isoformat()}
            for v in exam.variants
        ],
    }


@app.post("/api/exams")
def create_exam(data: ExamCreate, db: Session = Depends(get_db)):
    """Create a new exam"""
    exam = ExamCRUD.create(
        db,
        title=data.title,
        description=data.description,
        subject=data.subject,
        grade=data.grade,
        duration_minutes=data.duration_minutes,
    )
    return {"ok": True, "id": exam.id, "message": "Đã tạo đề thi"}


@app.put("/api/exams/{exam_id}")
def update_exam(exam_id: int, data: ExamUpdate, db: Session = Depends(get_db)):
    """Update an exam"""
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    exam = ExamCRUD.update(db, exam_id, **update_data)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    return {"ok": True, "message": "Đã cập nhật đề thi"}


@app.delete("/api/exams/{exam_id}")
def delete_exam(exam_id: int, db: Session = Depends(get_db)):
    """Delete an exam"""
    if ExamCRUD.delete(db, exam_id):
        return {"ok": True, "message": "Đã xóa đề thi"}
    raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")


@app.post("/api/exams/{exam_id}/questions")
def add_questions_to_exam(exam_id: int, question_ids: List[int], db: Session = Depends(get_db)):
    """Add questions to an exam"""
    exam = ExamCRUD.add_questions(db, exam_id, question_ids)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    return {"ok": True, "message": f"Đã thêm {len(question_ids)} câu hỏi"}


@app.delete("/api/exams/{exam_id}/questions/{question_id}")
def remove_question_from_exam(exam_id: int, question_id: int, db: Session = Depends(get_db)):
    """Remove a question from an exam"""
    if ExamCRUD.remove_question(db, exam_id, question_id):
        return {"ok": True, "message": "Đã xóa câu hỏi khỏi đề"}
    raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi trong đề")


@app.post("/api/exams/{exam_id}/variants")
def create_exam_variant(exam_id: int, variant_code: str, db: Session = Depends(get_db)):
    """Create a shuffled variant of an exam"""
    exam = ExamCRUD.get(db, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")

    # Get question IDs
    question_ids = [eq.question_id for eq in exam.exam_questions]
    if not question_ids:
        raise HTTPException(status_code=400, detail="Đề thi chưa có câu hỏi")

    # Shuffle questions
    shuffled_ids = question_ids.copy()
    random.shuffle(shuffled_ids)

    # Create answer mapping for MCQ
    answer_mapping = {}
    for qid in shuffled_ids:
        q = db.query(Question).filter(Question.id == qid).first()
        if q and q.options:
            try:
                opts = json.loads(q.options)
                if isinstance(opts, list) and len(opts) >= 2:
                    shuffled_opts = opts.copy()
                    random.shuffle(shuffled_opts)
                    # Map original answer to new position
                    if q.answer in opts:
                        old_idx = opts.index(q.answer)
                        new_idx = shuffled_opts.index(q.answer)
                        answer_mapping[str(qid)] = {
                            "original": chr(65 + old_idx),
                            "shuffled": chr(65 + new_idx),
                            "shuffled_options": shuffled_opts,
                        }
            except json.JSONDecodeError:
                pass

    variant = ExamCRUD.create_variant(
        db,
        exam_id=exam_id,
        variant_code=variant_code,
        question_order=json.dumps(shuffled_ids),
        answer_mapping=json.dumps(answer_mapping) if answer_mapping else None,
    )

    return {
        "ok": True,
        "variant_id": variant.id,
        "variant_code": variant_code,
        "message": f"Đã tạo phiên bản {variant_code}",
    }


# ============================================================================
# IMPORT/EXPORT APIs (JSON, Moodle XML, QTI)
# ============================================================================

@app.post("/api/import/json")
async def import_json(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from JSON file"""
    content = await file.read()
    try:
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="File JSON không hợp lệ")

    questions_data = data if isinstance(data, list) else data.get("questions", [])
    if not questions_data:
        raise HTTPException(status_code=400, detail="Không tìm thấy câu hỏi trong file")

    imported = []
    for q in questions_data:
        question = QuestionCRUD.create(
            db,
            content=q.get("content", q.get("question", "")),
            options=json.dumps(q.get("options", [])) if q.get("options") else None,
            answer=q.get("answer", ""),
            explanation=q.get("explanation", ""),
            subject=q.get("subject", "unknown"),
            topic=q.get("topic", ""),
            grade=q.get("grade", ""),
            question_type=q.get("question_type", q.get("type", "mcq")),
            difficulty=q.get("difficulty", "medium"),
            tags=json.dumps(q.get("tags", [])) if q.get("tags") else None,
            source="import_json",
        )
        imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi"}


@app.post("/api/import/moodle-xml")
async def import_moodle_xml(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from Moodle XML format"""
    import xml.etree.ElementTree as ET

    content = await file.read()
    try:
        root = ET.fromstring(content.decode("utf-8"))
    except ET.ParseError:
        raise HTTPException(status_code=400, detail="File XML không hợp lệ")

    imported = []
    for question_elem in root.findall(".//question"):
        qtype = question_elem.get("type", "multichoice")
        if qtype == "category":
            continue

        # Get question text
        name_elem = question_elem.find("name/text")
        questiontext_elem = question_elem.find("questiontext/text")

        content = ""
        if questiontext_elem is not None and questiontext_elem.text:
            content = questiontext_elem.text
        elif name_elem is not None and name_elem.text:
            content = name_elem.text

        if not content:
            continue

        # Get answers
        options = []
        correct_answer = ""
        for answer_elem in question_elem.findall("answer"):
            answer_text_elem = answer_elem.find("text")
            if answer_text_elem is not None and answer_text_elem.text:
                options.append(answer_text_elem.text)
                fraction = answer_elem.get("fraction", "0")
                if float(fraction) > 0:
                    correct_answer = answer_text_elem.text

        # Map Moodle question types
        question_type_map = {
            "multichoice": "mcq",
            "truefalse": "mcq",
            "shortanswer": "blank",
            "essay": "essay",
            "matching": "matching",
            "numerical": "blank",
        }

        question = QuestionCRUD.create(
            db,
            content=content,
            options=json.dumps(options) if options else None,
            answer=correct_answer,
            subject="imported",
            question_type=question_type_map.get(qtype, "mcq"),
            difficulty="medium",
            source="import_moodle",
        )
        imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi từ Moodle XML"}


@app.post("/api/import/qti")
async def import_qti(file: UploadFile, db: Session = Depends(get_db)):
    """Import questions from QTI (IMS Question & Test Interoperability) format"""
    import xml.etree.ElementTree as ET

    content = await file.read()
    try:
        root = ET.fromstring(content.decode("utf-8"))
    except ET.ParseError:
        raise HTTPException(status_code=400, detail="File QTI không hợp lệ")

    # Handle namespaces
    ns = {"qti": "http://www.imsglobal.org/xsd/imsqti_v2p1"}

    imported = []

    # Try QTI 2.1 format first
    for item in root.findall(".//qti:assessmentItem", ns) or root.findall(".//assessmentItem"):
        title = item.get("title", "")

        # Get item body
        item_body = item.find(".//qti:itemBody", ns) or item.find(".//itemBody")
        if item_body is None:
            continue

        content = "".join(item_body.itertext()).strip()
        if not content and title:
            content = title

        # Get choices
        options = []
        choice_interaction = item.find(".//qti:choiceInteraction", ns) or item.find(".//choiceInteraction")
        if choice_interaction is not None:
            for choice in choice_interaction.findall(".//qti:simpleChoice", ns) or choice_interaction.findall(".//simpleChoice"):
                choice_text = "".join(choice.itertext()).strip()
                if choice_text:
                    options.append(choice_text)

        # Get correct answer
        correct_answer = ""
        response_declaration = item.find(".//qti:responseDeclaration", ns) or item.find(".//responseDeclaration")
        if response_declaration is not None:
            correct_value = response_declaration.find(".//qti:value", ns) or response_declaration.find(".//value")
            if correct_value is not None and correct_value.text:
                # Find the option matching the identifier
                identifier = correct_value.text
                for choice in (choice_interaction.findall(".//qti:simpleChoice", ns) if choice_interaction is not None else []):
                    if choice.get("identifier") == identifier:
                        correct_answer = "".join(choice.itertext()).strip()
                        break

        if content:
            question = QuestionCRUD.create(
                db,
                content=content,
                options=json.dumps(options) if options else None,
                answer=correct_answer,
                subject="imported",
                question_type="mcq",
                difficulty="medium",
                source="import_qti",
            )
            imported.append(question.id)

    return {"ok": True, "count": len(imported), "message": f"Đã import {len(imported)} câu hỏi từ QTI"}


@app.get("/api/export/json")
def export_questions_json(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to JSON format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic, difficulty=difficulty)

    data = []
    for q in questions:
        data.append({
            "content": q.content,
            "options": json.loads(q.options) if q.options else [],
            "answer": q.answer,
            "explanation": q.explanation,
            "subject": q.subject,
            "topic": q.topic,
            "grade": q.grade,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "tags": json.loads(q.tags) if q.tags else [],
        })

    return StreamingResponse(
        io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=questions.json"},
    )


def _escape_xml(text: str) -> str:
    """Escape special XML characters"""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


@app.get("/api/export/moodle-xml")
def export_moodle_xml(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to Moodle XML format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic)

    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<quiz>\n'

    for q in questions:
        qtype = "multichoice" if q.question_type == "mcq" else "shortanswer"

        xml_content += f'  <question type="{qtype}">\n'
        xml_content += f'    <name><text>{_escape_xml(q.content[:100])}</text></name>\n'
        xml_content += f'    <questiontext format="html"><text><![CDATA[{q.content}]]></text></questiontext>\n'

        if q.options:
            try:
                options = json.loads(q.options)
                for opt in options:
                    fraction = "100" if opt == q.answer else "0"
                    xml_content += f'    <answer fraction="{fraction}"><text><![CDATA[{opt}]]></text></answer>\n'
            except json.JSONDecodeError:
                pass

        if q.explanation:
            xml_content += f'    <generalfeedback format="html"><text><![CDATA[{q.explanation}]]></text></generalfeedback>\n'

        xml_content += '  </question>\n'

    xml_content += '</quiz>'

    return StreamingResponse(
        io.BytesIO(xml_content.encode("utf-8")),
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=questions_moodle.xml"},
    )


@app.get("/api/export/qti")
def export_qti(
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Export questions to QTI 2.1 format"""
    questions = QuestionCRUD.get_all(db, limit=10000, subject=subject, topic=topic)

    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<assessmentTest xmlns="http://www.imsglobal.org/xsd/imsqti_v2p1"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.imsglobal.org/xsd/imsqti_v2p1 http://www.imsglobal.org/xsd/qti/qtiv2p1/imsqti_v2p1.xsd"
                identifier="test_export"
                title="Exported Questions">
'''

    for i, q in enumerate(questions, 1):
        item_id = f"item_{i}"
        xml_content += f'''  <assessmentItem identifier="{item_id}" title="{_escape_xml(q.content[:50])}" adaptive="false" timeDependent="false">
    <responseDeclaration identifier="RESPONSE" cardinality="single" baseType="identifier">
'''

        if q.options and q.answer:
            try:
                options = json.loads(q.options)
                correct_idx = options.index(q.answer) if q.answer in options else 0
                xml_content += f'      <correctResponse><value>choice_{correct_idx}</value></correctResponse>\n'
            except (json.JSONDecodeError, ValueError):
                pass

        xml_content += '''    </responseDeclaration>
    <itemBody>
'''
        xml_content += f'      <p>{_escape_xml(q.content)}</p>\n'

        if q.options:
            try:
                options = json.loads(q.options)
                xml_content += '      <choiceInteraction responseIdentifier="RESPONSE" shuffle="false" maxChoices="1">\n'
                for j, opt in enumerate(options):
                    xml_content += f'        <simpleChoice identifier="choice_{j}">{_escape_xml(opt)}</simpleChoice>\n'
                xml_content += '      </choiceInteraction>\n'
            except json.JSONDecodeError:
                pass

        xml_content += '''    </itemBody>
  </assessmentItem>
'''

    xml_content += '</assessmentTest>'

    return StreamingResponse(
        io.BytesIO(xml_content.encode("utf-8")),
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=questions_qti.xml"},
    )


# ============================================================================
# AI FEATURES: Difficulty Analysis, Suggestions, Quality Review
# ============================================================================

def _load_ai_settings() -> dict:
    """Load AI settings from file"""
    settings_file = Path("ai_settings.json")
    if settings_file.exists():
        with open(settings_file) as f:
            return json.load(f)
    return {}


async def _call_ai_engine(engine: str, prompt: str, settings: dict) -> str:
    """Helper function to call different AI engines"""
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout for large prompts
        if engine == "openai":
            # Try multiple key names for compatibility
            api_key = settings.get("openai_key") or settings.get("openai_api_key") or OPENAI_API_KEY
            base_url = settings.get("openai_base") or settings.get("openai_base_url") or OPENAI_API_BASE
            model = settings.get("openai_model") or OPENAI_MODEL

            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        elif engine == "gemini":
            api_key = settings.get("gemini_key") or settings.get("gemini_api_key") or GEMINI_API_KEY
            model = settings.get("gemini_model") or GEMINI_MODEL

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        elif engine == "ollama":
            base_url = settings.get("ollama_base") or settings.get("ollama_base_url") or OLLAMA_BASE
            model = settings.get("ollama_model") or OLLAMA_MODEL

            response = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]

        else:
            raise ValueError(f"Unknown AI engine: {engine}")


@app.post("/api/ai/analyze-difficulty")
async def ai_analyze_difficulty(data: AIAnalyzeRequest):
    """AI analyzes and estimates question difficulty"""
    settings = _load_ai_settings()
    engine = data.ai_engine

    prompt = f"""Phân tích độ khó của câu hỏi sau và trả về JSON với format:
{{
    "difficulty": "easy|medium|hard",
    "score": 0.0-1.0,
    "reasoning": "giải thích ngắn gọn",
    "factors": ["yếu tố 1", "yếu tố 2"]
}}

Câu hỏi: {data.content}
"""
    if data.options:
        prompt += f"\nCác lựa chọn: {data.options}"
    if data.subject:
        prompt += f"\nMôn học: {data.subject}"

    prompt += """

Đánh giá dựa trên:
- Độ phức tạp của khái niệm
- Số bước suy luận cần thiết
- Mức độ kiến thức yêu cầu
- Độ dài và phức tạp của đề bài

Chỉ trả về JSON, không giải thích thêm."""

    try:
        response = await _call_ai_engine(engine, prompt, settings)
        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {"ok": True, "analysis": result}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ai/suggest-similar")
async def ai_suggest_similar(data: AISuggestRequest):
    """AI suggests similar questions based on a sample"""
    settings = _load_ai_settings()
    engine = data.ai_engine

    prompt = f"""Dựa trên câu hỏi mẫu sau, hãy tạo {data.count} câu hỏi tương tự nhưng khác biệt về nội dung.

Câu hỏi mẫu: {data.content}
"""
    if data.subject:
        prompt += f"\nMôn học: {data.subject}"

    prompt += f"""

Yêu cầu:
- Giữ nguyên dạng câu hỏi và độ khó
- Thay đổi ngữ cảnh, số liệu, hoặc đối tượng
- Mỗi câu phải độc lập và có ý nghĩa

Trả về JSON array với format:
[
    {{
        "content": "nội dung câu hỏi",
        "options": ["A", "B", "C", "D"],
        "answer": "đáp án đúng",
        "explanation": "giải thích ngắn"
    }}
]

Tạo chính xác {data.count} câu hỏi. Chỉ trả về JSON array."""

    try:
        response = await _call_ai_engine(engine, prompt, settings)
        # Parse JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            suggestions = json.loads(json_match.group())
            return {"ok": True, "suggestions": suggestions}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ai/review-quality")
async def ai_review_quality(data: AIReviewRequest):
    """AI reviews question quality and identifies issues"""
    settings = _load_ai_settings()
    engine = data.ai_engine

    prompt = f"""Đánh giá chất lượng của câu hỏi sau và phát hiện các vấn đề.

Câu hỏi: {data.content}
"""
    if data.options:
        prompt += f"\nCác lựa chọn: {data.options}"
    if data.answer:
        prompt += f"\nĐáp án: {data.answer}"
    if data.subject:
        prompt += f"\nMôn học: {data.subject}"

    prompt += """

Kiểm tra các khía cạnh:
1. Ngữ pháp và chính tả
2. Độ rõ ràng của đề bài
3. Tính logic và chính xác
4. Các lựa chọn có hợp lý không (nếu có)
5. Đáp án có chính xác không

Trả về JSON với format:
{
    "quality_score": 0.0-1.0,
    "issues": [
        {
            "type": "grammar|clarity|logic|options|answer",
            "severity": "low|medium|high",
            "description": "mô tả vấn đề",
            "suggestion": "gợi ý sửa"
        }
    ],
    "summary": "tóm tắt đánh giá",
    "improved_version": "phiên bản cải thiện (nếu cần)"
}

Chỉ trả về JSON."""

    try:
        response = await _call_ai_engine(engine, prompt, settings)
        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            review = json.loads(json_match.group())
            return {"ok": True, "review": review}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Lỗi parse JSON: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================================
# WORD FILE PARSING API
# ============================================================================

@app.post("/api/parse-exam")
async def parse_exam_file(file: UploadFile):
    """Parse questions from uploaded Word document"""
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    # Try cell-based parser first (for ASMO Science format: 1 cell = 1 question)
    questions = _parse_cell_based_questions(doc)
    if questions:
        return {
            "ok": True,
            "filename": file.filename,
            "total_lines": len(questions),
            "questions": questions,
            "count": len(questions),
        }

    # Extract lines from document
    lines, table_options = _extract_docx_lines(doc)

    # Detect format and use appropriate parser
    # Check if this is a Math exam format (Question 1., Question 2., etc.)
    is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])

    # Check if this is an English Level exam format
    is_english_level_format = (
        any('Section A' in line or 'Section B' in line for line in lines[:15]) and
        file.filename and 'LEVEL' in file.filename.upper()
    )

    # Check if this is an EN-VIE bilingual format
    is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

    if is_math_format:
        questions = _parse_math_exam_questions(lines)
    elif is_english_level_format:
        questions = _parse_english_exam_questions(doc)
    elif is_envie_format:
        questions = _parse_envie_questions(doc)
    else:
        questions = _parse_bilingual_questions(lines, table_options)

    return {
        "ok": True,
        "filename": file.filename,
        "total_lines": len(lines),
        "questions": questions,
        "count": len(questions),
    }


# ============================================================================
# MATH EXAM PARSING
# ============================================================================

def _parse_math_exam_questions(lines: List[str]) -> List[dict]:
    """
    Parse Math exam questions with "Question X." format.
    Handles bilingual (English + Vietnamese) content and MCQ options A/B/C/D.
    Supports both multi-line format and single-line format (all content in one cell).
    """
    questions = []

    # Pattern to detect "Question X." header (standalone)
    question_header = re.compile(r'^Question\s+(\d+)\s*[.\)]?\s*$', re.IGNORECASE)
    # Pattern to detect options: A. xxx  B. xxx  C. xxx  D. xxx (with tabs or multiple spaces)
    option_pattern = re.compile(r'([A-D])\s*[.\)]\s*([^\t]+?)(?=\s{2,}[B-D]\s*[.\)]|\t[B-D]\s*[.\)]|$)', re.IGNORECASE)

    def extract_options_from_text(text: str) -> tuple:
        """
        Extract options from text that may contain A. xxx B. xxx C. xxx D. xxx
        Returns (content_without_options, list_of_options)
        Handles both spaced format (A. opt1  B. opt2) and compact format (A. opt1B. opt2)
        """
        # Try to find where options start - look for A. or A) pattern
        # Options are typically at the end of the text, separated by tabs or multiple spaces
        option_start_pattern = re.compile(r'(?:\s+|^)A\s*[.\)]\s*\S', re.IGNORECASE)
        match = option_start_pattern.search(text)

        if not match:
            return text, []

        # Split text into content and options part
        content_part = text[:match.start()].strip()
        options_part = text[match.start():].strip()

        # Extract individual options using different strategies
        options = []

        # Check if compact format (no spaces between options): "A. 2B. 4C. 5D. 3"
        # \S[B-D] means any non-whitespace character immediately followed by B/C/D
        is_compact = bool(re.search(r'\S[B-D]\s*[.\)]', options_part))

        if is_compact:
            # Strategy 1: Compact format - split by B. C. D. markers
            parts = re.split(r'([B-D])\s*[.\)]', options_part, flags=re.IGNORECASE)
            # First part after A.
            a_match = re.match(r'A\s*[.\)]\s*(.+?)$', parts[0].strip(), re.IGNORECASE)
            if a_match:
                options.append(a_match.group(1).strip())
            # Rest of options (B, C, D values)
            i = 1
            while i < len(parts) - 1:
                value = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if value:
                    options.append(value)
                i += 2
        else:
            # Strategy 2: Normal spaced format
            opt_matches = list(option_pattern.finditer(options_part))

            if opt_matches and len(opt_matches) >= 2:
                for m in opt_matches:
                    opt_text = m.group(2).strip()
                    # Clean up trailing whitespace and tabs
                    opt_text = re.sub(r'\s+$', '', opt_text)
                    if opt_text:
                        options.append(opt_text)

        return content_part, options

    def process_content_line(line: str, content_lines: list, options: list):
        """Process a line that may contain both content and options."""
        # Check if line has embedded options (A. B. C. D. pattern)
        if re.search(r'A\s*[.\)]\s*.+B\s*[.\)]', line, re.IGNORECASE):
            content_part, extracted_opts = extract_options_from_text(line)
            if content_part:
                content_lines.append(content_part)
            if extracted_opts:
                options.extend(extracted_opts)
        else:
            content_lines.append(line)

    # First, check if we have single-line format (each line contains full question with options)
    single_line_format = False
    for line in lines[:5]:
        if re.match(r'^Question\s+\d+\s*[.\)]?\s*.+A\s*[.\)]', line, re.IGNORECASE):
            single_line_format = True
            break

    if single_line_format:
        # Parse single-line format: "Question X. content... A. opt1 B. opt2 C. opt3 D. opt4"
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match question header with content
            combined = re.match(r'^Question\s+(\d+)\s*[.\)]?\s*(.+)$', line, re.IGNORECASE)
            if combined:
                q_num = int(combined.group(1))
                remaining = combined.group(2).strip()

                # Extract content and options
                content_part, options = extract_options_from_text(remaining)

                questions.append({
                    "question": content_part.strip(),  # No "Question X." prefix
                    "options": options[:4],  # Max 4 options
                    "answer": "",
                    "number": q_num
                })
    else:
        # Parse multi-line format
        current_question_num = None
        current_content_lines = []
        current_options = []

        def save_current_question():
            nonlocal current_question_num, current_content_lines, current_options

            if current_question_num is not None and (current_content_lines or current_options):
                question_text = "\n".join(current_content_lines)

                questions.append({
                    "question": question_text.strip(),
                    "options": current_options[:4],
                    "answer": "",
                    "number": current_question_num
                })

            current_question_num = None
            current_content_lines = []
            current_options = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for standalone "Question X." header
            header_match = question_header.match(line)
            if header_match:
                save_current_question()
                current_question_num = int(header_match.group(1))
                continue

            # Check for question header combined with content
            combined_header = re.match(r'^Question\s+(\d+)\s*[.\)]?\s*(.+)$', line, re.IGNORECASE)
            if combined_header:
                save_current_question()
                current_question_num = int(combined_header.group(1))
                remaining = combined_header.group(2).strip()
                if remaining:
                    process_content_line(remaining, current_content_lines, current_options)
                continue

            # If we're in a question, process content
            if current_question_num is not None:
                # Check for options line - first try compact format (A. 2B. 4C. 5D. 3)
                is_compact_options = bool(re.match(r'^A\s*[.\)]\s*\S', line, re.IGNORECASE) and
                                          re.search(r'\S[B-D]\s*[.\)]', line))
                if is_compact_options:
                    # Parse compact format options
                    parts = re.split(r'([B-D])\s*[.\)]', line, flags=re.IGNORECASE)
                    a_match = re.match(r'A\s*[.\)]\s*(.+?)$', parts[0].strip(), re.IGNORECASE)
                    if a_match:
                        current_options.append(a_match.group(1).strip())
                    i = 1
                    while i < len(parts) - 1:
                        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        if value:
                            current_options.append(value)
                        i += 2
                    continue

                # Check for spaced options line
                opt_matches = list(option_pattern.finditer(line))
                if opt_matches and len(opt_matches) >= 2:
                    for m in opt_matches:
                        opt_text = m.group(2).strip()
                        opt_text = re.sub(r'\s+$', '', opt_text)
                        if opt_text:
                            current_options.append(opt_text)
                    continue

                # Check for single option line (A. xxx)
                single_match = re.match(r'^([A-D])\s*[.\)]\s*(.+)$', line, re.IGNORECASE)
                if single_match:
                    current_options.append(single_match.group(2).strip())
                    continue

                # Otherwise process as content (may contain embedded options)
                process_content_line(line, current_content_lines, current_options)

        save_current_question()

    return questions


# ============================================================================
# ENGLISH EXAM PARSING
# ============================================================================

def _parse_english_exam_questions(doc: Document) -> List[dict]:
    """
    Parse English exam questions from Word document.
    Handles multiple formats specific to English Level exams:
    1. Nested 2x2 table options
    2. Paragraphs as options (reading comprehension, dialogue)
    3. Cloze passages with options in following table rows
    4. Orphan options (options without visible question text - may have image)
    5. Textbox content (dialogue context, additional question text)
    """
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    questions = []
    seen_texts = set()
    orphan_options = []  # Options without questions (text may be in image)

    # Track cloze passages and their options
    cloze_passage = None
    cloze_blanks = []
    cloze_options = []

    def extract_textbox_content(cell) -> str:
        """Extract text from textboxes in a cell (used for dialogue context)."""
        xml = cell._element.xml
        if 'w:txbxContent' not in xml:
            return ''

        matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
        texts = []
        seen = set()
        for m in matches:
            t_texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', m)
            content = ' '.join(t_texts).strip()
            # Deduplicate (textboxes often duplicated in Word)
            if content and content not in seen:
                seen.add(content)
                texts.append(content)
        return ' '.join(texts)

    def flush_cloze_questions():
        """Add cloze questions to the list when we have collected all options."""
        nonlocal cloze_passage, cloze_blanks, cloze_options
        if cloze_passage and cloze_options:
            for idx, blank_num in enumerate(cloze_blanks):
                if idx < len(cloze_options):
                    # Include full passage text for cloze questions
                    questions.append({
                        'question': f'Cloze ({blank_num}): {cloze_passage}',
                        'options': cloze_options[idx][:4]
                    })
            cloze_passage = None
            cloze_blanks = []
            cloze_options = []

    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                paras = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
                cell_text = cell.text.strip()
                textbox_content = extract_textbox_content(cell)

                nested_tables = cell._element.findall('.//w:tbl', ns)

                # Extract options from nested tables first
                options = []
                if nested_tables:
                    for nt in nested_tables:
                        rows_elem = nt.findall('.//w:tr', ns)
                        for nrow in rows_elem:
                            cells_elem = nrow.findall('.//w:tc', ns)
                            for nc in cells_elem:
                                t_elems = nc.findall('.//w:t', ns)
                                text = ''.join([t.text or '' for t in t_elems]).strip()
                                if text:
                                    options.append(text)

                # Skip completely empty cells
                if not cell_text and not options and not textbox_content:
                    continue

                # Better duplicate key: include textbox content
                para_opts_key = str(paras[1:5]) if len(paras) >= 5 else ''
                cell_key = (cell_text[:100] if cell_text else '') + str(options[:2]) + para_opts_key + textbox_content[:50]
                if cell_key in seen_texts:
                    continue
                seen_texts.add(cell_key)

                # Check for cloze passage (has numbered blanks like (31))
                numbered_blanks = re.findall(r'\((\d+)\)', cell_text)
                if len(numbered_blanks) >= 2 and not nested_tables:
                    # Flush any previous cloze questions first
                    flush_cloze_questions()

                    cloze_passage = cell_text
                    cloze_blanks = numbered_blanks
                    cloze_options = []
                    continue

                # Process nested table options
                if options:
                    # Normal question with nested table options (has question text)
                    if len(options) >= 4 and paras:
                        # First check if we have pending cloze - if so, this is NOT cloze options
                        # because it has question text
                        flush_cloze_questions()

                        q_text = ' '.join(paras)
                        # Include textbox content for dialogue/reading context
                        if textbox_content:
                            q_text = q_text + ' ' + textbox_content
                        # Remove option text that may have leaked into question text
                        for opt in options:
                            q_text = q_text.replace(opt, '').strip()

                        if q_text:
                            questions.append({
                                'question': q_text,
                                'options': options[:4]
                            })
                    # Standalone options table (for cloze or orphan)
                    elif len(options) >= 4 and not paras:
                        if cloze_passage:
                            cloze_options.append(options)
                            # Check if we have collected all cloze options
                            if len(cloze_options) >= len(cloze_blanks):
                                flush_cloze_questions()
                        else:
                            orphan_options.append(options)
                    continue

                # Paragraphs as options (reading comprehension, dialogue, antonyms)
                if len(paras) >= 5:
                    # Flush any pending cloze questions first
                    flush_cloze_questions()

                    q_text = paras[0]
                    opts = paras[1:5]
                    # Include textbox content for context (dialogue, etc.)
                    if textbox_content:
                        q_text = q_text + ' ' + textbox_content
                    if all(len(o) >= 2 for o in opts):
                        questions.append({
                            'question': q_text,
                            'options': opts
                        })
                    continue

    # Don't forget last cloze passage if not yet flushed
    flush_cloze_questions()

    # Handle orphan options - options without visible question text
    # (question text may be in an image)
    for opts in orphan_options:
        questions.append({
            'question': 'Question with options only (text may be in image)',
            'options': opts[:4]
        })

    return questions


def _parse_envie_questions(doc: Document) -> List[dict]:
    """
    Parse EN-VIE bilingual English exam questions.
    These files have different formats:
    1. Fill-blank questions followed by tab-separated options: "text\tB) text\tC) text"
    2. Questions ending with ? followed by paragraph options (no A/B/C markers)
    3. Matching questions with A) B) C) D) E) options
    """
    questions = []
    paragraphs = [p.text.strip() for p in doc.paragraphs]

    def extract_options_envie(line: str) -> List[str]:
        """Extract options from EN-VIE format line."""
        opts = []
        # Format: "optA\tB) optB\tC) optC\tD) optD" or "optA  B) optB C) optC"
        # Also handle format without tabs: "Beijing, China.	B) Athens, Greece.   C) Rome"
        if re.search(r'B\s*\)', line, re.IGNORECASE):
            # Find B) marker position
            b_match = re.search(r'B\s*\)', line, re.IGNORECASE)
            if b_match:
                # Option A is text before B)
                opt_a = line[:b_match.start()].strip().rstrip('\t ')
                if opt_a:
                    opts.append(opt_a)
                else:
                    # Image option - add placeholder
                    opts.append('[Image A]')
                # Find all B-E markers
                markers = list(re.finditer(r'([B-E])\s*\)', line, re.IGNORECASE))
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(line)
                    opt_text = line[start:end].strip().rstrip('\t ')
                    if opt_text:
                        opts.append(opt_text)
                    else:
                        # Image option - add placeholder
                        opts.append(f'[Image {m.group(1).upper()}]')
        # Format: "A) optA B) optB C) optC" (with A marker)
        elif re.match(r'^A\s*\)', line, re.IGNORECASE):
            markers = list(re.finditer(r'([A-E])\s*\)', line, re.IGNORECASE))
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
        # Format: "C) optC D) optD" or "D) optD E) optE" (continuation line)
        elif re.match(r'^[C-E]\s*\)', line, re.IGNORECASE):
            markers = list(re.finditer(r'([C-E])\s*\)', line, re.IGNORECASE))
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
        return opts

    def is_option_line_envie(line: str) -> bool:
        """Check if line contains options."""
        # Options with B) marker (tab or space separated)
        if re.search(r'B\s*\)', line, re.IGNORECASE):
            return True
        # Options starting with A)
        if re.match(r'^A\s*\)', line, re.IGNORECASE):
            return True
        # Image-only options: "B)\tC)\tD)"
        if re.match(r'^B\s*\)\s*\t', line, re.IGNORECASE):
            return True
        # Continuation line: "D) opt E) opt"
        if re.match(r'^[D-E]\s*\)', line, re.IGNORECASE):
            return True
        return False

    def has_fill_blank(text: str) -> bool:
        return '…' in text or '___' in text or '...' in text

    def is_sentence_with_blank(text: str) -> bool:
        """Check if text is a sentence containing a fill-blank (transform sentence)."""
        # Pattern: sentence with … marker, ending with period
        if has_fill_blank(text) and text.endswith('.'):
            return True
        return False

    def is_instruction_line(text: str) -> bool:
        """Check if line is an instruction (not a question)."""
        # If it has fill-blank marker, it's a question, not instruction
        if has_fill_blank(text):
            return False
        lower = text.lower()
        patterns = [
            'for each question',
            'read the following',
            'read and look',
            'read, look',
            'choose the correct',
            'choose the best',
            'answer the questions',
            'questions 1-',
            'questions (1-',
            'for questions',
        ]
        return any(p in lower for p in patterns)

    # Track question numbers we've seen to detect standalone number patterns
    seen_q_numbers = set()

    i = 0
    while i < len(paragraphs):
        line = paragraphs[i].strip()
        if not line:
            i += 1
            continue

        # Skip pure option lines (they belong to previous question)
        # Cloze option lines are handled separately in the textbox section
        if is_option_line_envie(line) and not has_fill_blank(line):
            # Check if this is a numbered cloze question "22. A) opt B) opt..."
            # Skip these as they're processed in the textbox section later
            is_numbered_cloze = re.match(r'^(\d+)\.\s*A\s*\)', line)
            if is_numbered_cloze:
                i += 1
                continue
            # Check if this is a standalone cloze option (A) at start with tab-separated B) C) D))
            # Also skip these - handled in textbox section
            is_standalone_cloze = re.match(r'^A\s*\)', line) and '\t' in line and re.search(r'B\s*\).*C\s*\).*D\s*\)', line)
            if is_standalone_cloze:
                i += 1
                continue
            # Regular option line - skip
            i += 1
            continue

        # Case 0a: Wallaby Q1-5 special format (must be BEFORE instruction skip)
        # Structure: "For each question (1-5)..." instruction
        # Then: passage paragraphs for Q1 (no number marker, can be multiple paragraphs)
        # Then: "2.", "3.", "4.", "5." markers (all consecutive)
        # Then: 15 option paragraphs (3 for each Q1-5)
        # Question content is in textboxes (Q2-5) or paragraphs before "2." (Q1)
        if 'question (1-5)' in line.lower() or 'questions (1-5)' in line.lower():
            # Extract textbox content for Q2-5 passages
            from lxml import etree
            try:
                body = doc.element.body
                xml_str = etree.tostring(body, encoding='unicode')
                textbox_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml_str, re.DOTALL)

                # Get unique textbox contents (they're often duplicated)
                textbox_contents = []
                seen_tb = set()
                for tb in textbox_matches:
                    texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', tb)
                    content = ''.join(texts).strip()
                    # Skip short content, headers, and cloze options
                    if content and len(content) > 30 and content not in seen_tb:
                        if not re.match(r'^\d+\.?\s*A\s*\)', content):  # Not cloze options
                            if 'Wallaby' not in content and 'Answer a maximum' not in content:
                                seen_tb.add(content)
                                textbox_contents.append(content)
            except:
                textbox_contents = []

            # Find all consecutive standalone numbers and passages before them
            j = i + 1
            passages_before_nums = []
            standalone_nums = []

            # First pass: collect everything before we see standalone numbers
            while j < len(paragraphs):
                next_para = paragraphs[j].strip()
                if not next_para:
                    j += 1
                    continue
                # Check for standalone number
                num_match = re.match(r'^(\d+)\.$', next_para)
                if num_match:
                    standalone_nums.append(int(num_match.group(1)))
                    j += 1
                    # Continue collecting more numbers
                    while j < len(paragraphs):
                        np = paragraphs[j].strip()
                        if not np:
                            j += 1
                            continue
                        nm = re.match(r'^(\d+)\.$', np)
                        if nm:
                            standalone_nums.append(int(nm.group(1)))
                            j += 1
                        else:
                            break
                    break
                else:
                    passages_before_nums.append(next_para)
                    j += 1

            # If we found numbers 2-5 pattern
            if standalone_nums and 2 in standalone_nums:
                # Count questions (1 + number of standalone markers)
                num_questions = 1 + len(standalone_nums)
                # Collect 3 options per question
                opts_list = []
                while j < len(paragraphs) and len(opts_list) < num_questions * 3:
                    opt_para = paragraphs[j].strip()
                    if not opt_para:
                        j += 1
                        continue
                    if is_instruction_line(opt_para) or is_option_line_envie(opt_para):
                        break
                    opts_list.append(opt_para)
                    j += 1

                # Build question contents
                # Q1: from paragraphs before "2." marker
                q1_content = ' '.join(passages_before_nums) if passages_before_nums else 'Question 1'

                # Q2-5: from textboxes (first 4 unique textbox contents)
                q_contents = [q1_content]
                for tb_idx in range(min(4, len(textbox_contents))):
                    # Truncate long passages for display
                    tb_text = textbox_contents[tb_idx]
                    if len(tb_text) > 200:
                        tb_text = tb_text[:200] + '...'
                    q_contents.append(tb_text)

                # Pad with placeholders if not enough textboxes
                while len(q_contents) < num_questions:
                    q_contents.append(f'Question {len(q_contents) + 1}')

                # Create questions from collected options
                if len(opts_list) >= num_questions * 3:
                    for q_idx in range(num_questions):
                        q_opts = opts_list[q_idx*3:(q_idx+1)*3]
                        if len(q_opts) == 3:
                            questions.append({
                                'question': q_contents[q_idx],
                                'options': q_opts
                            })
                            seen_q_numbers.add(str(q_idx + 1))
                    i = j
                    continue
            i += 1
            continue

        # Skip general instruction lines (after handling Q1-5 special case)
        if is_instruction_line(line):
            i += 1
            continue

        # Case 0b: Standalone question number "1." or "2." etc.
        # Followed by 3 option paragraphs (normal format)
        q_num_match = re.match(r'^(\d+)\.$', line)
        if q_num_match:
            q_num = q_num_match.group(1)
            if q_num not in seen_q_numbers:
                seen_q_numbers.add(q_num)
                # Collect next 3 non-empty paragraphs as options
                opts = []
                j = i + 1
                while j < len(paragraphs) and len(opts) < 3:
                    opt_line = paragraphs[j].strip()
                    if not opt_line:
                        j += 1
                        continue
                    # Stop if we hit another question number or option line
                    if re.match(r'^\d+\.$', opt_line) or is_option_line_envie(opt_line):
                        break
                    if is_instruction_line(opt_line):
                        j += 1
                        continue
                    opts.append(opt_line)
                    j += 1

                if len(opts) == 3:
                    questions.append({
                        'question': f'Question {q_num} (reading comprehension)',
                        'options': opts
                    })
                    i = j
                    continue
            i += 1
            continue

        # Find next non-empty line
        next_line = None
        next_idx = i + 1
        while next_idx < len(paragraphs):
            if paragraphs[next_idx].strip():
                next_line = paragraphs[next_idx].strip()
                break
            next_idx += 1

        # Case 1: Fill-blank question followed by option line(s)
        if has_fill_blank(line):
            # Case 1a: Options in tab-separated format
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for multi-line options (C) D) on next line)
                while j < len(paragraphs) and len(opts) < 4:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check if line starts with C) or D) (continuation)
                    if re.match(r'^[C-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if opts:
                    questions.append({
                        'question': line,
                        'options': opts
                    })
                    i = j
                    continue

            # Case 1b: Options as separate paragraphs (no A/B/C markers)
            # Common in Story format
            if next_line and not is_option_line_envie(next_line):
                opts = []
                j = i + 1
                while j < len(paragraphs) and len(opts) < 4:
                    opt_line = paragraphs[j].strip()
                    if not opt_line:
                        j += 1
                        continue
                    # Stop if we hit another question or option line
                    if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                        break
                    # Skip instruction lines
                    if is_instruction_line(opt_line):
                        j += 1
                        continue
                    opts.append(opt_line)
                    j += 1

                if len(opts) >= 3:  # Need at least 3 options
                    questions.append({
                        'question': line,
                        'options': opts[:4]
                    })
                    i = j
                    continue

        # Case 2: Question ending with ? followed by option line (text or image)
        if line.endswith('?') and len(line) > 10:
            # First check if next line is an option line (image options)
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                if opts:
                    questions.append({
                        'question': line,
                        'options': opts
                    })
                    i = next_idx + 1
                    continue

            # Otherwise collect next 3 paragraphs as options (no markers)
            # Common in Q1-5 sections with reading comprehension
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < 3:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                # Stop if we hit another question or option line
                if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                    break
                # Skip instruction lines
                if is_instruction_line(opt_line):
                    j += 1
                    continue
                # This looks like an option
                opts.append(opt_line)
                j += 1

            if len(opts) == 3:  # EN-VIE Q1-5 has exactly 3 options
                questions.append({
                    'question': line,
                    'options': opts
                })
                i = j
                continue

        # Case 2b: Dialogue question NOT ending with ? (Grey Kangaroo Q12, Q14, Q15)
        # Format: "I think we should come up with a plan B." followed by 3 response options
        # Distinguishing features:
        # - Short sentence ending with period (dialogue prompt)
        # - Following 3 lines are responses (not option markers like A) B))
        # - Must be in dialogue section (after "For each question (11-15)")
        if line.endswith('.') and len(line) > 15 and len(line) < 80 and not has_fill_blank(line):
            # First check for special format: next line has "optA\tB) optB" and following line has "C) optC"
            # This is Grey Kangaroo Q14 format
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                break

            if j < len(paragraphs):
                first_opt_line = paragraphs[j].strip()
                # Check for "optA\tB) optB" pattern
                ab_match = re.search(r'^(.+?)\tB\s*\)\s*(.+)$', first_opt_line)
                if ab_match:
                    opt_a = ab_match.group(1).strip()
                    opt_b = ab_match.group(2).strip()
                    # Look for C) option on next line
                    j += 1
                    while j < len(paragraphs):
                        np = paragraphs[j].strip()
                        if not np:
                            j += 1
                            continue
                        break
                    opt_c = None
                    if j < len(paragraphs):
                        c_line = paragraphs[j].strip()
                        c_match = re.match(r'^C\s*\)\s*(.+)$', c_line, re.IGNORECASE)
                        if c_match:
                            opt_c = c_match.group(1).strip()
                            j += 1
                    if opt_a and opt_b and opt_c:
                        questions.append({
                            'question': line,
                            'options': [opt_a, opt_b, opt_c]
                        })
                        i = j
                        continue

            # Otherwise check for dialogue responses (no markers)
            dialogue_opts = []
            j = i + 1
            while j < len(paragraphs) and len(dialogue_opts) < 4:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                # Stop if we hit a question, option line with markers, or instruction
                if opt_line.endswith('?') or has_fill_blank(opt_line) or is_instruction_line(opt_line):
                    break
                # Stop if line has B) C) markers (this is an option line for different question)
                if re.search(r'B\s*\)', opt_line):
                    break
                # Dialogue responses are short sentences (< 60 chars) ending with period
                if len(opt_line) < 70 and opt_line.endswith('.'):
                    dialogue_opts.append(opt_line)
                    j += 1
                else:
                    break

            # Dialogue questions have exactly 3-4 response options
            if len(dialogue_opts) == 3 or len(dialogue_opts) == 4:
                questions.append({
                    'question': line,
                    'options': dialogue_opts[:3]
                })
                i = j
                continue

        # Case 3: Matching question - description followed by option line with 5 choices
        # e.g., "Maria loves comfort food..." followed by "The Safari. B) Origo. C)..."
        # Skip numbered cloze format "22. A) ..."
        if len(line) > 30 and not line.endswith('?') and not has_fill_blank(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            if next_line and is_option_line_envie(next_line):
                # Check if options have 5 choices (matching format)
                opts = extract_options_envie(next_line)
                if len(opts) >= 4:  # Matching usually has 5 options
                    questions.append({
                        'question': line,
                        'options': opts[:5]
                    })
                    i = next_idx + 1
                    continue

        # Case 4: Question ending with , followed by paragraph options (incomplete sentence)
        # e.g., "According to the notice," followed by 3 options
        if line.endswith(',') and len(line) > 15:
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < 3:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                    break
                if is_instruction_line(opt_line):
                    j += 1
                    continue
                opts.append(opt_line)
                j += 1

            if len(opts) == 3:
                questions.append({
                    'question': line,
                    'options': opts
                })
                i = j
                continue

        # Case 5a: Reading comprehension with C) marker on option 2 line
        # Format: "Abbreviations in the likes of BRB and LOL" + opt1 + "opt2\tC) opt3"
        # Skip numbered cloze format "22. A) ..."
        # Skip if line ends with '.' AND has commas (looks like option list, not question)
        looks_like_option = line.endswith('.') and ',' in line and len(line) < 60
        if len(line) > 20 and len(line) < 150 and not line.endswith('?') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line) and not looks_like_option:
            j = i + 1
            opt1 = None
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                opt1 = np
                break

            if opt1 and j + 1 < len(paragraphs):
                opt2_line = paragraphs[j + 1].strip() if j + 1 < len(paragraphs) else ""
                c_marker = re.search(r'\tC\s*\)|(\s{2,})C\s*\)', opt2_line)
                if c_marker:
                    opt2 = opt2_line[:c_marker.start()].strip()
                    opt3_match = re.search(r'C\s*\)\s*(.+)$', opt2_line, re.IGNORECASE)
                    opt3 = opt3_match.group(1).strip() if opt3_match else ""

                    if opt2 and opt3:
                        questions.append({
                            'question': line,
                            'options': [opt1, opt2, opt3]
                        })
                        i = j + 2
                        continue

        # Case 5b-new: Grey Kangaroo Q1-5 reading comprehension (MUST come before 5b)
        # Format: stem line (no ? ending), then "opt1\tB) opt2" then "C) opt3"
        # Example: "Traditionally, child narrators are regarded as"
        #          "voices that employ a gloomy tone.\tB) innocent and genuine."
        #          "C) devoid of the depth needed to explore serious themes."
        if len(line) > 15 and len(line) < 80 and not line.endswith('?') and not line.endswith('.') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            # Check if next line has format "opt1\tB) opt2" (option A + B in one line)
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                break

            if j < len(paragraphs):
                first_opt_line = paragraphs[j].strip()
                # Check for "opt1\tB) opt2" or "opt1  B) opt2" pattern
                b_match = re.search(r'(.+?)(?:\t|\s{2,})B\s*\)\s*(.+)$', first_opt_line)
                if b_match:
                    opt_a = b_match.group(1).strip()
                    opt_b = b_match.group(2).strip()

                    # Look for C) option on next line
                    j += 1
                    while j < len(paragraphs):
                        np = paragraphs[j].strip()
                        if not np:
                            j += 1
                            continue
                        break

                    opt_c = None
                    if j < len(paragraphs):
                        c_line = paragraphs[j].strip()
                        c_match = re.match(r'^C\s*\)\s*(.+)$', c_line, re.IGNORECASE)
                        if c_match:
                            opt_c = c_match.group(1).strip()
                            j += 1

                    if opt_a and opt_b and opt_c:
                        questions.append({
                            'question': line,
                            'options': [opt_a, opt_b, opt_c]
                        })
                        i = j
                        continue

        # Case 5b: Reading comprehension with 3 plain paragraph options (Q12-15 format)
        # Stem is short (< 70 chars), followed by 3 options without markers
        # Each option starts with lowercase (continuation of stem sentence)
        # Skip numbered cloze format "22. A) ..."
        if len(line) > 8 and len(line) < 70 and not line.endswith('?') and not line.endswith('.') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            # Collect next 3 non-empty paragraphs
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < 3:
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Stop if we hit instruction or question-like line
                if is_instruction_line(np) or np.endswith('?') or is_option_line_envie(np):
                    break
                # Options should start with lowercase (continuation) or be short sentences
                if len(np) < 100:
                    opts.append(np)
                    j += 1
                else:
                    break

            if len(opts) == 3:
                questions.append({
                    'question': line,
                    'options': opts
                })
                i = j
                continue

        # Case 5c: Matching question - "Match each prefix..." or "Match the questions..." followed by table rows then options
        if 'match' in line.lower() and ('column' in line.lower() or 'prefix' in line.lower() or 'questions' in line.lower() or 'left' in line.lower()):
            # Skip table rows until we find options A) 1.../2.../
            j = i + 1
            table_rows = []
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Check for option line "A) 1e/2b/..."
                if re.match(r'^A\s*\)\s*\d', np):
                    break
                # Collect table rows
                if '\t' in np or re.match(r'^\w+\s+[a-e]\.', np):
                    table_rows.append(np)
                j += 1

            # Extract options from option lines
            opts = []
            while j < len(paragraphs) and len(opts) < 5:
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Extract A-E options
                if re.match(r'^[A-E]\s*\)', np):
                    markers = list(re.finditer(r'([A-E])\s*\)', np, re.IGNORECASE))
                    for idx, m in enumerate(markers):
                        start = m.end()
                        if idx + 1 < len(markers):
                            end = markers[idx + 1].start()
                        else:
                            end = len(np)
                        opt_text = np[start:end].strip()
                        if opt_text:
                            opts.append(opt_text)
                    j += 1
                    continue
                break

            if len(opts) >= 3:
                questions.append({
                    'question': line,
                    'options': opts[:5]
                })
                i = j
                continue

        # Case 5d: Question with options in image (no text options)
        # Format: "Pick the sentence with the correct punctuation." followed by another question
        # These questions have their options shown as images
        if line.endswith('.') and len(line) > 20 and len(line) < 80:
            # Check if this looks like a question (not a regular statement)
            lower_line = line.lower()
            is_question_like = any(w in lower_line for w in ['pick', 'choose', 'select', 'which', 'what'])
            # Check if next line is NOT an option line (options are in images)
            if is_question_like and next_line and not is_option_line_envie(next_line):
                # Check that next line is another question (not table data)
                next_is_question = len(next_line) > 20 and not re.match(r'^\w+\s+[a-e]\.', next_line)
                if next_is_question:
                    questions.append({
                        'question': line,
                        'options': ['[Option A in image]', '[Option B in image]', '[Option C in image]', '[Option D in image]', '[Option E in image]']
                    })
                    i += 1
                    continue

        # Case 6: Question + next line options with 5 choices (Wallaby Q31-50)
        # Format: "The Forbidden City is located in …" on one line
        # Then: "Beijing, China.\tB) Athens, Greece.   C) Rome..." on next line
        # Also handles multi-line options (D) E) on subsequent line)
        # Skip numbered cloze format "22. A) ..." which is handled by Case 7
        if len(line) > 15 and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for continuation line (D) E) options)
                while j < len(paragraphs) and len(opts) < 5:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check for D) E) continuation
                    if re.match(r'^[D-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if len(opts) >= 3:
                    questions.append({
                        'question': line,
                        'options': opts[:5]
                    })
                    i = j
                    continue

        # Case 7: Cloze question in paragraph format
        # Format: "22.\tA) neither\tB) both\tC) either\tD) whether"
        cloze_match = re.match(r'^(\d+)\.\s*A\s*\)', line)
        if cloze_match:
            q_num = cloze_match.group(1)
            opts = extract_options_envie(line[cloze_match.start():])
            # If A) not extracted properly, re-extract starting from A)
            a_pos = re.search(r'A\s*\)', line)
            if a_pos:
                opts_text = line[a_pos.start():]
                markers = list(re.finditer(r'([A-E])\s*\)', opts_text, re.IGNORECASE))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(opts_text)
                    opt_text = opts_text[start:end].strip().rstrip('\t ')
                    if opt_text:
                        opts.append(opt_text)
            if len(opts) >= 3:
                questions.append({
                    'question': f'Cloze question {q_num}',
                    'options': opts[:4]
                })
                i += 1
                continue

        # Case 8: Cloze options without number (Q25-30 format)
        # Format: "A) yet\tB) whereas\tC) also\tD) while" - only options, no question number
        # These appear after numbered cloze questions in the same section
        # Each line is OPTIONS for ONE cloze question (not question + options from next line)
        if re.match(r'^A\s*\)', line) and '\t' in line:
            opts_text = line
            markers = list(re.finditer(r'([A-E])\s*\)', opts_text, re.IGNORECASE))
            opts = []
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(opts_text)
                opt_text = opts_text[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
            if len(opts) >= 3:
                # This line IS the options for a cloze question (no question text, just options)
                # Question text is in the passage as a numbered blank like (26)
                questions.append({
                    'question': 'Cloze question (unnumbered)',
                    'options': opts[:4]
                })
                i += 1
                continue

        i += 1

    # Parse cloze questions from tables
    # Format: Table rows with "16.\tA) option" in first cell, B)/C)/D) in other cells
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if not cells or not cells[0]:
                continue

            # Check if first cell has numbered question format "16.\tA) option"
            first_cell = cells[0]
            match = re.match(r'^(\d+)\.\s*A\s*\)\s*(.+)$', first_cell.replace('\t', ' '))
            if match:
                q_num = match.group(1)
                opt_a = match.group(2).strip()
                opts = [opt_a]

                # Extract B, C, D options from other cells
                for cell_text in cells[1:]:
                    opt_match = re.match(r'^([B-E])\s*\)\s*(.+)$', cell_text.strip())
                    if opt_match:
                        opts.append(opt_match.group(2).strip())

                if len(opts) >= 3:
                    questions.append({
                        'question': f'Cloze question {q_num}',
                        'options': opts[:4]
                    })
                continue

            # Check for table with mixed question/options (Wallaby format)
            # Cell might have "Question?\nOption A." format
            for cell_text in cells:
                # Look for question ending with ? followed by option on new line
                if '?' in cell_text and '\n' in cell_text:
                    lines = cell_text.split('\n')
                    for li, line in enumerate(lines):
                        if line.strip().endswith('?'):
                            q_text = line.strip()
                            # Next line is option A
                            opts = []
                            if li + 1 < len(lines):
                                opt_a = lines[li + 1].strip().rstrip('.')
                                if opt_a:
                                    opts.append(opt_a)
                            # Get B, C options from other cells in same row
                            for other_cell in cells[1:]:
                                opt_match = re.match(r'^([B-E])\s*\)\s*(.+)$', other_cell.strip())
                                if opt_match:
                                    opts.append(opt_match.group(2).strip().rstrip('.'))

                            if q_text and len(opts) >= 3:
                                questions.append({
                                    'question': q_text,
                                    'options': opts[:5]
                                })
                            break

            # Check if row has only options (D, E) - belongs to previous question
            # Skip for now as it's handled above

            # Joey format: Row with "A) option" in cells + embedded question number
            # e.g., "A) Table tennis.\n34. Pick out the odd one." in first cell
            # Options B-E in other cells
            if cells and re.match(r'^A\s*\)', cells[0]):
                first_cell = cells[0]
                # Check if first cell has embedded question
                lines = first_cell.split('\n')
                opt_a_text = None
                embedded_q = None
                for ln in lines:
                    ln = ln.strip()
                    if re.match(r'^A\s*\)', ln):
                        # Extract option A text
                        opt_a_match = re.match(r'^A\s*\)\s*(.+?)\.?$', ln)
                        if opt_a_match:
                            opt_a_text = opt_a_match.group(1).strip()
                    elif re.match(r'^\d+\.', ln):
                        # This is embedded question for next row
                        embedded_q = re.sub(r'^\d+\.\s*', '', ln).strip()

                if opt_a_text:
                    opts = [opt_a_text]
                    # Get B-E options from other cells
                    for cell_text in cells[1:]:
                        opt_match = re.match(r'^([B-E])\s*\)\s*(.+?)\.?$', cell_text.strip())
                        if opt_match:
                            opts.append(opt_match.group(2).strip())

                    if len(opts) >= 3:
                        # This is options for a question (need to find question from context)
                        # Store for later matching or use placeholder
                        questions.append({
                            'question': 'Table question (options in table row)',
                            'options': opts[:5]
                        })

    # Parse cloze questions from textboxes and paragraphs
    # Textboxes may contain: "21. A) sparked B) spotted C) split D) squandered"
    from lxml import etree
    try:
        body = doc.element.body
        xml_str = etree.tostring(body, encoding='unicode')
        textbox_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml_str, re.DOTALL)

        seen_cloze_nums = set()
        cloze_opts_from_textbox = {}  # Store options by question number

        for match in textbox_matches:
            texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', match)
            content = ' '.join(texts).strip()

            # Look for cloze option format: "21. A) option B) option C) option D) option"
            cloze_match = re.match(r'^(\d+)\.\s*A\s*\)\s*', content)
            if cloze_match:
                q_num = int(cloze_match.group(1))
                if q_num in seen_cloze_nums:
                    continue
                seen_cloze_nums.add(q_num)

                # Extract options
                markers = list(re.finditer(r'([A-D])\s*\)\s*', content))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(content)
                    opt_text = content[start:end].strip()
                    if opt_text:
                        opts.append(opt_text)

                if len(opts) >= 3:
                    cloze_opts_from_textbox[q_num] = opts[:4]

        # Also check paragraphs for numbered cloze options (e.g., "22. A) neither B) both...")
        for para in doc.paragraphs:
            text = para.text.strip()
            opt_match = re.match(r'^(\d+)\.\s*A\s*\)', text)
            if opt_match:
                q_num = int(opt_match.group(1))
                if q_num not in cloze_opts_from_textbox:
                    # Extract options
                    markers = list(re.finditer(r'([A-D])\s*\)\s*', text))
                    opts = []
                    for idx, m in enumerate(markers):
                        start = m.end()
                        if idx + 1 < len(markers):
                            end = markers[idx + 1].start()
                        else:
                            end = len(text)
                        opt_text = text[start:end].strip()
                        if opt_text:
                            opts.append(opt_text)
                    if len(opts) >= 3:
                        cloze_opts_from_textbox[q_num] = opts[:4]

        # Collect unnumbered cloze option lines from paragraphs
        # These are lines like "A) yet B) whereas C) also D) while" in cloze section
        unnumbered_cloze_opts = []
        in_cloze_section = False
        for para in doc.paragraphs:
            text = para.text.strip()
            # Detect start of cloze section
            if 'space (21-30)' in text.lower() or '(21-30)' in text:
                in_cloze_section = True
                continue
            # Detect end of cloze section
            if in_cloze_section and ('questions 31-' in text.lower() or 'For questions 31' in text):
                in_cloze_section = False
                continue
            # Collect unnumbered option lines in cloze section
            if in_cloze_section and re.match(r'^A\s*\)', text):
                # This is an unnumbered cloze option line
                markers = list(re.finditer(r'([A-D])\s*\)\s*', text))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(text)
                    opt_text = text[start:end].strip()
                    if opt_text:
                        opts.append(opt_text)
                if len(opts) >= 3:
                    unnumbered_cloze_opts.append(opts[:4])

        # Find cloze numbers in passages (from textboxes)
        cloze_in_passage = set()
        for match in textbox_matches:
            texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', match)
            content = ''.join(texts).strip()
            # Passage blanks: "(26)" or "(27)"
            blanks = re.findall(r'\((\d+)\)', content)
            for b in blanks:
                cloze_in_passage.add(int(b))

        # Determine missing cloze numbers (numbers in passage but no options)
        cloze_with_options = set(cloze_opts_from_textbox.keys())
        missing_cloze = sorted(cloze_in_passage - cloze_with_options)

        # Assign unnumbered options to missing cloze numbers
        if missing_cloze and unnumbered_cloze_opts:
            for i, cloze_num in enumerate(missing_cloze):
                if i < len(unnumbered_cloze_opts):
                    cloze_opts_from_textbox[cloze_num] = unnumbered_cloze_opts[i]

        # Also handle cloze 30 which may be the last unnumbered option
        if 30 not in cloze_opts_from_textbox and unnumbered_cloze_opts:
            # Use the last available unnumbered option for cloze 30
            remaining_unnumbered = len(unnumbered_cloze_opts) - len(missing_cloze)
            if remaining_unnumbered > 0:
                cloze_opts_from_textbox[30] = unnumbered_cloze_opts[-1]

        # Add all cloze questions
        for q_num in sorted(cloze_opts_from_textbox.keys()):
            opts = cloze_opts_from_textbox[q_num]
            questions.append({
                'question': f'Cloze question {q_num}',
                'options': opts
            })

        # Add placeholder for any remaining missing cloze numbers (image-based)
        all_cloze_nums = set(cloze_opts_from_textbox.keys())
        still_missing = cloze_in_passage - all_cloze_nums
        if all_cloze_nums:
            max_cloze = max(all_cloze_nums)
            for cloze_num in sorted(still_missing):
                if 20 <= cloze_num <= max_cloze:
                    questions.append({
                        'question': f'Cloze question {cloze_num} (image-based)',
                        'options': ['[Option A in image]', '[Option B in image]', '[Option C in image]', '[Option D in image]']
                    })

    except Exception:
        pass  # lxml may not be available

    # Remove incorrectly parsed "A) ..." lines as questions
    questions = [q for q in questions if not q.get('question', '').startswith('A)')]

    # Sort cloze questions by their question number and insert at correct position
    def get_cloze_num(q):
        text = q.get('question', '')
        m = re.search(r'Cloze question (\d+)', text)
        if m:
            return int(m.group(1))
        return 999

    # Separate cloze questions from others
    cloze_questions = [q for q in questions if 'Cloze question' in q.get('question', '')]
    other_questions = [q for q in questions if 'Cloze question' not in q.get('question', '')]

    # Sort cloze questions by their number
    cloze_questions.sort(key=get_cloze_num)

    # Insert cloze questions at correct position based on cloze number
    # E.g., "Cloze question 21" should be inserted at position 20 (0-indexed)
    if cloze_questions:
        # Get the first cloze question number to determine insertion position
        first_cloze_num = get_cloze_num(cloze_questions[0])
        # Insert at position = first_cloze_num - 1 (0-indexed)
        # E.g., Cloze Q11 -> insert at index 10, Cloze Q21 -> insert at index 20
        insert_idx = first_cloze_num - 1
        # Make sure we don't go beyond the available questions
        insert_idx = min(insert_idx, len(other_questions))
        questions = other_questions[:insert_idx] + cloze_questions + other_questions[insert_idx:]
    else:
        questions = other_questions

    return questions


# ============================================================================
# CONVERT WORD TO EXCEL
# ============================================================================

@app.post("/convert-word-to-excel")
def convert_word_to_excel(
    file: UploadFile = Form(...),
    use_latex: str = Form("0")
) -> StreamingResponse:
    """Convert a Word file containing questions to Excel format.

    Args:
        file: The Word document to convert
        use_latex: "1" to convert math formulas to LaTeX, "0" for plain text
    """
    if not file.filename or not file.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    # Parse use_latex parameter
    latex_enabled = use_latex == "1"

    # Read the Word document
    try:
        content = file.file.read()
        doc = Document(io.BytesIO(content))

        # Try cell-based parser first (for ASMO Science format: 1 cell = 1 question)
        questions = _parse_cell_based_questions(doc)

        # Default format flags
        is_math_format = False
        is_english_level_format = False
        is_envie_format = False

        if not questions:
            # Use unified extraction for lines (handles paragraphs, textboxes, tables)
            lines, table_options = _extract_docx_lines(doc, include_textboxes=True, use_latex=latex_enabled)

            # Check if this is a Math exam format (Question 1., Question 2., etc.)
            is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])

            # Check if this is an English Level exam format
            # These have "Section A:", "Section B:" headers and questions in tables with nested option grids
            is_english_level_format = (
                any('Section A' in line or 'Section B' in line for line in lines[:15]) and
                file.filename and 'LEVEL' in file.filename.upper()
            )

            # Check if this is an EN-VIE bilingual format
            # These have "EN-VIE" in filename and specific patterns
            is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

            if is_math_format:
                questions = _parse_math_exam_questions(lines)
            elif is_english_level_format:
                questions = _parse_english_exam_questions(doc)
            elif is_envie_format:
                questions = _parse_envie_questions(doc)
            else:
                questions = _parse_bilingual_questions(lines, table_options)

        # Create Excel file
        try:
            import openpyxl
            from openpyxl import Workbook
        except ImportError:
            raise HTTPException(status_code=500, detail="openpyxl not installed")

        wb = Workbook()
        ws = wb.active
        ws.title = "Questions"

        # Headers matching the required format
        headers = [
            "Question Type",
            "Question",
            "Option 1",
            "Option 2",
            "Option 3",
            "Option 4",
            "Option 5",
            "Correct Answer",
            "Default Marks",
            "Default Time To Solve",
            "Difficulty Level",
            "Hint",
            "Solution"
        ]
        for col, header in enumerate(headers, 1):
            ws.cell(row=1, column=col, value=header)

        # Data rows
        for idx, q in enumerate(questions, 1):
            row = idx + 1
            options = q.get("options", [])

            # Question Type logic:
            # - If has options (A/B/C/D) → MSA (multiple choice)
            # - Math format without options → SAQ (short answer/fill-in)
            # - Non-math format without options but numbered → likely image-based MCQ → MSA
            # - Otherwise → SAQ
            has_options = len(options) >= 2
            is_numbered_question = q.get("number") is not None

            if has_options:
                # Has actual options → multiple choice
                question_type = "MSA"
            elif is_math_format:
                # Math format without options → short answer
                question_type = "SAQ"
            elif is_numbered_question:
                # Non-math format with number but no options = likely image-based MCQ
                question_type = "MSA"
            else:
                question_type = "SAQ"

            ws.cell(row=row, column=1, value=question_type)

            # Question content
            ws.cell(row=row, column=2, value=q.get("question", ""))

            # Options 1-5
            # Only add A/B/C/D placeholders for non-math formats with image-based questions
            if not options and is_numbered_question and not is_math_format:
                options = ["A", "B", "C", "D"]
            for opt_idx in range(5):
                if opt_idx < len(options):
                    ws.cell(row=row, column=3 + opt_idx, value=options[opt_idx])
                else:
                    ws.cell(row=row, column=3 + opt_idx, value="")

            # Correct Answer
            ws.cell(row=row, column=8, value=q.get("answer", ""))

            # Default Marks - empty by default
            ws.cell(row=row, column=9, value="")

            # Default Time To Solve - 30 seconds
            ws.cell(row=row, column=10, value=30)

            # Difficulty Level - EASY by default
            ws.cell(row=row, column=11, value="EASY")

            # Hint - empty
            ws.cell(row=row, column=12, value="")

            # Solution - empty
            ws.cell(row=row, column=13, value="")

        # Save to buffer
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={Path(file.filename).stem}.xlsx"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")


# ============================================================================
# ADDITIONAL WORD PARSING UTILITIES
# ============================================================================

def is_matching_section(text: str) -> bool:
    """Check if we're in a matching section (column A matches column B)."""
    lower = text.lower()
    return 'match' in lower or 'matching' in lower or 'column a' in lower


def is_matching_table_line(text: str) -> bool:
    """Check if line is part of a matching table (numbered items with ellipsis)."""
    # Pattern: "1. something …" or "a. something ..."
    if re.match(r'^[1-9a-z][.\)]\s*.+[…\.]{2,}', text, re.IGNORECASE):
        return True
    return False


def is_dialogue_completion(text: str) -> bool:
    """Check if text is a dialogue completion question (Complete the dialogue...)."""
    lower = text.lower()
    return 'complete the dialogue' in lower or 'suitable response' in lower


def is_blank_only_line(text: str) -> bool:
    """Check if line is only underscores/blanks (dialogue placeholder)."""
    stripped = text.strip().replace('_', '').replace('.', '').replace(' ', '')
    return len(stripped) == 0 and len(text.strip()) >= 3


def is_dialogue_blank_line(text: str) -> bool:
    """Check if line is a dialogue blank (speaker: ___ format)."""
    # Pattern: "Speaker:" or "A:" or "Name:" followed by blank
    if re.match(r'^[A-Z][a-zA-Z]*\s*:\s*[_\.…]+\s*$', text):
        return True
    return False


def is_dialogue_prompt_line(text: str) -> bool:
    """Check if line is dialogue speaker line."""
    # Pattern: "A:" or "Speaker:" followed by text
    if re.match(r'^[A-Z][a-zA-Z]*\s*:\s*.+', text):
        return True
    return False


def is_reading_passage_start(text: str) -> bool:
    """Check if text starts a reading passage section."""
    lower = text.lower()
    patterns = [
        'read the following',
        'read the passage',
        'reading comprehension',
        'read the text',
    ]
    return any(p in lower for p in patterns)


def is_question_with_single_word_options(text: str) -> bool:
    """Check if question has single-word options (vocabulary questions)."""
    # Pattern: options like "A. word  B. word  C. word  D. word"
    options = re.findall(r'[A-D]\s*[.\)]\s*(\S+)', text, re.IGNORECASE)
    if len(options) >= 3:
        # Check if all options are single words
        return all(len(opt.split()) == 1 for opt in options)
    return False


def extract_passage_questions(lines: List[str], start_idx: int) -> Tuple[List[dict], int]:
    """
    Extract questions from a reading passage section.
    Returns list of questions and the end index.
    """
    questions = []
    passage_lines = []
    i = start_idx

    # Collect passage text until we hit questions
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if line starts a numbered question
        if re.match(r'^\d+\s*[.\)]', line):
            break

        passage_lines.append(line)
        i += 1

    passage_text = "\n".join(passage_lines)

    # Now collect questions
    question_pattern = re.compile(r'^\s*(\d+)\s*[.\)]\s*(.*)$')

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        q_match = question_pattern.match(line)
        if q_match:
            q_num = q_match.group(1)
            q_content = q_match.group(2)

            # Collect options
            options = []
            j = i + 1
            while j < len(lines):
                opt_line = lines[j].strip()
                if not opt_line:
                    j += 1
                    continue
                if re.match(r'^[A-D]\s*[.\)]', opt_line, re.IGNORECASE):
                    opt_text = re.sub(r'^[A-D]\s*[.\)]\s*', '', opt_line, flags=re.IGNORECASE)
                    options.append(opt_text)
                    j += 1
                    if len(options) >= 4:
                        break
                else:
                    break

            if options:
                questions.append({
                    "question": f"(Passage) {q_num}. {q_content}",
                    "options": options,
                    "passage": passage_text[:200] + "..." if len(passage_text) > 200 else passage_text,
                })
                i = j
            else:
                i += 1
        else:
            # Not a question, might be end of section
            break

    return questions, i


def is_passage_with_blanks(text: str) -> bool:
    """Check if text is a passage paragraph with numbered blanks like (16)."""
    # Long text (> 100 chars) with embedded numbers like (16), (21)
    if len(text) > 100 and re.search(r'\(\d+\)', text):
        return True
    return False


def extract_cloze_questions(start_idx: int, lines_list: List[str]) -> Tuple[List[dict], int]:
    """Extract cloze passage questions (passage with numbered blanks + batched options)."""
    result = []
    i = start_idx
    line = lines_list[i].strip()

    # Check if this line has numbered blanks like __________(31)
    blank_nums = re.findall(r'_+\s*\((\d+)\)', line)
    if not blank_nums:
        return [], start_idx

    # Collect all passage lines with blanks
    passage_lines = [line]
    all_blank_nums = list(blank_nums)
    j = i + 1

    while j < len(lines_list):
        next_line = lines_list[j].strip()
        if not next_line:
            j += 1
            continue
        # Stop if we hit A/B/C/D options
        if 'A)' in next_line and 'B)' in next_line:
            break
        # Check for more blanks
        more_blanks = re.findall(r'_+\s*\((\d+)\)', next_line)
        if more_blanks:
            passage_lines.append(next_line)
            all_blank_nums.extend(more_blanks)
            j += 1
            continue
        # If line is short and part of passage, include it
        if len(next_line) < 150 and not re.match(r'^[A-E]\s*\)', next_line):
            passage_lines.append(next_line)
            j += 1
            continue
        break

    # Now collect options (one A/B/C/D line per blank)
    options_lines = []
    while j < len(lines_list) and len(options_lines) < len(all_blank_nums):
        next_line = lines_list[j].strip()
        if not next_line:
            j += 1
            continue
        if 'A)' in next_line and 'B)' in next_line:
            options_lines.append(next_line)
            j += 1
        else:
            break

    # Helper to extract options from line
    def extract_options_from_line(line: str) -> List[str]:
        opts = []
        marker_pattern = re.compile(r'(?:^|(?<=\s)|(?<=\t))([A-E])\s*[.\)]', re.IGNORECASE)
        markers = list(marker_pattern.finditer(line))
        if markers:
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
        return opts

    # Create one question per blank
    passage_text = " ".join(passage_lines)
    for idx, blank_num in enumerate(all_blank_nums):
        q_text = f"({blank_num}) {passage_text[:100]}..."
        opts = []
        if idx < len(options_lines):
            opts = extract_options_from_line(options_lines[idx])
        result.append({
            "question": q_text,
            "options": opts
        })

    return result, j


# ============================================================================
# ANALYZE EXAM WITH AI
# ============================================================================

@app.post("/api/analyze-exam")
async def analyze_exam_with_ai(
    file: UploadFile,
    ai_engine: str = Form("openai"),
):
    """Analyze an exam file using AI to extract and categorize questions."""
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    # Extract content
    text_content = _extract_docx_content(doc)

    # Use AI to analyze
    settings = _load_ai_settings()

    prompt = f"""Phân tích đề thi sau và trích xuất các câu hỏi. Trả về JSON array với format:
[
    {{
        "number": 1,
        "content": "nội dung câu hỏi",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
        "topic": "chủ đề",
        "difficulty": "easy|medium|hard"
    }}
]

Nội dung đề thi:
{text_content[:8000]}

Chỉ trả về JSON array, không giải thích."""

    try:
        response = await _call_ai_engine(ai_engine, prompt, settings)
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            questions = json.loads(json_match.group())
            return {
                "ok": True,
                "filename": file.filename,
                "questions": questions,
                "count": len(questions),
            }
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Note: /api/parse-exam endpoint is defined earlier in the file


@app.post("/api/generate-similar-exam")
async def generate_similar_exam(
    file: UploadFile,
    difficulty: str = Form("same"),
    subject: str = Form("auto"),
    bilingual: str = Form("auto"),
    ai_engine: str = Form("openai"),
):
    """Generate similar questions based on an exam file, one by one matching the original."""
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .docx")

    content = await file.read()
    doc = Document(io.BytesIO(content))

    # Try cell-based parser first (for ASMO Science format: 1 cell = 1 question)
    sample_questions = _parse_cell_based_questions(doc)

    if not sample_questions:
        # Extract lines and parse questions
        lines, table_options = _extract_docx_lines(doc)

        # Detect format and use appropriate parser
        is_math_format = any(re.match(r'^Question\s+\d+', line, re.IGNORECASE) for line in lines[:20])
        is_english_level_format = (
            any('Section A' in line or 'Section B' in line for line in lines[:15]) and
            file.filename and 'LEVEL' in file.filename.upper()
        )
        is_envie_format = file.filename and 'EN-VIE' in file.filename.upper()

        if is_math_format:
            sample_questions = _parse_math_exam_questions(lines)
        elif is_english_level_format:
            sample_questions = _parse_english_exam_questions(doc)
        elif is_envie_format:
            sample_questions = _parse_envie_questions(doc)
        else:
            sample_questions = _parse_bilingual_questions(lines, table_options)

    if not sample_questions:
        return {"ok": False, "error": "Không tìm thấy câu hỏi trong file"}

    # Build difficulty instruction
    difficulty_text = {
        "same": "tương đương về độ khó",
        "easier": "DỄ HƠN (đơn giản hơn, ít phức tạp hơn)",
        "harder": "KHÓ HƠN (phức tạp hơn, đòi hỏi tư duy cao hơn)"
    }.get(difficulty, "tương đương về độ khó")

    settings = _load_ai_settings()

    # Process in batches of 5 questions for faster response
    BATCH_SIZE = 5
    all_generated = []

    # Subject names mapping
    subject_names = {
        "science": "Science/Khoa học",
        "math": "Math/Toán học",
        "history": "History/Lịch sử",
        "geography": "Geography/Địa lý",
        "english": "English/Tiếng Anh",
        "general": "General",
        "auto": "General"
    }

    # Use user-selected subject or auto-detect
    if subject != "auto":
        detected_subject = subject
    else:
        # Auto-detect subject from sample questions content
        all_sample_text = ""
        for q in sample_questions[:10]:  # Check first 10 questions
            all_sample_text += q.get('question', '') + " " + " ".join(q.get('options', []))
        all_sample_lower = all_sample_text.lower()

        detected_subject = "general"
        if any(kw in all_sample_lower for kw in ['science', 'khoa học', 'biology', 'sinh học', 'chemistry', 'hóa học', 'physics', 'vật lý', 'animal', 'plant', 'cell', 'tế bào', 'organism', 'ecosystem', 'hệ sinh thái', 'energy', 'năng lượng', 'matter', 'vật chất', 'force', 'lực']):
            detected_subject = "science"
        elif any(kw in all_sample_lower for kw in ['math', 'toán', 'calculate', 'tính', 'equation', 'phương trình', 'number', 'số', 'triangle', 'tam giác', 'rectangle', 'hình chữ nhật', 'area', 'diện tích', 'perimeter', 'chu vi', 'fraction', 'phân số', 'multiply', 'nhân', 'divide', 'chia', 'add', 'cộng', 'subtract', 'trừ']):
            detected_subject = "math"
        elif any(kw in all_sample_lower for kw in ['history', 'lịch sử', 'war', 'chiến tranh', 'king', 'vua', 'dynasty', 'triều đại', 'century', 'thế kỷ', 'emperor', 'hoàng đế']):
            detected_subject = "history"
        elif any(kw in all_sample_lower for kw in ['geography', 'địa lý', 'country', 'quốc gia', 'continent', 'châu lục', 'river', 'sông', 'mountain', 'núi', 'capital', 'thủ đô', 'ocean', 'đại dương']):
            detected_subject = "geography"
        elif any(kw in all_sample_lower for kw in ['english', 'tiếng anh', 'vocabulary', 'từ vựng', 'grammar', 'ngữ pháp', 'verb', 'động từ', 'noun', 'danh từ', 'adjective', 'tính từ', 'sentence', 'câu', 'word', 'từ', 'meaning', 'nghĩa']):
            detected_subject = "english"

    for batch_start in range(0, len(sample_questions), BATCH_SIZE):
        batch = sample_questions[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(sample_questions) + BATCH_SIZE - 1) // BATCH_SIZE

        # Format batch questions with FULL content
        sample_text = ""
        for i, q in enumerate(batch, 1):
            q_content = q.get('question', '')  # Full content, no truncation
            q_options = q.get('options', [])
            # Format options with labels
            opts_text = ""
            if q_options:
                labels = ['A', 'B', 'C', 'D', 'E']
                opts_text = "\n".join([f"  {labels[j]}) {opt}" for j, opt in enumerate(q_options[:5])])
            sample_text += f"Q{i}: {q_content}\nOptions:\n{opts_text}\n\n"

        # Determine bilingual mode based on user selection or auto-detect
        sample_content = sample_text.lower()
        if bilingual == "bilingual":
            is_bilingual_mode = True
            language_mode = "bilingual"
        elif bilingual == "english":
            is_bilingual_mode = False
            language_mode = "english"
        elif bilingual == "vietnamese":
            is_bilingual_mode = False
            language_mode = "vietnamese"
        else:  # auto
            # Auto-detect bilingual format
            is_bilingual_mode = (
                'EN-VIE' in (file.filename or '').upper() or
                'tiếng việt' in sample_content or
                'vietnamese' in sample_content or
                'nghĩa là' in sample_content or
                'có nghĩa' in sample_content or
                'dịch sang' in sample_content or
                any(c in sample_content for c in 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
            )
            language_mode = "bilingual" if is_bilingual_mode else "english"

        bilingual_instruction = ""
        if language_mode == "bilingual":
            bilingual_instruction = """
=== CRITICAL: BILINGUAL FORMAT (SONG NGỮ) - MUST FOLLOW ===

IMPORTANT: The "content" field MUST contain BOTH English AND Vietnamese text.

REQUIRED FORMAT for "content" field:
[English question text]
[Vietnamese question text on NEW LINE]

CORRECT EXAMPLE of "content" value:
"What is the process by which plants make food?\\nThực vật sử dụng quá trình nào để tạo ra thức ăn?"

WRONG EXAMPLE (English only - DO NOT DO THIS):
"What is the process by which plants make food?"

WRONG EXAMPLE (Vietnamese only - DO NOT DO THIS):
"Thực vật sử dụng quá trình nào để tạo ra thức ăn?"

For OPTIONS: Include both languages separated by " / " like:
["Photosynthesis / Quang hợp", "Respiration / Hô hấp", "Digestion / Tiêu hóa", "Fermentation / Lên men"]

EVERY "content" field MUST have English text followed by \\n then Vietnamese text.
=============================================================
"""
        elif language_mode == "vietnamese":
            bilingual_instruction = """
=== LANGUAGE: VIETNAMESE ONLY (CHỈ TIẾNG VIỆT) ===
Create questions in VIETNAMESE only. Do not include English translations.
Tạo câu hỏi HOÀN TOÀN bằng tiếng Việt, không có tiếng Anh.
=================================================
"""

        subject_instruction = f"""
=== CRITICAL: SUBJECT/TOPIC REQUIREMENT ===
Detected subject: {subject_names[detected_subject]}
You MUST create questions about the SAME SUBJECT as the samples.
- If samples are about SCIENCE → create SCIENCE questions (plants, animals, energy, matter, cells, etc.)
- If samples are about MATH → create MATH questions (numbers, calculations, geometry, etc.)
- If samples are about HISTORY → create HISTORY questions
- DO NOT mix subjects. DO NOT create math questions for a science exam.
===========================================
"""

        prompt = f"""Create {len(batch)} NEW similar multiple-choice questions based on the samples below.
Difficulty level: {difficulty_text}
{subject_instruction}
{bilingual_instruction}
CRITICAL RULES:
1. SAME SUBJECT: Questions MUST be about {subject_names[detected_subject]} - the same topic as samples
2. SAME FORMAT: Copy the exact structure and style from samples (including bilingual format if present)
3. Each question MUST have exactly 4 options in the "options" array
4. Questions must be COMPLETE and SELF-CONTAINED with all necessary data

SAMPLE QUESTIONS (STUDY THESE CAREFULLY - match their subject and format):
{sample_text}

IMPORTANT JSON FORMAT:
- Return exactly {len(batch)} question objects in a JSON array
- Each object has: "content" (question text), "options" (array of 4 strings), "correct_answer" (A/B/C/D)
- The "options" array must contain 4 answer choices as separate strings
- DO NOT put each option as a separate question object
{"- FOR BILINGUAL: content MUST have both English and Vietnamese with newline between them" if language_mode == "bilingual" else ""}

{"CORRECT BILINGUAL EXAMPLE:" if language_mode == "bilingual" else "CORRECT EXAMPLE:"}
[{{"content":"{"What process do plants use to make food?\\nThực vật sử dụng quá trình nào để tạo ra thức ăn?" if language_mode == "bilingual" else "What shape has 5 faces?"}","options":["{"Photosynthesis / Quang hợp" if language_mode == "bilingual" else "Cube"}","{"Respiration / Hô hấp" if language_mode == "bilingual" else "Rectangle"}","{"Digestion / Tiêu hóa" if language_mode == "bilingual" else "Square"}","{"Fermentation / Lên men" if language_mode == "bilingual" else "Rhombus"}"],"correct_answer":"A"}}]

WRONG EXAMPLE (DO NOT DO THIS - options as separate questions):
[{{"content":"What shape has 5 faces?"}},{{"content":"A) Cube"}},{{"content":"B) Rectangle"}}]
{"WRONG EXAMPLE (DO NOT DO THIS - English only without Vietnamese):" if language_mode == "bilingual" else ""}
{"[{{\"content\":\"What process do plants use to make food?\",\"options\":[\"Photosynthesis\",\"Respiration\",\"Digestion\",\"Fermentation\"],\"correct_answer\":\"A\"}}]" if language_mode == "bilingual" else ""}

Return ONLY valid JSON array, no markdown, no explanation:"""

        try:
            response = await _call_ai_engine(ai_engine, prompt, settings)
            # Try to extract JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group()
                # Clean up common JSON issues from LLM responses
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing comma in objects
                try:
                    batch_questions = json.loads(json_str)

                    # Fix malformed responses where options are separate objects
                    # Detect: objects without "options" array but content starts with A), B), etc.
                    fixed_questions = []
                    current_question = None
                    current_options = []

                    for item in batch_questions:
                        content = item.get("content", "")
                        options = item.get("options", [])

                        # Check if this looks like an option line (A), B), C), D))
                        option_match = re.match(r'^([A-D])\s*[.)]\s*(.+)$', content.strip(), re.IGNORECASE)

                        if options and len(options) >= 2:
                            # This is a proper question with options
                            if current_question:
                                # Save previous question first
                                if current_options:
                                    current_question["options"] = current_options
                                fixed_questions.append(current_question)
                            fixed_questions.append(item)
                            current_question = None
                            current_options = []
                        elif option_match:
                            # This is an option line (A), B), etc.)
                            opt_text = option_match.group(2).strip()
                            current_options.append(opt_text)
                        elif content and not option_match:
                            # This is a question without options
                            if current_question:
                                # Save previous question
                                if current_options:
                                    current_question["options"] = current_options
                                if current_question.get("options"):
                                    fixed_questions.append(current_question)
                            current_question = item
                            current_options = []

                    # Don't forget the last question
                    if current_question:
                        if current_options:
                            current_question["options"] = current_options
                        if current_question.get("options"):
                            fixed_questions.append(current_question)

                    # Use fixed questions if we found malformed data
                    if fixed_questions:
                        all_generated.extend(fixed_questions)
                    else:
                        all_generated.extend(batch_questions)

                except json.JSONDecodeError:
                    # Try to fix truncated JSON by finding complete objects
                    # Find all complete JSON objects
                    objects = re.findall(r'\{[^{}]*"content"[^{}]*"options"[^{}]*\}', json_str)
                    for obj_str in objects:
                        try:
                            obj = json.loads(obj_str)
                            all_generated.append(obj)
                        except:
                            pass
                    if not objects:
                        raise ValueError("Invalid JSON response")
            else:
                # No JSON array found, try to parse line by line
                raise ValueError("No JSON array in response")
        except Exception as e:
            error_msg = str(e)[:100]
            # If batch fails, add placeholder
            for _ in batch:
                all_generated.append({
                    "content": f"Lỗi tạo câu hỏi: {error_msg}",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A"
                })

    if all_generated:
        return {
            "ok": True,
            "original_file": file.filename,
            "original_count": len(sample_questions),
            "generated_questions": all_generated,
            "count": len(all_generated),
            "difficulty": difficulty,
        }
    return {"ok": False, "error": "Không thể tạo câu hỏi. Vui lòng thử lại."}


# ============================================================================
# OMR (Optical Mark Recognition) - Chấm bài trắc nghiệm
# ============================================================================

# Cấu hình mẫu phiếu trả lời - International Kangaroo Contest
ANSWER_TEMPLATES = {
    # IKSC - Khoa học (Science Contest)
    # Lớp 1-2, 3-4: 24 câu (8 câu x 3đ + 8 câu x 4đ + 8 câu x 5đ = 96 điểm tối đa)
    # Lớp 5-12: 30 câu (10 câu x 3đ + 10 câu x 4đ + 10 câu x 5đ = 120 điểm tối đa)
    # Mỗi câu sai trừ 1 điểm
    "IKSC_PRE_ECOLIER": {
        "name": "Khoa học - Pre-Ecolier (Lớp 1-2)",
        "questions": 24,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {
            "type": "tiered",  # Điểm theo phần
            "tiers": [
                {"start": 1, "end": 8, "correct": 3, "wrong": -1},    # Câu 1-8: 3 điểm/câu
                {"start": 9, "end": 16, "correct": 4, "wrong": -1},   # Câu 9-16: 4 điểm/câu
                {"start": 17, "end": 24, "correct": 5, "wrong": -1},  # Câu 17-24: 5 điểm/câu
            ],
            "blank": 0,
            "base": 24  # Điểm khởi đầu
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
                {"start": 1, "end": 10, "correct": 3, "wrong": -1},   # Câu 1-10: 3 điểm/câu
                {"start": 11, "end": 20, "correct": 4, "wrong": -1},  # Câu 11-20: 4 điểm/câu
                {"start": 21, "end": 30, "correct": 5, "wrong": -1},  # Câu 21-30: 5 điểm/câu
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
    # Pre-Ecolier: 25 câu, 2đ/câu đúng, không trừ điểm sai, max 50 điểm
    # Ecolier: 30 câu, 2đ/câu đúng, không trừ điểm sai, max 60 điểm
    # Benjamin-Student: 50 câu, chỉ tính 40 câu tốt nhất, -0.5đ/câu sai, +10đ bonus
    "IKLC_PRE_ECOLIER": {
        "name": "Tiếng Anh - Pre-Ecolier (Lớp 1-2)",
        "questions": 25,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 2, "wrong": 0, "blank": 0, "base": 0}  # Max 50
    },
    "IKLC_ECOLIER": {
        "name": "Tiếng Anh - Ecolier (Lớp 3-4)",
        "questions": 30,
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 2, "wrong": 0, "blank": 0, "base": 0}  # Max 60
    },
    "IKLC_BENJAMIN": {
        "name": "Tiếng Anh - Benjamin (Lớp 5-6)",
        "questions": 50,
        "options": 5,
        "questions_per_row": 4,
        "layout": "column",  # Đọc theo cột: cột 1 có câu 1,5,9..., cột 2 có câu 2,6,10...
        "scoring": {
            "type": "best_of",
            "count_best": 40,  # Chỉ tính 40 câu tốt nhất
            "correct": 1.0,
            "wrong": -0.5,
            "blank": 0,
            "base": 10  # Bonus 10 điểm, max = 10 + 40*1.0 = 50
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
            "base": 10  # Max = 10 + 40*1.25 = 60
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
            "base": 10  # Max = 10 + 40*1.5 = 70
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
            "base": 10  # Max = 10 + 40*1.75 = 80
        }
    },
    # ASMO - Math
    # ASMO - Math (25 câu, theo lớp)
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
    # ASMO - Science (25 câu, theo Level: Level 1 = Lớp 1-2, Level 2 = Lớp 3-4, ...)
    "ASMO_SCIENCE_LEVEL_1": {"name": "ASMO Khoa học - Level 1 (Lớp 1-2)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_2": {"name": "ASMO Khoa học - Level 2 (Lớp 3-4)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_3": {"name": "ASMO Khoa học - Level 3 (Lớp 5-6)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_4": {"name": "ASMO Khoa học - Level 4 (Lớp 7-8)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_5": {"name": "ASMO Khoa học - Level 5 (Lớp 9-10)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_SCIENCE_LEVEL_6": {"name": "ASMO Khoa học - Level 6 (Lớp 11-12)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    # ASMO - English (theo Level: Level 1-3 = 50 câu, Level 4-6 = 60 câu)
    "ASMO_ENGLISH_LEVEL_1": {"name": "ASMO Tiếng Anh - Level 1 (Lớp 1-2)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_2": {"name": "ASMO Tiếng Anh - Level 2 (Lớp 3-4)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_3": {"name": "ASMO Tiếng Anh - Level 3 (Lớp 5-6)", "questions": 50, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_4": {"name": "ASMO Tiếng Anh - Level 4 (Lớp 7-8)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_5": {"name": "ASMO Tiếng Anh - Level 5 (Lớp 9-10)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    "ASMO_ENGLISH_LEVEL_6": {"name": "ASMO Tiếng Anh - Level 6 (Lớp 11-12)", "questions": 60, "options": 5, "questions_per_row": 4, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}},
    # SEAMO - Math (25 câu: 20 trắc nghiệm + 5 điền đáp án)
    "SEAMO_MATH_PAPER_A": {"name": "SEAMO Toán - Paper A (Lớp 1-2)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_B": {"name": "SEAMO Toán - Paper B (Lớp 3-4)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_C": {"name": "SEAMO Toán - Paper C (Lớp 5-6)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_D": {"name": "SEAMO Toán - Paper D (Lớp 7-8)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_E": {"name": "SEAMO Toán - Paper E (Lớp 9-10)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    "SEAMO_MATH_PAPER_F": {"name": "SEAMO Toán - Paper F (Lớp 11-12)", "questions": 25, "options": 5, "questions_per_row": 5, "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}, "mixed_format": {"mcq": 20, "fill_in": 5}},
    # Custom - cho phép người dùng tự định nghĩa
    "CUSTOM": {
        "name": "Tùy chỉnh",
        "questions": 30,  # Sẽ được override bởi số đáp án nhập vào
        "options": 5,
        "questions_per_row": 4,
        "scoring": {"correct": 1, "wrong": 0, "blank": 0, "base": 0}
    }
}


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


def _order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    import numpy as np
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left có tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]  # bottom-right có tổng lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image, pts):
    """Thực hiện perspective transform với 4 điểm"""
    import cv2
    import numpy as np

    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def _deskew_image(image):
    """Tự động căn chỉnh ảnh bị nghiêng"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Tìm đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image, 0

    # Tính góc nghiêng trung bình
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Chỉ lấy các đường gần ngang (±15 độ)
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return image, 0

    # Lấy góc trung vị
    median_angle = np.median(angles)

    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, median_angle


def _find_document_contour(image):
    """Tìm contour của tài liệu (phiếu trả lời)"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Làm dày cạnh
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def _preprocess_omr_image(image_bytes: bytes):
    """Tiền xử lý ảnh cho OMR với deskew và perspective correction"""
    import cv2
    import numpy as np

    # Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

    original = img.copy()

    # Bước 1: Tìm và căn chỉnh tài liệu nếu bị méo
    # CHÚ Ý: Chỉ áp dụng perspective transform khi contour bao phủ gần như toàn bộ ảnh
    # để tránh cắt mất nội dung (ví dụ: hàng cuối của phiếu 50 câu)
    doc_contour = _find_document_contour(img)
    if doc_contour is not None:
        contour_area = cv2.contourArea(doc_contour)
        img_area = img.shape[0] * img.shape[1]
        # Chỉ transform nếu contour bao phủ > 80% diện tích ảnh
        if contour_area > 0.8 * img_area:
            img = _four_point_transform(img, doc_contour)

    # Bước 2: Deskew (căn chỉnh góc nghiêng)
    img, skew_angle = _deskew_image(img)

    # Bước 3: Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bước 4: Tăng contrast bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bước 5: Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 6: Adaptive threshold (tốt hơn cho điều kiện ánh sáng khác nhau)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Bước 7: Morphological operations để làm sạch
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return img, gray, binary


def _find_answer_grid_region(gray_image, binary_image):
    """Tìm vùng chứa lưới đáp án trong ảnh dựa trên cấu trúc grid"""
    import cv2
    import numpy as np

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
    import cv2
    import numpy as np

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
    import numpy as np

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
    import cv2
    import numpy as np

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
    from collections import Counter
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
    import cv2
    import numpy as np

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
    import numpy as np

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
    import cv2
    import numpy as np

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
    import cv2
    import numpy as np

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
    import numpy as np

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


def _grade_single_sheet(image_bytes: bytes, answer_key: List[str], template_type: str = "IKSC_BENJAMIN", extract_info: bool = True):
    """Chấm một phiếu trả lời với thuật toán OMR cải tiến"""
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    option_labels = ["A", "B", "C", "D", "E"]

    # Trích xuất thông tin học sinh bằng OCR
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh với deskew và perspective correction
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # Thử phương pháp mới trước: phát hiện dựa trên grid
    rows, all_rects = _detect_bubbles_grid_based(gray, binary, template_type)

    questions = []
    use_new_method = False

    if rows and len(rows) >= 3:
        # Sử dụng phương pháp mới nếu phát hiện đủ hàng
        questions = _group_bubbles_to_questions_improved(rows, template_type)
        use_new_method = True

    # Fallback: Sử dụng phương pháp cũ nếu phương pháp mới không hiệu quả
    if len(questions) < num_questions * 0.3:
        bubbles = _detect_bubbles(binary, template_type)

        if len(bubbles) >= num_questions * 5 * 0.3:
            questions = _group_bubbles_to_questions(bubbles, template_type)
            use_new_method = False

    if len(questions) < num_questions * 0.3:
        return {
            "error": f"Không phát hiện đủ câu hỏi. Tìm thấy: {len(questions)}, cần: {num_questions}. "
                     f"Vui lòng đảm bảo ảnh rõ nét và phiếu được căn chỉnh đúng."
        }

    # Phân tích từng câu hỏi
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    # Tạo dict để tra cứu câu hỏi theo index (thay vì dùng vị trí trong mảng)
    questions_by_index = {q["index"]: q for q in questions}

    for q_idx in range(num_questions):
        q_num = q_idx + 1  # Số thứ tự câu hỏi (1-based)
        if q_num not in questions_by_index:
            # Không tìm thấy câu hỏi này
            student_answers.append(None)
            blank_count += 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        mean_vals = []  # Lưu mean value của từng option
        is_filled_list = []  # Lưu trạng thái is_filled của từng option
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            if use_new_method:
                # Phương pháp mới: bubble là dict
                result = _analyze_bubble_fill_improved(gray, bubble)
                if len(result) == 3:
                    is_filled, fill_ratio, mean_val = result
                else:
                    is_filled, fill_ratio = result
                    mean_val = 128.0
                mean_vals.append(mean_val)
                is_filled_list.append(is_filled)
            else:
                # Phương pháp cũ: bubble là tuple với contour
                is_filled, fill_ratio = _analyze_bubble_fill(binary, bubble[4])
                mean_vals.append(128.0)
                is_filled_list.append(is_filled)

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        # Ưu tiên option có is_filled=True và mean thấp nhất (được tô đậm nhất)
        filled_options = [(i, mean_vals[i]) for i in range(len(is_filled_list)) if is_filled_list[i]]
        if filled_options and selected_option is None:
            # Có option được đánh dấu filled nhưng chưa được chọn
            # Chọn option có mean thấp nhất (tối nhất = được tô)
            darkest_filled = min(filled_options, key=lambda x: x[1])
            selected_option = option_labels[darkest_filled[0]]
        elif len(filled_options) == 1:
            # Chỉ có 1 option filled -> chọn option đó
            selected_option = option_labels[filled_options[0][0]]

        # Phát hiện vùng tối bất thường (bóng/rìa ảnh)
        # Nếu nhiều options có mean rất thấp (<50), đây có thể là vùng tối
        dark_region_count = sum(1 for m in mean_vals if m < 50)
        is_dark_region = dark_region_count >= 3

        if is_dark_region:
            # Vùng tối: chọn option có mean CAO nhất (sáng nhất = không bị bóng che)
            # vì các vùng tối là do bóng, không phải do được tô
            max_mean = max(mean_vals)
            min_mean = min(mean_vals)

            # Chỉ chọn nếu có 1 option sáng hơn hẳn (chênh lệch > 50)
            if max_mean - min_mean > 50:
                bright_option_idx = mean_vals.index(max_mean)
                # Kiểm tra option sáng này có được tô không
                if fill_ratios[bright_option_idx] > 0.1:
                    selected_option = option_labels[bright_option_idx]
                else:
                    # Option sáng nhưng không được tô -> có thể là blank hoặc tô option khác
                    # Trong vùng tối, tìm option có score cao nhất trong các option không quá tối
                    valid_options = [(i, fill_ratios[i]) for i in range(len(mean_vals))
                                    if mean_vals[i] > 100 or fill_ratios[i] > 0.3]
                    if valid_options:
                        best_idx = max(valid_options, key=lambda x: x[1])[0]
                        selected_option = option_labels[best_idx]
        else:
            # Vùng bình thường: sử dụng logic cũ với cải tiến

            # Ngưỡng động: nếu max_fill > 0.25 và vượt trội hơn các option khác
            if selected_option is None and max_fill > 0.25:
                # Kiểm tra xem có một option nào vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)
                if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Option đầu lớn hơn 30% so với option thứ 2
                    selected_option = option_labels[fill_ratios.index(max_fill)]

            # Kiểm tra nếu có nhiều đáp án được chọn
            # Sử dụng ngưỡng động dựa trên max_fill
            if max_fill > 0.5:
                # Nếu có option được tô đậm, các option khác cần đạt ít nhất 60% của max
                filled_threshold = max_fill * 0.6
            else:
                filled_threshold = 0.35

            filled_count = sum(1 for r in fill_ratios if r > filled_threshold)
            if filled_count > 1:
                # Kiểm tra xem có 1 option rõ ràng vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)

                # Tính chênh lệch mean giữa option cao nhất và thấp nhất
                max_score_idx = fill_ratios.index(sorted_ratios[0])
                max_score_mean = mean_vals[max_score_idx]

                # Nếu option có score cao nhất cũng có mean thấp nhất -> đây là bubble được tô
                if max_score_mean == min(mean_vals) or sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Có 1 option vượt trội rõ ràng
                    selected_option = option_labels[max_score_idx]
                else:
                    # Kiểm tra thêm: nếu max > 0.5 và second < 0.4, vẫn chọn max
                    if sorted_ratios[0] > 0.5 and sorted_ratios[1] < 0.4:
                        selected_option = option_labels[fill_ratios.index(sorted_ratios[0])]
                    else:
                        # Phân tích thêm bằng mean value
                        # Option được tô sẽ có mean thấp hơn các option không tô
                        mean_diff = max(mean_vals) - min(mean_vals)
                        if mean_diff > 30:
                            # Có sự khác biệt rõ ràng về độ sáng
                            darkest_idx = mean_vals.index(min(mean_vals))
                            selected_option = option_labels[darkest_idx]
                        else:
                            selected_option = "MULTI"  # Đánh dấu chọn nhiều đáp án

        student_answers.append(selected_option)

        # So sánh với đáp án
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if selected_option is None:
            status = "blank"
            blank_count += 1
        elif selected_option == "MULTI":
            status = "invalid"
            wrong_count += 1
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            status = "correct"
            correct_count += 1
        else:
            status = "wrong"
            wrong_count += 1

        details.append({
            "q": q_idx + 1,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # Tính điểm
    if scoring.get("type") == "tiered":
        # Tính điểm theo phần (tiered scoring) - dùng cho IKSC
        score = scoring.get("base", 0)
        tiers = scoring.get("tiers", [])

        for detail in details:
            q_num = detail["q"]
            status = detail["status"]

            # Tìm tier phù hợp cho câu hỏi này
            tier_points = {"correct": 1, "wrong": 0}  # Default
            for tier in tiers:
                if tier["start"] <= q_num <= tier["end"]:
                    tier_points = tier
                    break

            if status == "correct":
                score += tier_points.get("correct", 1)
            elif status in ["wrong", "invalid"]:
                score += tier_points.get("wrong", 0)
            # blank: không cộng/trừ điểm

    elif scoring.get("type") == "best_of":
        # Tính điểm chỉ lấy N câu tốt nhất - dùng cho IKLC Benjamin-Student
        # Công thức: base + (correct * points) + (wrong * penalty)
        # Chỉ tính count_best câu có điểm cao nhất
        count_best = scoring.get("count_best", 40)
        correct_pts = scoring.get("correct", 1)
        wrong_pts = scoring.get("wrong", -0.5)

        # Tính điểm từng câu
        question_scores = []
        for detail in details:
            status = detail["status"]
            if status == "correct":
                question_scores.append(correct_pts)
            elif status in ["wrong", "invalid"]:
                question_scores.append(wrong_pts)
            else:  # blank
                question_scores.append(0)

        # Sắp xếp giảm dần và lấy N câu tốt nhất
        question_scores.sort(reverse=True)
        best_scores = question_scores[:count_best]

        score = scoring.get("base", 0) + sum(best_scores)

    else:
        # Tính điểm đơn giản (flat scoring) - dùng cho IKLC Pre-Ecolier, Ecolier và các kỳ thi khác
        score = (
            scoring.get("base", 0) +
            correct_count * scoring.get("correct", 1) +
            wrong_count * scoring.get("wrong", 0) +
            blank_count * scoring.get("blank", 0)
        )

    return {
        "answers": student_answers,
        "score": round(score, 2),
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details,
        "student_info": student_info,
        "detection_method": "grid_based" if use_new_method else "contour_based",
        "questions_detected": len(questions)
    }


def _detect_seamo_bubbles_fixed_grid(gray_image):
    """Phát hiện bubbles trong phiếu SEAMO với dynamic grid detection

    Sử dụng kết hợp:
    1. Phát hiện đường kẻ ngang để tìm vị trí các hàng
    2. Phát hiện đường kẻ dọc để tìm vị trí các cột
    3. Fallback về tọa độ cố định nếu không detect được
    """
    import cv2
    import numpy as np

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
    import cv2
    import numpy as np

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


def _grade_mixed_format_sheet(
    image_bytes: bytes,
    answer_key: List[str],
    template_type: str,
    extract_info: bool = True
):
    """Chấm phiếu trả lời có format hỗn hợp (trắc nghiệm + điền đáp án)

    Dùng cho SEAMO Math: 20 câu trắc nghiệm + 5 câu điền đáp án
    """
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    mixed_format = template.get("mixed_format", None)

    if not mixed_format:
        # Không có mixed format, dùng hàm chấm thông thường
        return _grade_single_sheet(image_bytes, answer_key, template_type, extract_info)

    mcq_count = mixed_format.get("mcq", 20)
    fill_in_count = mixed_format.get("fill_in", 5)

    # Trích xuất thông tin học sinh
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # ========== PHẦN 1: Chấm 20 câu trắc nghiệm bằng OMR ==========
    # Kiểm tra nếu là SEAMO, sử dụng fixed grid detection
    is_seamo = "SEAMO" in template_type.upper()

    mcq_questions = []
    if is_seamo:
        # SEAMO có layout cố định, dùng fixed grid
        mcq_questions = _detect_seamo_bubbles_fixed_grid(gray)
    else:
        # Các template khác dùng dynamic detection
        mcq_template_type = template_type
        rows, all_rects = _detect_bubbles_grid_based(gray, binary, mcq_template_type)

        if rows and len(rows) >= 2:
            mcq_questions = _group_bubbles_to_questions_improved(rows, mcq_template_type)
            # Chỉ lấy các câu từ 1 đến mcq_count
            mcq_questions = [q for q in mcq_questions if q["index"] <= mcq_count]

    # Chấm phần trắc nghiệm
    option_labels = ["A", "B", "C", "D", "E"]
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    questions_by_index = {q["index"]: q for q in mcq_questions}

    for q_idx in range(mcq_count):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if q_num not in questions_by_index:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "mcq",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            result_fill = _analyze_bubble_fill_improved(gray, bubble)
            if len(result_fill) == 3:
                is_filled, fill_ratio, mean_val = result_fill
            else:
                is_filled, fill_ratio = result_fill

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        if selected_option is None and max_fill > 0.25:
            sorted_ratios = sorted(fill_ratios, reverse=True)
            if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                selected_option = option_labels[fill_ratios.index(max_fill)]

        if selected_option is None:
            student_answers.append(None)
            blank_count += 1
            status = "blank"
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            student_answers.append(selected_option)
            correct_count += 1
            status = "correct"
        else:
            student_answers.append(selected_option)
            wrong_count += 1
            status = "wrong"

        details.append({
            "q": q_num,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "type": "mcq",
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # ========== PHẦN 2: Chấm 5 câu điền đáp án bằng OCR ==========
    try:
        reader = _get_easyocr_reader()
    except Exception as e:
        # Nếu không load được OCR, đánh dấu các câu fill-in là not_found
        for q_idx in range(mcq_count, num_questions):
            q_num = q_idx + 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0,
                "error": f"OCR error: {str(e)}"
            })

        # Tính điểm và trả về
        score = (
            scoring.get("base", 0) +
            correct_count * scoring.get("correct", 1) +
            wrong_count * scoring.get("wrong", 0) +
            blank_count * scoring.get("blank", 0)
        )

        return {
            "answers": student_answers,
            "score": round(score, 2),
            "correct": correct_count,
            "wrong": wrong_count,
            "blank": blank_count,
            "total": num_questions,
            "details": details,
            "student_info": student_info,
            "format": "mixed",
            "mcq_count": mcq_count,
            "fill_in_count": fill_in_count
        }

    # Tìm vùng chứa đáp án điền (thường ở phía dưới phiếu)
    # Phát hiện các ô điền đáp án
    height, width = gray.shape[:2]

    # Giả sử phần điền đáp án nằm ở 1/3 dưới của ảnh
    fill_in_region = gray[int(height * 0.6):, :]

    # Sử dụng OCR để đọc toàn bộ vùng
    try:
        ocr_results = reader.readtext(fill_in_region, detail=1, paragraph=False)
    except Exception:
        ocr_results = []

    # Tìm các đáp án số/chữ
    recognized_fill_ins = []
    for (bbox, text, confidence) in ocr_results:
        text = text.strip()
        # Lọc các text có vẻ là đáp án (số hoặc chữ ngắn)
        if text and len(text) <= 10:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            recognized_fill_ins.append({
                'text': text,
                'confidence': confidence,
                'cx': cx,
                'cy': cy + int(height * 0.6)  # Offset lại vị trí
            })

    # Sắp xếp theo vị trí (trái sang phải, trên xuống dưới)
    recognized_fill_ins.sort(key=lambda r: (r['cy'], r['cx']))

    # Gán đáp án cho các câu fill-in
    for i, q_idx in enumerate(range(mcq_count, num_questions)):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if i < len(recognized_fill_ins):
            recognized = recognized_fill_ins[i]['text']
            confidence = recognized_fill_ins[i]['confidence']

            # So sánh đáp án (có thể là số hoặc chữ)
            if correct_answer:
                # Chuẩn hóa để so sánh
                student_norm = recognized.upper().strip()
                correct_norm = str(correct_answer).upper().strip()

                if student_norm == correct_norm:
                    status = "correct"
                    correct_count += 1
                else:
                    status = "wrong"
                    wrong_count += 1
            else:
                status = "unknown"

            student_answers.append(recognized)
            details.append({
                "q": q_num,
                "student": recognized,
                "correct": correct_answer,
                "status": status,
                "type": "fill_in",
                "confidence": round(confidence, 3)
            })
        else:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0
            })

    # Tính điểm
    score = (
        scoring.get("base", 0) +
        correct_count * scoring.get("correct", 1) +
        wrong_count * scoring.get("wrong", 0) +
        blank_count * scoring.get("blank", 0)
    )

    return {
        "answers": student_answers,
        "score": round(score, 2),
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details,
        "student_info": student_info,
        "format": "mixed",
        "mcq_count": mcq_count,
        "fill_in_count": fill_in_count
    }


def _extract_answers_from_text(text: str, num_questions: int) -> dict:
    """Trích xuất đáp án từ text (PDF/Word)"""
    # Hỗ trợ các format: "1. A", "1) A", "1: A", "1 A", "Câu 1: A"
    answer_patterns = [
        r'(?:Câu\s*)?(\d+)\s*[.:)]\s*([A-Ea-e])',  # Câu 1: A, 1. A, 1) A
        r'(\d+)\s+([A-Ea-e])\b',  # 1 A
    ]

    found_answers = {}
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            q_num = int(match[0])
            answer = match[1].upper()
            if 1 <= q_num <= num_questions:
                found_answers[q_num] = answer

    return found_answers


@app.get("/api/answer-templates")
async def get_answer_templates():
    """Lấy danh sách mẫu phiếu trả lời"""
    templates = []
    for key, value in ANSWER_TEMPLATES.items():
        templates.append({
            "id": key,
            "name": value["name"],
            "questions": value["questions"],
            "options": value["options"],
            "scoring": value["scoring"]
        })
    return {"ok": True, "templates": templates}


def _parse_answer_key_for_template(answer_file_content: bytes, file_ext: str, template_type: str) -> List[str]:
    """Parse đáp án từ file cho một template cụ thể"""
    from collections import defaultdict

    template = ANSWER_TEMPLATES.get(template_type)
    if not template:
        return []

    num_questions = template["questions"]
    answers = []

    if file_ext in ["xlsx", "xls"]:
        # Đọc từ file Excel
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(answer_file_content))
        ws = wb.active

        for row in ws.iter_rows(min_row=2, max_col=2):
            if row[1].value:
                answers.append(str(row[1].value).strip().upper())

    elif file_ext == "pdf":
        # Đọc từ file PDF
        import fitz  # PyMuPDF

        pdf_doc = fitz.open(stream=answer_file_content, filetype="pdf")
        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text() + "\n"

        found_answers = {}

        # Kiểm tra nếu là file IKLC (Linguistic Kangaroo) với format đặc biệt
        is_iklc_format = "LINGUISTIC KANGAROO" in pdf_text.upper() or all(
            level in pdf_text for level in ["Joey", "Wallaby"]
        )

        if is_iklc_format and "IKLC" in template_type.upper():
            # Parse IKLC PDF với format nhiều cột theo vị trí x
            # Cột: Start (25 câu), Story (30 câu), Joey (50 câu), Wallaby (50 câu), Grey K. (50 câu), Red K. (50 câu)
            iklc_levels = [
                ("Start", 25),      # Pre-Ecolier (Lớp 1-2)
                ("Story", 30),      # Ecolier (Lớp 3-4)
                ("Joey", 50),       # Benjamin (Lớp 5-6)
                ("Wallaby", 50),    # Cadet (Lớp 7-8)
                ("Grey", 50),       # Junior (Lớp 9-10)
                ("Red", 50),        # Student (Lớp 11-12)
            ]

            level_map = {
                "IKLC_PRE_ECOLIER": 0,
                "IKLC_ECOLIER": 1,
                "IKLC_BENJAMIN": 2,
                "IKLC_CADET": 3,
                "IKLC_JUNIOR": 4,
                "IKLC_STUDENT": 5,
            }

            target_level_idx = level_map.get(template_type.upper(), -1)

            if target_level_idx >= 0:
                target_level_name, target_num_q = iklc_levels[target_level_idx]

                # Đọc tất cả text blocks với vị trí từ tất cả các trang
                all_blocks = []
                for page in pdf_doc:
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    x0 = span["bbox"][0]
                                    y0 = span["bbox"][1]
                                    if text:
                                        all_blocks.append({"text": text, "x": x0, "y": y0})

                # Tách số câu và đáp án
                numbers = []  # Số câu hỏi (1-50)
                answers_list = []  # Đáp án A-E

                for b in all_blocks:
                    text = b["text"]
                    if text.isdigit() and 1 <= int(text) <= 50:
                        numbers.append({"num": int(text), "x": b["x"], "y": b["y"]})
                    elif len(text) == 1 and text in "ABCDE":
                        answers_list.append({"ans": text, "x": b["x"], "y": b["y"]})
                    elif len(text) >= 1 and text[0] in "ABCDE" and "," in text:
                        # Trường hợp "B, C" -> lấy ký tự đầu
                        answers_list.append({"ans": text[0], "x": b["x"], "y": b["y"]})

                # Tìm vị trí x của số 1 cho mỗi cột (mỗi level bắt đầu từ câu 1)
                ones = [n for n in numbers if n["num"] == 1]
                ones.sort(key=lambda o: o["x"])

                # Có 6 cột (6 số 1), gán level theo thứ tự x
                # ones[0] = Start, ones[1] = Story, ones[2] = Joey, ...
                if len(ones) >= 6:
                    # Xác định x boundaries giữa các cột
                    x_boundaries = []
                    for i in range(len(ones) - 1):
                        mid_x = (ones[i]["x"] + ones[i + 1]["x"]) / 2
                        x_boundaries.append(mid_x)
                    x_boundaries.append(9999)  # Boundary cuối cùng

                    # Hàm xác định cột của một số dựa trên vị trí x
                    def get_column_idx(x):
                        for i, boundary in enumerate(x_boundaries):
                            if x < boundary:
                                return i
                        return len(x_boundaries) - 1

                    # Nhóm số câu theo cột
                    column_numbers = defaultdict(list)
                    for n in numbers:
                        col_idx = get_column_idx(n["x"])
                        column_numbers[col_idx].append(n)

                    # Lấy số câu của cột target
                    target_numbers = column_numbers.get(target_level_idx, [])

                    # Với mỗi số câu, tìm đáp án gần nhất bên phải
                    for num_block in target_numbers:
                        q_num = num_block["num"]
                        q_x = num_block["x"]
                        q_y = num_block["y"]

                        if q_num > target_num_q:
                            continue

                        # Tìm đáp án gần nhất: cùng y (tolerance 3px trước, sau đó 8px) và x lớn hơn số câu
                        best_answer = None
                        best_dist = 9999
                        best_y_diff = 9999

                        for ans_block in answers_list:
                            ans_x = ans_block["x"]
                            ans_y = ans_block["y"]

                            # Đáp án ở bên phải số câu và cùng hàng
                            y_diff = abs(ans_y - q_y)
                            if ans_x > q_x and y_diff < 8:
                                dist = ans_x - q_x
                                if dist < 50:  # Không quá xa
                                    # Ưu tiên đáp án cùng y hơn (y_diff nhỏ hơn)
                                    # Nếu y_diff gần bằng nhau (< 3px), chọn x gần nhất
                                    if y_diff < best_y_diff - 3 or (abs(y_diff - best_y_diff) <= 3 and dist < best_dist):
                                        best_dist = dist
                                        best_y_diff = y_diff
                                        best_answer = ans_block["ans"]

                        if best_answer and q_num not in found_answers:
                            found_answers[q_num] = best_answer

        pdf_doc.close()

        # Fallback: parse đơn giản
        if not found_answers:
            found_answers = _extract_answers_from_text(pdf_text, num_questions)

        for i in range(1, num_questions + 1):
            answers.append(found_answers.get(i, ""))

    elif file_ext in ["docx", "doc"]:
        # Đọc từ file Word
        doc = Document(io.BytesIO(answer_file_content))
        found_answers = {}

        level_keywords = {
            "pre_ecolier": ["preecolier", "pre-ecolier", "pre ecolier", "pre_ecolier"],
            "ecolier": ["ecolier"],
            "benjamin": ["benjamin"],
            "cadet": ["cadet"],
            "junior": ["junior"],
            "student": ["student"],
        }

        for table in doc.tables:
            if len(table.rows) > 1 and len(table.columns) >= 2:
                header = [cell.text.strip().lower() for cell in table.rows[0].cells]

                level_col = -1
                search_keywords = []
                is_ecolier_only = False

                for key, keywords in level_keywords.items():
                    if key in template_type.lower():
                        search_keywords = keywords
                        if key == "ecolier" and "pre" not in template_type.lower():
                            is_ecolier_only = True
                        break

                for col_idx, col_header in enumerate(header):
                    if is_ecolier_only:
                        if col_header == "ecolier" or (col_header.endswith("ecolier") and not col_header.startswith("pre")):
                            level_col = col_idx
                            break
                    else:
                        for keyword in search_keywords:
                            if keyword in col_header:
                                level_col = col_idx
                                break
                    if level_col >= 0:
                        break

                if level_col >= 0:
                    for row in table.rows[1:]:
                        try:
                            q_num = int(row.cells[0].text.strip())
                            answer = row.cells[level_col].text.strip().upper()
                            if answer and answer in "ABCDE":
                                found_answers[q_num] = answer
                        except (ValueError, IndexError):
                            continue

                if not found_answers and len(table.columns) == 2:
                    for row in table.rows[1:]:
                        try:
                            q_num = int(row.cells[0].text.strip())
                            answer = row.cells[1].text.strip().upper()
                            if answer and answer in "ABCDE":
                                found_answers[q_num] = answer
                        except (ValueError, IndexError):
                            continue

        if not found_answers:
            doc_text = ""
            for para in doc.paragraphs:
                doc_text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        doc_text += cell.text + " "
                    doc_text += "\n"
            found_answers = _extract_answers_from_text(doc_text, num_questions)

        for i in range(1, num_questions + 1):
            answers.append(found_answers.get(i, ""))

    return answers


@app.post("/api/grade-sheets")
async def grade_answer_sheets(
    files: List[UploadFile],
    template_type: str = Form("IKSC_BENJAMIN"),
    answer_key: str = Form(None),
    answer_file: UploadFile = None,
    auto_detect_template: bool = Form(True)  # Tự động nhận diện template từ phiếu
):
    """Chấm nhiều phiếu trả lời

    Nếu auto_detect_template=True (mặc định), hệ thống sẽ tự động nhận diện
    loại đề (IKSC/IKLC) và cấp độ (Benjamin, Cadet, etc.) từ phiếu trả lời
    và lấy đáp án tương ứng từ file đáp án.
    """
    import json

    # Đọc nội dung file đáp án (nếu có) để sử dụng sau
    answer_file_content = None
    answer_file_ext = None
    if answer_file and answer_file.filename:
        answer_file_content = await answer_file.read()
        answer_file_ext = answer_file.filename.lower().split(".")[-1]

    # Cache đáp án đã parse cho mỗi template
    answers_cache = {}

    def get_answers_for_template(tpl_type: str) -> List[str]:
        """Lấy đáp án cho một template, sử dụng cache"""
        if tpl_type in answers_cache:
            return answers_cache[tpl_type]

        template = ANSWER_TEMPLATES.get(tpl_type)
        if not template:
            return []

        num_questions = template["questions"]
        answers = []

        if answer_file_content and answer_file_ext:
            try:
                answers = _parse_answer_key_for_template(answer_file_content, answer_file_ext, tpl_type)
            except Exception:
                pass

        if not answers and answer_key:
            # Parse từ answer_key string
            try:
                answers = json.loads(answer_key)
                if isinstance(answers, str):
                    answers = list(answers.upper())
                answers = [str(a).upper() for a in answers]
            except json.JSONDecodeError:
                if "," in answer_key:
                    answers = [a.strip().upper() for a in answer_key.split(",")]
                else:
                    answers = list(answer_key.upper().replace(" ", ""))

        answers_cache[tpl_type] = answers
        return answers

    # Xử lý chế độ AUTO - tự động bật auto_detect_template
    if template_type == "AUTO":
        auto_detect_template = True
        template_type = "IKSC_BENJAMIN"  # Fallback mặc định

    # Validate template mặc định
    default_template = ANSWER_TEMPLATES.get(template_type)
    if not default_template:
        raise HTTPException(status_code=400, detail=f"Loại mẫu không hợp lệ: {template_type}")

    # Nếu không tự động nhận diện, kiểm tra đáp án trước
    if not auto_detect_template:
        answers = get_answers_for_template(template_type)
        num_questions = default_template["questions"]
        if len(answers) < num_questions:
            raise HTTPException(
                status_code=400,
                detail=f"Thiếu đáp án. Cần {num_questions} đáp án, chỉ có {len(answers)}"
            )

    # Chấm từng phiếu
    results = []
    all_scores = []

    for file in files:
        if not file.filename:
            continue

        # Kiểm tra định dạng file
        ext = file.filename.lower().split(".")[-1]
        supported_image_formats = ["jpg", "jpeg", "png", "bmp", "tiff"]

        if ext == "pdf":
            # Đọc PDF và chấm từng trang như một phiếu riêng
            try:
                import fitz  # PyMuPDF
                content = await file.read()
                pdf_doc = fitz.open(stream=content, filetype="pdf")

                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")

                    page_filename = f"{file.filename}_trang_{page_num + 1}"

                    # Tự động nhận diện template từ phiếu
                    actual_template = template_type
                    if auto_detect_template:
                        student_info = _extract_student_info_ocr(img_bytes)
                        detected = student_info.get("detected_template", "")
                        if detected and detected in ANSWER_TEMPLATES:
                            actual_template = detected

                    # Lấy đáp án cho template
                    answers = get_answers_for_template(actual_template)
                    tpl = ANSWER_TEMPLATES.get(actual_template, default_template)

                    if len(answers) < tpl["questions"]:
                        results.append({
                            "filename": page_filename,
                            "detected_template": actual_template,
                            "error": f"Không tìm thấy đáp án cho {tpl['name']}. Vui lòng kiểm tra file đáp án."
                        })
                        continue

                    # Kiểm tra nếu template có mixed format (VD: SEAMO Math)
                    if tpl.get("mixed_format"):
                        result = _grade_mixed_format_sheet(img_bytes, answers, actual_template)
                    else:
                        result = _grade_single_sheet(img_bytes, answers, actual_template)

                    if "error" in result:
                        results.append({
                            "filename": page_filename,
                            "detected_template": actual_template,
                            "error": result["error"]
                        })
                    else:
                        result["filename"] = page_filename
                        result["detected_template"] = actual_template
                        results.append(result)
                        all_scores.append(result["score"])

                pdf_doc.close()
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": f"Lỗi đọc PDF: {str(e)}"
                })
            continue

        elif ext not in supported_image_formats:
            results.append({
                "filename": file.filename,
                "error": "Định dạng file không được hỗ trợ"
            })
            continue

        try:
            content = await file.read()

            # Tự động nhận diện template từ phiếu
            actual_template = template_type
            if auto_detect_template:
                # Bước 1: Thử nhận diện từ OCR trước
                detected_info = _detect_template_from_image(content)
                detected = detected_info.get("detected_template", "")

                if detected and detected in ANSWER_TEMPLATES:
                    actual_template = detected
                else:
                    # Bước 2: Nếu OCR không thành công, phát hiện số câu hỏi trước
                    # Chạy preprocessing để đếm số bubbles/câu hỏi
                    try:
                        result_temp = _preprocess_omr_image(content)
                        if result_temp[0] is not None:
                            _, gray, binary = result_temp

                            # Thử với IKSC (layout row) trước
                            rows, _ = _detect_bubbles_grid_based(gray, binary, "IKSC_BENJAMIN")
                            questions_iksc = _group_bubbles_to_questions_improved(rows, "IKSC_BENJAMIN")
                            num_iksc = len(questions_iksc)

                            # Thử với IKLC (layout column) nếu IKSC không đủ
                            rows_iklc, _ = _detect_bubbles_grid_based(gray, binary, "IKLC_STUDENT")
                            questions_iklc = _group_bubbles_to_questions_improved(rows_iklc, "IKLC_STUDENT")
                            num_iklc = len(questions_iklc)

                            # Chọn template có nhiều câu hỏi hơn
                            num_questions_detected = max(num_iksc, num_iklc)
                            is_iklc_layout = num_iklc > num_iksc

                            # Xác định template dựa trên số câu và layout
                            if num_questions_detected >= 45 or is_iklc_layout:
                                # 50 câu hoặc layout IKLC -> IKLC
                                level = detected_info.get('detected_level', 'BENJAMIN')
                                if level not in ['BENJAMIN', 'CADET', 'JUNIOR', 'STUDENT']:
                                    level = 'STUDENT'  # Default cho 50 câu
                                actual_template = f"IKLC_{level}"
                                if actual_template not in ANSWER_TEMPLATES:
                                    actual_template = "IKLC_STUDENT"
                            elif num_questions_detected >= 25:
                                # 30 câu -> IKSC (Benjamin/Cadet/Junior/Student)
                                actual_template = f"IKSC_{detected_info.get('detected_level', 'BENJAMIN')}"
                                if actual_template not in ANSWER_TEMPLATES:
                                    actual_template = "IKSC_BENJAMIN"
                            elif num_questions_detected >= 20:
                                # 24 câu -> IKSC Pre-Ecolier/Ecolier
                                actual_template = "IKSC_PRE_ECOLIER"

                            # Nếu có detected_contest từ OCR, ưu tiên sử dụng
                            if detected_info.get("detected_contest"):
                                contest = detected_info["detected_contest"]
                                level = detected_info.get("detected_level", "")
                                if level:
                                    test_template = f"{contest}_{level}"
                                    if test_template in ANSWER_TEMPLATES:
                                        actual_template = test_template
                    except:
                        pass

            # Lấy đáp án cho template
            answers = get_answers_for_template(actual_template)
            tpl = ANSWER_TEMPLATES.get(actual_template, default_template)

            if len(answers) < tpl["questions"]:
                results.append({
                    "filename": file.filename,
                    "detected_template": actual_template,
                    "error": f"Không tìm thấy đáp án cho {tpl['name']}. Vui lòng kiểm tra file đáp án."
                })
                continue

            # Kiểm tra nếu template có mixed format (VD: SEAMO Math)
            if tpl.get("mixed_format"):
                result = _grade_mixed_format_sheet(content, answers, actual_template)
            else:
                result = _grade_single_sheet(content, answers, actual_template)

            if "error" in result:
                results.append({
                    "filename": file.filename,
                    "detected_template": actual_template,
                    "error": result["error"]
                })
            else:
                result["filename"] = file.filename
                result["detected_template"] = actual_template
                results.append(result)
                all_scores.append(result["score"])
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Lỗi xử lý: {str(e)}"
            })

    # Tính tổng kết
    summary = {
        "total_sheets": len(results),
        "graded": len(all_scores),
        "errors": len(results) - len(all_scores),
        "average_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        "highest": max(all_scores) if all_scores else 0,
        "lowest": min(all_scores) if all_scores else 0
    }

    return {
        "ok": True,
        "auto_detect": auto_detect_template,
        "results": results,
        "summary": summary
    }


@app.post("/api/grade-sheets/export")
async def export_grading_results(
    results: str = Form(...),
    template_type: str = Form("IKSC_BENJAMIN")
):
    """Xuất kết quả chấm bài ra Excel"""
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    try:
        data = json.loads(results)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Dữ liệu không hợp lệ")

    # Lấy số câu từ kết quả thực tế (từ details của bài đầu tiên có kết quả)
    # Thay vì dùng template_type có thể sai
    num_questions = 30  # Default
    for result in data:
        if "details" in result and result["details"]:
            num_questions = len(result["details"])
            break
        elif "total" in result:
            num_questions = result["total"]
            break

    # Fallback to template if no results
    if num_questions == 30:
        template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
        num_questions = template["questions"]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Kết quả chấm bài"

    # Styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    correct_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    wrong_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Headers - Thông tin học sinh + Điểm + Đáp án từng câu
    headers = [
        "STT",
        "Họ và tên",
        "Lớp",
        "Ngày sinh",
        "Số báo danh",
        "Trường",
        "Điểm",
        "Đúng",
        "Sai",
        "Bỏ trống"
    ]
    for i in range(1, num_questions + 1):
        headers.append(f"Câu {i}")

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    # Data rows
    for row_idx, result in enumerate(data, 2):
        if "error" in result:
            ws.cell(row=row_idx, column=1, value=row_idx - 1)
            ws.cell(row=row_idx, column=2, value=result.get("filename", ""))
            ws.cell(row=row_idx, column=7, value="Lỗi: " + result["error"])
            continue

        # Lấy thông tin học sinh
        student_info = result.get("student_info", {})

        ws.cell(row=row_idx, column=1, value=row_idx - 1).alignment = center_align
        ws.cell(row=row_idx, column=2, value=student_info.get("full_name", "")).alignment = left_align
        ws.cell(row=row_idx, column=3, value=student_info.get("class", "")).alignment = center_align
        ws.cell(row=row_idx, column=4, value=student_info.get("dob", "")).alignment = center_align
        ws.cell(row=row_idx, column=5, value=student_info.get("id_no", "")).alignment = center_align
        ws.cell(row=row_idx, column=6, value=student_info.get("school_name", "")).alignment = left_align
        ws.cell(row=row_idx, column=7, value=result.get("score", 0)).alignment = center_align
        ws.cell(row=row_idx, column=8, value=result.get("correct", 0)).alignment = center_align
        ws.cell(row=row_idx, column=9, value=result.get("wrong", 0)).alignment = center_align
        ws.cell(row=row_idx, column=10, value=result.get("blank", 0)).alignment = center_align

        # Chi tiết từng câu
        details = result.get("details", [])
        for detail in details:
            col = 10 + detail["q"]
            cell = ws.cell(row=row_idx, column=col, value=detail.get("student", ""))
            cell.alignment = center_align
            cell.border = thin_border

            if detail["status"] == "correct":
                cell.fill = correct_fill
            elif detail["status"] in ["wrong", "invalid"]:
                cell.fill = wrong_fill

    # Đáp án đúng ở hàng cuối
    answer_row = len(data) + 2
    ws.cell(row=answer_row, column=1, value="").alignment = center_align
    ws.cell(row=answer_row, column=2, value="ĐÁP ÁN ĐÚNG").font = Font(bold=True)

    if data and "details" in data[0]:
        for detail in data[0]["details"]:
            col = 10 + detail["q"]
            cell = ws.cell(row=answer_row, column=col, value=detail.get("correct", ""))
            cell.alignment = center_align
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    # Điều chỉnh độ rộng cột
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 25  # Họ và tên
    ws.column_dimensions["C"].width = 8   # Lớp
    ws.column_dimensions["D"].width = 12  # Ngày sinh
    ws.column_dimensions["E"].width = 12  # Số báo danh
    ws.column_dimensions["F"].width = 30  # Trường
    ws.column_dimensions["G"].width = 8   # Điểm
    ws.column_dimensions["H"].width = 6   # Đúng
    ws.column_dimensions["I"].width = 6   # Sai
    ws.column_dimensions["J"].width = 10  # Bỏ trống

    # Lưu file
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=ket_qua_cham_bai_{template_type}.xlsx"}
    )


# ============================================================================
# HANDWRITTEN ANSWER RECOGNITION (Nhận diện đáp án viết tay)
# ============================================================================

# Cache EasyOCR reader để tránh load lại model mỗi lần
_easyocr_reader = None

def _get_easyocr_reader():
    """Lấy hoặc tạo EasyOCR reader (singleton pattern)"""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        # Hỗ trợ tiếng Anh và tiếng Việt
        _easyocr_reader = easyocr.Reader(['en', 'vi'], gpu=False)
    return _easyocr_reader


def _preprocess_handwritten_image(image_bytes: bytes):
    """Tiền xử lý ảnh cho nhận diện chữ viết tay"""
    import cv2
    import numpy as np

    # Đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None, None

    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Giảm nhiễu
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    return image, denoised


def _detect_answer_boxes(gray_image, num_questions: int = 30):
    """Phát hiện các ô đáp án trong phiếu (dạng điền chữ)

    Trả về danh sách các vùng chứa đáp án, sắp xếp theo thứ tự câu hỏi
    """
    import cv2
    import numpy as np

    height, width = gray_image.shape[:2]

    # Binary threshold
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc các ô có kích thước phù hợp (ô đáp án thường là hình chữ nhật nhỏ)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Ô đáp án thường có:
        # - Kích thước vừa phải (không quá nhỏ, không quá lớn)
        # - Tỉ lệ gần vuông hoặc hơi ngang
        min_size = min(width, height) * 0.02
        max_size = min(width, height) * 0.15

        if (min_size < w < max_size and
            min_size < h < max_size and
            0.5 < aspect_ratio < 3.0 and
            area > min_size * min_size):
            boxes.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w // 2,
                'cy': y + h // 2
            })

    if not boxes:
        return []

    # Sắp xếp theo vị trí: từ trên xuống, từ trái sang phải
    # Nhóm theo hàng trước
    boxes.sort(key=lambda b: b['cy'])

    # Nhóm các ô theo hàng (tolerance = chiều cao trung bình / 2)
    avg_height = np.mean([b['h'] for b in boxes])
    row_tolerance = avg_height * 0.6

    rows = []
    current_row = [boxes[0]]

    for box in boxes[1:]:
        if abs(box['cy'] - current_row[-1]['cy']) < row_tolerance:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b['x']))
            current_row = [box]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b['x']))

    # Flatten và giới hạn số câu
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(row)

    return sorted_boxes[:num_questions]


def _recognize_handwritten_answer(reader, image, box, valid_answers: List[str] = None):
    """Nhận diện chữ viết tay trong một ô đáp án

    Args:
        reader: EasyOCR reader
        image: Ảnh gốc (grayscale hoặc color)
        box: Dict chứa x, y, w, h của ô
        valid_answers: Danh sách đáp án hợp lệ (e.g., ['A', 'B', 'C', 'D', 'E'])

    Returns:
        Tuple (recognized_text, confidence)
    """
    import cv2
    import numpy as np

    if valid_answers is None:
        valid_answers = ['A', 'B', 'C', 'D', 'E']

    # Cắt vùng ô đáp án với margin
    margin = 3
    x1 = max(0, box['x'] - margin)
    y1 = max(0, box['y'] - margin)
    x2 = min(image.shape[1], box['x'] + box['w'] + margin)
    y2 = min(image.shape[0], box['y'] + box['h'] + margin)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None, 0.0

    # Resize nếu quá nhỏ (EasyOCR cần ảnh đủ lớn)
    min_dim = 32
    if roi.shape[0] < min_dim or roi.shape[1] < min_dim:
        scale = max(min_dim / roi.shape[0], min_dim / roi.shape[1])
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Nhận diện bằng EasyOCR
    try:
        results = reader.readtext(roi, detail=1, paragraph=False)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    # Lấy kết quả có confidence cao nhất
    best_text = None
    best_confidence = 0.0

    for (bbox, text, confidence) in results:
        # Chuẩn hóa text
        text = text.strip().upper()

        # Loại bỏ ký tự không hợp lệ
        text = ''.join(c for c in text if c.isalnum())

        if not text:
            continue

        # Kiểm tra xem có match với đáp án hợp lệ không
        # Ưu tiên exact match
        for valid in valid_answers:
            if text == valid.upper():
                if confidence > best_confidence:
                    best_text = valid.upper()
                    best_confidence = confidence
                break
        else:
            # Nếu không exact match, lấy ký tự đầu tiên
            first_char = text[0]
            if first_char in [v.upper() for v in valid_answers]:
                if confidence > best_confidence:
                    best_text = first_char
                    best_confidence = confidence

    return best_text, best_confidence


def _grade_handwritten_sheet(
    image_bytes: bytes,
    answer_key: List[str],
    num_questions: int = 30,
    valid_answers: List[str] = None
):
    """Chấm một phiếu đáp án viết tay

    Args:
        image_bytes: Dữ liệu ảnh
        answer_key: Danh sách đáp án đúng
        num_questions: Số câu hỏi
        valid_answers: Danh sách đáp án hợp lệ

    Returns:
        Dict chứa kết quả chấm bài
    """
    if valid_answers is None:
        valid_answers = ['A', 'B', 'C', 'D', 'E']

    # Tiền xử lý ảnh
    original, processed = _preprocess_handwritten_image(image_bytes)

    if original is None:
        return {"error": "Không thể đọc ảnh"}

    # Lấy EasyOCR reader
    try:
        reader = _get_easyocr_reader()
    except Exception as e:
        return {"error": f"Không thể khởi tạo OCR: {str(e)}"}

    # Phương pháp 1: Phát hiện ô đáp án tự động
    boxes = _detect_answer_boxes(processed, num_questions)

    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    if boxes and len(boxes) >= num_questions * 0.5:
        # Có đủ ô đáp án -> nhận diện từng ô
        for q_idx in range(num_questions):
            if q_idx >= len(boxes):
                student_answers.append(None)
                blank_count += 1
                correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
                details.append({
                    "q": q_idx + 1,
                    "student": None,
                    "correct": correct_answer,
                    "status": "not_found",
                    "confidence": 0.0
                })
                continue

            box = boxes[q_idx]
            recognized, confidence = _recognize_handwritten_answer(
                reader, processed, box, valid_answers
            )

            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

            if recognized is None:
                student_answers.append(None)
                blank_count += 1
                status = "blank"
            elif correct_answer and recognized.upper() == correct_answer.upper():
                student_answers.append(recognized)
                correct_count += 1
                status = "correct"
            else:
                student_answers.append(recognized)
                wrong_count += 1
                status = "wrong"

            details.append({
                "q": q_idx + 1,
                "student": recognized,
                "correct": correct_answer,
                "status": status,
                "confidence": round(confidence, 3)
            })
    else:
        # Không phát hiện đủ ô -> scan toàn bộ ảnh
        try:
            results = reader.readtext(processed, detail=1, paragraph=False)
        except Exception as e:
            return {"error": f"Lỗi OCR: {str(e)}"}

        # Lọc và sắp xếp kết quả
        recognized_answers = []
        for (bbox, text, confidence) in results:
            text = text.strip().upper()
            text = ''.join(c for c in text if c.isalnum())

            if text and len(text) <= 2:  # Đáp án thường là 1-2 ký tự
                # Kiểm tra ký tự đầu có hợp lệ không
                first_char = text[0] if text else None
                if first_char in [v.upper() for v in valid_answers]:
                    # Lấy tọa độ trung tâm
                    cx = (bbox[0][0] + bbox[2][0]) / 2
                    cy = (bbox[0][1] + bbox[2][1]) / 2
                    recognized_answers.append({
                        'text': first_char,
                        'confidence': confidence,
                        'cx': cx,
                        'cy': cy
                    })

        # Sắp xếp theo vị trí
        recognized_answers.sort(key=lambda r: (r['cy'], r['cx']))

        for q_idx in range(num_questions):
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

            if q_idx >= len(recognized_answers):
                student_answers.append(None)
                blank_count += 1
                details.append({
                    "q": q_idx + 1,
                    "student": None,
                    "correct": correct_answer,
                    "status": "not_found",
                    "confidence": 0.0
                })
                continue

            answer_info = recognized_answers[q_idx]
            recognized = answer_info['text']
            confidence = answer_info['confidence']

            if correct_answer and recognized.upper() == correct_answer.upper():
                student_answers.append(recognized)
                correct_count += 1
                status = "correct"
            else:
                student_answers.append(recognized)
                wrong_count += 1
                status = "wrong"

            details.append({
                "q": q_idx + 1,
                "student": recognized,
                "correct": correct_answer,
                "status": status,
                "confidence": round(confidence, 3)
            })

    # Tính điểm (đơn giản: 1 điểm/câu đúng)
    score = correct_count

    return {
        "answers": student_answers,
        "score": score,
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details
    }


@app.post("/api/grade-handwritten")
async def grade_handwritten_sheets(
    files: List[UploadFile],
    answer_key: str = Form(...),
    num_questions: int = Form(30),
    valid_answers: str = Form("A,B,C,D,E"),
    scoring_correct: float = Form(1.0),
    scoring_wrong: float = Form(0.0),
    scoring_blank: float = Form(0.0)
):
    """Chấm nhiều phiếu đáp án viết tay

    Args:
        files: Danh sách file ảnh phiếu trả lời
        answer_key: Đáp án đúng (JSON array hoặc comma-separated)
        num_questions: Số câu hỏi
        valid_answers: Các đáp án hợp lệ (comma-separated, default: A,B,C,D,E)
        scoring_correct: Điểm cho câu đúng
        scoring_wrong: Điểm cho câu sai (thường là 0 hoặc âm)
        scoring_blank: Điểm cho câu bỏ trống
    """
    # Parse đáp án
    try:
        if answer_key.startswith('['):
            keys = json.loads(answer_key)
        else:
            keys = [k.strip().upper() for k in answer_key.split(',')]
    except Exception:
        raise HTTPException(status_code=400, detail="Định dạng đáp án không hợp lệ")

    # Parse valid answers
    valid_list = [v.strip().upper() for v in valid_answers.split(',')]

    results = []

    for file in files:
        try:
            image_bytes = await file.read()

            result = _grade_handwritten_sheet(
                image_bytes,
                keys,
                num_questions,
                valid_list
            )

            if "error" not in result:
                # Tính lại điểm theo công thức
                score = (
                    result["correct"] * scoring_correct +
                    result["wrong"] * scoring_wrong +
                    result["blank"] * scoring_blank
                )
                result["score"] = round(score, 2)

            result["filename"] = file.filename
            results.append(result)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    # Tính thống kê
    valid_results = [r for r in results if "error" not in r]
    summary = {
        "total_sheets": len(files),
        "successful": len(valid_results),
        "failed": len(files) - len(valid_results)
    }

    if valid_results:
        scores = [r["score"] for r in valid_results]
        summary["average_score"] = round(sum(scores) / len(scores), 2)
        summary["highest"] = max(scores)
        summary["lowest"] = min(scores)

    return {
        "ok": True,
        "results": results,
        "summary": summary
    }


@app.post("/api/grade-handwritten/export")
async def export_handwritten_results(
    results: str = Form(...),
    num_questions: int = Form(30)
):
    """Xuất kết quả chấm bài viết tay ra Excel"""
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    try:
        data = json.loads(results)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Dữ liệu không hợp lệ")

    # Lấy số câu từ kết quả thực tế
    for result in data:
        if "details" in result and result["details"]:
            num_questions = len(result["details"])
            break

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Kết quả chấm bài viết tay"

    # Styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    correct_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    wrong_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    low_conf_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Headers
    headers = ["STT", "Tên file", "Điểm", "Đúng", "Sai", "Bỏ trống"]
    for i in range(1, num_questions + 1):
        headers.append(f"Câu {i}")

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    # Data rows
    for row_idx, result in enumerate(data, 2):
        if "error" in result:
            ws.cell(row=row_idx, column=1, value=row_idx - 1)
            ws.cell(row=row_idx, column=2, value=result.get("filename", ""))
            ws.cell(row=row_idx, column=3, value="Lỗi: " + result["error"])
            continue

        ws.cell(row=row_idx, column=1, value=row_idx - 1).alignment = center_align
        ws.cell(row=row_idx, column=2, value=result.get("filename", "")).alignment = center_align
        ws.cell(row=row_idx, column=3, value=result.get("score", 0)).alignment = center_align
        ws.cell(row=row_idx, column=4, value=result.get("correct", 0)).alignment = center_align
        ws.cell(row=row_idx, column=5, value=result.get("wrong", 0)).alignment = center_align
        ws.cell(row=row_idx, column=6, value=result.get("blank", 0)).alignment = center_align

        # Chi tiết từng câu
        details = result.get("details", [])
        for detail in details:
            col = 6 + detail["q"]
            cell = ws.cell(row=row_idx, column=col, value=detail.get("student", ""))
            cell.alignment = center_align
            cell.border = thin_border

            # Màu theo status
            if detail["status"] == "correct":
                cell.fill = correct_fill
            elif detail["status"] in ["wrong", "invalid"]:
                cell.fill = wrong_fill

            # Đánh dấu confidence thấp
            confidence = detail.get("confidence", 1.0)
            if confidence < 0.5 and detail["status"] != "blank":
                cell.fill = low_conf_fill

    # Đáp án đúng ở hàng cuối
    answer_row = len(data) + 2
    ws.cell(row=answer_row, column=2, value="ĐÁP ÁN ĐÚNG").font = Font(bold=True)

    if data and "details" in data[0]:
        for detail in data[0]["details"]:
            col = 6 + detail["q"]
            cell = ws.cell(row=answer_row, column=col, value=detail.get("correct", ""))
            cell.alignment = center_align
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    # Điều chỉnh độ rộng cột
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 8
    ws.column_dimensions["D"].width = 6
    ws.column_dimensions["E"].width = 6
    ws.column_dimensions["F"].width = 10

    # Lưu file
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=ket_qua_cham_viet_tay.xlsx"}
    )
