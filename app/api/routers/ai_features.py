"""
AI Features API endpoints: Difficulty Analysis, Suggestions, Quality Review.
"""
import json
import re
from pathlib import Path
import httpx
from fastapi import APIRouter

from app.core import AIAnalyzeRequest, AISuggestRequest, AIReviewRequest
from app.services.ai import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE,
    GEMINI_API_KEY, GEMINI_MODEL,
    OLLAMA_BASE, OLLAMA_MODEL,
)

router = APIRouter(prefix="/api/ai", tags=["ai"])


def _load_ai_settings() -> dict:
    """Load AI settings from file"""
    settings_file = Path("ai_settings.json")
    if settings_file.exists():
        with open(settings_file) as f:
            return json.load(f)
    return {}


async def _call_ai_engine(engine: str, prompt: str, settings: dict) -> str:
    """Helper function to call different AI engines"""
    import app.services.ai.config as ai_config

    async with httpx.AsyncClient(timeout=300.0) as client:
        if engine == "openai":
            api_key = settings.get("openai_key") or settings.get("openai_api_key") or ai_config.OPENAI_API_KEY
            base_url = settings.get("openai_base") or settings.get("openai_base_url") or ai_config.OPENAI_API_BASE
            model = settings.get("openai_model") or ai_config.OPENAI_MODEL

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
            api_key = settings.get("gemini_key") or settings.get("gemini_api_key") or ai_config.GEMINI_API_KEY
            model = settings.get("gemini_model") or ai_config.GEMINI_MODEL

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        elif engine == "ollama":
            base_url = settings.get("ollama_base") or settings.get("ollama_base_url") or ai_config.OLLAMA_BASE
            model = settings.get("ollama_model") or ai_config.OLLAMA_MODEL

            response = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]

        else:
            raise ValueError(f"Unknown AI engine: {engine}")


@router.post("/analyze-difficulty")
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
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {"ok": True, "analysis": result}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/suggest-similar")
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
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            suggestions = json.loads(json_match.group())
            return {"ok": True, "suggestions": suggestions}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/review-quality")
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
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            review = json.loads(json_match.group())
            return {"ok": True, "review": review}
        return {"ok": False, "error": "Không thể phân tích kết quả AI"}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Lỗi parse JSON: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    ai_engine: str = "ollama"


@router.post("/chat")
async def ai_chat(data: ChatRequest):
    """Chat với AI trợ lý - hỗ trợ tạo đề, giải đáp câu hỏi"""
    settings = _load_ai_settings()
    engine = data.ai_engine

    # Build system prompt
    system_prompt = """Bạn là trợ lý AI chuyên hỗ trợ giáo viên trong việc:
1. Tạo câu hỏi và đề thi
2. Giải thích các khái niệm học thuật
3. Gợi ý cải thiện câu hỏi
4. Phân tích độ khó của đề
5. Hỗ trợ soạn giáo án

Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

FORMAT CÂU HỎI TRẮC NGHIỆM:
- Các đáp án A), B), C), D) chỉ chứa nội dung, KHÔNG thêm "(đúng)" hay đánh dấu gì
- Ghi đáp án đúng riêng ở cuối, ví dụ: "Đáp án: B"
- Ví dụ đúng:
  A) $x = 2$
  B) $x = 3$
  C) $x = 4$
  D) $x = 5$
  Đáp án: B

CÔNG THỨC TOÁN HỌC - LUÔN dùng LaTeX:
- Inline: $công thức$ (ví dụ: $x^2 + 2x + 1$)
- Block: $$công thức$$ (ví dụ: $$\\frac{a}{b}$$)
- Ví dụ: viết "$f'(x) = 3x^2$" thay vì "f'(x) = 3x^2\""""

    # Build conversation history
    messages = [{"role": "system", "content": system_prompt}]
    for msg in data.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": data.message})

    try:
        import app.services.ai.config as ai_config
        async with httpx.AsyncClient(timeout=300.0) as client:
            if engine == "openai":
                api_key = settings.get("openai_key") or settings.get("openai_api_key") or ai_config.OPENAI_API_KEY
                base_url = settings.get("openai_base") or settings.get("openai_base_url") or ai_config.OPENAI_API_BASE
                model = settings.get("openai_model") or ai_config.OPENAI_MODEL

                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                    },
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]

            elif engine == "gemini":
                api_key = settings.get("gemini_key") or settings.get("gemini_api_key") or ai_config.GEMINI_API_KEY
                model = settings.get("gemini_model") or ai_config.GEMINI_MODEL

                # Gemini format: proper multi-turn conversation
                gemini_contents = []
                for msg in messages:
                    if msg["role"] == "system":
                        # System message as first user message
                        gemini_contents.append({
                            "role": "user",
                            "parts": [{"text": f"[Hướng dẫn hệ thống]: {msg['content']}"}]
                        })
                        gemini_contents.append({
                            "role": "model",
                            "parts": [{"text": "Tôi hiểu và sẽ tuân theo hướng dẫn."}]
                        })
                    elif msg["role"] == "user":
                        gemini_contents.append({
                            "role": "user",
                            "parts": [{"text": msg["content"]}]
                        })
                    elif msg["role"] == "assistant":
                        gemini_contents.append({
                            "role": "model",
                            "parts": [{"text": msg["content"]}]
                        })

                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={"contents": gemini_contents},
                )
                response.raise_for_status()
                reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            elif engine == "ollama":
                base_url = settings.get("ollama_base") or settings.get("ollama_base_url") or ai_config.OLLAMA_BASE
                model = settings.get("ollama_model") or ai_config.OLLAMA_MODEL

                # Ollama với chat API - disable thinking for faster response
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": 1024,  # Limit response length
                        }
                    },
                )
                response.raise_for_status()
                reply = response.json()["message"]["content"]

            else:
                return {"ok": False, "error": f"Unknown AI engine: {engine}"}

        return {"ok": True, "reply": reply}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"API Error: {e.response.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
