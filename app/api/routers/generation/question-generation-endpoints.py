"""
Question generation endpoints.

POST /generate - Generate question variants from samples.
POST /generate-topic - Generate questions by topic using AI.
POST /auto-generate - Auto-generate questions with configurable AI ratio.
"""
import random
import re

from fastapi import APIRouter, Form

from app.core import GenerateRequest
from app.services.ai import call_ai
from app.services.text import normalize_question
from app.services.generation import (
    generate_variants,
    split_questions,
    load_questions_from_subject,
    build_topic_prompt,
    normalize_ai_blocks,
    retrieve_similar_questions,
)
from app.api.routers.generation.helpers import (
    _is_engine_available,
    _save_text_questions_to_bank,
    _parse_explanations,
)

router = APIRouter(tags=["generation"])


@router.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    """Generate question variants from samples."""
    engine = payload.ai_engine
    if payload.use_ai and not _is_engine_available(engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        questions = generate_variants(payload)
        saved = _save_text_questions_to_bank(
            questions,
            subject=payload.topic or "general",
            source="generated",
        )
        return {
            "questions": questions,
            "message": f"Chưa cấu hình {engine_names.get(engine, engine)} nên AI không được dùng.",
            "saved_to_bank": saved,
        }

    questions = generate_variants(payload)
    saved = _save_text_questions_to_bank(
        questions,
        subject=payload.topic or "general",
        source="generated-ai" if payload.use_ai else "generated",
    )

    if payload.use_ai and _is_engine_available(engine):
        src = {normalize_question(s) for s in payload.samples if s.strip()}
        out = {normalize_question(q) for q in questions if q.strip()}
        if out and out.issubset(src):
            return {
                "questions": questions,
                "message": "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản.",
                "saved_to_bank": saved,
            }
    return {"questions": questions, "saved_to_bank": saved}


@router.post("/generate-topic")
def generate_topic(
    subject: str = Form(...),
    grade: int = Form(1),
    qtype: str = Form("mcq"),
    count: int = Form(10),
    ai_engine: str = Form("ollama"),
    topic: str = Form(""),
    difficulty: str = Form("medium"),
    language: str = Form("vi"),
    use_rag: bool = Form(True),
    rag_count: int = Form(5),
) -> dict:
    """Generate questions by topic using AI."""
    if not _is_engine_available(ai_engine):
        engine_names = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": "Ollama"}
        return {"questions": [], "message": f"Chưa cấu hình {engine_names.get(ai_engine, ai_engine)} nên AI không được dùng."}

    count = max(1, min(50, count))
    grade = max(1, min(12, grade))
    rag_count = max(1, min(10, rag_count))

    # Retrieve RAG examples from question bank
    rag_examples = []
    if use_rag:
        rag_examples = retrieve_similar_questions(
            subject=subject,
            topic=topic,
            difficulty=difficulty,
            question_type=qtype,
            limit=rag_count,
        )

    prompt = build_topic_prompt(subject, grade, qtype, count, topic, difficulty, rag_examples=rag_examples, language=language)
    text, err = call_ai(prompt, ai_engine)

    if not text:
        msg = f"Không nhận được phản hồi từ AI. {err}" if err else "Không nhận được phản hồi từ AI."
        return {"questions": [], "answers": "", "message": msg}

    # Split explanations section first (if exists)
    explanations = ""
    explanation_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:LỜI GIẢI|Lời giải|EXPLANATIONS|Explanations|Giải thích)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    expl_match = explanation_pattern.search(text)
    if expl_match:
        explanations = text[expl_match.end():].strip()
        text = text[:expl_match.start()]

    # Split answers section from questions
    answers = ""
    answer_pattern = re.compile(
        r"\n\s*-{0,3}\s*(?:ĐÁP ÁN|Đáp án|đáp án|ANSWERS|Answers|Answer Key|answer key)\s*-{0,3}\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = answer_pattern.search(text)
    if match:
        raw_answers = text[match.end():].strip()
        text = text[:match.start()]
        # Clean up answers: extract only answer lines like "1. A", "2. B", etc.
        answer_lines = []
        for line in raw_answers.splitlines():
            line = line.strip()
            # Match patterns like "1. A", "1) B", "1: C", "Câu 1: A", etc.
            ans_match = re.match(r'^(?:Câu\s*)?(\d+)[\.\):\s]+([A-Da-d])\b', line, re.IGNORECASE)
            if ans_match:
                answer_lines.append(f"{ans_match.group(1)}. {ans_match.group(2).upper()}")
        answers = "\n".join(answer_lines) if answer_lines else raw_answers

    questions = normalize_ai_blocks(text)
    questions = [q for q in questions if q.strip()]
    final_questions = questions[:count]

    return {"questions": final_questions, "answers": answers, "explanations": explanations}


@router.post("/auto-generate")
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
    """Auto-generate questions with configurable AI ratio."""
    if samples_text.strip():
        samples = split_questions(samples_text)
    else:
        samples = load_questions_from_subject(subject)

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
    final_questions = questions[:count]

    saved = _save_text_questions_to_bank(
        final_questions,
        subject=subject,
        source="auto-generated",
    )

    result = {"questions": final_questions, "saved_to_bank": saved}
    if use_ai and _is_engine_available(ai_engine):
        src = {normalize_question(s) for s in samples if s.strip()}
        out = {normalize_question(q) for q in result["questions"] if q.strip()}
        if out and out.issubset(src):
            result["message"] = "AI đang trả về câu gần giống câu gốc. Hệ thống đã thêm biến thể đơn giản."
    return result
