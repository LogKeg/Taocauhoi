"""
AI prompt building for question generation.
"""
from typing import List, Optional

from app.core import TOPICS, TOPIC_AI_GUIDE, SUBJECTS, SUBJECT_TOPICS, QUESTION_TYPES


def build_ai_prompt(
    sample: Optional[str],
    topic_key: str,
    custom_keywords: List[str],
    paraphrase: bool,
    change_numbers: bool,
    change_context: bool,
    variants: int,
) -> str:
    """Build AI prompt for generating question variants."""
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


def build_topic_prompt(
    subject_key: str,
    grade: int,
    qtype: str,
    count: int,
    topic: str = "",
    difficulty: str = "medium",
    rag_examples: Optional[List[str]] = None,
) -> str:
    """Build AI prompt for generating questions by topic."""
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

        # Build RAG section for English
        rag_section_en = ""
        if rag_examples:
            rag_section_en = (
                "\n\nREFERENCE QUESTIONS (from question bank):\n\n"
                + "\n\n".join(f"Example {i+1}:\n{q}" for i, q in enumerate(rag_examples))
                + "\n\nCreate NEW questions similar in style to the examples above, DO NOT copy verbatim.\n"
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
            f"{rag_section_en}"
        )

    grade_text = f"Lớp {grade}"
    type_text = QUESTION_TYPES.get(qtype, "Trắc nghiệm")
    topic_text = f" về {topic_label}" if topic_label else ""
    diff_text = difficulty_map_vi.get(difficulty, "trung bình")
    mcq_rule = ""
    mcq_example = ""
    if qtype == "mcq":
        mcq_rule = "Mỗi câu có 4 đáp án A) B) C) D) trên các dòng riêng biệt ngay sau câu hỏi.\n"
        mcq_example = (
            "\n\nVÍ DỤ FORMAT ĐÚNG:\n"
            "Một đa thức bậc hai có dạng tổng quát là\n"
            r"A) $ax^2 + bx + c$" "\n"
            r"B) $ax^2 + b$" "\n"
            r"C) $ax + bx$" "\n"
            r"D) $ax + c$" "\n"
            "\n"
            r"Tìm nghiệm của phương trình $x^2 - 5x + 6 = 0$" "\n"
            "A) x = 2\n"
            "B) x = 3\n"
            "C) x = 1 và x = 6\n"
            "D) x = 2 và x = 3\n"
            "\n---ĐÁP ÁN---\n1. A\n2. D\n"
        )
    elif qtype == "blank":
        mcq_rule = "Dùng \"...\" cho chỗ trống.\n"

    # Build RAG section if examples provided
    rag_section = ""
    if rag_examples:
        rag_section = (
            "\n\nCÂU HỎI THAM KHẢO (từ ngân hàng đề):\n\n"
            + "\n\n".join(f"Ví dụ {i+1}:\n{q}" for i, q in enumerate(rag_examples))
            + "\n\nHãy tạo câu hỏi MỚI tương tự phong cách các ví dụ trên, KHÔNG sao chép nguyên văn.\n"
        )

    # Math formula instruction
    math_instruction = (
        "- Nếu có công thức toán, viết dạng LaTeX trong dấu $...$ (inline) hoặc $$...$$ (block).\n"
        r"  VD: $x^2$, $\frac{a}{b}$, $\sqrt{x}$, $\sum_{i=1}^{n}$, $\int_0^1 f(x)dx$" "\n"
    )

    return (
        f"Tạo {count} câu hỏi {type_text} môn {label} {grade_text}{topic_text}, độ khó: {diff_text}.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- KHÔNG đánh số câu hỏi (KHÔNG viết 1., 2., Câu 1, Câu 2)\n"
        "- KHÔNG viết 'Hãy cho biết:', 'Choose the correct option:'\n"
        f"{mcq_rule}"
        f"{math_instruction}"
        "- Mỗi câu hỏi cách nhau bằng một dòng trống\n"
        "- Cuối cùng viết ---ĐÁP ÁN--- rồi liệt kê: 1. A, 2. B, 3. C\n"
        f"{mcq_example}"
        f"{rag_section}"
    )
