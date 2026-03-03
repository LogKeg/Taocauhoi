"""
RAG retriever for question generation.
Retrieves similar questions from the database to use as context for AI generation.
"""
import json
from typing import List, Optional

from app.database import SessionLocal, QuestionCRUD


def retrieve_similar_questions(
    subject: str,
    topic: str = "",
    difficulty: str = "",
    question_type: str = "mcq",
    limit: int = 5,
) -> List[str]:
    """
    Retrieve similar questions from the database for RAG context.

    Args:
        subject: Subject key (e.g., 'toan', 'tieng_anh')
        topic: Topic key within the subject (optional)
        difficulty: Difficulty level - 'easy', 'medium', 'hard' (optional)
        question_type: Question type - 'mcq', 'blank', 'essay' (optional)
        limit: Maximum number of questions to retrieve

    Returns:
        List of formatted question strings with options
    """
    db = SessionLocal()
    try:
        questions = QuestionCRUD.get_all(
            db,
            limit=limit,
            subject=subject,
            topic=topic if topic else None,
            difficulty=difficulty if difficulty else None,
            question_type=question_type,
        )

        # Format questions with options
        result = []
        for q in questions:
            text = q.content

            # Add options if available
            if q.options:
                try:
                    opts = json.loads(q.options)
                    if isinstance(opts, list):
                        labels = ['A', 'B', 'C', 'D', 'E']
                        for i, opt in enumerate(opts[:5]):
                            text += f"\n{labels[i]}) {opt}"
                except (json.JSONDecodeError, TypeError):
                    pass

            result.append(text)

        return result
    finally:
        db.close()


def retrieve_by_content_similarity(
    sample_text: str,
    subject: str = "",
    limit: int = 3,
) -> List[str]:
    """
    Retrieve questions similar to a sample text (basic text matching).

    For now, this is a simple implementation that retrieves questions
    from the same subject. A more advanced implementation could use
    embedding-based similarity search.

    Args:
        sample_text: The sample question text to find similar questions for
        subject: Subject to filter by (optional)
        limit: Maximum number of questions to retrieve

    Returns:
        List of formatted question strings
    """
    db = SessionLocal()
    try:
        # Get questions from the same subject
        questions = QuestionCRUD.get_all(
            db,
            limit=limit * 2,  # Get more to filter
            subject=subject if subject else None,
        )

        # Basic relevance: prefer questions with similar length and structure
        sample_len = len(sample_text)
        sample_has_options = any(
            sample_text.find(f"{label})") != -1 or sample_text.find(f"{label}.") != -1
            for label in ['A', 'B', 'C', 'D']
        )

        scored = []
        for q in questions:
            score = 0
            content = q.content

            # Length similarity (within 50% of sample length)
            len_diff = abs(len(content) - sample_len) / max(sample_len, 1)
            if len_diff < 0.5:
                score += 2
            elif len_diff < 1.0:
                score += 1

            # Structure similarity (both MCQ or both not)
            q_has_options = bool(q.options)
            if sample_has_options == q_has_options:
                score += 2

            scored.append((score, q))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Format top results
        result = []
        for _, q in scored[:limit]:
            text = q.content
            if q.options:
                try:
                    opts = json.loads(q.options)
                    if isinstance(opts, list):
                        labels = ['A', 'B', 'C', 'D', 'E']
                        for i, opt in enumerate(opts[:5]):
                            text += f"\n{labels[i]}) {opt}"
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(text)

        return result
    finally:
        db.close()
