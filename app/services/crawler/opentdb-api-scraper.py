"""
Scraper for Open Trivia Database (opentdb.com) - Free trivia API.

Categories available:
- 17: Science & Nature
- 18: Science: Computers
- 19: Science: Mathematics
- 22: Geography
- 23: History
- 9: General Knowledge
"""
import html
import time
from typing import Dict, List, Optional, Tuple

import httpx

# Map OpenTDB categories to our subject names
CATEGORY_MAP = {
    17: "science",      # Science & Nature
    18: "informatics",  # Science: Computers
    19: "math",         # Science: Mathematics
    22: "geography",    # Geography
    23: "history",      # History
    9: "general",       # General Knowledge
    27: "biology",      # Animals -> biology
}

# Difficulty mapping
DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def _decode_html(text: str) -> str:
    """Decode HTML entities in text."""
    if not text:
        return ""
    # Decode HTML entities like &quot; &#039; etc.
    return html.unescape(text)


def fetch_questions(
    category: int,
    amount: int = 50,
    difficulty: Optional[str] = None
) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch questions from Open Trivia DB API.

    Args:
        category: Category ID (see CATEGORY_MAP)
        amount: Number of questions (max 50 per request)
        difficulty: easy, medium, or hard (optional)

    Returns:
        Tuple of (questions list, error message or None)
    """
    url = f"https://opentdb.com/api.php?amount={amount}&category={category}&type=multiple"
    if difficulty:
        url += f"&difficulty={difficulty}"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            if data.get("response_code") != 0:
                error_codes = {
                    1: "No results - not enough questions available",
                    2: "Invalid parameter",
                    3: "Token not found",
                    4: "Token exhausted",
                }
                code = data.get("response_code", -1)
                return [], error_codes.get(code, f"Unknown error code: {code}")

            questions = []
            for item in data.get("results", []):
                # Decode HTML entities
                question_text = _decode_html(item.get("question", ""))
                correct = _decode_html(item.get("correct_answer", ""))
                incorrect = [_decode_html(a) for a in item.get("incorrect_answers", [])]

                # Build options list with correct answer mixed in
                options = incorrect + [correct]
                # Shuffle options (simple approach: sort alphabetically)
                options = sorted(options)

                # Find correct answer index
                answer_idx = options.index(correct)
                answer_letter = chr(65 + answer_idx)  # A, B, C, D

                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": answer_letter,
                    "subject": CATEGORY_MAP.get(category, "general"),
                    "difficulty": DIFFICULTY_MAP.get(item.get("difficulty", "medium"), "medium"),
                    "source": "opentdb.com",
                    "grade": "international",
                })

            return questions, None

    except httpx.HTTPError as e:
        return [], f"HTTP error: {e}"
    except Exception as e:
        return [], f"Error: {e}"


def fetch_all_from_category(category: int, max_questions: int = 200) -> List[Dict]:
    """
    Fetch all available questions from a category.

    Args:
        category: Category ID
        max_questions: Maximum questions to fetch

    Returns:
        List of question dicts
    """
    all_questions = []
    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        # Fetch up to 50 per difficulty
        questions, error = fetch_questions(
            category=category,
            amount=min(50, max_questions - len(all_questions)),
            difficulty=difficulty
        )
        if questions:
            all_questions.extend(questions)
            print(f"    {difficulty}: {len(questions)} questions")

        # Respect rate limit
        time.sleep(1)

        if len(all_questions) >= max_questions:
            break

    return all_questions


def scrape_all_categories() -> Tuple[List[Dict], List[str]]:
    """
    Scrape all mapped categories from Open Trivia DB.

    Returns:
        Tuple of (all questions, list of errors)
    """
    all_questions = []
    errors = []

    for category_id, subject in CATEGORY_MAP.items():
        print(f"  Fetching category {category_id} ({subject})...")
        questions = fetch_all_from_category(category_id)

        if questions:
            all_questions.extend(questions)
            print(f"    Total: {len(questions)} questions")
        else:
            errors.append(f"No questions from category {category_id}")

        # Respect rate limit between categories
        time.sleep(2)

    return all_questions, errors


if __name__ == "__main__":
    print("Testing Open Trivia DB scraper...")
    questions, error = fetch_questions(category=17, amount=5)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Got {len(questions)} questions:")
        for q in questions:
            print(f"  - {q['question'][:60]}...")
            print(f"    Options: {q['options']}")
            print(f"    Answer: {q['answer']}")
