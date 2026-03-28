"""
Scraper for QuizAPI (quizapi.io) - Developer-focused quiz questions.

Features:
- IT/Tech focused: Linux, DevOps, Docker, SQL, Programming
- Free tier with API key
- Great for technical quizzes

Note: Requires API key from https://quizapi.io/
Set environment variable: QUIZAPI_KEY
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# Load .env if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# API key from environment
API_KEY = os.getenv("QUIZAPI_KEY", "")

# Map QuizAPI categories/tags to our subject names
CATEGORY_MAP = {
    "linux": "informatics",
    "devops": "informatics",
    "docker": "informatics",
    "sql": "informatics",
    "code": "informatics",
    "bash": "informatics",
    "kubernetes": "informatics",
    "cms": "informatics",
    "laravel": "informatics",
    "php": "informatics",
    "javascript": "informatics",
    "python": "informatics",
}

# Available tags/categories
AVAILABLE_TAGS = [
    "linux",
    "devops",
    "docker",
    "sql",
    "bash",
    "kubernetes",
    "php",
    "javascript",
]

# Difficulty mapping
DIFFICULTY_MAP = {
    "Easy": "easy",
    "Medium": "medium",
    "Hard": "hard",
}


def fetch_questions(
    tag: Optional[str] = None,
    limit: int = 20,
    difficulty: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch questions from QuizAPI.

    Args:
        tag: Category tag (linux, devops, docker, etc.)
        limit: Number of questions (max 20 per request for free tier)
        difficulty: Easy, Medium, or Hard
        api_key: API key (uses env var if not provided)

    Returns:
        Tuple of (questions list, error message or None)
    """
    key = api_key or API_KEY
    if not key:
        return [], "API key required. Set QUIZAPI_KEY environment variable."

    url = f"https://quizapi.io/api/v1/questions?api_key={key}&limit={min(limit, 20)}"

    if tag:
        url += f"&tags={tag}"
    if difficulty:
        url += f"&difficulty={difficulty.lower()}"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            result = response.json()

            # New API returns {success, data, meta}
            if isinstance(result, dict) and "data" in result:
                data = result["data"]
            else:
                data = result

            questions = []
            for item in data:
                # New API format uses "text" instead of "question"
                question_text = item.get("text", "") or item.get("question", "")
                diff = item.get("difficulty", "MEDIUM")
                tags = item.get("tags", [])
                category = item.get("category", "")

                # Get tag name - new format uses strings, old format uses objects
                if tags and isinstance(tags[0], dict):
                    tag_name = tags[0].get("name", "code")
                elif tags and isinstance(tags[0], str):
                    tag_name = tags[0]
                else:
                    tag_name = category or "code"

                # Build options from answers array (new format)
                answers_list = item.get("answers", [])
                options = []
                correct_idx = 0

                if isinstance(answers_list, list):
                    # New format: answers is array of {id, text, isCorrect}
                    for idx, ans in enumerate(answers_list):
                        options.append(ans.get("text", ""))
                        if ans.get("isCorrect", False):
                            correct_idx = idx
                elif isinstance(answers_list, dict):
                    # Old format: answers is dict {answer_a, answer_b, ...}
                    correct_answers = item.get("correct_answers", {})
                    idx = 0
                    for key_name in ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e", "answer_f"]:
                        answer = answers_list.get(key_name)
                        if answer:
                            options.append(answer)
                            correct_key = f"{key_name}_correct"
                            if correct_answers.get(correct_key) == "true":
                                correct_idx = idx
                            idx += 1

                if not options:
                    continue

                answer_letter = chr(65 + correct_idx)  # A, B, C, D, E, F

                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": answer_letter,
                    "subject": CATEGORY_MAP.get(tag_name.lower(), "informatics"),
                    "difficulty": DIFFICULTY_MAP.get(diff, "medium"),
                    "source": "quizapi.io",
                    "grade": "international",
                    "category_raw": tag_name,
                })

            return questions, None

    except httpx.HTTPError as e:
        return [], f"HTTP error: {e}"
    except Exception as e:
        return [], f"Error: {e}"


def fetch_all_tags(api_key: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
    """
    Fetch questions from all available tags.

    Args:
        api_key: API key (uses env var if not provided)

    Returns:
        Tuple of (all questions, list of errors)
    """
    all_questions = []
    errors = []

    for tag in AVAILABLE_TAGS:
        print(f"  Fetching tag: {tag}...")

        for difficulty in ["Easy", "Medium", "Hard"]:
            questions, error = fetch_questions(
                tag=tag,
                limit=10,
                difficulty=difficulty,
                api_key=api_key,
            )

            if error:
                errors.append(f"{tag}/{difficulty}: {error}")
                if "API key" in error:
                    return all_questions, errors
            elif questions:
                all_questions.extend(questions)
                print(f"    {difficulty}: {len(questions)} questions")

            # Rate limit
            time.sleep(0.5)

    return all_questions, errors


def scrape_all(api_key: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
    """
    Scrape all available questions from QuizAPI.

    Returns:
        Tuple of (all questions, list of errors)
    """
    print("Scraping QuizAPI...")
    return fetch_all_tags(api_key=api_key)


if __name__ == "__main__":
    print("Testing QuizAPI scraper...")
    if not API_KEY:
        print("Warning: QUIZAPI_KEY not set. Get a free key at https://quizapi.io/")
    questions, error = fetch_questions(tag="linux", limit=5)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Got {len(questions)} questions:")
        for q in questions:
            print(f"  - {q['question'][:60]}...")
            print(f"    Options: {q['options']}")
            print(f"    Answer: {q['answer']}")
