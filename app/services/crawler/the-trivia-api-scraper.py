"""
Scraper for The Trivia API (the-trivia-api.com) - Free trivia API with multiple categories.

Features:
- Free for non-commercial use
- Multiple categories and difficulties
- No API key required for basic usage
"""
import time
from typing import Dict, List, Optional, Tuple

import httpx

# Map The Trivia API categories to our subject names
CATEGORY_MAP = {
    "science": "science",
    "science_and_nature": "science",
    "general_knowledge": "general",
    "geography": "geography",
    "history": "history",
    "arts_and_literature": "literature",
    "music": "music",
    "film_and_tv": "entertainment",
    "sport_and_leisure": "sports",
    "food_and_drink": "general",
    "society_and_culture": "general",
}

# Available categories from the API
AVAILABLE_CATEGORIES = [
    "arts_and_literature",
    "film_and_tv",
    "food_and_drink",
    "general_knowledge",
    "geography",
    "history",
    "music",
    "science",
    "society_and_culture",
    "sport_and_leisure",
]

# Difficulty mapping
DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def fetch_questions(
    categories: Optional[List[str]] = None,
    limit: int = 10,
    difficulty: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch questions from The Trivia API.

    Args:
        categories: List of category names (optional)
        limit: Number of questions (max 50 per request)
        difficulty: easy, medium, or hard (optional)

    Returns:
        Tuple of (questions list, error message or None)
    """
    url = f"https://the-trivia-api.com/v2/questions?limit={min(limit, 50)}"

    if categories:
        url += f"&categories={','.join(categories)}"
    if difficulty:
        url += f"&difficulties={difficulty}"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            questions = []
            for item in data:
                question_text = item.get("question", {}).get("text", "")
                correct = item.get("correctAnswer", "")
                incorrect = item.get("incorrectAnswers", [])
                category = item.get("category", "general_knowledge")
                diff = item.get("difficulty", "medium")

                # Build options list
                options = incorrect + [correct]
                options = sorted(options)

                # Find correct answer index
                answer_idx = options.index(correct) if correct in options else 0
                answer_letter = chr(65 + answer_idx)  # A, B, C, D

                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": answer_letter,
                    "subject": CATEGORY_MAP.get(category, "general"),
                    "difficulty": DIFFICULTY_MAP.get(diff, "medium"),
                    "source": "the-trivia-api.com",
                    "grade": "international",
                    "category_raw": category,
                })

            return questions, None

    except httpx.HTTPError as e:
        return [], f"HTTP error: {e}"
    except Exception as e:
        return [], f"Error: {e}"


def fetch_all_categories(max_per_category: int = 50) -> Tuple[List[Dict], List[str]]:
    """
    Fetch questions from all available categories.

    Args:
        max_per_category: Max questions per category

    Returns:
        Tuple of (all questions, list of errors)
    """
    all_questions = []
    errors = []

    for category in AVAILABLE_CATEGORIES:
        print(f"  Fetching category: {category}...")

        for difficulty in ["easy", "medium", "hard"]:
            questions, error = fetch_questions(
                categories=[category],
                limit=min(20, max_per_category // 3),
                difficulty=difficulty,
            )

            if error:
                errors.append(f"{category}/{difficulty}: {error}")
            elif questions:
                all_questions.extend(questions)
                print(f"    {difficulty}: {len(questions)} questions")

            # Rate limit
            time.sleep(0.5)

    return all_questions, errors


def scrape_all() -> Tuple[List[Dict], List[str]]:
    """
    Scrape all available questions from The Trivia API.

    Returns:
        Tuple of (all questions, list of errors)
    """
    print("Scraping The Trivia API...")
    return fetch_all_categories(max_per_category=60)


if __name__ == "__main__":
    print("Testing The Trivia API scraper...")
    questions, error = fetch_questions(limit=5)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Got {len(questions)} questions:")
        for q in questions:
            print(f"  - {q['question'][:60]}...")
            print(f"    Options: {q['options']}")
            print(f"    Answer: {q['answer']}, Category: {q['category_raw']}")
