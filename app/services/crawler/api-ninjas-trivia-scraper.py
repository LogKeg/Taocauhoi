"""
Scraper for API Ninjas Trivia (api-ninjas.com/api/trivia) - General trivia questions.

Features:
- 100K+ questions (premium), 100 free
- Multiple categories
- Requires free API key

Note: Get free API key at https://api-ninjas.com/
Set environment variable: API_NINJAS_KEY
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
API_KEY = os.getenv("API_NINJAS_KEY", "")

# Map API Ninjas categories to our subject names
CATEGORY_MAP = {
    "artliterature": "literature",
    "language": "literature",
    "sciencenature": "science",
    "general": "general",
    "fooddrink": "general",
    "peopleplaces": "geography",
    "geography": "geography",
    "historyholidays": "history",
    "entertainment": "entertainment",
    "toysgames": "entertainment",
    "music": "music",
    "mathematics": "math",
    "religionmythology": "history",
    "sportsleisure": "sports",
}

# Available categories
AVAILABLE_CATEGORIES = [
    "artliterature",
    "language",
    "sciencenature",
    "general",
    "fooddrink",
    "peopleplaces",
    "geography",
    "historyholidays",
    "entertainment",
    "toysgames",
    "music",
    "mathematics",
    "religionmythology",
    "sportsleisure",
]


def fetch_questions(
    category: Optional[str] = None,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Fetch questions from API Ninjas Trivia.

    Args:
        category: Category name (optional, may not work on free tier)
        limit: Number of questions to fetch
        api_key: API key (uses env var if not provided)

    Returns:
        Tuple of (questions list, error message or None)

    Note: Free tier returns random questions. Category filter may not work.
    """
    key = api_key or API_KEY
    if not key:
        return [], "API key required. Set API_NINJAS_KEY environment variable."

    # Free tier doesn't support category filter well, just use base URL
    url = "https://api.api-ninjas.com/v1/trivia"
    headers = {"X-Api-Key": key}

    try:
        questions = []
        with httpx.Client(timeout=30) as client:
            # API returns 1 question per request, so we need multiple calls
            for _ in range(min(limit, 20)):
                response = client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                if not data:
                    continue

                item = data[0] if isinstance(data, list) else data
                question_text = item.get("question", "")
                answer_text = item.get("answer", "")
                cat = item.get("category", "") or "general"

                # API Ninjas returns open-ended questions (no multiple choice)
                questions.append({
                    "question": question_text,
                    "options": [],  # No options - open-ended
                    "answer": answer_text,
                    "subject": CATEGORY_MAP.get(cat.lower(), "general"),
                    "difficulty": "medium",  # API doesn't provide difficulty
                    "source": "api-ninjas.com",
                    "grade": "international",
                    "category_raw": cat,
                    "question_type": "short_answer",  # Mark as short answer
                })

                # Rate limit for free tier
                time.sleep(0.3)

        return questions, None

    except httpx.HTTPError as e:
        return [], f"HTTP error: {e}"
    except Exception as e:
        return [], f"Error: {e}"


def fetch_all_categories(
    max_per_category: int = 10,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Fetch questions from all available categories.

    Args:
        max_per_category: Max questions per category
        api_key: API key (uses env var if not provided)

    Returns:
        Tuple of (all questions, list of errors)
    """
    all_questions = []
    errors = []

    for category in AVAILABLE_CATEGORIES:
        print(f"  Fetching category: {category}...")

        questions, error = fetch_questions(
            category=category,
            limit=max_per_category,
            api_key=api_key,
        )

        if error:
            errors.append(f"{category}: {error}")
            if "API key" in error:
                return all_questions, errors
        elif questions:
            all_questions.extend(questions)
            print(f"    Got {len(questions)} questions")

        # Rate limit between categories
        time.sleep(1)

    return all_questions, errors


def scrape_all(api_key: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
    """
    Scrape all available questions from API Ninjas.

    Returns:
        Tuple of (all questions, list of errors)
    """
    print("Scraping API Ninjas Trivia...")
    return fetch_all_categories(max_per_category=5, api_key=api_key)


if __name__ == "__main__":
    print("Testing API Ninjas Trivia scraper...")
    if not API_KEY:
        print("Warning: API_NINJAS_KEY not set. Get a free key at https://api-ninjas.com/")
    questions, error = fetch_questions(category="sciencenature", limit=3)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Got {len(questions)} questions:")
        for q in questions:
            print(f"  - Q: {q['question'][:60]}...")
            print(f"    A: {q['answer']}")
            print(f"    Category: {q['category_raw']}")
