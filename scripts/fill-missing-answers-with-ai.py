#!/usr/bin/env python3
"""
Script to fill missing answers in question bank using AI.

Usage:
    python scripts/fill-missing-answers-with-ai.py --engine ollama --subject math --batch 10
    python scripts/fill-missing-answers-with-ai.py --engine ollama --all --batch 50
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from sqlalchemy import or_

from app.database import SessionLocal
from app.database.models import Question


def load_ai_settings() -> dict:
    """Load AI settings from file"""
    settings_file = Path("ai_settings.json")
    if settings_file.exists():
        with open(settings_file) as f:
            return json.load(f)
    return {}


async def call_ai_engine(engine: str, prompt: str, settings: dict) -> str:
    """Call AI engine to get answer"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        if engine == "ollama":
            base_url = settings.get("ollama_base", "http://localhost:11434")
            model = settings.get("ollama_model", "llama3.1:8b")

            response = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]

        elif engine == "gemini":
            api_key = settings.get("gemini_key", "")
            model = settings.get("gemini_model", "gemini-2.0-flash")

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        elif engine == "openai":
            api_key = settings.get("openai_key", "")
            base_url = settings.get("openai_base", "https://api.openai.com/v1")
            model = settings.get("openai_model", "gpt-4o-mini")

            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        elif engine == "claude":
            api_key = settings.get("anthropic_key", "")
            base_url = settings.get("anthropic_base", "https://api.anthropic.com")
            model = settings.get("anthropic_model", "claude-sonnet-4-20250514")

            response = await client.post(
                f"{base_url}/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]

        else:
            raise ValueError(f"Unknown engine: {engine}")


def build_prompt(question) -> str:
    """Build prompt for AI to answer the question"""
    options_text = ""
    if question.options:
        try:
            opts = json.loads(question.options) if isinstance(question.options, str) else question.options
            for i, opt in enumerate(opts):
                label = chr(65 + i)  # A, B, C, D
                # Clean option text
                opt_clean = opt.strip()[:200]  # Limit length
                options_text += f"\n{label}. {opt_clean}"
        except:
            options_text = str(question.options)[:500]

    # Clean question content
    content = question.content.strip()[:500]

    prompt = f"""Xác định đáp án đúng cho câu hỏi trắc nghiệm sau.

Câu hỏi: {content}

Các lựa chọn:{options_text}

Môn học: {question.subject}

CHỈ trả về MỘT chữ cái (A, B, C hoặc D) là đáp án đúng. KHÔNG giải thích."""

    return prompt


async def process_batch(questions: list, engine: str, settings: dict) -> tuple:
    """Process a batch of questions"""
    updated = 0
    errors = []

    for q in questions:
        try:
            prompt = build_prompt(q)
            response = await call_ai_engine(engine, prompt, settings)

            # Extract answer
            answer_match = re.search(r'\b([A-D])\b', response.upper())
            if answer_match:
                q.answer = answer_match.group(1)
                updated += 1
            else:
                errors.append(f"ID {q.id}: Cannot extract answer from: {response[:50]}")
        except Exception as e:
            errors.append(f"ID {q.id}: {str(e)[:100]}")

    return updated, errors


async def main():
    parser = argparse.ArgumentParser(description="Fill missing answers using AI")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "gemini", "openai", "claude"])
    parser.add_argument("--subject", help="Filter by subject")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--batch", type=int, default=10, help="Batch size")
    parser.add_argument("--max", type=int, default=0, help="Max questions to process (0=unlimited)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between batches")
    args = parser.parse_args()

    if not args.all and not args.subject:
        print("Error: Specify --subject or --all")
        return

    settings = load_ai_settings()
    db = SessionLocal()

    try:
        # Build query
        query = db.query(Question).filter(
            or_(Question.answer == None, Question.answer == "")
        )

        if args.subject:
            query = query.filter(Question.subject == args.subject)

        total_missing = query.count()
        print(f"Found {total_missing} questions without answers")

        if total_missing == 0:
            print("Nothing to process!")
            return

        # Process in batches
        processed = 0
        total_updated = 0
        all_errors = []

        while True:
            # Get batch
            questions = query.limit(args.batch).all()
            if not questions:
                break

            batch_updated, batch_errors = await process_batch(questions, args.engine, settings)

            # Commit changes
            db.commit()

            processed += len(questions)
            total_updated += batch_updated
            all_errors.extend(batch_errors)

            # Progress
            remaining = query.count()
            print(f"Processed: {processed}, Updated: {total_updated}, Remaining: {remaining}, Errors: {len(all_errors)}")

            # Check max limit
            if args.max > 0 and processed >= args.max:
                print(f"Reached max limit of {args.max}")
                break

            # Delay between batches
            if remaining > 0:
                await asyncio.sleep(args.delay)

        print(f"\n=== DONE ===")
        print(f"Total processed: {processed}")
        print(f"Total updated: {total_updated}")
        print(f"Total errors: {len(all_errors)}")

        if all_errors[:10]:
            print("\nSample errors:")
            for err in all_errors[:10]:
                print(f"  - {err}")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
