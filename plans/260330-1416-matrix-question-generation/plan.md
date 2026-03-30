---
title: "Matrix-based Question Generation (Ma tran de thi)"
description: "Allow teachers to define topic x difficulty matrix and batch-generate exam questions via AI"
status: pending
priority: P1
effort: 8h
branch: main
tags: [feature, ai, generation, exam-matrix]
created: 2026-03-28
---

# Matrix-based Question Generation

## Problem
Teachers must run "Tao theo chu de (AI)" multiple times to build one exam. Real Vietnamese exams follow a specification matrix (ma tran de thi) distributing questions across topics and difficulty levels.

## Solution
New tab "Tao theo ma tran" allowing users to build a topic x difficulty grid, then batch-generate all cells via AI in sequence.

## Architecture Overview
- **Frontend**: New tab section in `index.html` with dynamic matrix table
- **Backend**: New endpoint `POST /generate-matrix` that iterates matrix cells, calls existing `build_topic_prompt` + `call_ai` per cell, aggregates results
- **Schema**: New `MatrixCell` and `MatrixGenerateRequest` Pydantic models
- Reuses existing `build_topic_prompt`, `call_ai`, `normalize_ai_blocks`, RAG retrieval

## Phases

| # | Phase | Status | Effort |
|---|-------|--------|--------|
| 1 | [Backend: Schema + Endpoint](./phase-01-backend-schema-and-endpoint.md) | pending | 2h |
| 2 | [Frontend: Matrix UI Tab](./phase-02-frontend-matrix-ui-tab.md) | pending | 3h |
| 3 | [Integration + Bank/Export Support](./phase-03-integration-bank-export.md) | pending | 2h |
| 4 | [Testing + Polish](./phase-04-testing-and-polish.md) | pending | 1h |

## Key Dependencies
- Existing `build_topic_prompt` in `app/services/generation/prompt_builder.py`
- Existing `call_ai` in `app/services/ai/`
- Existing `normalize_ai_blocks` in `app/services/generation/`
- SUBJECTS / SUBJECT_TOPICS / QUESTION_TYPES in `app/core/constants.py`
- Current difficulty levels: easy/medium/hard (extend to add "very_hard" = "Van dung cao")

## Risk
- AI rate limits when generating many cells sequentially. Mitigation: sequential calls with progress feedback, max 50 total questions per matrix
- Large prompt context for many cells. Mitigation: one AI call per cell, not one massive prompt
