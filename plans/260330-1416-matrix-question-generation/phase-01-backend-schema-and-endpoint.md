# Phase 1: Backend - Schema + Endpoint

**Priority:** P1 | **Status:** pending | **Effort:** 2h

## Overview
Add Pydantic models for matrix input and a new endpoint that iterates cells, generating questions per cell using existing AI pipeline.

## Key Insights
- Current `generate-topic` endpoint already handles single (subject, topic, difficulty, count) generation
- We reuse `build_topic_prompt`, `call_ai`, `normalize_ai_blocks`, `retrieve_similar_questions`
- Each matrix cell = one AI call, keeping prompts focused and reliable
- Add "very_hard" difficulty level to match Vietnamese exam standards (Nhan biet / Thong hieu / Van dung / Van dung cao)

## Requirements

### Functional
- Accept matrix definition: subject, grade, qtype, language, engine, list of cells (topic + difficulty + count)
- Generate questions cell by cell, return aggregated result grouped by topic and difficulty
- Support RAG toggle per request (not per cell)
- Return answers and explanations alongside questions

### Non-functional
- Max 50 total questions per matrix request
- Sequential AI calls (no parallel) to avoid rate limits
- Each cell capped at 15 questions

## Related Code Files

### Modify
- `app/core/schemas.py` - Add `MatrixCell`, `MatrixGenerateRequest`
- `app/core/constants.py` - Add "very_hard" to difficulty maps in prompt_builder usage
- `app/services/generation/prompt_builder.py` - Add "very_hard" difficulty mapping
- `app/api/routers/generation/question-generation-endpoints.py` - Add `POST /generate-matrix`

### Create
- `app/api/routers/generation/matrix-generation-handler.py` - Core logic for matrix iteration (keep endpoint file clean)

## Implementation Steps

### 1. Add difficulty "very_hard"
In `app/services/generation/prompt_builder.py`:
- `difficulty_map_vi`: add `"very_hard": "rất khó (vận dụng cao)"`
- `difficulty_map_en`: add `"very_hard": "very hard/advanced application"`

### 2. Add Pydantic models in `app/core/schemas.py`
```python
class MatrixCell(BaseModel):
    topic: str = ""          # topic key from SUBJECT_TOPICS
    difficulty: str          # easy/medium/hard/very_hard
    count: int               # number of questions for this cell

class MatrixGenerateRequest(BaseModel):
    subject: str
    grade: int = 1
    qtype: str = "mcq"
    language: str = "vi"
    ai_engine: str = "ollama"
    use_rag: bool = True
    rag_count: int = 5
    cells: List[MatrixCell]
```

### 3. Export new models from `app/core/__init__.py`

### 4. Create `matrix-generation-handler.py`
Logic:
- Validate total count <= 50
- For each cell in `cells`:
  - Call `retrieve_similar_questions` if use_rag
  - Call `build_topic_prompt(subject, grade, qtype, cell.count, cell.topic, cell.difficulty, ...)`
  - Call `call_ai(prompt, ai_engine)`
  - Parse response: split questions, extract answers, extract explanations
  - Collect into result structure
- Return `{ sections: [{ topic, difficulty, questions, answers, explanations }], total_count }`

### 5. Add endpoint in `question-generation-endpoints.py`
```python
@router.post("/generate-matrix")
def generate_matrix(payload: MatrixGenerateRequest) -> dict:
    # delegate to matrix-generation-handler
```

## Todo List
- [ ] Add "very_hard" difficulty to prompt_builder
- [ ] Create MatrixCell + MatrixGenerateRequest schemas
- [ ] Export from core __init__
- [ ] Create matrix-generation-handler.py with cell iteration logic
- [ ] Add /generate-matrix endpoint
- [ ] Test with curl/httpie

## Success Criteria
- `POST /generate-matrix` with a 3-cell payload returns structured sections with questions per cell
- "very_hard" difficulty produces appropriate prompts
- Total question cap enforced

## Security Considerations
- Input validation: count per cell 1-15, total 1-50, grade 1-12
- Engine availability check before starting generation
