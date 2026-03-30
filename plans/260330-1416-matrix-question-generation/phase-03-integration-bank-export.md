# Phase 3: Integration - Bank + Export Support

**Priority:** P2 | **Status:** pending | **Effort:** 2h

## Overview
Ensure matrix-generated questions save to question bank with proper metadata (topic, difficulty per question) and export as structured exam documents.

## Key Insights
- Existing `_save_text_questions_to_bank` saves with flat subject/source/difficulty
- Matrix questions need per-question topic + difficulty metadata
- Export should produce formatted exam: title, sections by topic, answer key at end
- Existing ExamCreate schema can wrap matrix output as a saved exam

## Requirements

### Functional
- Auto-save each cell's questions to bank with correct topic + difficulty
- "Save as exam" option: create exam record linking all generated questions
- DOCX export: formatted with section headers per topic, numbered questions, answer key
- PDF export: same structure

### Non-functional
- Bank save happens server-side during generation (same pattern as generate-topic)

## Related Code Files

### Modify
- `app/api/routers/generation/matrix-generation-handler.py` - Add bank saving logic per cell
- `app/api/routers/generation/helpers.py` - May need enhanced save function with topic param
- `app/templates/index.html` - Export handler for matrix results

## Implementation Steps

### 1. Enhance bank saving in matrix handler
For each cell's generated questions, call `_save_text_questions_to_bank` with:
- `subject` from request
- `source` = "matrix-generated"
- `difficulty` = cell's difficulty
- Pass topic info via existing QuestionCRUD.create params

### 2. Add exam creation option
After generation completes, offer "Luu thanh de thi" button that:
- Creates exam record via existing ExamCreate
- Associates generated question IDs with exam

### 3. Structured export
Matrix results need custom export formatting:
- Title: "De thi [subject] - Lop [grade]"
- Per topic section: "Phan [n]: [topic_label]" with subsections by difficulty
- Numbered questions across entire exam
- Answer key at end: grouped by section

### 4. Frontend export handler
- Collect all questions from matrix result sections
- Number them sequentially
- Build text/DOCX/PDF with section headers

## Todo List
- [ ] Save questions to bank with per-cell topic + difficulty
- [ ] Add "Luu thanh de thi" button
- [ ] Implement structured DOCX export for matrix
- [ ] Implement structured PDF export for matrix
- [ ] Test bank saving with correct metadata

## Success Criteria
- Questions in bank show correct topic + difficulty from matrix cell
- Exported DOCX has proper section structure
- Exam record links to all generated questions
