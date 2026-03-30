# Phase 4: Testing + Polish

**Priority:** P2 | **Status:** pending | **Effort:** 1h

## Overview
End-to-end testing of matrix generation flow and UX polish.

## Implementation Steps

### 1. Backend testing
- Test `/generate-matrix` with various matrix sizes (1 cell, 5 cells, max cells)
- Test validation: total > 50 rejected, empty cells skipped, invalid difficulty rejected
- Test with each AI engine (ollama, openai, gemini)
- Test RAG on/off

### 2. Frontend testing
- Verify matrix table add/remove rows works
- Verify totals update correctly
- Verify progress display during generation
- Verify results render grouped properly
- Verify exports produce correct structure

### 3. Edge cases
- All cells have count=0 -> show validation message
- AI returns partial results for a cell -> include what was generated, show warning
- Engine unavailable -> show error before starting any generation
- Subject with no topics defined -> allow "all topics" row

### 4. UX polish
- Loading spinner per cell during generation
- Disable generate button during process
- Clear previous results when starting new generation
- Auto-scroll to results when done
- Keyboard navigation in matrix inputs (Tab moves to next cell)

## Todo List
- [ ] Test backend endpoint with various payloads
- [ ] Test frontend matrix interactions
- [ ] Test edge cases
- [ ] Polish UX (loading states, validation messages)
- [ ] Test exports

## Success Criteria
- Full flow works: build matrix -> generate -> view results -> export
- No console errors, no unhandled API errors
- Responsive on mobile
