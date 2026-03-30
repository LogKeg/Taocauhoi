# Phase 2: Frontend - Matrix UI Tab

**Priority:** P1 | **Status:** pending | **Effort:** 3h

## Overview
New sidebar tab "Tao theo ma tran" with interactive matrix builder. User selects subject/grade, adds topic rows, sets question counts per difficulty column, then generates.

## Key Insights
- App uses pure HTML/JS, no framework. All UI in single `index.html`
- Existing tab-topic pattern: sidebar nav-item + tab-panel section
- JS is inline at bottom of index.html
- Existing pattern: form fields in `.auto-grid`, results in `.panel.result-panel`
- Difficulty columns: Nhan biet (easy), Thong hieu (medium), Van dung (hard), Van dung cao (very_hard)

## Requirements

### Functional
- Sidebar nav button "Ma tran de" in "Tao de" section
- Subject, grade, qtype, language, engine, RAG toggle (reuse same pattern as tab-topic)
- Dynamic matrix table:
  - Columns: Chu de | Nhan biet | Thong hieu | Van dung | Van dung cao | Tong
  - Rows: one per selected topic (add/remove buttons)
  - Each cell = number input (default 0)
  - Footer row: column totals (auto-calculated)
- "Tao de theo ma tran" button
- Progress indicator showing which cell is being generated
- Result display: grouped by topic, each section showing difficulty label + questions
- Export buttons (TXT, DOCX, PDF) for full exam
- "Them tat ca vao ngan hang" button

### Non-functional
- Max 8 topic rows
- Responsive: matrix scrolls horizontally on mobile
- Total question count displayed and validated (max 50)

## Related Code Files

### Modify
- `app/templates/index.html` - Add nav-item + tab-panel section + JS logic

## Implementation Steps

### 1. Add sidebar nav button
After the "Tao theo chu de (AI)" button, add:
```html
<button class="nav-item" data-tab="tab-matrix">Ma tran de</button>
```

### 2. Add tab-panel section
Between `tab-topic` and `tab-convert` sections, add `#tab-matrix` panel containing:
- Header config: subject, grade, qtype, language, engine, RAG toggle (copy pattern from tab-topic)
- Matrix builder area:
  - "Them chu de" button that appends a row
  - Each row: topic dropdown (populated from SUBJECT_TOPICS[subject]) + 4 number inputs + row total + remove button
  - Footer: total per column + grand total
- Generate button
- Result panel (hidden initially)

### 3. JS: Matrix state management
```js
// matrixRows = [{ topicKey, easy, medium, hard, very_hard }]
// Functions: addMatrixRow(), removeMatrixRow(idx), updateMatrixTotals()
// On subject change: repopulate topic dropdowns
```

### 4. JS: Generate handler
- Collect cells from matrix (skip cells with count=0)
- Build FormData or JSON body for POST /generate-matrix
- Show progress: "Dang tao: [topic] - [difficulty] (x/y)..."
- On response: render sections grouped by topic/difficulty
- Show answers + explanations toggles per section

### 5. JS: Export integration
- Reuse existing export logic (DOCX/PDF/TXT) applied to matrix results
- Format: exam title + sections by topic + answers at end

### 6. CSS
- `.matrix-table` - bordered table with number inputs
- `.matrix-row-total`, `.matrix-col-total` - bold totals
- `.matrix-progress` - generation progress bar/text

## Todo List
- [ ] Add sidebar nav-item for matrix tab
- [ ] Create tab-panel HTML structure with config fields
- [ ] Build dynamic matrix table with add/remove rows
- [ ] Implement topic dropdown population on subject change
- [ ] Auto-calculate row and column totals
- [ ] Implement generate button handler with progress
- [ ] Render results grouped by topic/difficulty
- [ ] Add export buttons
- [ ] Add "add all to bank" button
- [ ] Mobile-responsive table scrolling
- [ ] Validate total <= 50 before submit

## Success Criteria
- User can build a matrix with 3+ topics, set counts, see totals
- Generate button calls backend, shows progress, displays grouped results
- Export produces properly structured exam document

## Risk Assessment
- Single HTML file is already large. If JS exceeds 200 lines for matrix logic, extract to separate `static/js/matrix.js`
- Topic dropdown must update when subject changes (same pattern as tab-topic)
