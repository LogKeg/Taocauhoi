---
phase: 3
title: "Frontend Image Display"
status: pending
effort: 1.5h
---

# Phase 3: Frontend Image Display

## Context

- Main template: `app/templates/index.html`
- KaTeX already loaded for LaTeX rendering
- `renderQuestionCard()` at line ~1158
- `bank-item` template at line ~2030
- Static files mounted at `/static`

## Requirements

1. Display images in question cards
2. Show thumbnails in question bank list
3. Full image in question detail modal
4. Serve images from uploads directory

## Related Code Files

**Modify:**
- `/Users/long/Downloads/Tạo đề online/app/main.py` (mount uploads)
- `/Users/long/Downloads/Tạo đề online/app/templates/index.html`

## Implementation Steps

### 1. Mount uploads directory (main.py)

After line 48:
```python
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
```

### 2. Update renderQuestionCard function (index.html ~line 1158)

```javascript
function renderQuestionCard(qText, options, answer, index, imageUrl = null) {
  const safeText = escapeHtml(qText).replace(/\n/g, '<br>');
  const imageHtml = imageUrl
    ? `<div class="question-image-container">
         <img src="/uploads/${imageUrl}" class="question-image" alt="Question image" loading="lazy" />
       </div>`
    : '';
  return `
    <div class="created-question" data-index="${index}" data-image="${imageUrl || ''}">
      <div class="q-header">
        <span class="q-number">Cau ${index + 1}</span>
        <div class="q-actions">
          <button class="edit-q-btn">Sua</button>
          <button class="add-to-bank-btn">Them vao ngan hang</button>
        </div>
      </div>
      ${imageHtml}
      <div class="q-content">
        <div class="q-text-display">${safeText}</div>
        ...
      </div>
    </div>`;
}
```

### 3. Update callers of renderQuestionCard

Line ~1264 (topic generation):
```javascript
return renderQuestionCard(questionText, optionLines, answerLine, i, q.image_url);
```

Line ~1501 (create from template):
```javascript
return renderQuestionCard(qText, opts, ans, i, q.image_url);
```

### 4. Update bank-item template (~line 2030)

```javascript
list.innerHTML = (data.questions || []).map(q => `
  <div class="bank-item" data-id="${q.id}">
    ${q.image_url ? `<img src="/uploads/${q.image_url}" class="bank-item-thumb" alt="" loading="lazy" />` : ''}
    <div class="bank-item-content">
      <strong>${q.subject || 'N/A'}</strong> -
      <span class="difficulty-${q.difficulty || 'medium'}">${q.difficulty || 'medium'}</span>
      ${q.grade ? `<span style="color:#6b7280;font-size:0.85em;">(Lop ${q.grade})</span>` : ''}
      <p>${escapeHtml((q.content || '').substring(0, 150))}${(q.content||'').length > 150 ? '...' : ''}</p>
    </div>
    <div class="bank-item-actions">
      <button class="edit-q" data-id="${q.id}">Sua</button>
      <button class="delete-q" data-id="${q.id}">Xoa</button>
    </div>
  </div>
`).join('') || '<p>Khong co cau hoi nao.</p>';
```

### 5. Add CSS styles

Add to existing styles or inline:
```css
.question-image-container {
  margin: 8px 0;
  text-align: center;
}

.question-image {
  max-width: 100%;
  max-height: 400px;
  border-radius: 4px;
  border: 1px solid var(--border);
}

.bank-item-thumb {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 4px;
  flex-shrink: 0;
}

.bank-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
```

### 6. Update question detail modal

In `openQuestionModal()` function, add image display:
```javascript
${q.image_url ? `<img src="/uploads/${q.image_url}" class="question-image" alt="" />` : ''}
```

## TODO

- [ ] Mount /uploads in main.py
- [ ] Update renderQuestionCard signature
- [ ] Update all renderQuestionCard callers
- [ ] Update bank-item template
- [ ] Add CSS for images
- [ ] Update question modal display
- [ ] Test with sample images

## Success Criteria

- Images display in generated questions
- Thumbnails show in bank list
- Full image in question modal
- Lazy loading works
- No layout breaking
