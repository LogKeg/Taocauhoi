---
phase: 5
title: "Manual Image Upload Support"
status: pending
effort: 0.5h
---

# Phase 5: Manual Image Upload Support

## Context

- Question add/edit modal exists in index.html
- "Them cau hoi" button triggers modal
- API endpoints in `app/api/routers/questions.py`

## Requirements

1. Add image upload input to question modal
2. Preview selected image before save
3. API accepts multipart form data with image
4. Clear/remove image option

## Related Code Files

**Modify:**
- `/Users/long/Downloads/Tạo đề online/app/templates/index.html` (question modal)
- `/Users/long/Downloads/Tạo đề online/app/api/routers/questions.py`

## Implementation Steps

### 1. Update question modal HTML

Find question add modal and add image input:

```html
<div class="form-group">
  <label for="questionImage">Hinh anh (tuy chon)</label>
  <input type="file" id="questionImage" accept="image/*" />
  <div id="questionImagePreview" style="display:none; margin-top:8px;">
    <img id="questionImageImg" style="max-width:200px; max-height:150px;" />
    <button type="button" id="clearQuestionImage" style="margin-left:8px;">Xoa</button>
  </div>
</div>
```

### 2. Add JavaScript for preview

```javascript
// Image preview
document.getElementById('questionImage').addEventListener('change', function(e) {
  const file = e.target.files[0];
  const preview = document.getElementById('questionImagePreview');
  const img = document.getElementById('questionImageImg');

  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      img.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    preview.style.display = 'none';
  }
});

// Clear image
document.getElementById('clearQuestionImage').addEventListener('click', function() {
  document.getElementById('questionImage').value = '';
  document.getElementById('questionImagePreview').style.display = 'none';
});
```

### 3. Update save question function

Change from JSON to FormData:

```javascript
async function saveQuestion() {
  const formData = new FormData();
  formData.append('content', document.getElementById('questionContent').value);
  formData.append('options', document.getElementById('questionOptions').value);
  formData.append('answer', document.getElementById('questionAnswer').value);
  formData.append('subject', document.getElementById('questionSubject').value);
  formData.append('grade', document.getElementById('questionGrade').value);
  formData.append('difficulty', document.getElementById('questionDifficulty').value);

  const imageFile = document.getElementById('questionImage').files[0];
  if (imageFile) {
    formData.append('image', imageFile);
  }

  const res = await fetch('/questions', {
    method: 'POST',
    body: formData,  // No Content-Type header, browser sets it with boundary
  });

  // ... handle response
}
```

### 4. Update API endpoint

```python
from fastapi import File, UploadFile

@router.post("/questions")
async def create_question(
    content: str = Form(...),
    options: str = Form(""),
    answer: str = Form(""),
    subject: str = Form("general"),
    grade: str = Form(""),
    difficulty: str = Form("medium"),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    # Create question
    question = Question(
        content=content,
        options=options,
        answer=answer,
        subject=subject,
        grade=grade,
        difficulty=difficulty,
    )
    db.add(question)
    db.flush()

    # Save image if provided
    if image and image.filename:
        from app.services.image import save_question_image
        image_bytes = await image.read()
        ext = Path(image.filename).suffix or ".png"
        image_url = save_question_image(question.id, image_bytes, f"main{ext}")
        question.image_url = image_url

    db.commit()
    db.refresh(question)

    return {"ok": True, "question": question}
```

### 5. Update edit question endpoint

Similar pattern for PUT /questions/{id}:
- Accept optional new image
- If new image, delete old and save new
- If clear flag, delete image and set image_url to None

## TODO

- [ ] Add image input to question modal HTML
- [ ] Add preview/clear JavaScript
- [ ] Convert save to FormData
- [ ] Update POST /questions API for multipart
- [ ] Update PUT /questions/{id} for image edit
- [ ] Test upload with various image types

## Success Criteria

- Image input visible in add question modal
- Preview shows selected image
- Clear button removes selection
- Save uploads image and sets image_url
- Edit can replace or remove image
