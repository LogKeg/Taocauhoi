---
phase: 1
title: "Database Schema Migration"
status: pending
effort: 1h
---

# Phase 1: Database Schema Migration

## Context

- Current schema: `app/database/models.py` (line 45-79)
- DB file: `question_bank.db`
- No image columns exist currently

## Requirements

Add `image_url` column to store relative path to image file.

## Architecture

```
Question model
├── id (existing)
├── content (existing)
├── image_url (NEW) → "images/123/main.png"
└── ... other fields
```

## Related Code Files

**Modify:**
- `/Users/long/Downloads/Tạo đề online/app/database/models.py`

## Implementation Steps

1. Add column to SQLAlchemy model:
```python
# After line 67 (source column)
image_url = Column(Text, nullable=True)  # Relative path: images/{id}/filename.png
```

2. Run migration on existing database:
```bash
sqlite3 question_bank.db "ALTER TABLE questions ADD COLUMN image_url TEXT;"
```

3. Update CRUD if needed (should work automatically via **kwargs)

## TODO

- [ ] Add `image_url` column to Question model in models.py
- [ ] Run ALTER TABLE on question_bank.db
- [ ] Verify existing questions unaffected
- [ ] Test create/update with image_url field

## Success Criteria

- Column exists in DB schema
- Existing questions have NULL image_url
- New questions can set image_url
- No errors on app startup
