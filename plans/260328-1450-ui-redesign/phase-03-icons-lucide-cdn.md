# Phase 3: Replace Emoji with Lucide Icons

## Context
- Current sidebar nav uses emoji icons (lines 38-83 in index.html)
- Emoji render inconsistently across OS/browsers and look unprofessional
- Lucide is ~70KB CDN, tree-shakeable, MIT license, 1500+ icons

## Key Decision
**Lucide via CDN** (`lucide.dev`): lightweight, no build step needed, consistent SVG icons.

```html
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
```
Then call `lucide.createIcons()` after DOM load.

## Icon Mapping

| Tab | Current | Lucide Icon |
|-----|---------|-------------|
| Tu de mau | clipboard | `file-text` |
| Theo chu de (AI) | robot | `brain` |
| Word -> Excel | arrows | `file-spreadsheet` |
| Ngan hang | books | `database` |
| Import | inbox | `upload` |
| Kho luu tru | folder | `archive` |
| Khung chuong trinh | book | `book-open` |
| OMR Scanner | check | `scan-line` |
| Viet tay | writing | `pen-tool` |
| Lich su | clock | `clock` |
| Cai dat | gear | `settings` |
| Blob | cloud | `cloud` |

## Implementation Steps

### 1. Add Lucide CDN to `<head>` in index.html
```html
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
```

### 2. Replace emoji spans with Lucide `<i>` tags
```html
<!-- Before -->
<span class="icon">gear-emoji</span>Cai dat
<!-- After -->
<i data-lucide="settings"></i>Cai dat
```

### 3. Initialize icons in JS
Add `lucide.createIcons()` call after DOMContentLoaded.

### 4. Style icon sizing
```css
.nav-item [data-lucide] {
  width: 18px;
  height: 18px;
  stroke-width: 2;
  flex-shrink: 0;
}
```

### 5. Use icons elsewhere
- Button icons (export, delete, add)
- Status indicators
- Empty states

## Todo
- [ ] Add Lucide CDN script tag
- [ ] Replace all emoji `<span class="icon">` with `<i data-lucide="...">`
- [ ] Add CSS sizing rules for Lucide icons
- [ ] Call `lucide.createIcons()` on load
- [ ] Add icons to key buttons (export, upload, delete, add)

## Success Criteria
- Zero emoji icons in navigation
- Consistent 18px mono-weight SVG icons throughout
- Icons render identically on Mac, Windows, Linux

## Risk
- Low. CDN dependency; could self-host the JS file as fallback.
- Lucide `createIcons()` must be called after any dynamic DOM updates (tab switches adding new elements).
