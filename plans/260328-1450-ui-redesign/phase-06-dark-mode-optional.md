# Phase 6: Dark Mode (Optional)

## Context
- Already using CSS variables for all colors -- ideal for dark mode swap
- Sidebar is already dark; main challenge is content area + panels

## Implementation Steps

### 1. Define dark palette in `:root` override
```css
@media (prefers-color-scheme: dark) {
  :root {
    --bg-main: #0f172a;
    --bg-panel: #1e293b;
    --ink: #e2e8f0;
    --ink-light: #94a3b8;
    --border: #334155;
    --border-light: #1e293b;
    --accent-subtle: rgba(99,102,241,0.15);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow: 0 4px 6px rgba(0,0,0,0.4);
  }
}
```

### 2. Manual toggle support
- Add `.dark` class on `<html>` as alternative to `prefers-color-scheme`
- Toggle button in header or settings tab
- Store preference in `localStorage`

### 3. Component adjustments
- Cards: remove white bg, use panel bg
- Inputs: darker bg `#0f172a`, lighter border
- Tables: darker zebra stripes
- Status badges: adjust bg opacity for dark bg

### 4. Images & icons
- Lucide icons inherit `currentColor` -- works automatically
- Any hardcoded colors in inline styles need audit

## Todo
- [ ] Add dark CSS variables via `prefers-color-scheme`
- [ ] Add manual `.dark` class toggle
- [ ] Adjust component-specific colors
- [ ] Test all tabs in dark mode
- [ ] Store preference in localStorage

## Success Criteria
- Automatic dark mode based on OS preference
- Manual override via toggle in settings
- No white flashes or unreadable text in dark mode

## Risk
- Low-medium. CSS variables make this straightforward. Main risk: inline styles in HTML that bypass variables.
