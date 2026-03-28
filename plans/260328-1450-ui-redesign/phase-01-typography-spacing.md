# Phase 1: Typography & Spacing Refinement

## Context
- File: `app/static/styles.css` (lines 5-50 for `:root` variables)
- Current: Inter font, 14px base, basic spacing via padding values

## Key Insights
- Spacing is inconsistent (mix of 12/16/20/24px without clear scale)
- No typographic scale defined; heading sizes set ad-hoc per component
- Line-height 1.5 is fine but letter-spacing not tuned for headings

## Implementation Steps

### 1. Add spacing scale to `:root`
```css
--space-xs: 4px;
--space-sm: 8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;
--space-2xl: 48px;
```

### 2. Add typographic scale
```css
--text-xs: 0.75rem;    /* 12px */
--text-sm: 0.8125rem;  /* 13px */
--text-base: 0.875rem; /* 14px */
--text-lg: 1rem;       /* 16px */
--text-xl: 1.25rem;    /* 20px */
--text-2xl: 1.5rem;    /* 24px */
```

### 3. Refine heading styles
- Panel h2: use `--text-xl`, `font-weight: 700`, `letter-spacing: -0.01em`
- Panel h3: use `--text-lg`, `font-weight: 600`
- Add subtle color to section titles (use accent for emphasis)

### 4. Normalize spacing
- Replace hardcoded padding values with spacing variables
- Consistent `--space-lg` (24px) for panel padding
- `--space-md` (16px) for form gaps
- `--space-sm` (8px) for tight element gaps

### 5. Improve body text readability
- Paragraph max-width: `65ch` inside panels
- `color: var(--ink-light)` for secondary text with `line-height: 1.6`

## Todo
- [ ] Define spacing + type scale in `:root`
- [ ] Update heading styles across panels
- [ ] Normalize padding/margin to use scale variables
- [ ] Test all 10+ tabs for visual consistency

## Success Criteria
- Consistent vertical rhythm across all tabs
- Clear visual hierarchy: h2 > h3 > body > caption
- No hardcoded spacing values outside `:root` definitions

## Risk
- Low. Variable changes propagate automatically. Manual check each tab after.
