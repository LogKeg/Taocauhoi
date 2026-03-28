# Phase 2: Component Polish (Buttons, Inputs, Cards, Tables)

## Context
- Files: `app/static/styles.css` lines 260-600 (panels, forms, buttons, badges)
- Current: solid foundation with gradients, but flat inputs, no hover depth on cards, tables unstyled

## Implementation Steps

### 1. Buttons
- Add `active` state: `transform: scale(0.97)` for tactile feedback
- Refine gradient: softer gradient angle (to bottom right), reduce shadow spread
- Add `disabled` styles: `opacity: 0.5; cursor: not-allowed; pointer-events: none`
- `.btn-ghost`: transparent bg, accent text, accent border on hover

### 2. Inputs & Selects
- Increase padding to `11px 14px` for better touch target
- Add subtle `background: var(--bg-main)` on unfocused state (instead of white)
- Focus ring: use `box-shadow: 0 0 0 3px rgba(99,102,241,0.15)` (softer)
- Add `placeholder` color styling: `color: var(--ink-light); opacity: 0.6`

### 3. Cards & Panels
- Add subtle top-border accent on `.panel`: `border-top: 3px solid var(--accent)` (optional per panel)
- Refine hover: `transform: translateY(-2px)` + enhanced shadow
- Add `.panel-flush`: variant with no padding (for tables inside panels)

### 4. Tables
- Add base table styles (currently missing explicit styles):
  ```css
  table { width: 100%; border-collapse: collapse; }
  th { background: var(--bg-main); font-weight: 600; text-align: left; }
  td, th { padding: 12px 16px; border-bottom: 1px solid var(--border-light); }
  tr:hover td { background: var(--accent-subtle); }
  ```
- Zebra striping: `tr:nth-child(even) { background: var(--bg-main); }`

### 5. Upload Areas
- Dashed border: `border: 2px dashed var(--border)` with larger radius
- Hover state: border-color accent, light accent bg
- Drag-over state via JS class `.drag-over`: accent border + bg tint

### 6. Modals
- Add backdrop blur: `backdrop-filter: blur(4px)`
- Smoother entrance: scale(0.95) -> scale(1) with opacity

## Todo
- [ ] Polish button states (active, disabled, ghost)
- [ ] Refine input focus styles
- [ ] Enhance card/panel hover effects
- [ ] Add table base styles
- [ ] Improve upload area styling
- [ ] Add modal backdrop blur + animation

## Success Criteria
- Every interactive element has visible hover/focus/active states
- Tables are readable with clear row separation
- Modals feel smooth and layered

## Risk
- Medium. Button style override `button { }` affects ALL buttons globally; must ensure nav-items and other button elements excluded via specificity or `:not()`.
