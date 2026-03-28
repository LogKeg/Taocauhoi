# Phase 4: Micro-interactions & Transitions

## Context
- Current: only `transition: all 0.15s ease` on a few elements
- Tab switches are instant (no fade), modals pop in without animation

## Implementation Steps

### 1. Global transition base
```css
* { transition-property: background-color, border-color, color, box-shadow, opacity, transform; }
```
No -- too broad. Instead, add explicit transitions to interactive elements only.

### 2. Tab content fade-in
```css
.tab-panel.active {
  animation: fadeSlideIn 0.2s ease-out;
}
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### 3. Card/panel entrance stagger
```css
.panel { animation: fadeIn 0.3s ease-out both; }
.panel:nth-child(2) { animation-delay: 0.05s; }
.panel:nth-child(3) { animation-delay: 0.1s; }
```

### 4. Button press feedback
```css
button:active, .btn:active {
  transform: scale(0.97);
  transition-duration: 0.05s;
}
```

### 5. Sidebar nav-item transition
- Already has 0.15s; add subtle `transform: translateX(2px)` on hover

### 6. Focus ring animation
```css
input:focus, select:focus, textarea:focus {
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
```

### 7. Loading states
- Add `.skeleton` class for loading placeholders
- Shimmer animation: gradient sweep left-to-right

### 8. Smooth scrollbar styling
```css
.content-scroll::-webkit-scrollbar { width: 6px; }
.content-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
```

## Todo
- [ ] Add tab panel fade-in animation
- [ ] Add card entrance stagger
- [ ] Refine button press feedback
- [ ] Add sidebar hover translateX
- [ ] Style scrollbars
- [ ] Add skeleton loading class

## Success Criteria
- Tab switches feel smooth, not jarring
- All interactive elements have perceptible feedback
- No animation longer than 300ms (keep snappy)

## Risk
- Low. CSS-only animations; no JS perf cost. Respect `prefers-reduced-motion`.
