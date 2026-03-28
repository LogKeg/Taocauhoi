# Phase 5: Responsive & Mobile Layout

## Context
- Current breakpoints: 900px (sidebar shrinks), 768px (minimal), 480px (chat only)
- Sidebar never fully collapses; no hamburger menu
- Form grids don't adapt well below 600px

## Implementation Steps

### 1. Mobile sidebar overlay (<768px)
```css
@media (max-width: 768px) {
  .sidebar {
    position: fixed; left: -260px; top: 56px; bottom: 0;
    z-index: 90; transition: left 0.3s ease;
    box-shadow: 4px 0 20px rgba(0,0,0,0.3);
  }
  .sidebar.open { left: 0; }
}
```
- Add hamburger button in header (hidden on desktop)
- JS: toggle `.open` class on sidebar

### 2. Header responsive
- Shrink logo text on mobile; hide "San sang" status text
- Hamburger icon left-aligned

### 3. Form grid breakpoints
```css
@media (max-width: 600px) {
  .form-grid { grid-template-columns: 1fr; }
  .form-inline { flex-direction: column; align-items: stretch; }
  .btn-group { flex-direction: column; }
}
```

### 4. Panel padding reduction on mobile
```css
@media (max-width: 600px) {
  .panel { padding: var(--space-md); }
  .content-scroll { padding: var(--space-md); }
}
```

### 5. Card grid single column on mobile
```css
@media (max-width: 600px) {
  .card-grid { grid-template-columns: 1fr; }
}
```

### 6. Touch targets
- Minimum 44px height for all interactive elements on mobile
- Larger checkboxes/radio buttons

### 7. Add sidebar overlay backdrop
```css
.sidebar-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.4);
  z-index: 89; display: none;
}
.sidebar.open ~ .sidebar-backdrop { display: block; }
```

## Todo
- [ ] Implement mobile sidebar with slide-in animation
- [ ] Add hamburger toggle button to header
- [ ] Fix form-grid to single column on mobile
- [ ] Reduce padding on mobile
- [ ] Ensure 44px minimum touch targets
- [ ] Add sidebar backdrop overlay
- [ ] Test on iOS Safari + Chrome Android

## Success Criteria
- Usable on 375px wide screens (iPhone SE)
- Sidebar slides in/out smoothly
- All forms fillable on mobile without horizontal scroll

## Risk
- Medium. JS change needed for hamburger toggle. Must not break existing tab-switching logic.
- Fixed positioning + z-index may conflict with modals.
