---
title: "UI/UX Polish - Pure CSS Redesign"
description: "Professional UI polish for Tao de online without changing framework"
status: pending
priority: P2
effort: 8h
branch: main
tags: [ui, css, polish, responsive]
created: 2026-03-28
---

# UI/UX Polish Plan

## Current State
- Single CSS file (2322 lines), single HTML template (4189 lines)
- Good foundation: CSS variables, flexbox/grid, gradient accents
- Weak: no icons (emoji only), limited responsive, no transitions on content, inconsistent spacing

## Phases

### Phase 1: CSS Variables & Typography (1h) - [HIGH IMPACT]
- Status: pending
- [phase-01-typography-spacing.md](phase-01-typography-spacing.md)

### Phase 2: Component Polish (2h) - [HIGH IMPACT]
- Status: pending
- [phase-02-component-polish.md](phase-02-component-polish.md)

### Phase 3: Icons - Lucide via CDN (1h) - [HIGH IMPACT]
- Status: pending
- [phase-03-icons-lucide-cdn.md](phase-03-icons-lucide-cdn.md)

### Phase 4: Micro-interactions & Transitions (1h)
- Status: pending
- [phase-04-micro-interactions-transitions.md](phase-04-micro-interactions-transitions.md)

### Phase 5: Responsive & Mobile (2h)
- Status: pending
- [phase-05-responsive-mobile-layout.md](phase-05-responsive-mobile-layout.md)

### Phase 6: Dark Mode (optional) (1h)
- Status: pending
- [phase-06-dark-mode-optional.md](phase-06-dark-mode-optional.md)

## Key Files
- `app/static/styles.css` - main stylesheet
- `app/templates/index.html` - single-page template

## Priority Order
1. Phase 3 (icons) + Phase 1 (typography) - biggest visual jump, least risk
2. Phase 2 (components) - polishes every panel/button/table
3. Phase 4 (animations) - adds perceived quality
4. Phase 5 (responsive) - mobile support
5. Phase 6 (dark mode) - optional nice-to-have

## Constraints
- No CSS framework added; stay pure CSS
- No JS framework changes
- Must not break existing 10+ tab layouts
- Keep file under 200 lines after modularization (split CSS into partials if needed)
