# Task & Bug Tracker

## Active Tasks

## Bugs

## Improvements (Nice-to-Have)

- Auto-published model menagerie (replace manual Google Drive). Design
  notes: `.project-context/research/menagerie_revamp.md`. Hybrid CI
  (smoke gallery on PR + full on release) -> GitHub Pages, PDFs as
  release assets, generalize `build_torchlens_theme_gallery.py` as
  template.

- Per-grad_fn auto-computed memory cost. Once GradFnLog has
  saved_for_backward refs from the backward-pass sprint, memory cost per
  grad_fn = sum of saved tensor sizes + output gradient shapes plus
  type-specific contributions. Currently using explicit peak-memory
  capture per backward sweep. Design notes:
  `.project-context/research/backward_pass_sprint.md` (parking lot).

- Fastlog gradient support. Predicate-selected gradient capture in
  fastlog. Slow-path backward needs to settle first. Design context:
  `.project-context/research/backward_pass_sprint.md`.

## Completed (recent)
