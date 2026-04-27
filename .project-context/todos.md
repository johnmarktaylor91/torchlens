# Task & Bug Tracker

## Active Tasks

## Bugs

## Improvements (Nice-to-Have)

- Rethink the parameter name `activation_postfunc` itself. Current name is
  awkward (`-postfunc` suffix) and the semantic shifts after the upcoming
  raw-vs-transformed split (it is a "transform" hook, not a "post-processing
  function"). Candidates: `activation_transform`, `activation_hook`,
  `transform_activation`. Keep `activation_postfunc` as a deprecated alias
  for at least one minor release. Defer to a UX-focused naming pass after
  the activation_postfunc refactor sprint lands.

- Estimated autograd_saved_bytes via static formula (no graph required).
  Companion to the introspection-based `autograd_saved_bytes` field shipped
  in the grab-bag sprint: a per-op lookup table keyed on forward function
  name + input/output tensor shapes that returns the expected bytes autograd
  WOULD save if `requires_grad` were on. Useful for what-if estimation in
  `inference_mode` / `no_grad` workflows. Maintenance cost: needs PyTorch
  version pinning + tests for table accuracy across releases. Defer until a
  user actually asks; introspection covers the 90% case.

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
