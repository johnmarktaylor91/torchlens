# Task & Bug Tracker

## Active Tasks

## Bugs

## Improvements (Nice-to-Have)

### Multi-trace V2 (deferred from 2026-04-27 sprint design)

These were scoped OUT of the multi-trace MVP (TraceBundle + TraceOverlay)
but are natural follow-ons. Pick up after MVP ships.

- **Branch-divergence detection.** Soft version of "counterfactual
  tracing." Given a TraceOverlay, surface "branch-divergence nodes"
  where execution paths split, expose as a queryable list. NOT true
  forced-branch enumeration -- just detection of where forward paths
  diverged across the bundled inputs.

- **True counterfactual branch enumeration** (forced-branch execution).
  Programmatically force both arms of every Python `if` in a dynamic
  network. Hard problem -- requires either symbolic tracing
  (incompatible with TorchLens's runtime model), bytecode hacking
  (fragile), or an intervention API where the user explicitly drives
  forced inputs to coax branches. Defer indefinitely; the
  branch-divergence detection above is the practical 80% solution.

- **Streaming aggregate over a dataloader.** `tl.aggregate(model,
  dataloader, metrics=[...])` -- consumes traces from a generator,
  computes running per-node statistics (mean, var, RDM,
  dimensionality), discards raw activations. For "10K images through
  ResNet50" workflows where holding traces is impractical. Function,
  not a class.

- **Interactive viewer for bundles/overlays.** D3.js or anywidget-based
  Jupyter widget. Pan/zoom, hover tooltips, click-to-expand node
  visualizations, path highlighting, export to standalone HTML. After
  graphviz MVP is stable.

- **Custom node visualizations.** Pluggable per-node display: PCA-RGB
  for conv layers, MDS scatter for linear layers, histograms for
  activations, dimensionality estimates. Bundle/Overlay would accept
  `node_display={'conv.*': 'pca_rgb', ...}` mappings.

- **Convenience constructors with intervention APIs.** `tl.zero(node)`,
  `tl.steer(node, vector)`, `tl.compare(model, input, {...})`,
  `tl.sweep(model, input, {...})`. Interventions produce a new
  ModelLog (never mix clean and intervened activations in a Bundle).
  Out of scope for MVP; design once Bundle/Overlay are stable.

- **Rename ModelLog -> Trace** (one-way door). Clean public-API
  migration -- `tl.trace()` constructor, `Trace` class. Major
  breaking change to a published-and-cited package. Do as a separate
  deliberate migration, never bundled with feature work.

### Other improvements

- Rethink the parameter name `activation_postfunc` itself. Current name is
  awkward (`-postfunc` suffix) and the semantic now reads as a "transform"
  hook, not a "post-processing function" (after the raw-vs-transformed
  split landed in PR #166). Candidates: `activation_transform`,
  `activation_hook`, `transform_activation`. Keep `activation_postfunc` as
  a deprecated alias for at least one minor release. Defer to a
  UX-focused naming pass.

- Estimated autograd_saved_bytes via static formula (no graph required).
  Companion to the introspection-based `autograd_saved_bytes` field
  shipped in PR #165: a per-op lookup table keyed on forward function
  name + input/output tensor shapes that returns the expected bytes
  autograd WOULD save if `requires_grad` were on. Useful for what-if
  estimation in `inference_mode` / `no_grad` workflows. Maintenance cost:
  needs PyTorch version pinning + tests for table accuracy across
  releases. Defer until a user actually asks; introspection covers the
  90% case.

- Auto-published model menagerie (replace manual Google Drive). Design
  notes: `.project-context/research/menagerie_revamp.md`. Hybrid CI
  (smoke gallery on PR + full on release) -> GitHub Pages, PDFs as
  release assets, generalize `build_torchlens_theme_gallery.py` as
  template.

- Per-grad_fn auto-computed memory cost. Once GradFnLog has
  saved_for_backward refs from the backward-pass sprint, memory cost per
  grad_fn = sum of saved tensor sizes + output gradient shapes plus
  type-specific contributions. Distinct from the per-op
  `autograd_saved_bytes` shipped in PR #165 -- this is a per-grad_fn
  view (backward-graph node accounting). Currently using explicit
  peak-memory capture per backward sweep. Design notes:
  `.project-context/research/backward_pass_sprint.md` (parking lot).

- Fastlog gradient support (PR C). Predicate-selected gradient capture
  in fastlog. As of 2026-04-27, slow-path backward IS settled (PRs
  #161-163, #165), so this is unblocked architecturally -- gated only
  on JMT direction to dispatch (a research-and-spec sprint). Design
  context: `.project-context/research/backward_pass_sprint.md`. Once
  shipped, fastlog `gradient_postfunc` parity (mirroring slow path
  behavior added in PR #166) becomes the natural followup; do NOT add
  `gradient_postfunc` to fastlog BEFORE gradient capture lands per
  research conclusion in
  `.project-context/research/fastlog_postfunc_parity_2026-04-27.md`.

- Document activation_postfunc / gradient_postfunc portable-save
  persistence story. PR #166 made the callable-drop / repr-keep
  behavior explicit in code, but the user-facing rationale is not
  surfaced in the public docstring or README. Add a short note
  explaining that `torchlens.save` strips the callable for portability
  and retains only `activation_postfunc_repr` /
  `gradient_postfunc_repr` strings as a record. Source: postfunc
  review Finding #7 (`.project-context/research/postfunc_review_2026-04-27.md`).

- Reduce first-call cost of `patch_detached_references()`. Profiling
  audit measured 16.7s cumulative / 8.64s self time on a small smoke
  model -- 70% of total runtime. The cost is one-time (caches across
  calls) but cold-start UX is rough. Options: idempotent re-entry
  guard, narrower `sys.modules` crawl, or an opt-out for environments
  that don't need detached-import patching. Source: profiling audit
  Finding #5 (`.project-context/research/profiling_audit_2026-04-27.md`).

- Bound the AST/file cache for long-running services. Caches in
  `torchlens/postprocess/ast_branches.py` persist by filename and
  aren't bounded. The 50-iteration leak loop confirmed flat behavior
  in normal use (no leak), but daemon-style processes that touch many
  unique source files could accumulate parsed ASTs over time. Add an
  LRU cap and/or a public cache-clear API. Source: profiling audit
  Finding #6.

## Completed (recent)

### 2026-04-27 grab-bag + activation_postfunc + perf sprint

Versions shipped: 2.10.0 -> 2.13.0 (six PRs).

- PR #164 -> 2.10.0: `extra_data` + `input_metadata` plumbing on
  LayerLog / LayerPassLog / ModelLog (open-ended user dicts).
- PR #165 -> 2.11.0: `autograd_saved_bytes` per-op introspection
  (distinct from existing `saved_activation_memory` -- autograd's
  saves vs torchlens's own).
- PR #166 -> 2.12.0: `activation_postfunc` raw-vs-transformed split,
  `train_mode` hardening, `ActivationPostfunc` / `GradientPostfunc`
  type aliases, `TorchLensPostfuncError`, README + docstring expansion.
- PR #167 -> 2.12.1: two-pass mode in-place module fix (regression
  source: commit 326b8a90, fast-mode pass-through detector mistakenly
  treating in-place modules as pass-throughs because input labels were
  read AFTER `orig_forward()` already mutated them).
- PR #168 -> 2.12.2: perf bundle, 3-of-3 fixes (bytecode
  column-offset cache, Step 5 branch-attribution fast-skip, CUDA probe
  guard for CPU runs). Shipped via Claude general-purpose agent after
  codex hit daily quota.
- PR #169 -> 2.13.0: fastlog `activation_postfunc` parity. Intentional
  architectural divergence from slow path per
  `.project-context/research/fastlog_postfunc_parity_2026-04-27.md`
  (parallel `transformed_*_payload` fields on `ActivationRecord`,
  postfunc runs in `_storage_resolver` post-predicate, predicates
  still see raw metadata, no `gradient_postfunc` until fastlog gets
  gradient capture). Also via Claude agent.

Research reports committed in `.project-context/research/`:
- `postfunc_review_2026-04-27.md`
- `fastlog_postfunc_parity_2026-04-27.md`
- `profiling_audit_2026-04-27.md`
- `two_pass_diagnostic_2026-04-27.md`

Other artifacts:
- `~/.claude/CLAUDE.md` hardened with a codex-quota-exhaustion
  fallback procedure (added 2026-04-27 incident section under Codex
  Dispatch).
- `.gitignore` extended to ignore `/modelgraph.*` and
  `/backward_modelgraph.*` (default `vis_outpath` filenames that were
  polluting the repo root).
