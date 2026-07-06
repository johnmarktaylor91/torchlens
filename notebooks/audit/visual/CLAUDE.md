# Visual Audit Pack — Agent & Maintainer Guide

`visual_audit.pdf` is a scrollable montage of the complete TorchLens visual
language. Its acceptance bar: after a few minutes of scrolling, there are
**zero surprises** about how any TorchLens render can look. It is a debugging
and polishing instrument — every page states what is demonstrated and what the
eye should check.

## Regenerating

```bash
python notebooks/audit/visual/generate_visual_pack.py
```

Produces (all untracked/regenerable, per `notebooks/audit/.gitignore`):
- `visual_audit.pdf` — the stapled pack
- `_pages/` — per-page intermediates (PNG renders + per-page PDFs)

and regenerates the **tracked** `coverage_matrix.md` (auto-generated — never
hand-edit it; edit `AXES` / page `covers=` tags in the script instead).

Runtime is minutes (torchvision resnet18/resnet50/mobilenet_v2 are constructed
with `weights=None` — no downloads; structure is all the pack needs). Run
renders sequentially; do not parallelize torch processes.

## Structure

Single render-then-staple script (a locked design decision):
- `generate_visual_pack.py` — the page/section specs (`SECTIONS`), coverage
  axes (`AXES` + `NA_AXES`), trace cache, and main loop.
- `_pagekit.py` — page composition (header + caption + PNG panels embedded at
  native resolution; matplotlib's PDF backend preserves full raster detail
  when zooming).
- `_visual_models.py` — models beyond the shared `../_models.py` ZOO
  (recurrent cells, weight-tied loops, block stacks, mini transformer /
  inception, degenerate cases). Do not edit `../_models.py` from here; it is
  shared with the audit notebooks.

### Section map

| Section | Contents |
|---------|----------|
| A | Node & edge vocabulary: baseline anatomy, buffers, multi-I/O, edge multiplicity, legend |
| B | Layout & direction: direction, order_siblings, large-graph regression, dot vs rank |
| C | Containers, module focus, call depth, skip_fn/collapse_fn/node_spec_fn hooks, override dicts |
| D | Loop rolling & recurrence: unrolled-vs-rolled, back-edges, pass-count sweep, fused-kernel contrast, loops with branching |
| E | Collapse & run folding: none/auto/max, float-t filmstrip, fold_runs, ellipsis grammar, segments, remainder labels, known artifacts, plan/schedule diagnostics |
| F | Node content: node_mode presets, overlays (incl. the NaN debugger), label fields, code panel, typography, raw I/O thumbnails, input-transform summary |
| G | Themes (all five presets) |
| H | Backward & combined graphs |
| I | Control flow & interventions |
| J | Real architectures at page scale (resnet18 auto, transformer, inception) |
| K | Degenerate & edge cases |

## The visual-grammar cheat sheet

These three claims look similar and must never be conflated:

- **`(xN)`** — true recurrence: the SAME parameters applied N times
  (rolled mode / loop rolling, Section D).
- **`+N more Class`** — ellipsis from run folding: N further DISTINCT
  same-class instances, each with its own parameters (Section E).
- **dashed segment box** — adjacency-only range: "these consecutive siblings
  live here" — NOT a real module, never carries a single class name for mixed
  content (Section E).

Also: collapsed boxes carry an honest `N layers total` remainder that includes
buffer leaves; ops with hidden buffer dependencies get a double border
(`peripheries=2`).

## How to extend when a new visual feature ships

A new draw() kwarg, node kind, edge style, or label form is **not done** until
it is in this pack:

1. Add the new axis tag(s) to `AXES` in `generate_visual_pack.py` (or to
   `NA_AXES` with an honest rationale if it truly has no visual identity).
2. Add a `Page` (or panel on an existing page) to `SECTIONS` demonstrating it
   on the SMALLEST model that shows the phenomenon. Write the caption for a
   reader who has never seen TorchLens: what is shown, what to check.
3. Declare the axis in the page's `covers=` list.
4. Re-run the script; confirm the console reports no GAPs and
   `coverage_matrix.md` has no `UNCOVERED — DEFECT` rows.
5. Re-run the two critic passes (see below) before calling it shipped.
6. Commit script + regenerated `coverage_matrix.md` together
   (`feat(audit-viz): ...`).

### Critic passes

- **Completeness critic:** independently enumerate every draw() kwarg x value
  and every node/edge kind emitted by `torchlens/visualization/rendering.py`;
  diff against `AXES` + `NA_AXES`. Anything unrepresented is a defect.
- **Fresh-eyes clarity critic:** scroll the PDF as someone who has never seen
  TorchLens. Every page must be self-explanatory from its caption alone; flag
  any caption that doesn't match what the panels actually show.

## Known traps

- **Import the checkout under audit:** `python3 path/to/script.py` puts the
  SCRIPT directory (not the cwd) at `sys.path[0]`, so a pip `-e` install of a
  DIFFERENT checkout can silently supply the renderer. The script injects the
  repo root into `sys.path` and hard-fails if `torchlens.__file__` resolves
  elsewhere. Do not remove that guard.
- **Feature preconditions** (a panel silently showing nothing usually means a
  missing trace flag, not a broken renderer):
  - intervention site/cone/hook styling marks PLANNED interventions:
    `tl.trace(..., intervention_ready=True)` then `trace.set(...)`. Live
    `intervene=` fire records do not style nodes.
  - raw-input thumbnails need `tl.trace(model, raw, transform=fn)` (raw input
    is only stored when a transform produced the model-ready tensor).
  - container labels/collapsed need `intervention_ready=True`;
    `show_containers='nodes'` needs `capture_container_structure=True`;
    'cluster' currently always falls back to labels; 'collapsed' only merges
    homogeneous containers larger than `container_max_inline`.
  - `show_input_transform_summary` renders only when the trace carries an
    input-preprocessing record (bridge-populated; the pack injects a demo one).
  - `collapse_fn`/`skip_fn`: the Module predicate matches on `.class_name`;
    `skip_fn(layer) is True` means SKIP (inputs/outputs may never be skipped).
- **Train-mode BatchNorm noise:** BN models traced in train mode scatter
  orphaned dashed buffer-update ops when buffers are hidden; `.eval()` your
  demo models unless that artifact is the point of the page.
- **AI-review image caps:** when inspecting pages with a vision model, convert
  with `pdftoppm -png -r 100` and keep each image <= 2000 px on the longest
  side (crop or lower `-r`); multi-image requests reject larger panels.
- **Huge renders:** full unrolled torchvision graphs exceed Graphviz's cairo
  bitmap cap (~32k px) and get scaled down — expected for the deliberately
  unreadable `collapse='none'` panels.
- **Model weights:** keep `weights=None` on torchvision constructors. The pack
  must not download checkpoints.
- **Page-render time:** resnet50 `collapse='max'` and the resnet18 filmstrip
  dominate runtime. Reuse the trace cache (one trace, many draws) when adding
  panels.
- **Randomness:** the script seeds torch (`torch.manual_seed`) so
  branch-taking models (e.g. conditional demos) render deterministically.
