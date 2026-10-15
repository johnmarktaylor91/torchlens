# attribution/ - Implementation Guide

Input-attribution methods over live models (lazy public submodule,
`tl.attribution`). Gradient-based attribution runs the USER's model directly
(with a temporary `eval()` context and fresh input leaves), not a replay of a
captured Trace.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public re-exports only; the 12-name `__all__` below |
| `_core.py` | `AttributionError`, `AttributionResult`, input normalization / leaf minting / baseline validation machinery, and the input-level methods (`saliency`, `input_x_grad`, `integrated_gradients`, `smoothgrad`); IG always reports its completeness residual |
| `_layer.py` | Layer-level methods: `layer_attribution`, `layer_integrated_gradients`, `layer_conductance`, the shared layer capture and input-path machinery, and the CAM overlay renderer |
| `_cam.py` | `grad_cam`: feature maps, token grids (through `_cam_grid.py`), integer-input grids, and the input grid the map is upsampled onto |
| `_cam_grid.py` | Token-grid derivation for `grad_cam` from a metadata-only trace (grid flatten and grid view patterns, prefix tokens from the token concatenation), the `reshape=` escape hatch and their coded refusals |
| `_cam_choice.py` | Automatic `grad_cam` layer choice (`layer=None`): once-firing map or token-grid outputs that dominate the output, deepest with a nonzero grid gradient |
| `_entry.py` | Integer inputs for the layer methods: finds the innermost modules that turn caller integers into floats (embeddings), substitutes the path values at their outputs by prepended forward hooks, and builds baselines from reference ids |
| `_site.py` | `SiteTarget` (attribute a hidden layer's output or one unit of it) and its route: `_call_model` returns the site activation, `_scalarize_output` selects the unit; coded refusals for missing sites, bad units or calls, and sites the input does not reach |
| `_curves.py` | `deletion_curve` and `insertion_curve`: rank input features by attribution (elements, pixels, patches, tokens), perturb them onto a baseline in steps, score each step; area and AOPC; each `CurveStep` carries its replayable `tl.when(...)` intervention |
| `_randomization.py` | `randomization_test`: the model-parameter randomization test (cascading or independent, top down) on deep copies inside a forked random state |
| `_eval_maps.py` | Shared by the evaluation battery: aligns results (including `grad_cam` maps) with the input, Spearman rank correlation and SSIM |
| `_occlusion.py` | Selection-addressed scored occlusion through `Trace.fork().do(...)`, with explicit zeros/mean/blur baselines |
| `_viz.py` | `overlay`: bridges layer-scoped results (or explicit mappings) onto `Trace.draw(color_by=...)`; uncovered nodes return `None` (unencoded) |

## Public surface (`__all__`, 12 names)

`AttributionError`, `AttributionResult`, `saliency`, `input_x_grad`,
`integrated_gradients`, `smoothgrad`, `layer_attribution`,
`layer_integrated_gradients`, `layer_conductance`, `grad_cam`, `occlusion`,
`overlay`.

## Gotchas

- `AttributionError` subclasses `ValueError`; argument/baseline problems
  refuse through it, not bare exceptions.
- Inputs are normalized through `_PreparedInputs` and replaced by fresh
  requires-grad leaves (`_make_input_leaves`); nested input trees keep
  identity-interning so repeated tensor objects stay aliased.
- Baselines are validated against the exact input tree shape
  (`_validate_baseline_tree`); repeated-reference baselines have their own
  validation path.
- The model's `training` flag is saved/restored by `_temporarily_eval` —
  do not add code paths that return without restoring it.
- Stage-4b names and metadata are DOCUMENTED-UNSTABLE and lockstep-indexed in
  `docs/reference/glossary.md`.

## Tests

`tests/test_attribution.py`, `tests/test_attribution_layer.py`,
`tests/test_backward_attribution_unchanged.py`.
