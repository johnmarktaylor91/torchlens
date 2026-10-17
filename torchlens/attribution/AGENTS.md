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
| `_curve_settings.py` | The curves' settings: feature grouping (`features`, `patch_size`), baselines and step counts, with their coded refusals |
| `_randomization.py` | `randomization_test`: the model-parameter randomization test (cascading or independent, top down) on deep copies inside a forked random state |
| `_road.py` | `road_curve`: ROAD (remove and debias), the deletion curve with removed pixels filled by noisy linear imputation (8-neighbour weights, conjugate gradients per plane, seeded noise per step) through the curves' imputer seam |
| `_causal_audit.py` | `causal_audit`: ablates each unit of a layer attribution's hidden site alone (one `site_sweep` member per unit, any ablation) and the top-k, bottom-k and random-k sets together (one `Trace.fork().do` per set); rank agreement is Spearman; every row and set carries its replayable `(tl.units(...), edit)` pair; dispatches a claim to its plan (layer, direct-logit, read or EDGE table); `extra["sets_not_measured"]` stays empty for an EDGE table, since every set is patched |
| `_audit_sites.py` | The causal audit's plan for claims spread over sites (direct-logit, read and EDGE tables; `_audit_edges._EdgesPlan` extends it): units with their site, component and claim; under zero ablation every unit, whole-site or element, is one candidate of one `site_sweep` (replay engine; `_swept_scores`, also used by the layer plan), and "mean" or baseline units and every set run one `Trace.fork().do` each; the recorded `(site label or tl.units(...), edit)` pairs |
| `_audit_logits.py` | Grounding a `tl.mechinterp` direct-logit-attribution table: the unembedding head's output op, the native-logit check, one whole-site unit per component row, the constant row reported apart; coded refusals |
| `_audit_reads.py` | Grounding a one-backward `ReadTable`: one target, `activation_x_grad` `ACT` rows at site or element grain, the seed target measured on a fork, rows without a score reported apart; a table of `EDGE` rows goes to `_audit_edges.py` |
| `_audit_edges.py` | Grounding an `EDGE` table from `tl.mechinterp.edge_scores`: the caller's `corrupted=` and `metric=` (the table records neither), checks of example count, method, reduction, edge names and sender shapes against the component graph rebuilt on the audit's trace, one unit per `ok` row patched alone with `tl.mechinterp.patch_edges` (`M(patched) - M(clean)`, reduced as the table was), every set patched in one `patch_edges` call into one receiver or several, its recorded pairs read from the patched fork (one per receiver, in write order), `edges=` subsets; coded refusals |
| `_audit_claims.py` | The causal audit's plan: setting checks, the one intervention-ready trace, grounding the claim at one module call, the edit (zero, per-unit batch mean, or a baseline) and the target score of a trace or fork, with their coded refusals |
| `_eval_maps.py` | Shared by the evaluation battery: aligns results (including `grad_cam` maps) with the input, Spearman rank correlation and SSIM |
| `_occlusion.py` | Selection-addressed scored occlusion through `Trace.fork().do(...)`, with explicit zeros/mean/blur baselines |
| `graph/` | Attribution graphs: `build_graph` (one prompt's capture plus transcoders; target seeds at feature inputs and logits, one-backward reads in chunks of 256 targets, each chunk's gradients contracted into its adjacency rows and released before the next read so the gradient buffer peaks at one chunk (internal `_BuildCounters`: reads and that peak in bytes), frozen = every transcoder-written MLP output plus the caller's `frozen=` sites), `AttributionGraph`/`GraphNode` (feature, error, token, logit nodes; adjacency rows are targets; `constants` encoder_bias, decoder_bias and the remainder `other`), transcoder specs in `_transcoders.py` (per-layer, or cross-layer with mapping-valued `W_dec`/`b_dec`); closed-form fixtures in `tests/_ag1_fixtures.py`; on Hugging Face Gemma-2 the pattern and norm-scale sites are op labels (the `softmax` in each `self_attn` under eager attention, the `rsqrt` in each `Gemma2RMSNorm`), the pre-softcap logits are `lm_head`, and token nodes read `model.embed_tokens` (the sqrt(d_model) scale after it cancels in each edge; `tests/test_ag2_gemma2_sites.py`); a transcoder key is a module name (reads its call input) or an op label (reads that op's output, such as the Gemma-2 norm's value before its gain; needs mapping-valued `W_dec`/`b_dec` and must run before every write; `_read_sites.py`: `op_read_label`, `check_upstream`, `tests/test_ag3a_read_sites.py`); `frozen=` also takes facet selectors resolved through `intervention.hooks._resolve_facet_specs`, each facet the whole output of one op (`_read_sites.facet_labels`; `frozen=[tl.facet("pattern"), tl.facet("inv_rms")]` on Gemma-2, `tests/test_ag3a_gemma2_facets.py`) |
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
- Cross-layer transcoders on Hugging Face Gemma-2 (`tests/test_ag3_cross_layer_gemma2.py`):
  circuit-tracer's `hook_resid_mid` read is the residual add whose output enters
  `pre_feedforward_layernorm`, keyed by its op label, and `hook_mlp_out` is the
  `post_feedforward_layernorm` output. A module's `input_ops` are layer labels without a pass,
  and that add can be a multi-pass layer (it is on the tiny random Gemma-2 of the tests, not on
  Gemma-2 2B), where the bare label refuses with `read_target_invalid`; key the transcoder by
  the pass-qualified label of the pass that runs just before the norm (`tests/_ag3_fixtures.py`,
  `read_label`). A module-name read key whose input op is a multi-pass layer reaches the same
  refusal through `_one_op`. A reader's `b_dec`
  mapping names the same writes as its `W_dec`; give each output layer's decoder bias to one
  reader only (zeros elsewhere). Restricting a ReLU transcoder to its active features builds the
  same graph bitwise (an inactive feature is no node and writes nothing), which is how a large
  cross-layer set fits in memory.

## Tests

`tests/test_attribution.py`, `tests/test_attribution_layer.py`, `tests/test_ag1_attribution_graph.py`, `tests/test_ag2_chunked_assembly.py`, `tests/test_ag2_gemma2_sites.py`, `tests/test_ag3a_read_sites.py`, `tests/test_ag3a_gemma2_facets.py`,
`tests/test_backward_attribution_unchanged.py`.
