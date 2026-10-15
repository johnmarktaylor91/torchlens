# TorchLens Migration Policy

## `.tlspec` Compatibility

TorchLens 2.16.0 wrote two public `.tlspec` directory formats:

- Intervention specs with `spec.json` containing `format_version`.
- Portable `ModelLog` bundles with `manifest.json` containing `io_format_version`.

TorchLens does not auto-migrate them in place; readers dispatch by detected
format.

**Erratum (2026-10-14).** This section used to say that these 2.16.0 formats
"are permanently readable". That sentence is retracted. Current releases still
load 2.16.0 intervention specs, but they refuse 2.16.0 portable `ModelLog`
bundles (`io_format_version` 2) with `tl.errors.ArtifactVersionBelowFloorError`
(code `artifact_version_below_floor`): saved traces older than `tlspec_version`
6 are below the rehydration floor. To read such a bundle, load it with
TorchLens 2.17.0 (`pip install torchlens==2.17.0` in a separate environment), the
verified reader for that format; `torchlens.ecosystem.compat_window()` lists the
verified reader for each retired format.

New writers introduced during the Phase 11 schema graduation emit the unified
manifest format. The polymorphic loader detects the on-disk format and dispatches
to the appropriate reader. During the Phase 11.0 transition, intervention specs
also include `kind: "intervention"` in `manifest.json` while preserving the
2.16.0 fields.

An optional future utility, `torchlens.io.migrate_tlspec(path, dest)`, may write
an upgraded copy in the unified format. It will not be required for loading
existing 2.16.0 files.

## Visualization Collapse Engine

`Trace.draw(collapse=...)` is one v2 smart-collapse surface: `"none"`, `"auto"`, `"max"`,
or a float `t` in `[0.0, 1.0]` on the public monotone schedule (`collapse="max"` is the v2
mode that may emit segment boxes). There is no engine-selection environment variable.

Two S5 grain metric rows are intentionally rebaselined: `deeplabv3_resnet50` and
`convnext_tiny` are superseded by the flip-gate visual ruling. For DeepLabV3,
v1's lower variance came from a degenerate opaque ASPP cut; v2 exposes the ASPP
branches. For ConvNeXt Tiny, v2's extra downsample detail is treated as a
legitimate overview detail rather than a blocking regression.

F11 (collapse memo D5/D8/D9, hard changes -- no warn shims per the clean-v2
alias posture):

- `collapse_order(weights=...)` / `Trace.collapse_order(weights=...)` is
  REMOVED. The parameter was documented-inert (scores were never
  weight-sensitive on this surface); drop the argument.
- The compute gate for smart collapse is now the rendered universe `U` of
  the full plan plus a measured `(U, W)` work estimator, not raw op count:
  `module=` focus, `vis_call_depth`, and rolled mode now genuinely re-admit
  the quality planner. Over-budget requests degrade to a deterministic
  compact fallback plan (disclosed, coded `collapse_budget_fallback`) and
  never to an uncollapsed wall; only pathological inputs (20x
  `COLLAPSE_OPTIMIZER_MAX_OPS` raw ops) still decline outright.
- `collapse="auto"` is UNFROZEN to the documented contract: the first
  in-band point of the typed event ladder, else the disclosed strongest
  point (`band_missed`, `strongest_plan_count`). Interior float levels map
  geometrically in realized count, `t` strictly increases, and no-op steps
  coalesce; `t=0.0`/`t=1.0` stay byte-identical to `"none"`/`"max"`.
  Renders drawn with interior floats or `auto` may therefore change; the
  regenerated `docs/images/collapse` gallery is the reference.

## run() Declared-State Restore (default behavior change)

The default live `trace.run(inputs=...)` now snapshot-restores the model's declared
state (named parameters plus registered buffers, alias topology preserved) around the
run, so repeated `run()` calls leave the model bit-identical. Previously, forward-pass
mutations of declared state (BatchNorm running statistics, buffer counters, `no_grad`
in-forward parameter updates) persisted on the live model across `run()` calls. Pass
`carry_state=True` (provisional spelling, documented-unstable) for the old persistence;
the report discloses the choice as `report.state_carried`.

## `resample_ablate` -> `scramble_elements` (honest rename, same bytes)

The elementwise-scramble helper is renamed: `resample_ablate` implied the
field's "resampling ablation" (replacing a site's value with another run's
COHERENT donor value), but the helper has always been an elementwise iid
scramble from a flattened source -- a structure-destroying noise baseline.
The canonical constructor is now `torchlens.intervention.scramble_elements`
(same bytes, same seeds, same draws; specs constructed through either
spelling carry helper name `"scramble_elements"`).

Which "resample" do you mean?

| You want | Use |
|---|---|
| Elementwise iid scramble (noise baseline) | `scramble_elements(source, seed=)` |
| Coherent donor patch at the same site ("resampling ablation") | `tl.patch_from(other_trace)` |
| Batch-row permutation | `torchlens.intervention.permute_batch` (shipped, lane F02) |
| Per-row donor resampling | `torchlens.intervention.resample_rows_from` (shipped, lane F02) |

Saved artifacts: an intervention spec or pickled edit that persists the
helper name `"resample_ablate"` no longer loads; it refuses with
`intervention_helper_renamed`, whose remedy names
`torchlens.intervention.scramble_elements`. Rebuild the edit with
`scramble_elements(...)` (same arguments, same bytes) and save it again. The
`resample_ablate` spelling itself is REMOVED, with no alias: `tl.resample_ablate`, `torchlens.intervention.resample_ablate`, and
`torchlens.intervention.helpers.resample_ablate` no longer exist.

| Old spelling | New spelling |
|---|---|
| `tl.resample_ablate(...)` | `torchlens.intervention.scramble_elements(...)` |
| `torchlens.intervention.resample_ablate(...)` | `torchlens.intervention.scramble_elements(...)` |

## Legacy `summary()` spellings removed (clean break, no aliases)

`Trace.summary()` and `tl.summary()` no longer accept the legacy keyword
spellings or the legacy `level=` preset names, and the historical renderer
behind them is deleted. Each removed spelling raises `InvalidArgumentError`
(`summary_option_invalid` for keywords, `summary_level_invalid` for level
presets) whose message names the replacement; `tl.summary()` refuses before
running any capture. Unknown option names now refuse typed with the nearest
grammar option instead of a bare `TypeError`.

| Old spelling | New spelling |
|---|---|
| `preset=` | `view=` (`"overview"` / `"compute"`) for columns, `level=` for row grain |
| `fields=` | `columns=` |
| `show_ops=True`, `include_ops=True` | `level="op"` |
| `mode=` | `level="op"`, or `fold_repeats=False` to unfold repeated runs |
| `print_to=fn` | `report.print(file=...)`, or `fn(str(report))` |
| `count_fma_as_two=True` / `False` | `flop_convention="fma2"` / `"fma1"` |
| `show_input_preprocessing_details=True` | `trace.provenance()` and `trace.input_preprocessor` (`verified`, `source`, `identifier`) |
| `level="overview"` | `view="overview"` (the default) |
| `level="compute"`, `level="cost"` | `view="compute"` |
| `level="graph"` | `trace.to_agent_json()` (op rows, edges, hierarchy) or `trace.draw()` |
| `level="memory"` | `trace.profile(sort_by="activation_memory")`; memory totals stay in the summary footer |
| `level="control_flow"` | `trace.conditional_records` and the `conditional_*` columns of `trace.to_pandas()` |
| `level="waterfall"` | `trace.profile(level="op")`, or `trace.to_pandas()` in execution order |
| `level="output"` | `trace.output_table()` |

The old renderer also printed a partial live table when `summary()` was called
from inside a running forward (a hook or a module's `forward`). That preview is
gone: `trace.summary()` mid-forward now raises `CaptureContextError`
(`trace_not_finished`). Use `repr(trace)` for the live op count, and call
`summary()` after `trace(...)` returns.

## `tl.utils.flop_count(count_fma_as_two=)` removed (clean break, no alias)

`tl.utils.flop_count()` takes the same `flop_convention=` vocabulary as
`summary()`. The old keyword raises `InvalidArgumentError`
(`flop_count_option_invalid`) naming the replacement, before any capture runs.

| Old spelling | New spelling |
|---|---|
| `flop_count(model, x, count_fma_as_two=True)` | `flop_count(model, x, flop_convention="fma2")` (the default) |
| `flop_count(model, x, count_fma_as_two=False)` | `flop_count(model, x, flop_convention="fma1")` |

## Steering bridges become converters (clean break, no aliases)

The contrastive direction math lives in TorchLens only:
`torchlens.intervention.direction_from_pairs(pos_traces, neg_traces, site, method=)` returns
a `Direction`, and the steering-vectors and repeng bridges only convert it. The old trainers
are removed; calling them raises `AttributeError`.

| Old spelling | New spelling |
|---|---|
| `tl.bridge.steering_vectors.vector(log_pos, site, negative_log=log_neg)["vector"]` | `torchlens.intervention.direction_from_pairs(log_pos, log_neg, site).vector` |
| `tl.bridge.steering_vectors.vector(..., layer=i)["steering_vector"]` | `direction_from_pairs(log_pos, log_neg, site).to_steering_vectors(layer=i)` |
| `tl.bridge.steering_vectors.vector(..., trainer=sv.pca_aggregator())` | `direction_from_pairs(..., method="pca")` |
| `tl.bridge.repeng.control_vector(log_pos, site, negative_log=log_neg, layer=i)["control_vector"]` | `direction_from_pairs(log_pos, log_neg, site, method="pca").to_repeng(layer=i)` |
| `tl.bridge.dialz.vector(log, site, negative_log=..., layer=i, method=...)["steering_vector"]` | `direction_from_pairs(log_pos, log_neg, site, method=...).to_dialz()` |

The native `"pca"` is the uncentered first principal axis of the paired differences (the
steering-vectors `pca_aggregator` definition), not repeng's centered `pca_diff`, so a repeng
vector built the old way and the new one differ; `pca_center` and `umap` have no
replacement. The explicit `attention_mask=` / `negative_attention_mask=` arguments are gone:
the builder reads each trace's saved `attention_mask` input.

For dialz, the `dialz` extra now needs dialz 1.1 or later (`dialz>=1.1,<2`; it was 0.2). dialz's
own `pca` (scikit-learn PCA, which centres the differences), `pca_center` and `umap` training
methods are not reproduced; only `mean_diff` equals dialz's `mean_diff`.

## Removed member spellings name their replacement (typed redirect)

The alias members and public subpackage names deleted by the 2026-10-01
shim removal and the F38 sweep, and the deleted
`VisualizationTheme.legend_items` field, are still gone (no alias). Looking
one up now raises `FacadeTeachingError` (code `facade_redirect`, an
`AttributeError` subclass, so `getattr(obj, name, default)` still degrades)
whose message names the replacement. (For removed module-level names,
`from torchlens import <name>` shows a plain `ImportError` instead: CPython
replaces the attribute error on that path. Attribute access such as
`tl.resample_ablate` carries the redirect.)

| Removed | Use |
|---|---|
| `Trace.replay`, `Trace.replay_from`, `Trace.rerun` | `trace.push`, `trace.push_from`, `trace.run` |
| `Trace.validate_saved_outs` | `trace.validate_forward_pass` |
| `Trace.conditional_then_entry_edges` / `_elif_` / `_else_` | `trace.conditional_arm_entry_edges` |
| `Bundle.replay`, `Bundle.rerun` | `bundle.push`, `bundle.run` |
| `VisualizationOptions.mode`, `.max_module_depth`, `.layout_engine`, `.node_mode` | `.view`, `.depth`, `.layout`, `.node_style` |
| `VisualizationTheme.legend_items` | legends derive from the encoding channels; style via `semantic_palette` and `ramp` |
| `torchlens.validation.validate_saved_outs`, `.validate_trace_saved_outs` | `torchlens.validation.validate(model, x, scope="saved")`, or `trace.validate_forward_pass(...)` |
| `torchlens.io.get_model_metadata` | `torchlens.io.log_model_metadata` |
| `torchlens.observers.record_span` | `torchlens.observers.span` |
| `torchlens.intervention.intervening`, `.replay_from` | `torchlens.intervention.without_op`, `.push_from` |

## Episode ledger grammar v2 (the C07X coordinated tlspec-v9 amendment)

The persisted episode ledger (`trace.annotations["episode"]`) moves to
grammar v2, versioned family-locally by the new required header field
`episode_ledger_version` (= 2). Future episode grammar changes bump the
family version only, never the tlspec version.

- `cache_len` is DELETED. It was a derived arithmetic guess (prompt length
  + step), wrong on real KV-cache feeds and never a measured cache fact.
  The replacement is the measured GENERIC carried-state witness: the
  channel-keyed `entry_state_digest` / `exit_state_digest` row slots, where
  `None` means NOT MEASURED (the only value any shipped writer emits today).
- `tokens` is renamed to the generic `step_output`, typed by the header's
  declared `step_output_kind` (`tokens` default / `digest` / `none`).
- New header slots: `step_output_from`, `step_axis`, `step_join`,
  `capture_digest`, `intervention_digest`; new row slot `fire_count`;
  `perturbed` joins the fidelity-basis vocabulary. All entry-dark until
  their writer lanes (F40b / F40c / F42 / F-WITNESS) land.

Compatibility: no released torchlens ever wrote grammar v1 to a real
artifact family older than the current line, and loads are fail-closed in
both directions — a v1 payload (carrying `cache_len`/`tokens`) QUARANTINES
typed (`episode_ledger_incoherent`) with one warning; it never silently
normalizes. Re-capture with a current TorchLens to produce v2 ledgers.

D3 AUTHORIZATION (C07-owner line, foldA s5 item 3(vi)): lane F40b — the
episode-family derivation rebuild that activates `step_output_from` /
`step_output_kind` on this grammar — is authorized to SELF-AUTHOR its
MIGRATIONS row in this file when it lands, without a further C07-owner
sign-off; the grammar carriers it activates are the ones this amendment
declared.

## Episode derivation on the declared source and kind (lane F40b, self-authored under the D3 line above)

The episode evidence derivation is DECLARATION-DRIVEN as of this lane
(foldA D8): `EpisodeSpec` declares `step_output_kind`
(`tokens` default / `digest` / `none`), `step_output_from` (a root-output
slot path for dict/`ModelOutput`/tuple roots), and the generic `step_axis`
(renamed from `token_axis`, remove-and-rename per interim policy — every
episode spelling is DOCUMENTED-UNSTABLE, no shim owed). Per-step evidence
is TAIL-ALIGNED along `step_axis` (the last `n_steps` positions), so a root
returning what real `generate()` returns — prompt+completion — and
float-emitting stepped roots (under `digest`/`none`) now capture instead of
refusing; the four historical shape-guessing refusals survive only as
declaration-mismatch teaching refusals. The header's `step_output_from` /
`step_axis` slots are written whenever a kind consumes a source, and
`capture_digest` is minted at settlement (`mint_capture_digest`, schema tag
`episode_capture_digest_v1`) binding the ledger to its product; lane F42
consumes it. Value-free save policies are admitted exactly for
`step_output_kind="none"` (kind-conditional save rule). Loads CONSUME the
family version field: any `episode_ledger_version` other than 2 —
including the version-less pre-v2 grammar with `tokens`/`cache_len` rows —
quarantines typed (`episode_ledger_incoherent`) with the grammar named,
and never normalizes.

## Bound-method roots and the fail-closed rerun identity gate (lane F41)

`tl.trace` (and `tl.validate`) accept a bound method of an `nn.Module` as
the capture root (the ruled root contract): the owner resolves via
`method.__self__` and registers as the `owner` submodule of a
TorchLens-authored wrapper root, the method is called exactly once, the
capture's identity reads `type(owner).__name__`, and two persisted
identity facts disclose the synthetic root
(`Trace.root_entry_point="bound_method:<owner_qualname>.<method>"`; the
root op record carries `tl_authored_root=True`). Other callables
(closures, bare functions) keep refusing `model_type_unsupported`, now
teaching the ruled spelling. `EpisodeSpec.stepped_module` gains a `None`
default that resolves to the owner on bound-method episode captures and
refuses typed on module roots.

Behavior change at the rerun/append identity gate (fail-closed, foldA
D10/D11): the gate now REQUIRES the root entry-point fact. A legacy
artifact saved before the fact existed refuses rerun/append typed
(`root_entry_point_unavailable`) instead of silently passing on
class+weights evidence alone, and bound-method captures refuse
rerun/append (`rerun_entry_point_unsupported`, interim posture — rerun
executes the supplied module's `forward`, a different entry point than
the captured bound method). Replay engines and loaded-sparse `run()` are
unaffected. Remedy for legacy artifacts: re-capture with a current
TorchLens, or use the replay engine.

## Op-Level Buffer Accessor Iteration (behavior change)

Op-level buffer accessors (`op.buffer_sources` / `op.buffer_sinks` and friends)
now share the layer-accessor access model: iteration yields the resolved `Op`
records themselves, and `.get(...)` resolves on the same 0-based-position /
label basis as `__getitem__`. Previously iteration yielded internal call-index
keys and `.get(key)` resolved those keys, so code written against the old model
(`for key in acc: acc.get(key)`) now receives `Op` records directly -- iterate
the accessor and read each record's `.label` (or index by position/label). The
partition semantics are unchanged: sources are the buffer-read parents in
order, sinks the buffer-write children in order.

## Op accessor iteration and indexing share one basis (lane C02, BREAKING)

Iterating an op accessor now yields `Op` records, and `get`/`[]`/iteration
all share ONE 0-based, pass-qualified basis. Historically iteration yielded
1-based ints while indexing was 0-based -- a silent off-by-one for any
consumer that mixed the two. Code that iterated accessors for ints must read
`op.label`/`op.raw_index` off the yielded records instead.

## One-voice repr/str (lanes C02 + F10, BREAKING for string-parsers)

`Trace`, `PartialTrace`, `Bundle`, `EdgeUseRecord`, `CollectiveJoin`, and
every accessor render through the one-voice grammar: `repr` is one line
(was 5-21), `str` is a bounded card, and `repr == str` no longer holds on
value records. Module/profile trees emit ASCII rails; the stats-line hazard
marker is ASCII `!`; the number formatter keeps trailing zeros under the
precision law; `tensor_stats_summary` drops `neg=` from its default line.
Exact-string consumers of the old dumps must re-pin (the in-tree consumers
were swept in the same change).

## Model Explorer export schema v3 (lane F15)

The Model Explorer emitter moves to the schema v3 family: the top-level v2
`schema`/`disclaimer` keys are REMOVED, namespaces derive from the recorded
`module_call_stack` (percent-escaped, pass-qualified), and universal
`site_key|ordinal` node ids are appended ALWAYS with an `id_fidelity` stamp.
Readers of the v2 JSON must re-export; the pinned vendor harness
(`dist/worker.js`) is the contract oracle.

## `tl.export.tensorboard` requires `step` (lane F26)

The exporter's `step` keyword lost its default: pass the global step
explicitly (`tl.export.tensorboard(log, writer, step=n)`). A stepless call
silently landed every export on one x-coordinate, which the TensorBoard
frontend renders as a single point; requiring the keyword makes the time
axis an explicit user fact.

## MCP tools renamed onto the nine-tool agent registry plus the ledger trio (lane F29)

The MCP server's `load_overview` and `agent_dump` tools are REMOVED
(remove-and-rename, no aliases). The registry now serves twelve tools:
`torchlens_doctor`, `torchlens_api_map`, `torchlens_overview` (was
`load_overview`), `torchlens_dump` (was `agent_dump`), `torchlens_explain`,
`torchlens_query_sites`, `torchlens_payload_stats`, `torchlens_compare`,
`torchlens_schema`, and the ledger family
(`torchlens_ledger_overview`/`_entry`/`_evidence`). Separately,
`import torchlens` no longer imports torch (deferred to first use) -- import
order can no longer be used to force torch initialization.

## `TorchLensLitModel` removed (lane F31)

The LIT stub class `TorchLensLitModel` is REMOVED -- it never worked against
real LIT (real LIT rejects it at construction). The real adapters are
`torchlens.bridge.lit.model(net, tokenizer, ...)`, `bridge.lit.dataset`, and
`bridge.lit.layout`, each returning native LIT objects. Sixteen typed
`lit_*` refusal codes enter the error contract.

## Submodule advertisement sweep (lane F38)

Sixty-seven zero-evidence advertisement rows were removed across sixteen
submodule `__all__` lists: 66 names are DEMOTED (still importable at their
modules, no longer advertised), the duplicate `label` advertisement is
deduped, and `validate_trace_saved_outs` is DELETED outright. Top-level
`torchlens.__all__` is untouched by this sweep.

## `extract_dataset` runs no_grad + eval by default (lane A11, behavior change)

Extraction forwards now run under `no_grad` in eval mode with exact
mode/flag restore afterward. The old default read train-mode activations and
mutated BatchNorm running statistics during harvest -- silently corrupting
RDMs downstream. Train-mode extraction, where genuinely wanted, must now be
requested explicitly.

## rsatoolbox descriptor retirement: "neuroid" -> `feature_index` (lane F22)

`bridge.rsatoolbox.dataset` (now a delegation to `tl.neuro.datasets`)
retires the "neuroid" channel-descriptor name for the neutral
`feature_index` (authorized retirement). Legacy shaping/`input_shape`/
integer-`presentation` spellings are preserved for existing readers; the
previously silent flatten is now disclosed as `pool="flatten"`.

## `VisualizationTheme.legend_items` deleted (lane F12)

The dead `legend_items` theme field is DELETED. Themes carry
`semantic_palette`/`ramp`/`neutral_aggregate_fill`; legends derive from the
active encoding channels, never from a static theme list.

## Root typing leaks removed (lane P02)

`torchlens.Any`, `torchlens.TYPE_CHECKING`, and `torchlens.annotations` --
accidental typing re-exports, never API -- are removed from the root
namespace. Import them from `typing`/`__future__`.

## Structure-only capture admits all-meta models (lane F33, default flip)

`CaptureOptions(structure_only=True)` now ADMITS meta-materialized models
when the substrate is uniform (all-meta inputs and state); mixed real/meta
substrates refuse typed (`structure_only_substrate_mismatch`) in both
directions. Previously all meta models refused at entry. Value-bearing
claims remain hypotheses until `discharge_against` corroborates them; the
parity gate (real digest == meta digest + CORROBORATED discharge) is the
acceptance authority.

## Extraction manifest v2 (lanes C04/F18)

`tl.extract_dataset` writes manifest v2 (model identity, transform
disclosure, input-preprocessing block, per-site identity with L1 site keys).
v1 artifacts remain readable through `load_extraction`/`open_extraction`;
resume continuation across the version boundary refuses typed
(`extraction_resume_*`) rather than grafting mixed-manifest shards.

## Gaussian RDM values changed: `repgeom.rdm(metric="gaussian")` now matches thingsvision (2026-10-14, numeric fix)

`torchlens.repgeom.rdm(features, metric="gaussian")` now returns
`1 - exp(-D / mean(D))`, where `D` is the squared Euclidean distance between
stimuli and `mean(D)` is its mean over the whole matrix. This is the formula
thingsvision's `gaussian_kernel` uses and the one the docstring always cited.
Earlier releases divided by `2 * mean(D)`, so every gaussian RDM value differs
from earlier releases; other metrics are unchanged. For three stimuli with one
feature each, at 0, 1 and 3, the off-diagonal entries for the pairs (0, 1),
(0, 3) and (1, 3) were 0.148, 0.765 and 0.474 and are now 0.275, 0.945 and
0.724. Recompute any stored gaussian RDM, and any comparison score built on
one, with the new release.

## `trace.ops` covers only counted ops (2026-10-14, behavior change)

`len`, iteration, integer indexing, `keys()`/`values()`/`items()` and `to_pandas()`
of `trace.ops` now cover only the ops `trace.num_ops` counts, so
`len(trace.ops) == trace.num_ops`. Input, output and buffer rows are no longer in
that view. Code that read `trace.ops` as every row should read `trace.layer_list`
(or use `len(trace)` and `trace[int]`). `trace.input_ops`, `trace.output_ops`,
`trace.saved_ops` and `trace.saved_grad_ops` are unchanged.

## Zero-fire rules warn by default on every intervention door (2026-10-14, default flip)

`spec.bind(model)` and `torchlens.intervention.steer_generate(...)` no longer raise
`bind_zero_fire` by default when a rule never fires. The default `on_zero_fire` is now
`"warn"` at every door: the rule is recorded on `.last_report.zero_fire_rule_ids` and a
`TorchLensWarning` with code `bind_zero_fire` is emitted. Code that relied on the refusal
passes `on_zero_fire="error"`, which raises `BindingRuntimeError` (`bind_zero_fire`) after
the call exactly as before. `on_zero_fire="disclose"` keeps recording without a warning.

## `on_zero_fire` on `tl.trace` and `Trace.run` (2026-10-14, new keyword)

`tl.trace(..., intervene=..., on_zero_fire=...)` and `Trace.run(..., on_zero_fire=...)`
take the same knob as `spec.bind`. A capture-time `intervene=` that fired nowhere now warns
with code `trace_zero_fire` (it was an uncoded `UserWarning` with the same message start),
and a rerun hook entry that fired nowhere keeps warning `rerun_zero_fire`. `"error"` raises
`SiteResolutionError` with the same code; `"disclose"` records only. Rerun edits that cannot
fire live still refuse `rerun_selector_unmatchable_live` whatever the knob says.

## `bind_zero_fire_policy_invalid` renamed to `zero_fire_policy_invalid` (2026-10-14, code rename)

An `on_zero_fire` value outside `"warn"`, `"error"` and `"disclose"` now refuses with code
`zero_fire_policy_invalid` at every door, raised as `InvalidArgumentError` (a `ValueError`)
instead of `BindingPreflightError`. Branch on the new code; `"warn"` is now a valid value.

## A hooked `run(inputs=..., fast=True)` reports `unverifiable` (2026-10-14, verdict change)

`trace.run(inputs=..., fast=True)` on a live trace that applies its staged hooks or value
replacements used to report `path_faithfulness=verified`. Its output is an edited forward,
not the model's own, and no hook-aware check exists yet, so the report now reads
`unverifiable` (`poisoned=True`) with a `user_intervention_not_replayable` diagnostic
(stage `hooked_run_unverified`, detail `pristine=false`). Runs with nothing staged keep
their verdicts.

## Unarmed capture verification reads "not checked" (2026-10-14, wording)

`tl.report.explain` and the export honesty preamble now say "not checked" (was "not
recorded") for a default capture's `capture_verified=None`. The preamble header token
`verified=not_recorded` is unchanged. A default capture that builds a tensor from host data
(`torch.from_numpy`, `torch.from_dlpack`, `torch.frombuffer`, `torch.as_tensor` on
non-tensor data) also records a `host_tensor_source` row in
`trace.annotations["capture_advisories"]`.

## Bundle baseline auto-detection reads `Trace.pristine` (2026-10-14, behavior change)

Before: a Bundle picked the one member whose `Trace.state` was `TraceState.PRISTINE` and whose
spec had never been edited. A capture with
`tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))` kept that state, so it
could be picked, or make the choice ambiguous; assigning `trace.state = TraceState.PRISTINE` made
any member eligible.

After: a member is a candidate only when `trace.pristine is True`, derived from the evidence the
trace carries. Intervened members, members loaded from artifacts older than tlspec 9 and
non-torch members (`pristine is None`) are never auto baselines. If your bundle relied on
auto-detection with such members, pass the baseline explicitly:
`tl.Bundle({...}, baseline="clean")` or `bundle.most_changed(baseline="clean")`. Code that read
`trace.state is TraceState.PRISTINE` to mean "never intervened" should read
`trace.pristine is True`.

## Agent tools page at most 150 rows (2026-10-14, limit change)

A request with `max_rows` between 151 and 200 now refuses `agent_limit_invalid`; pass
`max_rows<=150` and follow `data.next` for the rest. `torchlens_overview` serves up to 12,000
tokens by default (was 4,000). Both numbers are configuration set from a corpus measurement
(`tools/agent_evals/BUDGETS.md`), not a guaranteed API limit.

## Patch-grid receipts plot effects (2026-10-14, value change)

Receipts from `tviz.receipt_from_patch_grid` change value: `effects` are now `grid.values` minus
`grid.receipts["corrupted_metric"]`. Pass `joint_effect` as a delta from the same run (for a
whole-layer grid: `float(joint.values[i]) - joint.receipts["corrupted_metric"]`). A hand-built
`PatchGrid` must set `receipts["corrupted_metric"]` or the call refuses `tv_receipt_invalid`. The
default `intervention` wording is now "clean activation of one head patched into the corrupted
run" (was "head contribution zeroed"); the grid title and joint line no longer say "ablation".

## `bridge.sae_lens.decode` removed; zero-variance splice FVE is `nan` (2026-10-15, removal and value change)

`torchlens.bridge.sae_lens.decode(log, site, sae)` is removed: it passed the site activation to
`sae.decode` as if it were latents. Use `torchlens.bridge.sae_lens.reconstruct(log, site, sae)`
for the reconstruction, or `sae.decode(latents)` on latents you hold.
`SpliceResult.fraction_variance_explained` is `nan` (with `zero_variance=True`) when the site's
total variance is zero; it was 0.0.

## `save="stats"` is a save level (2026-10-15, spelling change)

A bare `save="stats"` string used to be a substring selector for layers with "stats" in their
names; it is now the reduce-and-discard save level (`tl.stats.site_spines(trace)` reads it). To
select those layers by name, write `save=["stats"]`.
