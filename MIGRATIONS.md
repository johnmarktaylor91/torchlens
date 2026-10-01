# TorchLens Migration Policy

## `.tlspec` Compatibility

TorchLens 2.16.0 wrote two public `.tlspec` directory formats:

- Intervention specs with `spec.json` containing `format_version`.
- Portable `ModelLog` bundles with `manifest.json` containing `io_format_version`.

These 2.16.0 formats are permanently readable. TorchLens does not auto-migrate
them in place; readers dispatch by detected format and preserve support for the
legacy schemas.

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
| Batch-row permutation | planned stochastic-edit verb (not this helper) |
| Per-row donor resampling | planned stochastic-edit verb (not this helper) |

Compatibility: saved intervention specs and pickles that persist the helper
name `"resample_ablate"` keep loading (they reconstruct through the renamed
constructor). No runtime deprecation shim is added; the old top-level
spelling is retired with the facade export flip.

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
