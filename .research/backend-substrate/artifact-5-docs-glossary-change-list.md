# M0.1a Artifact 5: Docs/Glossary Change List

Date: 2026-06-12
Plan source: `.research/jax-tinygrad-sprint_PLAN.md` v13, M0.1a artifact 5.
Scope: design artifact only. No code, tests, ruff, pytest, or benchmark work.

## Purpose

This artifact is the merge-gate checklist for backend-substrate documentation. The plan's rule is
strict: no public backend surface lands unless the same PR updates the glossary, docs, and notebooks
that teach or rely on that surface. This document does not change user docs. It names the required
future edits, the source anchors they must stay true to, and the fixture inventory that will verify
docs and public API do not drift.

Current documentation is torch-first in three different ways:

- The glossary defines `tl.trace(model, x)` as a real PyTorch forward pass and the `Trace` object as
  containing PyTorch modules, params, buffers, autograd gradients, and metadata
  (`.project-context/torchlens_glossary.md:19`, `.project-context/torchlens_glossary.md:20`,
  `.project-context/torchlens_glossary.md:24`, `.project-context/torchlens_glossary.md:26`,
  `.project-context/torchlens_glossary.md:28`).
- The limitations docs explain TorchLens as Python-level PyTorch wrapper capture and explicitly say
  pure JAX workflows have no equivalent (`docs/LIMITATIONS.md:3`, `docs/LIMITATIONS.md:4`,
  `docs/LIMITATIONS.md:62`, `docs/LIMITATIONS.md:63`).
- Public code already has an internal backend tag for torch and MLX only, but no public `backend=`
  kwarg or registry docs (`torchlens/data_classes/trace.py:1007`,
  `torchlens/data_classes/trace.py:1351`, `torchlens/user_funcs.py:1383`,
  `torchlens/user_funcs.py:1422`, `torchlens/user_funcs.py:1664`).

## Merge-Gate Rule

Every implementation PR that changes one of these public names or user-visible semantics must include
the matching docs and glossary edits in that same PR:

| Public change | Required docs in same PR |
|---|---|
| Add or change `backend=` on `tl.trace()` or `tl.validate()` | Glossary signature rows, README/backend quickstart, limitations backend matrix, migration notes, and audit notebook README. |
| Add `tl.backends.*` namespaces or `tl.backends.jax.GradOptions` | Glossary backend section, package API docs, JAX preview page or subsection, and notebook/example import cells. |
| Add `Trace.backend`, `module_identity_mode`, `param_source`, or `derived_grads` | Glossary `Trace` identity/gradient sections, dataframe/report docs if displayed, save/load docs if serialized, and backward docs for non-true-backward wording. |
| Change `.tlspec` manifest schema behavior | Save/load docs, glossary portable-save section, limitations, state/architecture docs, and schema validation docs. |
| Change MLX or tinygrad capability errors | Glossary backend integration row, limitations backend matrix, and public unsupported-surface examples. |
| Add JAX or tinygrad examples | Notebook/gallery index, limitations, dependency install docs, and backend capability matrix. |

The gate is not satisfied by TODOs. If the public name is importable or the behavior is reachable, the
docs must describe the exact current behavior, including unsupported cases.

## Glossary Change List

The glossary is currently a target-name reference that admits it may differ from source
(`.project-context/torchlens_glossary.md:5`, `.project-context/torchlens_glossary.md:6`). Backend
substrate work must tighten that convention for backend names: backend-related entries must describe
only shipped behavior, with preview/future items clearly marked as such.

Required glossary edits:

| Glossary area | Current anchor | Required change |
|---|---|---|
| Top-level vocabulary | `tl.trace(model, x)` says PyTorch forward pass (`.project-context/torchlens_glossary.md:19`). | Change to backend-resolved capture. State torch eager remains default, MLX is technical preview, JAX is explicit `backend="jax"` preview after M1, tinygrad only after M2 funding. |
| Trace identity | `_backend_name` is private/provisional (`.project-context/torchlens_glossary.md:47`). | Replace with public `Trace.backend: BackendName`; list `"torch"`, `"mlx"`, `"jax"`, and future `"tinygrad"` only when registered. |
| Trace params/modules | `Module` and `Param` entries are PyTorch-specific (`.project-context/torchlens_glossary.md:24`, `.project-context/torchlens_glossary.md:26`). | Add `module_identity_mode` values `torch_module`, `pytree_module`, and `function_root`; add `param_source` values `native-module`, `pytree-derived`, and `none`. |
| Trace signature | `tl.trace(...)` row has no `backend=` and embeds MLX dispatch prose (`.project-context/torchlens_glossary.md:940`). | Add `backend: BackendName | None = None`; move MLX/JAX/tinygrad capability wording to backend integration rows. |
| Fastlog signature | `tl.fastlog.record(...)` is documented as general sparse capture (`.project-context/torchlens_glossary.md:944`). | State `tl.record()` / fastlog is torch-only in backend v1; non-torch backends raise canonical unsupported errors. |
| Validation signature | `tl.validate(...)` has no `backend=` and all scopes sound torch-autograd shaped (`.project-context/torchlens_glossary.md:960`). | Add `backend=None`; document bool contract for all backends and JAX M1 per-equation replay/perturbation. |
| Backward records | `GradFn` and `GradFnCall` are top-level vocabulary (`.project-context/torchlens_glossary.md:28`, `.project-context/torchlens_glossary.md:29`). | Mark them as true-backward/autograd records. JAX M1 `derived_grads` is not a `GradFn` graph and must not populate backward accessors. |
| Backend integration | Current rows mention `Trace._backend_name`, MLX hard rejections, and MLX pin (`.project-context/torchlens_glossary.md:1103`, `.project-context/torchlens_glossary.md:1106`). | Replace with `BackendSpec`, `BackendName`, resolution order, capability error names, and per-backend matrix. |
| Portable save scrub | Current scrub entry names unsupported MLX tensors (`.project-context/torchlens_glossary.md:1108`, `.project-context/torchlens_glossary.md:1110`). | Add the three serialization version axes and audit-only non-torch payload access behavior from artifact 3. |

New glossary entries required before public backend API ships:

- `BackendName`: literal backend identifier used by `backend=` and `Trace.backend`.
- `BackendSpec`: registry object owning resolution, capture, validation, serialization policy, and
  capability errors.
- `Trace.backend`: public backend tag persisted through portable state.
- `Trace.module_identity_mode`: explicit mode for module-like records.
- `Trace.param_source`: source of parameter records.
- `Trace.derived_grads`: JAX leaf-gradient preview results from a second JAX AD execution.
- `tl.backends.jax.GradOptions`: JAX derived-gradient configuration object.
- `backend_runtime`: manifest runtime fingerprint object.
- `payload_policy`: manifest policy for `full`, `audit_only`, or `metadata_only` payloads.
- `BackendUnsupportedError`, `BackendMismatchError`, `BackendAmbiguityError`,
  `BackendPayloadUnsupportedError`, and `BackendRuntimeCompatibilityError`.

## Documentation Change List

| File | Current anchor | Required backend-substrate edit |
|---|---|---|
| `README.md` | Overview says every intermediate operation in a PyTorch model (`README.md:20`, `README.md:22`, `README.md:68`). | Add a short backend-status section. Keep torch as default/stable; MLX technical preview; JAX functional preview only when M1 lands; tinygrad discovery/funded status only after gate. |
| `README.md` | Intervention publishing says `.tlspec` has JSON metadata plus tensor sidecars (`README.md:129`, `README.md:130`). | Add backend-conditional payload wording: non-torch preview bundles may be audit-only and fail loudly on payload materialization. |
| `README.md` | Fastlog section positions fastlog as a general lighter path (`README.md:348`, `README.md:352`). | State fastlog/`tl.record()` is torch-only for backend v1. |
| `README.md` | Training section uses `train_mode=True` and raw `backward()` language (`README.md:373`, `README.md:376`). | Keep torch-specific; do not imply JAX derived gradients are graph-connected PyTorch training activations. |
| `docs/LIMITATIONS.md` | Wrapper model assumptions and JAX "no equivalent" language (`docs/LIMITATIONS.md:3`, `docs/LIMITATIONS.md:4`, `docs/LIMITATIONS.md:62`, `docs/LIMITATIONS.md:63`). | Split into backend-specific limitations: torch wrapper capture, JAX jaxpr-first capture, MLX preview, tinygrad funded/discovery state. Replace "no equivalent" with precise JAX preview limitations only after M1. |
| `docs/backward.md` | Backward capture is described through PyTorch autograd sidecar/projection (`docs/backward.md:3`, `docs/backward.md:6`). | Add backend matrix: torch true backward capture; MLX unsupported; JAX derived-gradient preview is not true backward capture; tinygrad depends on payload gate. |
| `docs/backward.md` | Current future wording claims MLX T0/T1 path and vjp composition (`docs/backward.md:139`, `docs/backward.md:140`). | Replace with v13 wording: JAX M1 leaf-level derived gradients only; JAX T1 is research; no committed "T1 yes via vjp composition" language. |
| `docs/performance.md` | Decision tree recommends `tl.trace`, `tl.record`, `storage=tl.to_disk` generally (`docs/performance.md:3`, `docs/performance.md:15`). | Add backend notes: JAX interpreter capture cost, `tl.record()` torch-only, non-torch storage may be audit-only. |
| `docs/performance.md` | Portable `.tlspec` says tensor sidecars and callables not portable (`docs/performance.md:107`, `docs/performance.md:109`). | Add manifest schema v2 wording and payload-policy caveat. |
| `docs/migration/penzai.md` | Pure JAX workflows have no equivalent and TorchLens is PyTorch-only (`docs/migration/penzai.md:14`, `docs/migration/penzai.md:18`). | Rewrite after M1 as "Penzai remains better for model surgery; TorchLens JAX preview captures declared functional calls as jaxpr-derived traces." |
| `docs/migration/v2.0_api_changes.md` | Selective capture and kwarg migration table has no backend axis (`docs/migration/v2.0_api_changes.md:110`, `docs/migration/v2.0_api_changes.md:118`). | Add `backend=` migration note and the generated kwarg matrix reference. |
| `docs/migration/v2.0_api_changes.md` | Validation and backward graph examples are torch-shaped (`docs/migration/v2.0_api_changes.md:176`, `docs/migration/v2.0_api_changes.md:222`). | State legacy torch examples remain torch; non-torch validation/drawing routes through backend capabilities. |
| `.project-context/state_of_torchlens.md` | Current state says PyTorch eager execution and `ModelLog` (`.project-context/state_of_torchlens.md:3`, `.project-context/state_of_torchlens.md:8`). | Update to current `Trace`/backend language after substrate lands; fix stale paths and include backend registry. |
| `.project-context/state_of_torchlens.md` | `.tlspec` section says safetensors tensor blobs only (`.project-context/state_of_torchlens.md:128`, `.project-context/state_of_torchlens.md:132`). | Add the three serialization version axes and schema v2 backend runtime/payload policy. |
| `notebooks/audit/README.md` | Audit notebook index includes save/load, backward, fastlog, validation topics (`notebooks/audit/README.md:12`, `notebooks/audit/README.md:15`, `notebooks/audit/README.md:19`). | Add backend audit notebook rows or explicit "torch-only" labels for existing notebooks when backend examples are not yet shipped. |
| `examples/5min/README.md` | Save/load example is generic `.tlspec` round-trip (`examples/5min/README.md:10`). | Mark as torch materialized-payload example; add JAX audit-only example only after M1 serialization fixture exists. |

## Notebook and Example Change List

Existing notebooks are not line-addressable in this artifact because they are `.ipynb` JSON and this
round is docs-only, but their index files are. Backend PRs must update these user-facing collections:

- `notebooks/audit/05_save_and_load.ipynb`: add schema-axis and payload-policy cells when schema v2
  lands.
- `notebooks/audit/06_backward_and_gradients.ipynb`: add a warning cell that JAX `derived_grads`
  is separate from true backward capture when JAX M1 lands.
- `notebooks/audit/08_fastlog.ipynb`: label fastlog as torch-only in the backend v1 era.
- `notebooks/audit/12_validation_stats_reporting.ipynb`: add `backend=` validation examples only
  once registry dispatch is implemented.
- `examples/5min/save_load.ipynb`: keep torch materialized-payload round-trip; add non-torch
  audit-only access failure only after the fixture exists.
- New JAX preview example, only after M1: functional call with explicit `backend="jax"`, explicit
  statics/params, `GradOptions`, real `validate() -> bool`, and clear unsupported nested-jaxpr/effect
  examples.
- New tinygrad example, only after M2 go decision: include exact version pin and payload capability
  wording from S0.G.

## Wording Contracts

These wording rules are part of the gate:

1. Do not describe JAX M1 as true backward capture. Use "derived-gradient preview" and say it is a
   second JAX AD execution over declared leaves.
2. Do not say "T1 via vjp composition" in committed roadmap docs. Plan v13 moved JAX T1 to research.
3. Do not describe non-torch `.tlspec` payloads as materializable unless a backend codec exists.
   Use "audit-only" and document typed access failures.
4. Do not call JAX wrapper-based. JAX capture is jaxpr-first unless S0.J evidence reopens the
   wrapper fallback at the impact gate.
5. Do not make `tl.record()` look backend-general. It remains torch-only in v13.
6. Do not hide module identity differences. Use `function_root` for raw JAX functions and reserve
   `pytree_module` for an adapter such as Equinox.
7. Do not use old internal names such as `_backend_name` for new public docs once `Trace.backend`
   lands.
8. Do not conflate the on-disk family, manifest schema, and pickled object-state version.

## Per-Item Decisions

1. The glossary is the authoritative public-name checklist for backend substrate, but backend-related
   entries must describe shipped behavior rather than target-only names.
2. `backend=` docs must ship with the first implementation PR that exposes the kwarg, not later in a
   docs sweep.
3. JAX docs must present explicit `backend="jax"` as the initial public spelling unless the impact
   gate changes autorouting.
4. JAX `derived_grads` docs live beside backward docs but are explicitly not true backward graph
   docs.
5. `.tlspec` docs must use the three-axis vocabulary from artifact 3.
6. Existing torch notebooks can remain torch-first, but their indexes must say so once non-torch
   backends become public.
7. MLX docs must move from ad hoc hard-rejection prose to capability/error language once
   `BackendSpec` canonical errors land.
8. tinygrad docs remain discovery/future wording until the impact gate funds M2 implementation.

## Open Questions For The Impact Gate

1. Should the first JAX public docs be a standalone `docs/jax.md` page, a `docs/backends.md` page, or
   a subsection in limitations plus one notebook?
2. Does `backend=None` remain torch/MLX-only in docs for all of M1, or does the gate allow callable
   autorouting to JAX?
3. Should non-torch audit-only `.tlspec` examples appear in the README, or only in save/load docs to
   avoid implying materialized payload parity?
4. Should `tl.backends.jax.GradOptions` be shown in the top-level glossary signature row or only in
   a backend-specific section?
5. Which notebook becomes the canonical JAX preview: a new audit notebook, a 5-minute gallery
   notebook, or both?
6. Should the Penzai migration doc position TorchLens JAX preview as complementary or keep an
   explicit recommendation to use Penzai for model surgery?

## Fixture/Test Inventory For Implementation

- Docs API truth test: generated check that glossary `tl.trace` and `tl.validate` signatures include
  `backend=` once the code does.
- Backend wording grep: fail on committed docs phrases "JAX true backward", "T1 via vjp composition",
  and unqualified "TorchLens is PyTorch-only" after JAX M1 public docs land.
- Glossary backend entry test: assert `BackendName`, `BackendSpec`, `Trace.backend`,
  `module_identity_mode`, `param_source`, `derived_grads`, `backend_runtime`, and `payload_policy`
  entries exist when the corresponding code names are exported.
- Notebook index test: every backend notebook/example has an index row and every torch-only audit
  notebook is labelled torch-only once non-torch backends are public.
- Serialization docs test: docs mention on-disk family, manifest schema, and pickled object-state
  version together on the save/load page.
- Fastlog docs test: docs mentioning `tl.record()` or `tl.fastlog.record()` also mention torch-only
  capability after registry errors land.
- Validation docs test: docs describing `tl.validate(..., backend="jax")` assert a bool return and
  do not direct users to diagnostics as validation.
- Payload docs test: docs for non-torch `.tlspec` examples include an audit-only access-failure
  example until a materializing codec ships.
- Migration docs test: Penzai and limitations docs no longer contradict public JAX preview once M1
  is merged.
- README smoke snippet inventory: torch default snippet, explicit JAX preview snippet, and
  non-torch unsupported-surface snippet are listed as runnable or intentionally skipped by marker.
