# M0.1a Artifact 3: Serialization Contract With Three Version Axes

Date: 2026-06-12
Plan source: `.research/jax-tinygrad-sprint_PLAN.md` v13, M0.1a artifact 3.
Scope: design artifact only. No code, tests, ruff, pytest, or benchmark work.

## Purpose

This artifact separates the three serialization version axes named in plan v13 §2.7 and defines
the minimum schema changes needed for backend substrate work without weakening existing torch
`.tlspec` compatibility.

Current code intentionally mixes public JSON manifests with pickled object state:

- the public format detector returns `"v2.0_unified"` whenever `manifest.json` has both `kind`
  and `tlspec_version` (`torchlens/io/__init__.py:18`, `torchlens/io/__init__.py:38`,
  `torchlens/io/__init__.py:42`);
- unified trace saves write `manifest.json` plus `metadata.pkl` after scrubbing tensor blobs
  (`torchlens/_io/bundle.py:229`, `torchlens/_io/bundle.py:292`,
  `torchlens/_io/bundle.py:297`, `torchlens/_io/bundle.py:303`);
- load dispatch reads the public manifest kind, then still converts trace manifests through
  `Manifest.from_dict()` and unpickles scrubbed state (`torchlens/_io/bundle.py:430`,
  `torchlens/_io/bundle.py:439`, `torchlens/_io/bundle.py:574`,
  `torchlens/_io/bundle.py:580`, `torchlens/_io/bundle.py:512`);
- pickled object state has its own `_io.TLSPEC_VERSION = 4` and per-object
  `tlspec_version` validation (`torchlens/_io/__init__.py:25`,
  `torchlens/_io/__init__.py:27`, `torchlens/_io/__init__.py:81`,
  `torchlens/data_classes/trace.py:3002`, `torchlens/data_classes/trace.py:3007`).

Implementation contract: M0.2 must not rename the public family marker. It must add a manifest
schema v2 that is backend-aware, while leaving torch v1 manifests loadable and only bumping the
pickled object-state version if serialized object fields actually change.

## Three Version Axes

| Axis | Current source | Current value | Backend-substrate contract |
|---|---|---:|---|
| On-disk family detector | `detect_tlspec_format()` checks `kind` + `tlspec_version` (`torchlens/io/__init__.py:38`, `torchlens/io/__init__.py:42`). | `"v2.0_unified"` string. | Unchanged. New backend manifests still detect as `"v2.0_unified"` so public routing and old docs do not fork by backend. |
| Manifest schema | `_TlSpecWriter` writes `TLSPEC_VERSION = 1` and `TLSPEC_SCHEMA_VERSION = 1` (`torchlens/_io/tlspec.py:23`, `torchlens/_io/tlspec.py:24`, `torchlens/_io/tlspec.py:280`, `torchlens/_io/tlspec.py:287`). `validate_tlspec()` loads `tlspec_manifest_v1.json` (`torchlens/validation/__init__.py:25`, `torchlens/validation/__init__.py:62`). | Public JSON schema v1. | Add `torchlens/schemas/tlspec_manifest_v2.json`; write `schema_version: 2` for backend-aware manifests. Keep `tlspec_version: 1` unless the family detector semantics change, which this sprint must not do. |
| Pickled object-state schema | `_io.TLSPEC_VERSION = 4`; `Trace.__getstate__()` stores that value; `Trace.__setstate__()` validates/default-fills it (`torchlens/_io/__init__.py:27`, `torchlens/data_classes/trace.py:3002`, `torchlens/data_classes/trace.py:3007`, `torchlens/data_classes/trace.py:3008`). | Pickled state v4. | Do not bump for manifest-only changes. Bump to v5 only if `Trace`, `Op`, `Layer`, `Param`, module, grad-fn, or accessor pickled fields change; then add default-fill and legacy normalization before any new fixture is accepted. |

The naming rule is mandatory: public prose must call axis 1 the "on-disk family", axis 2 the
"manifest schema", and axis 3 the "pickled object-state version". New code comments and tests
must not call all three "tlspec_version" without qualification.

## Manifest V2 Field Contract

Current unified manifests require torch-specific fields and torch-shaped fingerprints:

- `torch_version` is required by both the bundled JSON schema and runtime validator
  (`torchlens/schemas/tlspec_manifest_v1.json:7`, `torchlens/schemas/tlspec_manifest_v1.json:13`,
  `torchlens/validation/__init__.py:93`, `torchlens/validation/__init__.py:98`);
- `Manifest.from_dict()` requires `torch_version` as a non-empty string before trace load
  (`torchlens/_io/manifest.py:219`, `torchlens/_io/manifest.py:221`,
  `torchlens/_io/manifest.py:235`);
- `model_fingerprint` requires `parameter_meta_hash`, `buffer_meta_hash`, and
  `class_qualname` (`torchlens/schemas/tlspec_manifest_v1.json:59`,
  `torchlens/schemas/tlspec_manifest_v1.json:62`, `torchlens/validation/__init__.py:126`,
  `torchlens/validation/__init__.py:142`);
- compatibility policy parses and compares PyTorch major/minor versions unconditionally
  (`torchlens/_io/manifest.py:376`, `torchlens/_io/manifest.py:407`,
  `torchlens/_io/manifest.py:424`).

Schema v2 must add these fields:

| Field | Required | Type | Contract |
|---|---|---|---|
| `backend` | Yes | string enum | One of `"torch"`, `"mlx"`, `"jax"`, future `"tinygrad"`. For torch v2 it must equal `trace.backend`, currently persisted in `Trace.PORTABLE_STATE_SPEC` (`torchlens/data_classes/trace.py:1007`, `torchlens/data_classes/trace.py:1028`, `torchlens/data_classes/trace.py:1032`). |
| `backend_runtime` | Yes | object | Backend-owned runtime fingerprint with `name`, `version`, `runtime_config`, `device_summary`, and `compat_policy`. Torch v2 may include current `torch_version` here as well as the legacy top-level field. JAX M1 must include JAX/JAXLIB versions, x64/promotion/precision config, PRNG implementation, device/sharding summary, primitive-table id, and async-barrier policy. |
| `torch_version` | Yes but nullable | string or null | Non-null only when `backend == "torch"` or when a torch compatibility dependency is genuinely needed to read torch-shaped payload metadata. Non-torch manifests must set `null`, not omit it. |
| `model_fingerprint` | Yes but backend-conditional shape | object or null | Torch keeps v1 shape. JAX function-root uses a backend-neutral fingerprint with callable identity, closed-jaxpr digest, treedef digest, const/static digests, param-leaf digest if `params=` is declared, and differentiated-leaf digest if derived grads are present. MLX/tinygrad must use backend-owned fingerprint objects after their discovery gates. |
| `backward_summary` | Yes but nullable | object or null | Torch true-backward traces keep v1 summary. JAX M1 without true backward capture uses `null` plus derived-gradient metadata elsewhere; it must not fake `has_backward_pass`. Metadata-only backends use `null`. |
| `derived_gradient_summary` | Backend-conditional | object or null | Required non-null when JAX `trace.derived_grads` exists. Contains loss identity, differentiated leaf paths, repeated-gradient digest, config/runtime fingerprints, and aux-output equality summary. It is not a backward graph summary. |
| `payload_policy` | Yes | object | Declares whether payloads are `full`, `audit_only`, or `metadata_only`; whether direct materialization is supported after load; and which payload kinds are present. JAX M1 non-torch payloads are audit-only until a codec exists. |

Top-level `kind`, `created_at`, `torchlens_version`, `python_version`, `schema_version`,
`model_signature`, `sites`, `spec_compat_info`, `body_format`, `body_index`, `save_level`,
`optional_dependencies`, and `intervention_compat_metadata` remain present. `additionalProperties`
may remain true as in v1 (`torchlens/schemas/tlspec_manifest_v1.json:5`,
`torchlens/schemas/tlspec_manifest_v1.json:6`), but v2 runtime validation must fail closed on
unknown backend names and unsupported schema versions.

## Payload and Body Contract

Current tensor body support is torch/safetensors-specific:

- `TensorEntry` stores storage `backend`, `shape`, `dtype`, `device_at_save`, `layout`, bytes, and
  sha256 (`torchlens/_io/manifest.py:30`, `torchlens/_io/manifest.py:44`,
  `torchlens/_io/manifest.py:60`);
- trace save writes supported tensors as safetensors and records skipped unsupported tensors when
  `strict=False` (`torchlens/_io/bundle.py:258`, `torchlens/_io/bundle.py:271`);
- lazy refs materialize to `torch.Tensor` and resolve dtypes via `getattr(torch, dtype_name)`
  (`torchlens/_io/lazy.py:27`, `torchlens/_io/lazy.py:71`,
  `torchlens/_io/rehydrate.py:566`, `torchlens/_io/rehydrate.py:662`,
  `torchlens/_io/rehydrate.py:681`);
- `body_format` is validated as only `"safetensors"` (`torchlens/validation/__init__.py:103`,
  `torchlens/schemas/tlspec_manifest_v1.json:120`).

Contracts:

1. Torch payload behavior stays unchanged: `out`, `grad`, saved args, RNG states, child versions,
   and transformed payloads continue to use current blob policies (`torchlens/data_classes/op.py:802`,
   `torchlens/data_classes/op.py:812`, `torchlens/data_classes/op.py:834`,
   `torchlens/data_classes/op.py:835`, `torchlens/data_classes/op.py:853`).
2. Non-torch payloads are audit-only until a backend codec is explicitly implemented. Their
   manifests may record array metadata and digests, but `tl.load(...).layer.out` materialization
   must raise a backend payload unsupported error rather than returning an incorrect `torch.Tensor`.
3. `body_format` v2 becomes backend-conditional: `"safetensors"` for torch payloads, `"audit_only"`
   for JAX M1 non-torch payload manifests, and future codec literals only after registry approval.
4. `body_index[*].intended_use` must grow through schema v2 rather than reusing torch grad names
   for derived gradients. Proposed additions: `jax_equation_out`, `jax_input_leaf`,
   `jax_const_leaf`, `jax_param_leaf`, `jax_derived_grad`, and backend-owned `audit_record`.
5. Access-after-load failures for audit-only payloads must be loud and typed. Silent `None`,
   accidental empty tensors, or torch dtype coercion are forbidden.

## Save-Level Contract

Current save levels are `audit`, `executable_with_callables`, and `portable`
(`torchlens/_io/tlspec.py:26`, `torchlens/_io/tlspec.py:583`), and trace save maps them to payload
inclusion policies (`torchlens/_io/bundle.py:202`, `torchlens/_io/bundle.py:203`,
`torchlens/_io/bundle.py:208`).

| Save level | Torch current behavior | JAX M1 contract | MLX/tinygrad contract |
|---|---|---|---|
| `audit` | Drops outs, grads, saved args, RNG states (`torchlens/_io/bundle.py:203`). | Allowed. Writes schema v2 audit metadata, closed-jaxpr digest, equation/site inventory, runtime fingerprint, no materializable payload promise. | Allowed if backend can write metadata without side effects. |
| `portable` | Saves portable tensor sidecars where supported and excludes executable callables by scrub policy. | Allowed only as audit-only until non-torch payload codec exists; docs must say non-torch payloads do not materialize after load. | Depends on S0.G payload decision. |
| `executable_with_callables` | Forces saved args and RNG states for replay (`torchlens/_io/bundle.py:208`). | Rejected for JAX M1 unless the impact gate explicitly accepts callable serialization. JAX executable replay should come from declared source inputs plus jaxpr interpreter, not pickled Python callables. | Rejected until a backend-specific executable policy exists. |

## Loader and Compatibility Contract

Schema v2 requires loader dispatch before torch-specific manifest parsing:

1. `_load_unified_tlspec()` must read `kind`, `schema_version`, and `backend` from raw JSON before
   constructing the current `Manifest` dataclass. Today trace loading calls `Manifest.from_dict()`
   directly (`torchlens/_io/bundle.py:580`, `torchlens/_io/bundle.py:583`), which cannot accept
   nullable `torch_version` or backend-shaped fingerprints.
2. A schema-v1 trace with no `backend` is interpreted as torch for compatibility. This is an
   inference from current writes, because `Trace.backend` defaults to `"torch"`
   (`torchlens/data_classes/trace.py:1351`) and v1 writer always records `torch_version`
   (`torchlens/_io/tlspec.py:286`).
3. `enforce_version_policy()` remains torch-only for v1/v2 torch manifests. For non-torch v2,
   compatibility moves to `BackendSpec.serialization_policy` and `backend_runtime.compat_policy`;
   unconditional torch parse/reject must not run.
4. `validate_tlspec()` must choose `tlspec_manifest_v1.json` or `tlspec_manifest_v2.json` by
   `schema_version`. Current code always loads v1 (`torchlens/validation/__init__.py:48`,
   `torchlens/validation/__init__.py:53`, `torchlens/validation/__init__.py:62`).
5. Loader tests must cover all three axes independently: family `"v2.0_unified"` with manifest
   schema v1/v2, plus pickled object-state v4/v5 if changed.

## Pickled Object-State Contract

Current portable object state is governed by `PORTABLE_STATE_SPEC` dictionaries and per-class
`__getstate__`/`__setstate__` methods. Trace keeps `backend` and drops callables/source model refs
(`torchlens/data_classes/trace.py:1028`, `torchlens/data_classes/trace.py:1032`,
`torchlens/data_classes/trace.py:1048`, `torchlens/data_classes/trace.py:1063`). Ops keep graph
metadata but drop `func`, `grad_fn_handle`, and `grad_fn` (`torchlens/data_classes/op.py:845`,
`torchlens/data_classes/op.py:866`, `torchlens/data_classes/op.py:867`) while preserving
container paths and graph edges (`torchlens/data_classes/op.py:871`, `torchlens/data_classes/op.py:893`,
`torchlens/data_classes/op.py:894`).

Contracts:

- Manifest-only schema v2 work must not bump `_io.TLSPEC_VERSION`.
- Adding stored fields such as `module_identity_mode`, `param_source`, `derived_grads`,
  `primitive_name`, `operation_group_id`, `backend_runtime`, or backend-neutral dtype refs is a
  pickled object-state change and requires a v5 bump unless the field is manifest-only.
- Every new pickled field needs a `PORTABLE_STATE_SPEC` entry and a `default_fill_state()` default
  for older fixtures before the implementation PR can land.
- Backward-compatible default for missing `backend` in old object state is `"torch"` only for
  schema v1 or plain legacy fixtures. New schema v2 manifests missing `backend` are invalid.
- Non-torch object state must not store executable callables in portable mode.

## Per-Item Decisions

1. Keep the on-disk family detector string `"v2.0_unified"` unchanged for backend-aware manifests.
2. Add manifest schema v2 via `schema_version: 2`; do not repurpose top-level `tlspec_version`.
3. Do not bump pickled object-state version for manifest-only fields; bump from 4 to 5 only when
   serialized Trace/Op/Layer/Param/etc. fields are added.
4. Make `backend` and `backend_runtime` mandatory in schema v2.
5. Make `torch_version`, torch-shaped `model_fingerprint`, and `backward_summary` backend
   conditional and nullable for non-torch.
6. Treat JAX M1 non-torch payloads as audit-only after load until a real non-torch codec exists.
7. Keep JAX derived gradients separate from true backward summary.

## Open Questions For The Impact Gate

1. Should schema v2 write `tlspec_version: 1` or introduce a less confusing alias such as
   `manifest_schema_family_version` while retaining `tlspec_version` for detector compatibility?
2. What is the exact JAX `backend_runtime` compatibility policy: hard pin JAX/JAXLIB patch version,
   minor-version warning, or import-time probe plus primitive-table id?
3. Should JAX audit-only payloads include digests of concrete arrays in `body_index`, or only in a
   backend-specific `jax_payload_manifest` object?
4. Is `portable` an acceptable label for non-torch audit-only payloads, or should JAX M1 force
   `level="audit"` until materialization exists?
5. Which future object fields are definitely manifest-only versus pickled state: `backend_runtime`,
   `module_identity_mode`, `param_source`, `derived_grads`, and primitive/equation metadata?

## Fixture/Test Inventory For Implementation

- Schema selector tests: `validate_tlspec()` loads v1 for schema v1 and v2 for schema v2 while both
  detect as `"v2.0_unified"`.
- Backcompat tests: existing v1 torch fixtures load unchanged; v1 manifests without `backend`
  infer torch only during load.
- Schema v2 torch fixture: writes `backend="torch"`, non-null `torch_version`, v1-shaped
  `model_fingerprint`, and current `backward_summary`; round-trips equal current torch behavior.
- Schema v2 JAX audit fixture: `backend="jax"`, `torch_version=null`, backend-shaped fingerprint,
  `backward_summary=null`, non-null `backend_runtime`, and audit-only payload policy.
- Loader dispatch test: non-torch v2 trace does not call `Manifest.from_dict()` or
  `enforce_version_policy()` before backend serialization dispatch.
- Access-after-load test: JAX audit-only payload access raises typed backend payload unsupported
  error, not `None`, not `torch.Tensor`, and not a torch dtype parse failure.
- Pickled-state axis test: if `_io.TLSPEC_VERSION` remains 4, schema v2 manifest-only fixtures keep
  object state v4; if it bumps to 5, v4 fixtures default-fill every new field.
- Corruption tests: missing `backend`, unknown backend, missing `backend_runtime`, non-null
  `backward_summary` on JAX M1, and torch manifest with `torch_version=null` all fail closed.
- Derived-gradient schema tests: JAX derived-gradient metadata validates while
  `backward_summary is None` and loaded trace still has `has_backward_pass=False`.
- Body-index tests: new intended-use literals are accepted only in schema v2; v1 validation rejects
  them.
