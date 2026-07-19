# Sparse runnable `.tlspec` frozen contract

Status: **AUTHORITATIVE AND FROZEN** for `sparse_recorded_taken_path_v2` (this document is the
explicit, versioned contract amendment superseding the frozen `sparse_recorded_taken_path_v1`
text; v1 artifacts remain covered by the legacy posture in section 1a).

This is the single implementation contract for the complete sparse runnable `.tlspec` surface. The
definitions in `torchlens.runnable` are its behavior-free typed mirror. A disagreement between this
document and those types is a release-blocking schema defect. Neither may change without an explicit,
versioned contract amendment.

The shipped surface includes the sparse producer, safe loader/resolver, state binder, transactional
executor, control-flow honesty checks, optional weight payloads, and optional selected-activation
attestation. It does not change the ordinary capture path or analysis save levels.

## 1. Versions and enum vocabularies

| Name | Frozen value |
|---|---|
| sparse capability/schema | `sparse_recorded_taken_path_v2` |
| call recipe | `non_tensor_args_tensor_slots_and_context_v2` |
| callable-ref schema | integer `1` |
| state binding | `module_path_role_v1` |
| input binding | `model_site_io_role_v1` |
| control-witness schema | `scalar_bool_and_arm_entry_v1` |
| initializer policy | `torchlens_role_init_v2` |
| optional weight payload schema | `state_dict_v1` |
| **required** non-persistent buffer payload schema | `runnable_nonpersistent_buffer_v1` |
| optional activation payload schema | `selected_activation_v2` |

The constant is exactly
`RUNNABLE_TLSPEC_SCHEMA_VERSION = "sparse_recorded_taken_path_v2"`. It is a capability version, not
the existing whole-bundle `TLSPEC_VERSION` or JSON manifest schema version.

v2 execution-context records are **REQUIRED and EXPLICIT**: every call descriptor carries a
`CallExecutionContext` (per-device autocast with disabled state written affirmatively, plus
grad/inference mode) and the descriptor carries one capture-scoped `AmbientExecutionContext`.
The parser REJECTS their absence. An absent context record therefore only ever means a legacy v1
artifact, which is analysis-only (section 1a) -- absence is never interpreted as "disabled" or
any other default.

## 1a. Legacy `sparse_recorded_taken_path_v1` posture

A v1 artifact loads for ordinary analysis. Its `.run()` capability is a typed readiness refusal
(`run_capability_unavailable`) naming the missing execution-context class; the actionable remedy
is re-capturing and re-saving under v2. A compatibility replay of a v1 artifact is admissible only
when every v2-required fact is provable from immutable v1 artifact metadata -- caller ambient
state, current backend defaults, an output dtype/value, or the absence of an old field is never
such a proof -- and no such prover ships in this revision. Legacy payload blob families stay
unbound on an analysis-only legacy load.

The complete `TensorSlotRole` values are `model_input`, `parameter`, `buffer`, `intermediate`,
`constant_like_tensor`, `rng_source`, and `output`. `constant_like_tensor` is a classification, not
permission to serialize a tensor. Rung 1 normally rejects it because no general value-free constant
initializer exists.

The complete `StateSlotRole` values are `weight`, `bias`, `norm_scale`, `norm_offset`,
`running_mean`, `running_var`, `counter`, and `generic_buffer`. There is no catch-all parameter role.

The complete `WitnessCompleteness` values are `complete`, `incomplete_scalar_escape`,
`incomplete_opaque_side_effect`, and `incomplete_unobserved_predicate`. If several incomplete
conditions apply, the graph value is the first in capture-event order and diagnostics name every
site. `complete` attests witness coverage, not numerical equivalence.

Other fixed sets are:

| Enum | Values |
|---|---|
| `PathFaithfulness` | `verified`, `diverged`, `unverifiable` |
| `DivergencePolicy` | `raise`, `return_diverged` |
| `ReadinessStatus` | `ready`, `unavailable` |
| `RunProvider` | `live`, `loaded_sparse`, `loaded_analysis` |
| `StateSource` | `live_model_state`, `embedded_capture_state`, `user_state_dict`, `random_initialization`, `not_applicable` |
| `NumericAttestationStatus` | `attested`, `numeric_attestation_failed`, `not_applicable`, `not_present` |
| `ResolverStatus` | `resolved_exact`, `resolved_alias`, `unavailable` |

`loaded_analysis` only explains unavailability. Old or pre-descriptor bundles are never promoted.

## 2. Safe non-tensor literal grammar

Every non-tensor argument leaf uses one of these tagged nodes. Untagged JSON is not a runnable call
recipe.

| Node | Exact fields | Constraint |
|---|---|---|
| `LiteralAtom` | `kind`, `value` | kind is `none`, `bool`, `int`, `float`, `str`, or `ellipsis`; value has exactly that Python type (`None` for `none` and for `ellipsis` -- the atom KIND, not the wire value, disambiguates a real `None` index from a `...` index) |
| `LiteralSlice` | `start`, `stop`, `step` | each component is a `LiteralAtom` restricted to kind `none` or `int`; decodes to `slice(start, stop, step)` |
| `LiteralTorchSymbol` | `qualname` | non-callable symbol below an explicitly allowlisted stock torch root |
| `LiteralSequence` | `kind`, `items` | kind is `list` or `tuple`; items are ordered literal nodes |
| `LiteralMapping` | `entries` | ordered `LiteralMappingEntry` nodes; duplicate keys invalid |
| `LiteralMappingEntry` | `key`, `value` | key is `LiteralAtom` or recursive `LiteralTupleKey`; value is any literal node |
| `LiteralTupleKey` | `items` | ordered `LiteralAtom` or `LiteralTupleKey` nodes only |

Floats round-trip without coercion, including non-finite values. Booleans are not integers. Dict
order, tuple versus list, key types, nesting, and `None` are preserved. A Python `slice` (e.g. the
key of `x[:, 3:]`), a bare `Ellipsis` (`x[..., 0]`), and a bare `None` newaxis index (`x[:, None]`)
are inert value types with no callables or imports, so they round-trip exactly through the
`LiteralSlice` node and the `none`/`ellipsis` `LiteralAtom` kinds -- including inside the tuple key
`__getitem__` produces for multi-axis indexing (`LiteralSequence` of kind `tuple`). Bytes, complex
values, sets, arbitrary enums/objects, tensors, callables, classes, import references, pickles, and
opaque reprs are outside the grammar and fail preflight with `unsupported_literal`.

`LiteralTorchSymbol.qualname` is resolved only through an explicit non-callable allowlist. It never
authorizes `importlib`, arbitrary attribute walking, custom modules, or a callable fallback.

## 3. Callable key and capture-spine coupling

Rung 1 retains the existing `FunctionRegistryKey` shape exactly:

| Field | Type | Contract |
|---|---|---|
| `namespace` | `torch | torch.Tensor | torch.nn.functional | operator | custom` | existing vocabulary; runnable preflight rejects `custom` |
| `qualname` | `str` | qualified name below the namespace |
| `dispatch_kind` | `function | method | dunder | namespace_alias` | receiver/dispatch contract |
| `version` | `int` | must equal `1` |
| `import_path` | `str | null` | must be null; non-null is `untrusted_custom_import` |

`callable_registry` deduplicates keys as `CallableRegistryEntry(registry_id: str,
key: FunctionRegistryKey)`. IDs are unique; every computational call names one; unused entries are
invalid.

The sole future capture-spine coupling is this contract; Stage 0 does not implement it:

```python
FunctionCallRef.func_id: FunctionRegistryKey | None
Op.func_id: FunctionRegistryKey | None
```

Capture must populate the first where it already computes the live key, copy it to cooked `Op`, and
retain it with `FieldPolicy.KEEP`. The `Op` field, canonical FIELD_ORDER, and compatibility gates
change together. No raw capture graph is serialized and no other spine widening is authorized.

## 4. Authoritative descriptor

The manifest `run` member is a `SparseRunDescriptor` with exactly these fields. JSON arrays encode
typed tuples and enums encode as string values. Runnable capability is never inferred from other
fields.

| Field | Type / exact value |
|---|---|
| `capability` | `sparse_recorded_taken_path_v2` |
| `backend` | string; only `torch` executes in rung 1 |
| `call_recipe` | `non_tensor_args_tensor_slots_and_context_v2` |
| `callable_ref_schema` | integer `1` |
| `state_binding` | `module_path_role_v1` |
| `input_binding` | `model_site_io_role_v1` |
| `control_witness` | `scalar_bool_and_arm_entry_v1` |
| `initializer_policy_version` | `torchlens_role_init_v2` |
| `payload_layers` | `PayloadLayersDescriptor` |
| `callable_registry` | tuple of `CallableRegistryEntry` |
| `calls` | tuple of `RunnableCallDescriptor` |
| `tensor_slots` | tuple of `TensorSlotDescriptor` |
| `control_witnesses` | tuple of `ControlWitness` |
| `witness_completeness` | `WitnessCompleteness` |
| `rng_profile` | `RunnableRngProfile` |
| `ambient_context` | `AmbientExecutionContext` (REQUIRED, explicit) |
| `compatibility` | `RunnableCompatibility` |
| `preflight` | `ProducerPreflight` |
| `unsupported_sites` | tuple of `RunnableDiagnostic`; empty for a runnable claim |

`PayloadLayersDescriptor` has exactly `weights`, `nonpersistent_buffers`, and `activations`.
Weights and non-persistent buffers use `PayloadLayerDescriptor(present: bool, schema: str)`.
Activations use that same two-field form while absent; when present,
`ActivationPayloadLayerDescriptor` additionally carries exact `members`,
`original_input_digests`, `capture_state_digests`, and `input_fingerprints`. Each member
identifies its blob, slot, call, op label, `out`/`transformed_out` field, and logical byte digest.
Each `InputAttestationFingerprint` records the physical identity of one live capture-time
model-input slot: logical byte digest, device type/index, layout, exact sizes/strides/storage
offset, contiguity/channels-last flags, conjugate/negative bits, base-Tensor-vs-subclass
classification, grad/inference metadata, and data-pointer alignment class -- captured from the
live in-memory value that seeded the captured forward, never from an archived payload (payload
serialization contiguifies strides). Both sides fingerprint the EXECUTED-clone basis (the
retained capture-time input clone; the run-time defensive clone): a physical difference that
survives the clone (a channels-last memory format, an alignment-class change) makes the run
changed-input-for-attestation (`not_applicable`, path untouched), while a difference the clone
erases (a storage offset, non-dense slicing) leaves execution physically identical to capture, so
such a run honestly stays eligible with the byte-exact tripwire still armed. Schemas are respectively `state_dict_v1`,
`runnable_nonpersistent_buffer_v1`, and `selected_activation_v2`. Any payload bytes/references
live outside the sparse core.

### Execution-context records

`AmbientExecutionContext` (one per descriptor, capture-scoped) records exactly: `default_dtype`,
`default_device`, `float32_matmul_precision`, `deterministic_algorithms` (+`_warn_only`),
`cuda_matmul_allow_tf32`, `cudnn_allow_tf32`, `cudnn_deterministic`, `cudnn_benchmark`,
`cudnn_enabled`, `flash_sdp_enabled`, `mem_efficient_sdp_enabled`, `math_sdp_enabled`, and
`attestation_ineligible_context`. Every control the producing runtime exposes is recorded
affirmatively (explicit `false`); `null` means only "the producer runtime did not expose this
control" (feature-detected through `utils/_torch_compat.py` with named `HAS_*` flags).

`attestation_ineligible_context` is the POSITIVE capture-time marking for a nondeterministic
execution context: `cudnn.benchmark=true`, or a documented CUDA-nondeterministic op (the
transpose-conv atomicAdd family and the documented index/scatter accumulation set) running on a
CUDA device without `use_deterministic_algorithms(True)`. Such a capture can replay and verify
its path, but byte-exact numeric attestation is `not_applicable` -- fail-safe, never a false
`attested` and never a spurious `numeric_attestation_failed`. Users who want transpose-conv
attestation on CUDA should capture (and replay) under `torch.use_deterministic_algorithms(True)`.

`CallExecutionContext` (one REQUIRED per call descriptor) records the per-device autocast state
(`device_type`, `enabled`, portable dtype name -- disabled state written affirmatively) and the
grad/inference mode at the call's capture-time execution point. Replay enters exactly this context
tightly around the resolved call (actively entering `enabled=False` autocast so a caller's ambient
autocast cannot contaminate a disabled capture) and restores the caller's context on every exit.
Context entry never saves/restores RNG. A recorded context the runtime cannot enter or restore is
a typed refusal (`execution_context_unavailable`), never a silent ambient passthrough.

`RunnableCompatibility` has exactly `torchlens_version: str`, `python_version: str`,
`backend_version: str`, `descriptor_version: str`, `call_recipe_version: str`,
`callable_ref_schema_version: int`, and `initializer_policy_version: str`. The last four repeat
frozen descriptor values intentionally for diagnostics.

### Call descriptor

`RunnableCallDescriptor` has exactly:

```text
call_id: str
op_labels: tuple[str, ...]
registry_id: str
dispatch_kind: function | method | dunder | namespace_alias
argument_names: tuple[str, ...]
num_positional_args: int
num_keyword_args: int
tensor_arguments: tuple[TensorArgumentRef, ...]
literal_arguments: tuple[LiteralArgumentRef, ...]
output_slot_ids: tuple[str, ...]
parent_call_ids: tuple[str, ...]
is_inplace: bool
runtime_fingerprint: str
execution_context: CallExecutionContext
```

`TensorArgumentRef` is `argument_path: tuple[str | int, ...]` plus `slot_id: str`.
`LiteralArgumentRef` has the same path plus `value: NonTensorLiteral`. A path begins with `args`, a
positional index or `kwargs`, a keyword name, then container components. No path occurs in both
lists; together they completely describe args/kwargs. Parent IDs and list order define schedule;
existing graph edge-use/version metadata remains authoritative for repeated uses and mutation.

`runtime_fingerprint` is a non-executable digest of signature facts and call recipe, including
the canonical serialized `execution_context`. Its algorithm is producer-versioned and diagnostic;
the registry key is callable identity.

### Tensor slot

`TensorSlotDescriptor` has exactly:

```text
slot_id: str
role: TensorSlotRole
use_sites: tuple[TensorUseSite, ...]
shape: tuple[int, ...]
dtype: str
rank: int
device_type: str
device_index: int | None
mutable: bool
version_of: str | None
producer_slot_id: str | None
output_path: tuple[str | int, ...] | None
input_binding: InputSlotBinding | None
state_binding: StateSlotBinding | None
```

`rank == len(shape)` and dimensions are non-negative. Null device index means any index of the
required device class. `version_of` links mutation versions; `producer_slot_id` links produced
views/uses. Output/container leaves require `output_path`. `TensorUseSite` is exactly `call_id: str`
and `argument_path: tuple[str | int, ...]`.

`InputSlotBinding` is exactly:

```text
io_role: model_input
model_ref: str
model_site_position: str | int | tuple[str | int, ...]
container_record_id: int
container_path: tuple[str | int, ...]
```

It exists only on `model_input` slots. Binding uses the model-site/container path, never display
order.

`StateSlotBinding` is exactly:

```text
module_path: str
state_dict_name: str
semantic_role: StateSlotRole
trainable: bool
persistent: bool
alias_group: str | None
```

It exists only on `parameter`/`buffer` slots. Parameters are persistent. Non-persistent buffers are
not sourced from a `state_dict` but still need a role initializer. Alias members have identical
shape/dtype and share one allocation.

### Control witness

`ControlWitnessKind` values are `scalar_bool`, `conditional_arm_entry`, `loop_predicate`,
`shape_structure_fact`, and `tensor_derived_scalar_literal`. A witness has exactly
`witness_id: str`, `kind: ControlWitnessKind`, `order: int`, `call_id: str | None`,
`site_label: str`, and `observed_value: NonTensorLiteral`.

IDs are unique and order is dense zero-based. Scalar/loop predicates use boolean `LiteralAtom`;
arm entry uses a stable arm identity; shape/structure uses the literal grammar. Missing/opaque facts
use completeness plus diagnostics, never an opaque payload.

A `tensor_derived_scalar_literal` witness records a tensor->Python-scalar escape
(`.item()`/`int()`/`float()`/`aten._local_scalar_dense`) whose derived scalar was baked into a
downstream op as a literal constant. `site_label` is the runtime slot of the scalar-shaped internal
sink that produced the escaped scalar, `call_id` is that sink's recomputing call, and
`observed_value` is the scalar's capture-time value. At run time the executor recomputes the sink
slot: if its value differs from the witnessed capture-time value the baked literal may be stale, so
the run reports `path_faithfulness=unverifiable` and `numeric_attestation=not_applicable` instead of a
false `verified`/`attested`. The ORIGINAL/unchanged input recomputes the same scalar and still
reports `verified` + `attested`; `witness_completeness` stays `complete` because the downgrade is
input-conditional at run time, not a static descriptor flag.

## 5. Producer preflight and no-payload invariant

Preflight is whole-graph and fail-closed. After diagnostic failure a producer may write ordinary
analysis output but must not write runnable capability. It rejects when:

1. A computational group lacks a key, uses another ref schema, requests custom/import code, or is
   outside the stock resolver protocol.
2. An argument violates the literal grammar, contains a tensor literal, has an ambiguous path, or
   lacks arity/names/dispatch/fingerprint.
3. A tensor use lacks stable slot ID, role/path, shape/dtype/rank/placement, producer/version relation,
   or required input/state/output binding.
4. An input/output container is missing, opaque, unreconstructable, or lacks complete
   `ContainerRecord`/`ModelSite` paths.
5. State lacks canonical module path/name/role/shape/dtype/persistence/trainability/coherent aliases.
6. A tensor constant is neither input nor state and has no allowed value-free initializer.
7. A control site lacks completeness classification or a witness required by that classification.
8. A declared optional weight layer cannot represent state mutation after its snapshot.
9. Any forbidden sparse-core payload or executable reference survives final scrubbing.
10. Bound state alias topology is unsupported (r37): two DISTINCT live state tensor objects whose
    touched bytes overlap, or whose relation cannot be proven, refuse with
    `state_alias_topology_unsupported` (detection stage `producer_state_alias_topology`, details
    `reason`/`left_state`/`right_state`/`relation`). The topology is captured from the LIVE model
    objects before capture-state cloning erases it. Repeated live object IDENTITY (tied weights,
    double-registered buffers) is NOT refused: it becomes a shared `alias_group`, serialized and
    staged as ONE allocation so `a is b` and in-place propagation semantics replay exactly.
    Proved-disjoint views of one storage serialize independently and stay admitted. The v2 schema
    deliberately carries no backing-storage/view recipe; representing overlapping distinct views
    is a future versioned schema bump, not an implicit encoding.
11. The model output has ZERO tensor slots (r37, `zero_tensor_slot_output`): an all-literal tree,
    a literal root, or empty containers pass the losslessness proof but leave no output Op to
    carry the root `ContainerSpec`, so a loaded run would reconstruct `None`. Unrepresentable
    means refuse at save with `missing_output_container_contract`, uniformly with every other
    output refusal.

The hard invariant is:

> The sparse core contains zero tensor values, tensor blob files, tensor blob references,
> executable callables, and import instructions. Tensor payloads live only in the declared
> external blob families: the optional `state_dict_v1` and `selected_activation_v2` families,
> and the REQUIRED `runnable_nonpersistent_buffer_v1` family.

Forbidden sparse-core content includes op outputs/transformed outputs, inputs, activations,
gradients, child tensor versions, tensor args/templates, parameters, buffers, `state_dict`, state
snapshots, `_buffer_initial_values`, live handles/models, tensor RNG snapshots, callable pickles,
executable code, and custom import paths. Call recipes and the core descriptor never carry tensor
values or references; the external families are the only tensor carriers.

`runnable_nonpersistent_buffer_v1` is a REQUIRED external payload family whenever the taken path
uses a non-persistent registered buffer slot. It is written unconditionally -- it is NOT gated on
`include_weights` or `include_activations` -- because a used non-persistent buffer is declared
state (section 11) without which the artifact cannot replay. **Privacy note (prominent):** a
DEFAULT runnable save of such a model therefore carries user tensor data (the capture-time
non-persistent buffer values, which may hold arbitrary cached data) even with both include flags
false. The save discloses this: the family is manifest-visible
(`payload_layers.nonpersistent_buffers.present=true`) and the producer emits a one-time warning
when the family is non-empty.

Optional weights are declared as the external `state_dict_v1` blob family. With
`include_weights=True`, it contains one full capture-time `state_dict`: all named parameters and
persistent buffers keyed by canonical state records. It contains no gradients, RNG state,
callables, model handles, or per-call snapshots. Optional activations are independently declared as
`selected_activation_v2`: exactly the payloads retained by capture-time `save=`, never a new
selector and never part of the sparse call recipe.
Tensor arguments, RNG tensors, callables/code/imports remain forbidden, and payload-only runnable
artifacts are invalid.

### Output losslessness (invariant I1)

A runnable save requires a PROVED lossless model output: a bare-Tensor root, or a positive
traversal proof establishing the exact root kind, recursively supported child kinds, fully
encodable literal leaves, and a bijection between the walked tensor leaves and unique typed spec
paths. Runnable model-output traversal never relies on the generic BFS fallback, and a childless
leaf that merely *contains* tensors is not reconstructable. Sets, frozensets, their subclasses,
and unordered/opaque containers are unsupported runnable outputs at every depth and cardinality
(including zero and one tensor) and are uniformly refused at save with
`missing_output_container_contract`. Ordinary analysis capture is unaffected. Proof failure is a
typed producer refusal -- refuse-unless-proved, never accept-unless-flagged.

r37 adds ONE per-kind reconstruction capability table (`CONTAINER_KIND_CAPABILITIES` in
`torchlens/ir/container.py`) shared by spec construction, this producer proof, and the runtime
independent recompute; a kind existing in only one site fails a coverage meta-test. The
instance-state rules it encodes:

- `tuple`/`list`/`literal` are structurally stateless (exact builtins, no instance `__dict__`).
- `namedtuple` uses NAMEDTUPLE-SPECIFIC helpers (never the dataclass helper, whose
  no-`__dict__` interpretation is inverted for tuple storage): an instance carrying non-field,
  non-`None` `__dict__` state refuses at save (`namedtuple_instance_state`); at load, a RESOLVED
  namedtuple type that CAN carry per-instance state (no `__slots__ = ()`) is treated as lossy
  even when the persisted `lossy_reconstruction` flag says `False` (forged-flag defense). Plain
  `collections.namedtuple`, `typing.NamedTuple`, `__slots__ = ()` subclasses, and supported
  `torch.return_types` stay admitted.
- `dataclass`/`hf_model_output` keep their r25/r27 type-level recompute.
- `dict` admits only the exact trusted bases (`dict`/`OrderedDict`/`defaultdict` with an
  allowlisted factory); an instance carrying extra non-`None` `__dict__` state refuses at save
  (`mapping_instance_state`). The load-time recompute for this kind is the persisted flag plus
  the exact-type gate (extra instance state is not type-observable on the trusted bases); honest
  captures never produce a stateful instance because the save refuses it.
- `registered` requires the registration's explicit `state_complete=True` declaration
  (`tl.register_container(..., state_complete=True)`) before an instance carrying extra
  `__dict__` state may save (`registered_container_instance_state` otherwise); the declaration is
  the trusted statement that `unflatten` restores everything `flatten` observed.
- `opaque` is always a typed save refusal.

## 6. Resolver protocol

Resolution is non-executing, once per unique entry, and atomic: every computational group attaches
or none does. Analysis loading survives failure; `run()` raises one `ReattachError` with the complete
readiness report.

The torch ladder is fixed:

1. Exact `getattr` on allowlisted roots only: `torch`, `torch.Tensor`, `torch.nn.functional`,
   `operator`, and explicitly enumerated stock namespaces; prefer public surfaces.
2. Explicit producer-version-bounded aliases in `utils/_torch_compat.py`, including
   `Tensor`/`_TensorBase`/`TensorBase` and enumerated private-to-public mappings. Aliases never
   reinterpret exact matches. Current-runtime target availability remains capability-detected.
3. Cached reverse-index diagnostics over `get_orig_torch_funcs()` for supported bare forms, ranked
   by namespace and guarded by function name/recorded arity; ambiguity hard-fails.
4. Translate through `_state._decorated_to_orig`; a wrapper result is an invariant failure. Resolve
   and execute under `pause_logging()`.

No custom import, `eval`, arbitrary walk, pickle tier, or artifact code is permitted. Non-torch
backends report `unsupported_backend_replay`; analysis load still succeeds. Attachment is
`O(unique callable refs)`.

`ResolverRecord` is exactly `registry_id: str`, `status: ResolverStatus`,
`recorded_key: FunctionRegistryKey`, `resolved_qualname: str | None`, `provenance: str`, and
`diagnostics: tuple[RunnableDiagnostic, ...]`.

## 7. Error taxonomy

The exact `RunnableErrorCode` values are:

```text
run_capability_unavailable
sparse_preflight_failed
sparse_core_tensor_payload
unsupported_literal
unsupported_tensor_constant
missing_tensor_slot
missing_input_container_contract
missing_output_container_contract
missing_control_classification
missing_callable_ref
unsupported_backend_replay
unsupported_ref_schema
unresolved_qualname
ambiguous_qualname
callable_moved_or_renamed
callable_removed
private_api_unavailable
signature_drift
runtime_signature_drift
semantic_drift
wrapper_shadowed
untrusted_custom_import
state_missing_key
state_unexpected_key
state_shape_mismatch
state_dtype_mismatch
state_role_mismatch
state_module_path_mismatch
state_alias_conflict
input_tree_mismatch
input_shape_mismatch
input_dtype_mismatch
call_arity_mismatch
call_structure_mismatch
output_structure_mismatch
output_shape_mismatch
output_dtype_mismatch
slot_production_mismatch
mutation_version_mismatch
scalar_bool_divergence
conditional_arm_divergence
loop_predicate_divergence
input_alias_topology_unresolved
state_alias_topology_unsupported
execution_context_unavailable
context_field_invalid
numeric_attestation_failed
poisoned_run_refused
```

`input_alias_topology_unresolved` is an unverifiability CEILING, not a contradiction: the
three-valued alias engine (section 11) could prove neither overlap nor disjointness for a
same-storage input pair, so the run reports `unverifiable` with `not_applicable` attestation --
never `diverged` by assumption and never `verified`. `execution_context_unavailable` is the typed
refusal for a recorded execution context the producer could not capture or the runtime cannot
enter/restore.

`state_alias_topology_unsupported` (r37) is a SAVE-time producer refusal: two distinct live
bound-state tensor objects overlap in touched bytes (or their relation is unprovable), which the
v2 value-only state encoding cannot represent (section 5, rule 10). `context_field_invalid`
(r37, INV-4) is the PARSE-time refusal for a persisted ambient/per-call execution-context VALUE
outside its closed vocabulary -- device literals against the `type[:index]` grammar, dtype
literals against the live dtype table, matmul precision against exactly
`highest|high|medium`, strict Booleans -- surfaced as an `unavailable` readiness diagnostic at
detection stage `context_parse_validation` before any torch setter, staging, or callable can
observe the bytes (the ambient-apply guards remain as a second belt). Device-UNAVAILABILITY (a
CUDA artifact on a CPU-only host) deliberately reuses `run_capability_unavailable` -- it is a
runtime capability fact, not a new class -- as a readiness diagnostic at detection stage
`readiness_device_capability` naming the slot and device.

`callable_moved_or_renamed` is a successful alias diagnostic. `runtime_signature_drift` rolls back
but is compatibility failure, not path divergence. `semantic_drift` comes only from the independent
live-model oracle and never rebaselines an artifact.

`RunnableDiagnostic` is exactly:

```text
code: RunnableErrorCode
message: str
registry_id: str | None
affected_op_labels: tuple[str, ...]
recorded_runtime: str | None
current_runtime: str | None
detection_stage: str
resolver_provenance: str | None
analysis_load_available: bool
details: tuple[tuple[str, str], ...]
```

Exception classes are `RunnableTLSPECError(TorchLensError)`,
`RunnablePreflightError(ConfigurationError, ValueError)`,
`RunCapabilityUnavailableError(CompatibilityError, RuntimeError)`,
`ReattachError(CompatibilityError, RuntimeError)`,
`StateBindingError(ConfigurationError, ValueError)`,
`RunPreconditionError(ConfigurationError, ValueError)`,
`RuntimeSignatureDriftError(CompatibilityError, RuntimeError)`,
`PathDivergenceError(ValidationError, RuntimeError)`,
`NumericAttestationError(ValidationError, RuntimeError)`, and
`PoisonedRunError(ValidationError, RuntimeError)`. Each also subclasses `RunnableTLSPECError`. The
machine code is in its diagnostic/report; exception text is not a compatibility surface.

## 8. Readiness and result shapes

`ReadinessReport` is exactly:

```text
status: ReadinessStatus
provider: RunProvider
backend: str
capability: str | None
resolver_records: tuple[ResolverRecord, ...]
state_sources_available: tuple[StateSource, ...]
witness_completeness: WitnessCompleteness | None
diagnostics: tuple[RunnableDiagnostic, ...]
```

Analysis-only loaded traces use `loaded_analysis`, `unavailable`, null capability, and only
`not_applicable` as a state source. Non-torch descriptors keep their capability but report
unavailable. Live providers list `live_model_state`; loaded sparse providers list every actually
available source in run precedence order.

`RunResult` is exactly `output: Any`, `trace: Trace`, `report: RunReport`. Output is the reconstructed
model output; trace is a transactional run fork with new in-memory activations; source Trace is
unchanged.

`RunReport` is exactly:

```text
readiness: ReadinessReport
state_source: StateSource
initializer_policy_version: str | None
seed: int | None
random_filled_slot_ids: tuple[str, ...]
contract_checks: tuple[ContractCheck, ...]
path_faithfulness: PathFaithfulness
first_mismatch: RunnableDiagnostic | None
numeric_attestation: NumericAttestationStatus
poisoned: bool
```

`ContractCheck` is `name: str`, `passed: bool`, `diagnostic: RunnableDiagnostic | None`, ordered by
execution. Random reports name the policy and every random-filled slot, including alias members,
and never call those values original/recovered/reconstructed/trained/capture-time weights.

## 9. Runtime API and state lifecycle

The unified transactional `run` provider and staged-state surface is:

```python
def run(
    self,
    inputs: Any,
    *,
    seed: int | None = None,
    on_divergence: DivergencePolicy = DivergencePolicy.RAISE,
) -> RunResult: ...

def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...
```

Equivalent string-enum values may be accepted and normalized. There is no rung-1 `strict=`.

`Trace.run` is the single verb. Live Trace uses the live model/state and alignment checks. Loaded
sparse Trace binds new inputs/state and executes the taken-path DAG. Both return `RunResult` without
mutating the source. Analysis-only loaded Trace raises `RunCapabilityUnavailableError` with
`run_capability_unavailable`. This is not module/model reconstruction, source emission, or execution
of untaken branches.

For migration compatibility, the older intervention spelling `run(model, x, ...)` continues to
return its in-place rerun Trace. New provider-neutral code uses the explicit `inputs=` keyword;
loaded sparse traces always dispatch to the sparse provider. The live provider forks first and then
delegates unchanged to `save_new_outs`, retaining its graph-alignment tripwire.

Inputs require the recorded tree, leaf paths, shapes, and dtypes. Binding follows model site,
container record, and path, never display order. Seeds are cloned before in-place calls. Call
construction fills literal/tensor paths, preserves receiver/dispatch/aliases/versions, and checks
arity/structure/shape/dtype. Inputs/state seed the overlay; intermediates come from this run, never
archived activations.

`load_state_dict` strictly validates and atomically stages without executing or changing the core.
A later successful call replaces staging; failure preserves prior staging. Run state precedence is:

1. explicitly staged user state;
2. embedded capture state;
3. random initialization.

Exact canonical `state_dict_name` maps first, then module path, role, shape, dtype,
persistence/trainability, and alias coherence are verified. Missing, unexpected, shape/dtype/role/
module mismatches and alias conflicts are errors. Positional/shape-only matching is forbidden.
Strict staging never silently random-fills a missing key. Intermediates are never initialized.

`tl.save(trace, path, level="runnable", include_weights=True)` is the confirmed Stage 7 spelling.
The default is false and writes no weight entries. When true, load decodes the schema-versioned
weight family and validates it atomically through this same strict contract before returning the
Trace. A run that uses it reports `embedded_capture_state` with no random-filled slots. Explicitly
staged user state retains higher effective precedence. Neither source writes tensor values into the
sparse core or presents a reconstructed model object.

`include_activations=True` is independent and composes with `include_weights`. It writes exactly the
capture-selected retained `out`/`transformed_out` payloads plus slot membership and byte digests.
Load exposes immutable records through `Trace.archived_activations` for offline inspection. The
scheduler never reads them. Original-input runs with embedded or byte-equivalent staged capture
state compare freshly recomputed raw saved slots against the archive before exposure and report
`attested`; first mismatch is `numeric_attestation_failed` and rolls back. Changed-input,
random-state, and non-equivalent-state runs report `not_applicable`.

## 10. N1-a initializer and seed

`torchlens_role_init_v2` is the v1 role table below plus degenerate totality: a legal
`numel() == 0` slot (any shape containing a zero dimension) allocates and returns immediately
with ZERO generator consumption -- provably, for every role -- and every nonempty Kaiming slot
requires finite positive `fan_in`. The initializer contract is validated centrally at producer
preflight AND defensively at runtime; an unsupported contract fails typed, never by division or
backend sampling. No previously successful v1 reproduction changes. The role table is:

| Role | Policy |
|---|---|
| `weight` | `kaiming_normal` |
| `bias` | `zeros` |
| `norm_scale` | `ones` |
| `norm_offset` | `zeros` |
| `running_mean` | `zeros` |
| `running_var` | `ones` |
| `counter` | `zeros` |
| `generic_buffer` | `zeros` |

`kaiming_normal` means independent zero-mean normal values with standard deviation
`sqrt(2 / fan_in)`. Rank >=2 uses the product of dimensions after dimension zero; rank 1 uses
`max(1, shape[0])`. Scalar weight, unsupported dtype, or incompatible role fails preflight. Zero/one
casts exactly to dtype; counter zero is integral.

Alias groups initialize once, ordered by lexicographically smallest slot ID; unaliased slots use
slot-ID order. Every alias member is still reported random-filled.

`seed` controls state and runtime RNG/source slots through isolated run-local backend generators. A
fixed seed, descriptor, inputs, backend/runtime version, and device reproduces both without changing
global RNG. Null seed uses normal entropy. The report records it. This is architecture execution,
never original-weight or original-random-draw recovery.

Seeded-RNG isolation is TOTAL over the generators the run actually seeds: the executor seeds only
the CPU generator plus each individually forked CUDA device generator (never a global
seed-everything primitive), and the fork/restore set covers every CUDA device when CUDA is
initialized or the descriptor's capture metadata names a CUDA device -- including devices that
appear only as produced intermediates or RNG sources, not just bound inputs/state. Restoration
runs in `finally` on success, divergence, callable exception, and numeric-attestation rollback. A
post-run tripwire asserts CUDA initialization did not flip during a seeded run whose fork set
excluded it. Generators this executor does not seed (MPS/XPU/other accelerators) are never
touched by a seeded run, so no state can leak into them.

## 11. Honesty, divergence, poison, and exactness

All checks occur inside the transaction before exposure. `verified` requires every contract and
complete witnesses. `diverged` means an observed contradiction. `unverifiable` means completion
without contradiction but incomplete witnesses. No mismatch alone never promotes to verified.

Default `on_divergence=raise` stops at first contradiction, rolls back all updates, and raises
`PathDivergenceError`. The only opt-in is `return_diverged`: it may finish the recorded schedule but
returns a permanently, monotonically poisoned result/Trace. Contradiction is diverged; incomplete
proof is unverifiable; both have `poisoned=true`. The mark cannot be cleared by another run.

Validation, runnable export, faithful comparison, path-assuming intervention chaining, and any
model-faithful presentation reject poison with `PoisonedRunError`/`poisoned_run_refused`. The LIVE
refresh provider finalizes through the SAME spine as the sparse provider (r37 corr2-5): the
monotonic Trace path-status mark, the shared divergence-policy enforcement (with `on_divergence`
threaded from the public `run` surface), and the one report finalizer whose `poisoned` flag is
DERIVED solely from `path_faithfulness is not verified` -- no provider carries a caller Boolean,
and direct report construction outside the finalizer is meta-tested away.

There are exactly two independent reproduction oracles:

1. Load original state, supply original input, and compare sparse output/intermediates to a separate
   live-model run. References are runtime/test inputs, never sparse payload. Contract-compatible
   disagreement is `semantic_drift`.
2. When activation payload exists, original-input plus embedded/equivalent real state compares every
   saved selected activation byte-exactly before exposure. First mismatch is
   `numeric_attestation_failed`, identifies slot/call/digest, and rolls back. Blobs never seed run.

Changed-input/random-state activation runs and sparse-only runs are `not_applicable`; no sparse run
silently claims a numeric pass. Unsaved slots have no numeric claim. Sparse-only promises
contract/witness honesty, not numerical reproduction.

### Attestation lattice (invariant I3)

Numeric attestation is DOWNSTREAM of the settled path verdict. Eligibility is derived from the
provisional path-faithfulness verdict computed from ALL non-numeric contract checks and static/
dynamic ceilings: a verdict that is not `verified` makes numeric attestation `not_applicable`
before any archive byte is read. `attested` therefore implies `verified` and not poisoned -- the
report constructor asserts this invariant, so the contradictory combination is unrepresentable.
Every FUTURE contract check automatically caps attestation through the same derivation; there is
no parallel eligibility flag list. Eligibility additionally requires: exact logical input digests
AND exact physical input fingerprints (compared against the value that actually seeds execution),
capture-equivalent persistent state plus validated capture-embedded non-persistent buffer values,
a deterministic raw selection, and a capture context not positively marked
`attestation_ineligible_context`.

### Event-lifecycle discharge (invariant I2)

Every observed capture dispatch event must be DISCHARGED: it ends as an accounted modeled call, an
exact witness, an audited opaque boundary, or an explicit INCOMPLETE reason. In particular, an op
that RAISED during the captured forward whose exception was caught before forward completion
(`try/except` numerical fallbacks -- Cholesky-with-jitter, robust inversion, safe-log guards) is a
`caught_exception_control` fact: the taken path was decided by whether an op raised, a channel no
tensor witness can see, so the producer downgrades `witness_completeness` and EVERY run of that
artifact -- original or changed input -- reports `unverifiable` + `not_applicable`, never
`verified`. The discharge rule is owner-accounted: a raised or host-returning subevent whose
enclosing wrapper owner became an accounted runnable call is discharged (replaying the owner
replays its internal fallback); there are no exception-type or framework-file exemptions. A
successful host/`None`-returning unaccounted event likewise needs an exact witness or audited
boundary or it downgrades completeness.

r37 makes the ledger EXHAUSTIVE over dispatch outcomes (INV-1): a `returned_tensor` event is
never implicitly discharged. An unowned MUTATING dispatch records `opaque_side_effect`; an
unowned VALUE-PRODUCING non-mutating dispatch records `unmodeled_tensor_return` (its product can
bake into a later traced call as an unwitnessed constant -- the corr2-1 class and its
non-mutating twin); the only audited `returned_tensor` rows are pure views (`aten.detach` /
`aten.alias`, the C-level `.data` accessor) and a span-CONTAINED `aten.as_strided` (DLPack /
array-interop restride; an out-of-span restride stays an incomplete fact). An unhandled outcome
value is a hard internal error. Producer preflight enforces observed events == explicit
dispositions.

Escape-source attribution is a SINGLE-EXIT positive ladder: a labeled source witnesses by its
slot digest; an unlabeled source resolves through the direct registered-state storage alias or
the propagated dispatch-origin ledger (every in-scope dispatch registers each tensor result with
the union of its operands' origins); the only other exit is a fail-closed opaque record ->
INCOMPLETE. An orphan-pruned source label falls back to its census-recorded LEAF-origin basis
(the terminal inputs/state its VALUE derives from), each leaf itself re-resolved or the escape
stays INCOMPLETE. BANNED forever as discharge/attribution mechanisms (r36 hon2_1/hon2_2/hon2_3,
measured): scalar value equality/collision (a value match may only ADD a witness), `.item()`
re-extraction on unknown-arity operands, and ANY autograd-graph structural argument as an
operand-totality proof (non-differentiable-dtype and detached operands leave no autograd slot at
all, so "every leaf is a param" proves nothing). Pure param-derived host reads (`w.sum()`,
`w * 2`) recover `verified` ONLY through positive origin resolution; `.data`-of-input escapes
attribute to the input slot and keep the ORIGINAL input `verified` while any changed input
restales the witness. Raw seeded-RNG products taint their origins and can never launder
attribution.

A future replayable exception witness may recover `verified` ONLY when all five preconditions
hold: (1) exact pre-call argument binding recorded; (2) purity proof -- no RNG consumption, no
`out=`/in-place mutation, no allocator-visible or global side effect before the raise; (3)
exception-identity witness -- replay raises the same portable exception type at the same site with
recorded-handler compatibility; (4) full execution-context restoration around the probe; (5) an
RNG bracket proving zero generator advance. This recovery is DEFERRED (not shipped); until it
lands every in-forward caught raise ceilings at `unverifiable`.

### Host nondeterminism channel vocabulary (r37)

The replayable engines are exactly the module-global Python `random` engine and the legacy global
`numpy.random` singleton: their consumption is snapshot-detected, records the capture seed, and a
matching-seed replay stays `verified`/`attested` while an off-seed or seedless run ceilings.

Every OTHER host channel is monitored over this FROZEN vocabulary, and ANY capture-time touch is
permanently unreplayable -- `host_rng_consumed=true` with NO identifiable seed, so every run of
the artifact (any input, any seed) reports `unverifiable` + `not_applicable`:

1. non-global `random.Random` / `random.SystemRandom` draw primitives (`random`, `getrandbits`,
   `randbytes`) -- private instances and subclasses included (class-level monitoring);
2. any C call bound to a `numpy.random.Generator` / `BitGenerator` / `RandomState` receiver
   (a capture-scoped chained `sys.setprofile` classifier -- an OUTSIDE-held generator drawn
   in-forward is observed without discovering its holder; the formerly declared residual is
   CLOSED), plus `numpy.random.default_rng` construction;
3. the `secrets` family (funnels through `SystemRandom` and the import-time `random._urandom`
   alias -- monitored directly because `secrets.token_bytes` bypasses the `os.urandom`
   attribute);
4. `os.urandom` / `os.getrandom` and `uuid.uuid4` (feeds through `os.urandom`);
5. the full clock family: `time`, `time_ns`, `monotonic`, `monotonic_ns`, `perf_counter`,
   `perf_counter_ns`, `process_time`, `process_time_ns`, `thread_time`, `thread_time_ns`,
   `clock_gettime`, `clock_gettime_ns` (any in-forward user read ceilings; a forward that reads
   no clock records nothing).

TorchLens's own per-op timing reads are excluded by EXACT module ownership of the caller frame
(the registered `torchlens.*` module globals), never by filename strings or frame ancestry -- a
user callback's code stays user code even when invoked from a TorchLens frame, and a plain
deterministic capture records nothing (the critical over-trigger pin). Monitor UNCERTAINTY
(installation, chaining, restoration, or classification failure) downgrades capture completeness
to INCOMPLETE -- it never reads as no-consumption. Absence of a touch proves no touch of THIS
NAMED vocabulary; it does not claim environmental determinism for channels outside it (residual
tail: direct `/dev/urandom` file reads, user C-extension RNGs, ctypes -- outside any sane
monitor). Entropy/instance channels mark from any thread; clock channels mark only on the
capture owner thread.

A pruned `.data`-alias BOOL control predicate whose leaf origins resolve positively (e.g.
`bool(self.gate.data > 0.5)` -> the gate's state digest) is witnessed by that basis: the
original state stays `verified` and a changed staged state restales the witness -- the pre-r37
"unattributable pruned bool" fail-closed exception no longer applies to positively-resolved
cases.

### Three-valued input alias topology

Runtime input aliasing against the de-aliased capture is judged by ONE shared three-valued
touched-byte engine (`torchlens.utils.tensor_utils`, r37 INV-2): identity (`a is b`) and PROVED
overlap of recorded-disjoint inputs are observed contradictions (`diverged`); PROVED disjointness
passes; anything unproven is `unknown`, which adds the `input_alias_topology_unresolved` ceiling
-- `unverifiable` path, `not_applicable` attestation -- never `overlap` by assumption and never
`verified`.

All proofs run on ABSOLUTE, device-scoped byte addresses (`storage.data_ptr()` + offset +
min/max stride contributions; negative and zero strides sound). Storage-object identity or
pointer (in)equality NEVER decides anything: `torch.from_numpy` / `torch.frombuffer` / DLPack
views own DISTINCT torch storage objects over genuinely overlapping host memory, so the former
"distinct storages are trivially disjoint" rule was unsound (r36 hon1_1) and is REMOVED. Proof
layers: disjoint absolute byte intervals; identical absolute geometry; an
element-grid-normalized residue/GCD proof on absolute coordinates (disjointness only; applicable
when both views share element size and byte starts congruent on the shared grid); and bounded
exact enumeration up to 65,536 logical elements per view, computed in PURE Python integers so
the verdict is identical under an implicit CPU default, a process-global meta default, and
nested `torch.device(...)` modes (r36 corr2-2). Distinct device address spaces are disjoint by
construction (cross-device aliasing is not constructible through public torch APIs); the device
key prevents spurious cross-device numerical collisions. A meta tensor (`data_ptr() == 0`) or
any unprovable footprint is `unknown`, never a proof. No bounding-interval overlap alone is an
overlap proof, and no complexity cap is a disjointness proof. Alias admission runs BELOW the
hard layout precondition choke point, so the engine's domain is admitted plain strided tensors
by construction.

### Execution-context equivalence and documented residuals

Replay equivalence is EXPLICIT, never ambient: the per-call `CallExecutionContext` and the
capture-scoped `AmbientExecutionContext` are recorded explicitly and restored transactionally
(section 4). The consciously documented residual list -- contexts NOT captured, by decision --
is: thread/intra-op parallelism configuration (can change reduction bytes; forcing
`set_num_threads` is a process-global hazard, so attestation is same-parallelism-config by
contract); environment identity (CPU ISA dispatch, library builds, GPU architecture/driver,
allocator) -- attestation is same-environment by contract; CUDA stream identity (inert for a
serially replayed DAG); and autotuner cache state (subsumed by the `cudnn.benchmark` positive
ineligibility marking). Where one of these residuals changes bytes, the byte-exact attestation
tripwire fails loud (`numeric_attestation_failed`), never falsely `attested`.

### Declared state model boundary

The declared state model is exactly the capture-time `state_dict` (named parameters plus
persistent buffers) PLUS the capture-time values of every used non-persistent registered buffer
(shipped as the required `runnable_nonpersistent_buffer_v1` family), together with the taken-path
DAG. `verified` asserts faithful reproduction against oracle 1: a *separate, fresh* live-model
run whose instance received that declared state -- including the non-persistent buffer values,
injected before its forward -- on the given inputs. It does not assert that a specific already-run
model *instance* will reproduce the replay on a later call.

A model whose `forward` is not a pure function of `(inputs, state_dict)` because it carries hidden
mutable state in unregistered Python attributes — an arbitrary attribute, or a retained
activation-derived handle such as a kept `numpy()`/`untyped_storage()` view or a retained detached
tensor — that evolves *across* forwards is outside the declared state model. That hidden cross-forward
state is not captured and cannot be, so replaying the captured taken path faithfully reproduces the
captured forward (and matches a fresh instance on the given inputs) but is not expected to reproduce
that same instance's subsequent, differently-branched forwards. Such a case stays `verified`; it is
not a divergence, because the divergence exists only against a re-run of the same mutated instance,
never against oracle 1.

The in-scope counterpart is a host write that occurs *within* the captured forward and corrupts the
captured computation: a host write through a zero-copy alias into a captured activation's storage, or
into a registered parameter/buffer's storage, changes what the taken-path DAG consumed. These are
witnessed — observable writes are caught by whole-storage byte comparison (including per-consumption
sampling), and the only unobservable surface (a raw `data_ptr()` pointer) fails closed to
`unverifiable`. Parameters and buffers are witnessed identically: a bytes-changed-but-version-static
storage during the forward is an opaque host write-back (`unverifiable`), while a read-only exposure
of either stays `verified`.

State ALIAS topology is part of the declared model (r37): repeated live object identity across
state names reproduces exactly (one serialized value, one staged allocation per `alias_group`,
`a is b` preserved); distinct-object overlap or an unprovable relation refuses at save
(`state_alias_topology_unsupported`, section 5 rule 10); proved-disjoint views of one storage
serialize independently.

Device placement (r37): payload blobs keep their `map_location` transport placement through load
and analysis. Readiness capability-checks every recorded slot device WITHOUT allocating (a
CPU-only host loads a CUDA artifact for analysis and reports `unavailable` with a
`run_capability_unavailable` diagnostic at `readiness_device_capability`; `.run()` refuses typed
before any callable). At run preparation ONE atomic staging pass -- the sole execution-placement
authority -- moves embedded capture state, staged user state, and required non-persistent
buffers to their recorded slot devices, once per shared value (alias groups keep one allocation),
preserving dtype and `requires_grad`, publishing nothing on failure. A post-staging
in-transaction `state_device` tripwire catches a broken staging hook typed -- never a mid-call
`runtime_signature_drift`. Defensive run clones MIRROR fidelity: runtime input clones restore
the live leaf's `requires_grad` (the recorded grad context stays semantically live and the exact
original input remains attestation-eligible; an intentionally changed flag stays a physical
input change -> `not_applicable`), while state clones restore the RECORDED per-slot trainable
bit.

The recorded DEFAULT DEVICE is entered as a scoped `with torch.device(recorded)` context nested
above the caller's mode stack -- never via `torch.set_default_device`, whose process-global
DeviceContext mutation leaked/corrupted caller modes (r36 corr2-3) -- so context-manager exit
restores the caller's exact state on every path by construction; a feature-probed mode-stack
depth postcondition is the tripwire. Ambient `torch.inference_mode()` capture is supported: the
per-call `inference_mode=true` context is recorded and re-entered at replay, with all
TorchLens-owned `_version` reads routed through one safe accessor (an unavailable version
degrades bookkeeping to its conservative fallback, never a capture crash; the replay mutation
tripwire keeps its version-independent alias leg for version-less inference tensors).

A third, pathological boundary is an autograd property read off an input-*derived view* rather than
the input leaf. Direct-leaf autograd reads (`requires_grad`, gradient presence, `_version`,
`output_nr`) are witnessed in the input contract, so a model that branches on a *leaf* input's
autograd state fails closed when that state differs at run time. A model that instead branches on
`x.view(-1).requires_grad` — an autograd property read through a non-leaf view of an input — is a
documented residual and may stay `verified` even when the branch would differ. This is accepted
rather than closed because a view's `requires_grad` is always equal to its base leaf's, so no
faithful model gains information by reading it off the view instead of the leaf; and closing it would
require instrumenting TorchLens's own per-op autograd reads on the core capture hot path, imposing
capture-wide risk for a case no real model exercises. The residual is the conservative engineering
choice, not a divergence the runnable path claims to catch.

## 12. Optional-payload API spelling and docs lockstep

`include_weights` and `include_activations` are confirmed on `tl.save`/`Trace.save` with
`level="runnable"`. Both default false and are independent. Weights mean the full `state_dict`
(named parameters plus persistent buffers), not only trainable weights. Activations mean exactly the
already-retained capture-time `save=` selection, not a second selector.

The complete implementation includes `load_state_dict`, transient state sources, initializer
reporting, `run`, `RunResult`, transactional run forks, sparse input/call/output reconstruction,
three-state `path_faithfulness`, strict divergence rollback, monotonically poisoned opt-in results,
the external weight family, the required non-persistent buffer family, and inspection-only selected
activation attestation. This contract plus `CLAUDE.md`/`AGENTS.md`, FIELD_ORDER, schema, and API
tests move together. Until a curated public glossary ships, this document is the authoritative
public glossary entry for sparse runnable execution.

Glossary of v2 vocabulary introduced by this amendment (canonical here per the lockstep rule):

- `sparse_recorded_taken_path_v2` -- the required capability whose context records are explicit;
  absence of a context record only ever means legacy v1 (analysis-only).
- `non_tensor_args_tensor_slots_and_context_v2` -- the call recipe carrying the REQUIRED
  `CallExecutionContext` per call.
- `CallExecutionContext` / `AutocastDeviceContext` -- per-call autocast (explicit disabled) and
  grad/inference mode, entered tightly around each resolved call at replay.
- `AmbientExecutionContext` -- the capture-scoped backend context record (defaults, matmul
  precision, determinism, TF32/cuDNN flags, SDP toggles) restored transactionally around the run;
  carries the positive `attestation_ineligible_context` nondeterministic-context marking.
- `selected_activation_v2` -- the activation payload schema whose eligibility metadata includes
  physical `InputAttestationFingerprint` records (sizes/strides/offset, memory-format flags,
  conj/neg bits, subclass class, grad/inference metadata, data-pointer alignment class).
- `runnable_nonpersistent_buffer_v1` -- the REQUIRED external family carrying capture-time values
  of used non-persistent buffers; part of the declared state model (section 11).
- `torchlens_role_init_v2` -- the initializer policy with degenerate totality (empty slots consume
  zero RNG; nonempty Kaiming requires finite positive fan-in).
- `input_alias_topology_unresolved` -- the unverifiability ceiling for an unproven input alias
  relation (r37: absolute device-scoped byte addresses; distinct storage objects are never
  trivially disjoint).
- `execution_context_unavailable` -- the typed refusal for uncapturable/unrestorable context.
- `state_alias_topology_unsupported` (r37) -- the save-time refusal for distinct-object
  overlapping/unprovable bound-state alias topology (section 5 rule 10).
- `context_field_invalid` (r37) -- the parse-time refusal for a persisted execution-context value
  outside its closed vocabulary (section 7).
- `CONTAINER_KIND_CAPABILITIES` (r37) -- the one per-kind output reconstruction capability table
  shared by spec construction, producer proof, and runtime recompute (section 5).
- `tl.register_container(..., state_complete=...)` (r37) -- the explicit trusted declaration that
  a registered container's hooks round-trip all per-instance state; without it, runnable saves
  refuse instances carrying extra state.
- `stage_state_to_slot_devices` (r37) -- the single atomic run-preparation staging authority for
  recorded slot devices (section 11, declared state model).

## 13. Resolver compatibility release gate

Every release runs readiness over a representative torch corpus covering the menagerie classics and
test-suite model families, deduplicates by complete `FunctionRegistryKey`, and reports exact, alias,
unresolved, and ambiguous counts. The threshold is **zero unresolved or ambiguous torch keys**. A
nonzero result is release-blocking unless each key is explicitly documented here with a bounded
compatibility disposition; filtering, skipping, or reporting only successful keys is forbidden.

The checked-in fast gate covers linear, convolution, normalization, pooling, embedding, recurrent,
attention, Tensor-method, operator, and special-function families. The Stage 9 release report records
the expanded classics run, including all unsuccessful model attempts as well as every unavailable
unique key. A sweep of the entire 10,000+ menagerie catalog is deliberately separate and deferred;
the classics plus test-suite corpus is the runnable release gate.
