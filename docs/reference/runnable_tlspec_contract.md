# Sparse runnable `.tlspec` frozen contract

Status: **AUTHORITATIVE AND FROZEN** for `sparse_recorded_taken_path_v1`.

This is the single implementation contract for sparse runnable `.tlspec` Stages 2 and later. The
definitions in `torchlens.runnable` are its behavior-free typed mirror. A disagreement between this
document and those types is a release-blocking schema defect. Neither may change without an explicit,
versioned contract amendment.

Stages 2--4 now provide the sparse producer, safe loader/resolver, and non-executing state binder.
There is still no DAG executor or `Trace.run` behavior, and these stages change no capture path,
analysis save level, bundle, or validation result.

## 1. Versions and enum vocabularies

| Name | Frozen value |
|---|---|
| sparse capability/schema | `sparse_recorded_taken_path_v1` |
| call recipe | `non_tensor_args_and_tensor_slots_v1` |
| callable-ref schema | integer `1` |
| state binding | `module_path_role_v1` |
| input binding | `model_site_io_role_v1` |
| control-witness schema | `scalar_bool_and_arm_entry_v1` |
| initializer policy | `torchlens_role_init_v1` |
| optional weight payload schema | `state_dict_v1` |
| optional activation payload schema | `selected_activation_v1` |

The constant is exactly
`RUNNABLE_TLSPEC_SCHEMA_VERSION = "sparse_recorded_taken_path_v1"`. It is a capability version, not
the existing whole-bundle `TLSPEC_VERSION` or JSON manifest schema version.

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
| `NumericAttestationStatus` | `passed`, `failed`, `not_applicable`, `not_present` |
| `ResolverStatus` | `resolved_exact`, `resolved_alias`, `unavailable` |

`loaded_analysis` only explains unavailability. Old or pre-descriptor bundles are never promoted.

## 2. Safe non-tensor literal grammar

Every non-tensor argument leaf uses one of these tagged nodes. Untagged JSON is not a runnable call
recipe.

| Node | Exact fields | Constraint |
|---|---|---|
| `LiteralAtom` | `kind`, `value` | kind is `none`, `bool`, `int`, `float`, or `str`; value has exactly that Python type (`None` for `none`) |
| `LiteralTorchSymbol` | `qualname` | non-callable symbol below an explicitly allowlisted stock torch root |
| `LiteralSequence` | `kind`, `items` | kind is `list` or `tuple`; items are ordered literal nodes |
| `LiteralMapping` | `entries` | ordered `LiteralMappingEntry` nodes; duplicate keys invalid |
| `LiteralMappingEntry` | `key`, `value` | key is `LiteralAtom` or recursive `LiteralTupleKey`; value is any literal node |
| `LiteralTupleKey` | `items` | ordered `LiteralAtom` or `LiteralTupleKey` nodes only |

Floats round-trip without coercion, including non-finite values. Booleans are not integers. Dict
order, tuple versus list, key types, nesting, and `None` are preserved. Bytes, complex values, sets,
arbitrary enums/objects, tensors, callables, classes, import references, pickles, and opaque reprs are
outside the grammar and fail preflight with `unsupported_literal`.

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
| `capability` | `sparse_recorded_taken_path_v1` |
| `backend` | string; only `torch` executes in rung 1 |
| `call_recipe` | `non_tensor_args_and_tensor_slots_v1` |
| `callable_ref_schema` | integer `1` |
| `state_binding` | `module_path_role_v1` |
| `input_binding` | `model_site_io_role_v1` |
| `control_witness` | `scalar_bool_and_arm_entry_v1` |
| `initializer_policy_version` | `torchlens_role_init_v1` |
| `payload_layers` | `PayloadLayersDescriptor` |
| `callable_registry` | tuple of `CallableRegistryEntry` |
| `calls` | tuple of `RunnableCallDescriptor` |
| `tensor_slots` | tuple of `TensorSlotDescriptor` |
| `control_witnesses` | tuple of `ControlWitness` |
| `witness_completeness` | `WitnessCompleteness` |
| `compatibility` | `RunnableCompatibility` |
| `preflight` | `ProducerPreflight` |
| `unsupported_sites` | tuple of `RunnableDiagnostic`; empty for a runnable claim |

`PayloadLayersDescriptor` has exactly `weights` and `activations`. Each is
`PayloadLayerDescriptor(present: bool, schema: str)`. Schemas are respectively `state_dict_v1` and
`selected_activation_v1`, even when absent. Sparse default has both `present=false`; any payload
bytes/references live outside the sparse core.

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
```

`TensorArgumentRef` is `argument_path: tuple[str | int, ...]` plus `slot_id: str`.
`LiteralArgumentRef` has the same path plus `value: NonTensorLiteral`. A path begins with `args`, a
positional index or `kwargs`, a keyword name, then container components. No path occurs in both
lists; together they completely describe args/kwargs. Parent IDs and list order define schedule;
existing graph edge-use/version metadata remains authoritative for repeated uses and mutation.

`runtime_fingerprint` is a non-executable digest of signature facts and call recipe. Its algorithm
is producer-versioned and diagnostic; the registry key is callable identity.

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

`ControlWitnessKind` values are `scalar_bool`, `conditional_arm_entry`, `loop_predicate`, and
`shape_structure_fact`. A witness has exactly `witness_id: str`, `kind: ControlWitnessKind`,
`order: int`, `call_id: str | None`, `site_label: str`, and
`observed_value: NonTensorLiteral`.

IDs are unique and order is dense zero-based. Scalar/loop predicates use boolean `LiteralAtom`;
arm entry uses a stable arm identity; shape/structure uses the literal grammar. Missing/opaque facts
use completeness plus diagnostics, never an opaque payload.

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

The hard invariant is:

> The sparse core contains zero tensor values, tensor blob files, tensor blob references,
> executable callables, and import instructions.

Forbidden content includes op outputs/transformed outputs, inputs, activations, gradients, child
tensor versions, tensor args/templates, parameters, buffers, `state_dict`, state snapshots,
`_buffer_initial_values`, live handles/models, tensor RNG snapshots, callable pickles, executable
code, and custom import paths.

Optional weights/activations, when later implemented, are declared external blob families. Even
then tensor arguments, RNG tensors, callables/code/imports remain forbidden. Payload-only runnable
artifacts are invalid.

## 6. Resolver protocol

Resolution is non-executing, once per unique entry, and atomic: every computational group attaches
or none does. Analysis loading survives failure; `run()` raises one `ReattachError` with the complete
readiness report.

The torch ladder is fixed:

1. Exact `getattr` on allowlisted roots only: `torch`, `torch.Tensor`, `torch.nn.functional`,
   `operator`, and explicitly enumerated stock namespaces; prefer public surfaces.
2. Explicit version-bounded aliases in `utils/_torch_compat.py`, including
   `Tensor`/`_TensorBase`/`TensorBase` and enumerated private-to-public mappings. Aliases never
   reinterpret exact matches.
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
numeric_attestation_failed
poisoned_run_refused
```

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

Stage 4 adds `load_state_dict`; Stage 5 adds the unified transactional `run` provider surface:

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

## 10. N1-a initializer and seed

`torchlens_role_init_v1` is:

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

## 11. Honesty, divergence, poison, and exactness

All checks occur inside the transaction before exposure. `verified` requires every contract and
complete witnesses. `diverged` means an observed contradiction. `unverifiable` means completion
without contradiction but incomplete witnesses. No mismatch alone never promotes to verified.

Default `on_divergence=raise` stops at first contradiction, rolls back all updates, and raises
`PathDivergenceError`. The only opt-in is `return_diverged`: it may finish the recorded schedule but
returns a permanently, monotonically poisoned result/Trace. Contradiction is diverged; incomplete
proof is unverifiable; both have `poisoned=true`. The mark cannot be cleared by another run.

Validation, runnable export, faithful comparison, path-assuming intervention chaining, and any
model-faithful presentation reject poison with `PoisonedRunError`/`poisoned_run_refused`. This is the
target for live run too; temporary provider gaps must remain explicit.

There are exactly two independent reproduction oracles:

1. Load original state, supply original input, and compare sparse output/intermediates to a separate
   live-model run. References are runtime/test inputs, never sparse payload. Contract-compatible
   disagreement is `semantic_drift`.
2. When activation payload exists, original-input plus embedded/equivalent real state compares every
   saved selected activation byte-exactly before exposure. First mismatch is
   `numeric_attestation_failed`, identifies slot/call/digest, and rolls back. Blobs never seed run.

Changed-input/random-state activation runs are `not_applicable`; no activation layer is
`not_present`. Unsaved slots have no numeric claim. Sparse-only promises contract/witness honesty,
not numerical reproduction.

## 12. Parked API spelling and docs lockstep

**TODO — JMT confirmation at the optional-payload sprint kickoff:** public spellings including
`include_weights=`, `include_activations=`, `level="runnable"`, `save_level`, and placement on
`tl.save`/`Trace.save` are proposals, not frozen. Stages 7/8 must not treat them as settled. Frozen
behavior is sparse default, independent layers, and reuse of capture-time `save=` for activations.

Stage 0 introduced public error names and an importable schema/type module. Stage 4 documents
`load_state_dict`, transient state sources, and initializer reporting. Stage 5 implements `run`,
`RunResult`, transactional run forks, sparse input/call/output reconstruction, and populated
three-state `path_faithfulness`. Stage 6 remains responsible for strict divergence raising and
poisoned opt-in results. This contract plus `CLAUDE.md`/`AGENTS.md`, FIELD_ORDER, schema, and API
tests move together. Until a curated public glossary ships, this document is the authoritative
public glossary entry for sparse runnable execution.
