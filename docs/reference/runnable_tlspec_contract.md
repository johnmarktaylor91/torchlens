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
authorizes `importlib`, arbitrary attribute walking, custom modules, or a callable fallback. It
resolves by direct module `__dict__` lookup over the non-callable symbol allowlist; it never invokes
the `torch` module `__getattr__` or a lazy submodule import / deprecated-attr `replacement()` call
(r42 secC_1 / r45): EVERY attacker-derived top-level-`torch` name resolution on the load / decode /
exec path routes through the single shared helper `torch_attr()` (`torch.__dict__.get`), so an
attacker qualname (`onnx`, `_dynamo`, `_inductor`, `_export`, deprecated `has_cuda`) triggers no
unrequested import and leaks no raw native error. This is pinned by an AST immunizer that fails on any
bare `getattr(torch, <non-literal>)` in load/decode/exec code, so a future decode site cannot
reintroduce the PEP-562 lazy-import / deprecated-`replacement()` side effect. The sibling torch-symbol
validators (`_validated_dtype_literal`, `_getattr_allowlisted`, `_dtype_from_manifest_string`, the
autocast dtype apply) route through the same helper for the `torch` module root; class roots
(`torch.Tensor` / `_VariableFunctions` / `torch._C`), the proxying `torch._VF` module, `torch.backends`,
and literal-name `getattr(torch, "...")` module-layout constants -- which carry no lazy-import hazard --
stay out of the helper's scope. A malformed `torch.device(...)` literal
payload raises a typed `RunPreconditionError` (`unsupported_literal`) at decode (r41 secC -- never a
raw torch error); a malformed device qualname additionally refuses at descriptor parse, degrading the
load to analysis-only with the typed diagnostic intact per the corr2_3 disposition.

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
`cudnn_enabled`, `flash_sdp_enabled`, `mem_efficient_sdp_enabled`, `math_sdp_enabled`,
`grad_enabled`, `inference_mode`, `fill_uninitialized_memory`, and
`attestation_ineligible_context`. Every control the producing runtime exposes is recorded
affirmatively (explicit `false`); `null` means only "the producer runtime did not expose this
control" (feature-detected through `utils/_torch_compat.py` with named `HAS_*` flags).

`AmbientExecutionContext` additionally records `grad_enabled` (required), `inference_mode`
(required), and `fill_uninitialized_memory` (feature-detected; `null` when the runtime does
not expose `torch.utils.deterministic.fill_uninitialized_memory`). The producer records the
global autograd/inference mode and the deterministic uninitialized-memory fill flag with the
ambient snapshot. Replay restores the global autograd/inference mode as scoped contexts around
the whole sparse run; `CallExecutionContext` remains the tighter per-call record. A v2
descriptor missing these fields refuses at parse (`context_field_invalid`, detection stage
`context_parse_validation`) and loads analysis-only: absent context is never defaulted, because
a defaulted grad mode could bless a different-context comparison as `verified`. A Python READ
of `torch.is_grad_enabled()` / `is_inference_mode_enabled()` is deliberately never witnessed or
ceilinged: library code reads the flag constantly, and recording plus restoring the global makes
a fresh-instance replay take the same branch, so a read witness would be a mass over-trigger for
zero honesty gain.

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

### Load-side structural integer anchoring (r53 free_1)

The load-side parser anchors every persisted descriptor integer that can scale an allocation to the
structure it must describe, BEFORE readiness resolution, signature binding, state staging, or any
allocation whose size is a function of that integer. This bounds the whole "a parsed integer scales
an allocation" class at the parse layer rather than at each individual allocation site, so a hostile
`manifest.json` integer (`num_positional_args=1e10`, `shape=[1e9,1e9]`) can never drive an
80 GB-8 EB single-shot allocation on the default-untrusting `tl.load(path)` / `tl.load(path).run(...)`
path.

Per-call arity anchoring: `num_positional_args` and `num_keyword_args` are non-negative;
`num_positional_args` equals the count of distinct first-level positional argument roots `("args", i)`,
which are dense over `[0, num_positional_args)` (verified by an allocation-free length/min/max
pigeonhole, never by materializing `set(range(n))`); `num_keyword_args` equals the count of distinct
first-level keyword argument names; every argument path is rooted at `args`/`kwargs` with a
non-negative integer or string first component. These are load-side anchors of facts the producer
already guarantees (the run-time tripwire has always required exactly this dense arity), so they
never over-trigger on a producible artifact.

Per-slot shape anchoring: `rank == len(shape)` with `rank >= 0` and `rank` within a structural
ceiling (64, which no real tensor exceeds); every dimension is non-negative; and the total byte
product `prod(shape) * itemsize(dtype)` is computed in bounded Python integers and must fit signed
64-bit (torch storage sizes are int64 quantities). There is deliberately NO absolute byte cap at
parse: real multi-GiB state slots (large-model embeddings) must keep loading; run-preparation
magnitude gating is the allocation preflight (section 7), and embedded-payload slots stay
value-anchored downstream by the strict binder and the safetensors header.

Structural violations surface as the frozen `call_arity_mismatch` (arity) and `state_shape_mismatch`
(shape/rank/product) codes at detection stage `descriptor_parse`, degrading the load to
analysis-only with the diagnostic intact -- the same analysis-only disposition as `context_field_invalid`.
The `.tlspec` manifest JSON schema additionally declares `minimum: 0` on the arity, rank, and shape
integers; the parser stays authoritative for the density and product cross-checks the schema cannot
express.

### Preflight rejection rules

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
independent recompute; a kind existing in only one site fails a coverage meta-test. The load-time
recompute and sparse reconstruction are additionally coupled through ONE shared plain-substitution
predicate (`_reconstruction_would_substitute_plain`), so any future gate/reconstruction criterion
divergence is a failing coverage meta-test rather than a silent false `verified` (r49 secB_1). The
instance-state rules it encodes:

- `tuple`/`list`/`literal` are structurally stateless (exact builtins, no instance `__dict__`).
- `namedtuple` uses NAMEDTUPLE-SPECIFIC helpers (never the dataclass helper, whose
  no-`__dict__` interpretation is inverted for tuple storage). r39 corr1-1: the SAVE refusal keys
  on the TYPE-LEVEL criterion the load recompute already uses -- a namedtuple TYPE that CAN carry
  per-instance state (no `__slots__ = ()`) refuses at save (`namedtuple_instance_state`) EVEN when
  the captured instance's `__dict__` is currently empty, because load cannot see the original
  instance and (secB_1) a persisted "no extras" flag is not an independent proof. Producer and
  consumer therefore share one criterion (closing the r37 instance-vs-type disagreement that let
  an unslotted subclass save a permanently-`unverifiable` artifact). At load, that same resolved
  type is lossy even when the persisted `lossy_reconstruction` flag says `False` (forged-flag
  defense). Plain `collections.namedtuple`, `typing.NamedTuple`, `__slots__ = ()` subclasses, and
  supported `torch.return_types` structseqs stay admitted. The load recompute additionally treats a
  namedtuple type that is neither a generated namedtuple nor a trusted structseq as lossy (sparse
  reconstruction would substitute a plain `tuple`), so gate and reconstruction share ONE substitution
  criterion and a non-`FunctionType`-`__new__` tuple subclass can no longer forge a false `verified`
  (r49 secB_1).
  **Structseq resolution authority (r39 secB_1).** The structseq reconstruction branch -- the one
  path that invokes an arbitrary resolved type's own `__new__` -- trusts a type ONLY when the
  SPEC-AWARE gate holds: `spec.type_module == "torch.return_types"` AND resolution began from the
  real already-loaded `sys.modules["torch.return_types"]` AND re-resolving `spec.type_qualname`
  from that module yields the IDENTICAL class AND the class carries the tuple-subclass structseq
  markers. Trust follows the RESOLUTION AUTHORITY, never the resolved class's spoofable
  `__module__` attribute, so a non-torch tuple subclass with a forged `__module__` (or an
  alias-module pointing at a genuine structseq) is refused before its `__new__` runs.
- `dataclass`/`hf_model_output` keep their r25/r27 type-level recompute. **`dataclass` custom-init
  authority (r47 secB_1).** The plain-`dataclass` lossy-reconstruction recompute now flags a
  user-authored `__init__` the same way it flags `__post_init__`: because sparse reconstruction
  intentionally does not invoke either, constructor-computed non-field state cannot be proven, so a
  dataclass whose WINNING `__init__` is not the dataclasses-GENERATED one (detected by the generated
  init's feature-detected `co_filename` marker) is unverifiable for runnable output reconstruction.
  Generated field-mirroring initializers (incl. a dataclass that generates its own init over an evil
  base) stay lossless. A dataclass whose `__new__` is not an inert allocator (`object.__new__`), or
  whose fields are not inertly settable, is likewise lossy at load: sparse reconstruction substitutes a
  plain container for such a type and drops any `__new__`-computed state, so the recompute mirrors
  `_rebuild_container_from_spec` exactly through the shared substitution predicate (r49 secB_1). A
  custom `__init__` compiled in a `<string>`/exec context yet resolvable at the
  spec's `type_module`/`type_qualname` reads as generated -- a narrow documented residual, fail-safe in
  the realistic file-defined case. **Metaclass-`__call__` authority (r51 secB_1).** A `dataclass` or
  `hf_model_output` type whose METACLASS (`type(container_type)`) defines a `__call__` other than the
  builtin `type.__call__` is lossy at load, in the same spirit as the `__post_init__` /
  foreign-`__init__` signals: a custom metaclass `__call__` can compute a dropped tensor-derived
  instance attribute that non-invoking reconstruction (`cls.__new__(cls)` + inert field writes)
  bypasses, and it is not type-observable without INVOKING the metaclass constructor (the SEC1
  surface). The signal is applied to the load-time forged-flag recompute directly (mirroring
  `_dataclass_has_foreign_init`), NOT to the shared plain-substitution predicate, so reconstruction
  still rebuilds the correct inert type while the honesty verdict fail-closes to `unverifiable`. A
  plain-`type`-metaclass dataclass, a real namedtuple, and a standard `ModelOutput` (metaclass
  `type`, or a `__call__`-free `ABCMeta`) stay VERIFIED-eligible (no over-trigger); the namedtuple
  kind is already covered by `namedtuple_type_can_carry_instance_state`. **`hf_model_output` trust
  authority (r42 secB_1).** The lossy-reconstruction recompute trusts an `hf_model_output` type's
  field-mirroring init ONLY by RESOLUTION AUTHORITY: the loaded type is identically re-resolvable
  (identity, not name) from the genuine `transformers` package via `spec.type_module` /
  `spec.type_qualname` in `sys.modules`. A spoofable `__module__` string or a loose `transformers*`
  prefix (e.g. `transformers_evil`) is never sufficient to suppress the lossy-reconstruction
  recompute; an unresolved or non-identical type fails closed to lossy (`unverifiable`). An
  `hf_model_output` type whose `__new__` is not the inert `dict.__new__`, that is not a `dict`
  subclass, or whose fields are not inertly settable is likewise lossy through the same shared
  substitution predicate (reconstruction substitutes a plain container, dropping `__new__`-computed
  state), mirroring `_rebuild_container_from_spec` (r49 secB_1).
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
input_arity_extra
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

`input_arity_extra` (r43, corr1_1) is raised when a loaded sparse `.run()` call carries MORE
top-level positional/keyword input sites than the capture recorded: the descriptor encodes a
finite concrete site set (even for Python variadic signatures), so an extra runtime argument is
outside the recorded taken path and must never report `verified`.

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

Host allocation infeasibility (r53 free_1) likewise reuses `run_capability_unavailable` -- it is
the same "this host cannot execute this artifact" runtime capability fact, not a new class -- as a
run-preparation diagnostic at detection stage `state_allocation_preflight` naming the target device
and the per-device required and available byte totals. Before random role initialization allocates
any slot, the byte total of the slots that fall to `torchlens_role_init_v2` is summed per target
device (once per alias group) and compared against a never-under-estimating live budget: the target
CUDA device's free memory plus the caching allocator's reusable reserve via `torch.cuda.mem_get_info`;
for host-backed devices the available host memory plus free swap (`psutil`, else `/proc/meminfo`
`MemAvailable + SwapFree`); else a static 1 TiB defense ceiling. A total above the budget could
never have been allocated, so refusing it typed can never over-trigger -- it strictly replaces the
`MemoryError`/OOM-kill the raw allocator would otherwise raise, and a legitimate large-model slot
(there is NO static per-slot byte cap) is never refused. The refusal fires before the first
allocation; a residual allocator failure inside staging is wrapped into the same typed diagnostic,
never surfaced as a raw allocator error. `call_arity_mismatch` and `state_shape_mismatch`
additionally originate at detection stage `descriptor_parse` (section 5), degrading the load to
analysis-only with the diagnostic intact.

The same host-infeasibility fact also fires at detection stage `op_allocation_preflight` (r55
free_1): before the taken-path DAG executes, every recorded op-output slot (roles `intermediate`
and `output`) is compared **per slot** against the identical never-under-estimating live budget,
and a single recorded output larger than the whole device budget is refused typed before it can
allocate. It is compared ADDITIONALLY as a per-device **retention floor** (r59, detection stage
`run_retention_preflight`): loaded-sparse replay retains a materialized clone of every taken-path
op output (one `Op.out` clone per `(call, output_slot_id)` at bind, plus one reconstruct clone per
model-`output` slot) for the life of the returned trace, so the SUM of recorded op-output bytes is
a guaranteed lower bound on replay memory. A descriptor whose per-device floor exceeds the device
budget could never complete on this host and is refused typed at `run_retention_preflight` before
any call executes -- closing the accumulation seam (r58 free_1/free_2) where many honest,
individually-feasible output slots together exhaust the host, each passing the per-slot bound and
each clone passing its per-clone byte guard. A long trace with many small outputs is still never
over-refused: the floor UNDER-counts true retention (it ignores live slot values, raw outputs, and
attestation/witness snapshots) and the budget never under-estimates, so a refusal only ever names a
guaranteed mid-replay allocator death.
This closes the op-execution literal-argument seam the r53 state-slot / arity gates did not cover:
an attacker who edits a taken-path size literal (`torch.arange(n)`/`zeros(n)`) to a huge value can
no longer drive an allocation bomb on the default `tl.load(path).run(inputs)` path. Symmetrically,
the parser bounds a literal integer's magnitude under the SAME signed-64-bit ceiling the slot-shape
parser enforces, so a literal can never be more extreme than a slot dimension is allowed to be; an
over-ceiling literal refuses at `descriptor_parse` (`state_shape_mismatch`) and degrades the load to
analysis-only. The PRIMARY, op-agnostic layer (r55 free_1/sec_1; r57 allowlist deletion) runs
immediately before **every taken-path call carrying a numeric literal OR a tensor operand** (r61: a
call with neither has no size source and is skipped): the resolved callable is
projected under
`FakeTensorMode(allow_non_fake_inputs=True)`, which computes the call's output shape/bytes WITHOUT
allocating (fake tensors never allocate -- a `torch.zeros(10**12)` factory, which has no tensor
operand and so cannot be bounded by a per-argument `.to("meta")` pass, projects `4e12` bytes and is
refused). A projected per-device NEW-allocation total above the same live budget refuses typed at
`op_allocation_preflight` before the real call. This closes the literal-only tamper the
recorded-output-slot bound cannot (an honest small output slot with an inflated size literal). It
FAILS OPEN by design: a data-dependent op with no fake/meta implementation
(`nonzero`/`unique` raise `DynamicOutputShapeException`), or an unavailable `FakeTensorMode`, does
NOT refuse -- the run falls through to the recorded-output-slot bound and the parse-time literal
gate -- so a legitimate data-dependent op is never over-refused (the r51 over-catch anti-pattern).
Zero-numel amplifiers are NOT part of this fail-open residual: `mm`/`matmul`/`einsum` on
`[N,0] @ [0,N]` are shape-driven (the output size follows from input shapes alone) and fully
projectable, so they are bounded by the projection, not by the fallback.
The projection runs for **every taken-path call carrying a size source: a non-bool integer or
finite-float literal, or any tensor operand** (float coverage closes `interpolate(scale_factor=...)`;
r61 closes the has-literal-only gate, whose premise -- no literal implies output sizes are bounded by
live tensor shapes -- is false for shape amplifiers: `outer`/`kron`/broadcast-`mul`/`cartesian_prod`/
`tensordot(dims=0)`/`einsum`/`diag`, including the zero-numel family `mm`/`matmul`/`einsum` on
`[N,0] @ [0,N]` where numel is a product and a 0 dim hides arbitrarily large sibling dims, amplify
small or empty inputs into out-of-budget outputs with no literal present -- and any numel/arity
threshold above "has a tensor operand" gaps the same way). A call with neither a literal nor a tensor
operand has no size source -- no fake/meta kernel can size an output tree from it -- and is the only
skipped shape; every amplifier carries a tensor operand, so no amplifier is ever pre-filtered out.
"Size-relevance" is decided structurally by
the projection, not by an op-name family list (the r55 size-driving allowlist is deleted, closing the
r56 `pad`/`constant_pad_nd`/`*_window`/`tril_indices`/`triu_indices`/`one_hot` gate-incompleteness
class). **Pure views, input-returning ops, and in-place ops are excluded structurally**: an output
tensor whose fake `untyped_storage()` aliases a fake-input storage contributes zero new bytes, so
only newly materialized tensors are charged -- by construction, not an op/view list. An unreadable
output storage is charged (fail-closed: an unreadable materializer never masquerades as a view). A
call whose numeric literal drives a coupled-shape validation (`fold` `output_size`) fails the
projection open, but the real op raises its own consistency check before allocating -- caught as
`runtime_signature_drift`, never an allocation bomb. The allocation invariant
thus covers literal sizes, tensor slots, output slots, and staged state uniformly, before allocation:
the fake-tensor projection is the primary per-call layer and the run-preparation output-slot bound is
its fail-open fallback.

Output COUNT (r59 free_1). The per-op byte budget is structurally blind to how MANY output tensors
a call produces: `torch.tensor_split(x, N)` returns N mostly-empty tensors (~0 bytes) off a single
int literal, and PROJECTING it to "see" its size itself builds the N-tensor fake tree -- the defence
executing the attack (linear-in-N CPU/memory before any refusal). r59 count-instruments the
projection: the `FakeTensorMode` subclass counts fake tensor leaves produced by each
`__torch_dispatch__` and refuses typed at `op_output_count_preflight` the instant the running total
passes `max(recorded_count * 8, 4096)` -- DURING fanout construction, before the whole fake OR real
output tree materializes, so a huge N aborts at `ceiling + 1` fakes and can never self-DoS. A
bind-time aggregate accountant (`max(sum recorded * 8, 4096)` over realized leaves) is the backstop
for any projection-skipped path (a data-dependent op that fails open, or a call with no size source
-- neither a tensor operand nor a numeric literal, r61). Both constants are load-bearing and must never be "simplified": the FLOOR carries honest
LOW-arity DECOMPOSITIONS (a legit `interpolate`/`batch_norm`/`einsum` projects up to 55/28/12 fakes
per 1 recorded output), and the MARGIN carries HIGH-arity headroom (a legit `unbind` recording 2048
outputs needs a 16384 ceiling). The count sentinel BYPASSES the projection fail-open;
`DynamicOutputShapeException`-style data-dependent failures still fail open. The gate is op-agnostic
-- it closes the arity-vs-bytes blindness as a CLASS, with no op-name list or arity-estimator
registry (which would be the r55 size-driving allowlist reborn).

Re-materialization BYTES (r59 free_2). A view whose logical numel exceeds its physical storage
(`expand`/`broadcast_to`/`as_strided`) is correctly charged zero NEW bytes by the projection, but
the framework's own bind-time snapshot clone (`value.detach().clone()` onto the fork `Op.out`, plus
the attestation/witness/reconstruct snapshots) re-materializes exactly the allocation the exclusion
assumed away -- an attacker inflates the size literal while leaving the recorded slot honest and
small, and the clone OOM-kills the victim. r59 routes every TorchLens-owned op-output snapshot clone
through a `guarded_clone` that compares `numel * itemsize` (no allocation) against the SAME per-device
live budget and refuses typed at `clone_allocation_preflight` BEFORE the clone allocates and before
any shape-mismatch check. r61 (corr_2) extends the same byte-guard core to **every TorchLens-owned
replay/staging re-materialization clone**: the accepted runtime **input mirror clone** (a tampered
input slot shape plus a runtime `expand` view materializes the logical extent at the mirror, with no
prior byte bound), the **state staging clones** (user `load_state_dict` binder staging -- where the
strict binder accepts an expanded view and materializes it with user/embedded state never entering
the run-prep representative sum -- plus the embedded-state and non-persistent-buffer bind clones),
and the run-time state clone are all routed through the one core and refuse typed at the same
`clone_allocation_preflight` stage. The capture-side save-time snapshot is deliberately not routed:
it clones the live model's own values, so no artifact-driven amplification exists there. The guard
is provably non-regressive: an honest clone equals an honest recorded or declared slot that already
passed the run-prep bound, so a faithful run -- including honest expanded-view user state -- is never
refused; only a tampered view or oversized supplied value is refused before materialization.

Allocation-classed projection/execution failures (r59 section 2.4). A projection failure that is
itself an allocation failure -- `MemoryError`, `torch.OutOfMemoryError`, or a `RuntimeError` carrying
an allocator signature (`std::bad_alloc`/`DefaultCPUAllocator`/`can't allocate memory`/`CUDA out of
memory`) -- fails CLOSED at `op_allocation_preflight` instead of failing open: a projection allocates
strictly less than the real op, so a projection that cannot even fake-project is proof the real
call's identical prelude dies too, and failing open just runs the death twice. Every OTHER projection
failure keeps today's fail-open. Symmetrically, an allocation-classed exception from the REAL call
raises `run_capability_unavailable` at detection stage `op_allocation_execution` (never a misleading
`runtime_signature_drift`) -- an allocator death is a capability fact, not signature drift. Both are
re-typings of already-failing paths; neither can refuse a completable run. Together these gates close
the allocation-DoS class as a WHOLE -- output-count blindness (r58 free_1), the re-materialization /
bind-clone seam (r58 free_2), and the honest-slot accumulation seam are all CLOSED, not patched
op-by-op. Accepted r59 residual: a hostile output-count literal can still buy bounded typed-refusal
LATENCY (seconds to tens of seconds at int64-limit magnitudes) inside a single C++ operator prelude
that NO in-process gate -- step count or wall clock -- can preempt (the O(N) work completes in one C++
frame before the first `__torch_dispatch__`); it cannot allocate past the gates and always terminates
in a typed refusal, so cumulative-projected-bytes, projection-step, and wall-clock bounds are
DIAGNOSTIC-ONLY and never hard refusals (a legit deep model or slow-CI host would otherwise false-refuse).

`callable_moved_or_renamed` is a successful alias diagnostic. `runtime_signature_drift` rolls back
but is compatibility failure, not path divergence. `semantic_drift` comes only from the independent
live-model oracle and never rebaselines an artifact. A malformed `torch.device(...)` literal
payload raises `RunPreconditionError` (`unsupported_literal`) at decode -- every branch of the
torch-symbol decoder is typed (r41 secC); a malformed device qualname additionally refuses at
descriptor parse and degrades the load to analysis-only per the corr2_3 disposition.

Load trust-boundary invariants (r55). An artifact device string is closed-vocabulary DATA, never a
materialization authority: every non-torch payload codec routes an artifact-supplied
`logical_device`/`device_at_save` token through one shared closed device grammar
(`cpu|gpu|cuda|mps|clang|metal|npu|xpu|xla|tpu|rocm|hip|llvm|python` plus an optional pure-integer
index, after unwrapping `Device(...)`/`Place(...)`/`DeviceType....` reprs); any path/URL/scheme
token (`disk:`, `file:`, a leading `/`/`\`, `..`) is refused to the runtime default and never
reaches a backend tensor constructor (closes r54 sec_2, tinygrad arbitrary file write). A caller
`map_location` remains trusted input and is unaffected. Slot `device_type` is validated against the
closed torch device-type vocabulary at parse for the same reason. Rehydrate resolves any invoked
protocol setter (`_internal_set`) off the CLASS via `inspect.getattr_static`, never off the
attacker-controllable instance state, and every portable `__setstate__` additionally refuses an
incoming state key that shadows a class-owned plain method (a NARROW filter: `@property`/descriptor
field names are deliberately never caught, so there is no legitimate-key over-refusal) -- closing
the read-then-call enabler (r54 sec_3). Finally, every manifest/metadata/format-detection JSON read
passes a byte ceiling and a string-aware nesting-depth prescan before stdlib `json.loads`, and the
recursive literal parser carries an independent depth counter; an over-nested or over-size artifact
degrades typed (malformed-descriptor / analysis-only disposition) instead of escaping as an uncaught
`RecursionError` (r54 free_2). These are format limits, not model-size limits.

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
nondeterministic_sources: tuple[str, ...]
```

`ContractCheck` is `name: str`, `passed: bool`, `diagnostic: RunnableDiagnostic | None`, ordered by
execution. Random reports name the policy and every random-filled slot, including alias members,
and never call those values original/recovered/reconstructed/trained/capture-time weights.

"Deterministic raw selection" excludes seeded-RNG products and uninitialized-memory family
products (section 11) whose bytes were not fully overwritten by a total writer, unless the
recorded ambient context proves deterministic fill (`deterministic_algorithms` true and
`fill_uninitialized_memory` not false). `RunReport.nondeterministic_sources` is the closed,
sorted, deduplicated declared-source vocabulary `seeded_rng | host_rng | uninitialized_alloc`,
derived only by the single report finalizer; it distinguishes a declared-nondeterministic
path-only `verified` from a deterministic one and never alters verdict semantics.

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

**Dual observer routes and disabled-mode coverage (r39 hon2_1).** Every known tensor->host VALUE
exit has TWO independent routes to the ONE escape-source ledger: the aten dispatch census (primary)
and a mode-independent Python belt. The belt method-patches the scalar numeric protocol
(`item`, `__bool__`, `__int__`, `__float__`, `__index__`, `__complex__`) and the pure predicates
(`equal`, `allclose`, `is_nonzero`, in both `torch.Tensor.*` and `torch.*` spellings); it records
its operands ONLY when the census dispatch mode is NOT on the active stack (i.e. inside a
`torch._C._DisableTorchDispatch()` / `_disable_current_modes()` region that popped the census),
so it complements -- never pre-empts -- the census. String/format spellings (`str`/`repr`/`print`/
f-string/`format`) are NOT patched exits: TorchLens intercepts its own tensor `__repr__`/`__str__`/
`_str` and extracts values under `pause_logging()`, so a captured tensor's stringification is
recorded as a value escape AT that interception (a string NaN guard therefore ceilings a changed
run exactly like a `.numpy()` escape). A REQUIRED belt observer that fails to install or restore
is itself a fail-closed INCOMPLETE fact; `__repr__`/`__str__`/`__format__` transitivity is pinned
by regression tests, and a future uncovered escape spelling or eager mode-disable site is a RED
coverage meta-test, not a false `verified`.

A future replayable exception witness may recover `verified` ONLY when all five preconditions
hold: (1) exact pre-call argument binding recorded; (2) purity proof -- no RNG consumption, no
`out=`/in-place mutation, no allocator-visible or global side effect before the raise; (3)
exception-identity witness -- replay raises the same portable exception type at the same site with
recorded-handler compatibility; (4) full execution-context restoration around the probe; (5) an
RNG bracket proving zero generator advance. This recovery is DEFERRED (not shipped); until it
lands every in-forward caught raise ceilings at `unverifiable`.

### Host nondeterminism channel vocabulary (r37, extended r39)

The replayable engines are exactly the module-global Python `random` engine and the legacy global
`numpy.random` singleton: their consumption is snapshot-detected, records the capture seed, and a
matching-seed replay stays `verified`/`attested` while an off-seed or seedless run ceilings.

Every OTHER host channel is monitored over a FROZEN, DATA-DRIVEN registry (r39: each row declares
its family, matcher, observation strategy, and thread scope; the runtime classifiers are built
from the registry, and a coverage meta-test makes a new stdlib RNG/clock endpoint a FAILING test
rather than a silent gap). ANY capture-time touch is permanently unreplayable --
`host_rng_consumed=true` with NO identifiable seed, so every run of the artifact (any input, any
seed) reports `unverifiable` + `not_applicable`:

1. non-global `random.Random` / `random.SystemRandom` / bare `_random.Random` draw primitives
   (`random`, `getrandbits`, `randbytes`) -- private instances and subclasses included
   (class-level monitoring; the bare C base `_random.Random()` channel is patched too, r39);
2. numpy `Generator` / `BitGenerator` / `RandomState` INSTANCE draws, witnessed by a chained
   `sys.setprofile` (owner thread) + `threading.setprofile`
   (threads started in-window) receiver classifier (the immutable numpy Generator classes cannot
   be class-patched, so the profile-hook receiver typing is the mechanism -- an externally-held
   generator drawn on the owner or an in-window helper thread is caught, including the
   hon1_1/corr2_2 cross-thread case), belted by a cheap thread-independent before/after state
   digest of any generator or bare BitGenerator the MODEL itself holds. A STATELESS
   `random.Random` subclass -- `random.SystemRandom`, whose `getstate()` raises
   `NotImplementedError` by design -- is monitored-not-digestible (r55 corr_1): mere possession
   of an UNDRAWN instance never ceilings a deterministic capture, while its draws stay
   class-patch witnessed; any OTHER `getstate()` failure still fails closed to
   `inventory_state_read_failed`. The model inventory walks
   every reference edge that can be followed WITHOUT executing user-defined code:
   instance `__dict__`/`__slots__` surfaces (never invoking a property or `__getattr__` --
   r42 corr2_1; including PRIVATE `__slots__` entries resolved through CPython name mangling:
   `__rng` -> `_Class__rng`, with trailing-dunder and all-underscore-class names not mangled and
   the declaring MRO class as the mangling authority -- r61 corr_1; the MANGLED private
   descriptor is preferred over any raw class-dict key, and only an inert slot member
   descriptor is ever read -- a post-hoc raw shadow entry, or any non-descriptor value planted
   at either key, neither hides the real slot value nor gets its `__get__` invoked, r63),
   Mapping/Collection container
   protocols (every `collections.abc.Mapping`
   contributes BOTH its keys/values AND, when the mapping object is a custom (non-stdlib) inert
   holder, its own `__dict__`/`__slots__` values; every non-leaf `collections.abc.Collection`
   likewise contributes BOTH its elements AND its own custom-subclass `__dict__`/`__slots__`
   values -- deque, OrderedDict, Counter, defaultdict, ChainMap, UserList/UserDict, namedtuple,
   and any custom Sequence/Mapping, including a generator held as `self.rng` on a container
   SUBCLASS; r45/r47 hon1_1), inspectable queue buffers (`queue.Queue`/`LifoQueue`/`PriorityQueue`
   via a non-mutating read of the internal deque), class-MRO `__dict__` surfaces of user-defined
   classes (raw mappingproxy reads -- the descriptor protocol never fires; torch/stdlib/numpy
   implementation classes are trusted leaves), weak references (`weakref.ref`/`WeakMethod`
   referents through the base C dereference, weak containers through the container protocols),
   and -- r55 C6 (corr_2/corr_4), superseding the r53 hand-maintained callable-interior
   vocabulary -- EVERY remaining node's reference edges through the AUTHORITATIVE
   `gc.get_referents` enumerator: CPython `tp_traverse`, pure C field enumeration that cannot
   execute Python, exposing every inert reference field an object type declares (closure cells,
   `__defaults__`/`__kwdefaults__`, `__annotations__`, `functools` wrapper `__wrapped__` chains,
   `partial`/`property`/`staticmethod`/`classmethod` interiors, bound-method
   `__func__`/`__self__` -- including a BOUND C METHOD's `__self__` receiver (r57 C6:
   `gen.standard_normal.__self__` IS the generator, so a cached
   `self.sample = self.rng.standard_normal`, alone or behind a `partial` / closure cell /
   `staticmethod` / `classmethod` / dict / list value, recovers the RNG; a module-level C
   function -- module or absent `__self__`, e.g. `math.sqrt` / `np.array` / `torch.relu` --
   contributes nothing and is never generically expanded) -- and function and callable-instance
   `__dict__`/`__slots__` surfaces) minus
   exactly two documented exclusion families: referents whose identity is a loaded module's
   `__dict__` (shared namespaces), and the AMBIENT-BRIDGE leaves (modules, code objects,
   frames, the C `_abc_data` ABC-cache slot -- r56 amb_1). Tensors, ndarrays, and numpy
   scalars are NOT leaves (r61 hon_1): their numeric buffer / autograd / storage internals are
   never walked (`gc.get_referents` is never invoked on them), but their Python instance
   `__dict__`/`__slots__` surfaces AND their user-defined class surface are walked like every
   other holder's (r63: the node's class flows through the same trusted-leaf-gated class-MRO
   branch, so a class-attribute generator on a user-defined Tensor / Parameter / ndarray
   subclass is inventoried while trusted torch/numpy/stdlib implementation classes remain
   leaves) -- a generator stashed
   on a parameter (`weight.rng = default_rng()`), a plain tensor attribute, or an
   ndarray-SUBCLASS instance attribute is inventoried. A bound C
   callable's receiver passes through these SAME walls, so the r47/r56 ambient-escape
   exclusions are unaffected (a numeric-payload receiver is enqueued and reduces to its
   instance state on visit). This enumerator is ROOTED at the model and
   feeds the same cycle-guarded, node-capped walk -- it is not a process-wide `gc.get_objects()`
   scan -- and a new inert hiding field is unreachable only if CPython itself cannot traverse it
   for garbage collection: reachability holds by construction, not by table maintenance. The
   inventory never invokes properties, descriptors, `__getattr__`, or arbitrary
   callables (immunizer-pinned: hostile property/`__getattr__`/descriptor counters stay at
   zero). Every reachable `nn.Module` -- registered or held UNREGISTERED behind any walked
   edge -- is descended through the same surfaces (r51 hon1_1).
   Descent is gated on `collections.abc.Collection` (Sized), so a one-shot iterator / generator
   attribute is NEVER consumed. An opaque queue with no non-mutating payload snapshot is SKIPPED only
   when it is non-mutatingly PROVABLY EMPTY at inventory time (`empty()` is exactly `True`, else
   `qsize()` is integer `0`; any exception / negative / disagreement fails closed -- r47 hon1_2). The
   emptiness probe is CLOCK-NEUTRAL: the monitor suppresses its OWN transitive channel marks during
   the probe (a monitor-initiated read is not a model host read, enforced at the single `_mark` choke
   point), so an empty `multiprocessing.Queue` -- whose `.empty()` reads a poll clock through
   `multiprocessing.connection` -- narrows correctly, matching `queue.SimpleQueue` (r49 hon1_1);
   user/model/worker clock reads outside the probe still mark. A
   NON-EMPTY or unknown non-enumerable model-reachable container
   (`queue.SimpleQueue`, `multiprocessing.Queue`, any object exposing the queue protocol with no
   non-mutating snapshot) fails closed to INCOMPLETE (`inventory_opaque_container`) rather than
   reading as no-consumption -- a documented conservative over-trigger for a deterministic model
   holding a non-empty opaque queue of non-RNG payloads, since a generator inside it drawn on a
   pre-existing worker would otherwise be unwitnessed. Cycle-safe and unbounded for any realistic
   model -- rooted per-object `gc.get_referents` enumeration, never a process-wide
   `gc.get_objects()` scan, and never treating unrelated worker threads as evidence;
   exhaustion of the defensive sweep cap downgrades capture completeness to INCOMPLETE
   (`inventory_budget_exhausted`) -- a truncated inventory never reads as no-consumption;
3. UNSEEDED numpy generator CONSTRUCTION, via `numpy.random.default_rng` and the writable
   `numpy.random.bit_generator.randbits` construction-entropy alias (r39): an unseeded
   `PCG64()` / `default_rng()` built on any thread marks;
4. the `secrets` family (funnels through `SystemRandom` and the import-time `random._urandom`
   alias -- monitored directly because `secrets.token_bytes` bypasses the `os.urandom`
   attribute);
5. `os.urandom` / `os.getrandom` and `uuid.uuid4` (feeds through `os.urandom`);
6. the clock family (a classified bounded-namespace inventory, r39): the current-clock `time.*`
   counters (`time`, `time_ns`, `monotonic`, `monotonic_ns`, `perf_counter`, `perf_counter_ns`,
   `process_time`, `process_time_ns`, `thread_time`, `thread_time_ns`, `clock_gettime`,
   `clock_gettime_ns`); the implicit-now converters `time.localtime` / `gmtime` / `asctime` /
   `ctime` / `strftime` (marking only when called with NO explicit-time argument -- an explicit
   time argument is a pure transform); the immutable `datetime.datetime.now` / `utcnow` / `today`
   and `datetime.date.today` (c_call identity, since these extension-type methods cannot be
   class-patched); and `os.times` / `resource.getrusage`. A forward that reads no clock records
   nothing. The datetime current-clock readers include inherited C readers on `datetime.datetime`
   and `datetime.date` SUBCLASSES; classification is based on receiver type/subclass identity at
   the Python-visible `c_call`, not exact base-class object id (r42 hon1_1). A subclass method that
   genuinely OVERRIDES the current-clock reader is not attributed to the base reader merely because
   of its name.

Every module-attr channel is ADDITIONALLY identity-registered at monitor install (r41): a held
pre-window reference to the original builtin (the idiomatic `from time import time` /
`from os import urandom` at the top of a model or helper file) is classified by `c_call`
identity on the owner and every in-window profile-hooked thread, with TorchLens's own frames
excluded by exact module-globals ownership. The implicit-now converters decode the call site's
positional argument count from the caller's bytecode, so a held `localtime(t)` /
`strftime(fmt, t)` remains a pure transform; an undecodable call site (a star-call) marks
fail-closed.

TorchLens's own per-op timing reads are excluded by EXACT module ownership of the caller frame
(the registered `torchlens.*` module globals), never by filename strings or frame ancestry -- a
user callback's code stays user code even when invoked from a TorchLens frame, and a plain
deterministic capture records nothing (the critical over-trigger pin; the legacy `mtrand._rand`
singleton and its underlying bit generator are identity-exempt, so a SEEDED `np.random` model
stays `verified`). Monitor UNCERTAINTY (patch/inventory install, hook chaining, exact restoration,
or event classification failure) downgrades capture completeness to INCOMPLETE -- it never reads
as no-consumption.

**Positively-covered thread qualification (r39).** Entropy / instance / construction / clock
positives mark on ANY COVERED thread (thread-independent module/class patches; the cheap
model-attribute generator digest; and every profile-hooked thread -- the owner plus threads
started in-window). An IN-WINDOW cross-thread external-generator draw (hon1_1/corr2_2) is caught
by `threading.setprofile`; an owner-thread numpy instance draw and the immutable `datetime`
readers by `sys.setprofile`; a model-held generator on any thread by its state digest; unseeded
construction and Python `random` by the construction/class patches. Held-reference spellings of
the module-attr channels mark on the owner and every in-window hooked thread by original-builtin
identity; module-attr patched spellings remain thread-independent. The monitor does NOT ceiling a
capture merely because a benign background thread (a DataLoader worker, a Jupyter history thread, a
pytest plugin thread) is alive, and it does NOT run a process-wide `gc` scan per capture (the
r39-draft inventory cost ~900 ms/capture and perturbed the peak-memory measurement -- removed).

Absence of a touch proves no touch of THIS NAMED vocabulary; it does not claim environmental
determinism for channels outside it. The residual tail is exactly: (i) direct `/dev/urandom` file
reads; (ii) ctypes / user C-extension entropy or clock reads that never cross a Python-visible
call surface, including C-mediated indirect calls of held builtins (a
`functools.partial(time.time)()` invoked from C emits no Python-visible call of the monitored
builtin); (iii) legacy `RandomState()` C-level CONSTRUCTION entropy (its DRAWS stay
digest/profile-witnessed); (iv) a generator drawn on a PRE-EXISTING
(already-running, non-owner, non-hooked) thread -- which `threading.setprofile` cannot reach on
Python <= 3.11 -- that is reachable only BY EXECUTING USER CODE (a property/descriptor `__get__`
body, `__getattr__`, or a callable's return value) or held ONLY in a SHARED module-global
namespace (the r55 C6 shared-namespace exclusion, explicit: loaded-module `__dict__` identities
are never expanded). Every INERTLY-followable model-rooted reference edge IS
walked by the model inventory (the r55 authoritative `gc.get_referents` enumeration in item 2
above -- CPython `tp_traverse`, complete by construction), so descriptor-held, weakref-held,
class-attribute, closure/default/kwdefault/annotation/partial/property-interior,
`functools`-wrapper (`__wrapped__`), bound-C-method (`self.sample = self.rng.standard_normal`
-- the `__self__` receiver, r57 C6), and callable-instance holders -- like the earlier
container-protocol, container-subclass, and
unregistered-`nn.Module` holders (r42/r45/r47/r51) -- are all witnessed on any thread; only a
NON-EMPTY OPAQUE queue's contents remain a conservative fail-closed residual, never a false
negative. Also of this class: a bare one-shot iterator attribute, which cannot be inspected
without consuming it -- the same class
as the adversarial draw+`state`-restore a cooperative model does not exercise; (v) a
held-reference module-builtin call on a PRE-EXISTING (non-hooked) thread -- the module-attr
patched spelling stays thread-independent; and (vi) a held-reference implicit-now converter
explicitly passed `None` (`localtime(None)`) decodes as a one-argument transform call site (the
patched spelling catches it). `datetime.now()` / `localtime()` are NOT residual (covered above).
Future all-thread coverage is `sys.monitoring` (PEP 669, 3.12+, interpreter-wide).

A pruned `.data`-alias BOOL control predicate whose leaf origins resolve positively (e.g.
`bool(self.gate.data > 0.5)` -> the gate's state digest) is witnessed by that basis: the
original state stays `verified` and a changed staged state restales the witness -- the pre-r37
"unattributable pruned bool" fail-closed exception no longer applies to positively-resolved
cases.

### Uninitialized-memory value sources (r53 hon_2)

The uninitialized-memory op family -- the `empty` factory family (`empty`, `empty_like`,
`empty_permuted`, `empty_strided`, `new_empty`, `new_empty_strided`, and their torch-level
spellings including `empty_quantized`), the SIZE-FORM legacy `Tensor.new` allocator (r55
hon_1: `new(sizes)`/`new(int...)`/`new(torch.Size)` returns allocator garbage byte-identical
to `new_empty` and has NO aten spelling; the DATA form `new([values])`/`new(tensor)` is a
deterministic copy constructor and is NEVER tainted -- consumers gate the size-vs-data
argument form through the shared predicate, and an UNDECIDABLE form -- e.g. a decoded
int-tuple, since the portable literal grammar erases `torch.Size` to a plain tuple -- fails
closed to tainted), plus a `resize_`/`resize_as_` that GROWS its receiver
beyond its pre-call element count -- produces bytes that are not a function of the recorded
computation. Family products (and anything value-derived from them) are nondeterministic value
sources of kind `uninitialized_alloc`, UNLESS the governing ambient context records
`deterministic_algorithms` true with `fill_uninitialized_memory` not false (torch then fills
deterministically), or the product has zero elements. Taint propagates along the value DAG and
is removed exactly by a TOTAL WRITE of the tainted bytes independent of their prior content:
an `out=` destination, `copy_`, `zero_`, `fill_`, or an RNG fill (which replaces the
`uninitialized_alloc` taint with the RNG source classification). The `out=` sanitization is
ALIASING-AWARE (r55 hon_2): an `out=` destination that is ALSO a value operand of the same op --
`torch.add(a, x, out=a)`, whose result `a + x` READS `a`'s prior uninitialized bytes -- is NOT a
prior-independent total write, so its taint is PRESERVED (fail closed, r35 unknown-alias precedent);
only an `out=` to a destination not itself read, and the pure `copy_`/`zero_`/`fill_` receiver
total-writers, drop taint. A partial or unprovable
in-place write propagates taint (unprovable is never a proof of cleanliness). Consequences,
in parity with seeded-RNG sources at every consumer: a control fact (scalar-bool, conditional
arm, loop predicate, tensor-derived scalar witness) whose source is tainted ceilings the run
`unverifiable` -- including a family op whose control-driving chain was orphan-pruned from the
descriptor, which the producer records through the pruned-nondeterministic-control side table;
an archived activation or output slot reached by taint makes numeric attestation
`not_applicable` upfront (never a contradictory `numeric_attestation_failed`); a byte mismatch
on an UNTAINTED slot still raises `numeric_attestation_failed`. A tainted value that reaches
only the model output leaves path faithfulness `verified` (path-only) and is declared through
`RunReport.nondeterministic_sources`. Escape attribution treats `uninitialized_alloc` origins
as unattributable (fail-closed), like raw seeded-RNG output.

No `torch.Tag` marks uninitialized allocation, so the family is a closed name table in ONE
shared predicate block (`utils/rng.py`) consumed by all three recognition layers (the load-side
value-source classifier, the producer origin ledger, and the pruned-orphan control walk) and
defended over BOTH spelling surfaces by drift meta-tests (r55 hon_1): the aten-namespace test
(a new `empty*`/`resize*` aten name that is neither tabled as family nor allowlisted as
justified-non-family is a failing test) AND the Python-`torch.Tensor`-method test (a
`new`/`new_*`/`*empty*`/`*resize*` Tensor method -- with or without an aten spelling -- that is
neither tabled nor justified-non-family is a failing test, so a future python-only uninit
factory cannot slip both surfaces). At the dispatch level the size-form `Tensor.new`
redispatches `aten.empty.memory_format` (pinned by a live decomposition test), so the producer
origin ledger covers it transitively; the python-method recognition in the completeness witness
declares the same family for the qualname surface the load-side classifier sees. A
partially-written uninitialized buffer is an accepted conservatism: a slice-filled cache that
in fact fully covers its bytes reports attestation `not_applicable` (and `unverifiable` if
branched on) rather than proving interval coverage -- never a false `verified`; interval-
coverage refinement is deferred until a real-model sighting.

### Cross-thread captured-tensor qualification (r43)

TorchLens runnable capture is single-owner-thread by design. During the forward window, any
non-owner thread that performs a Python-visible tensor-to-host value escape, storage/pointer
exposure, stringification, or model-input metadata read on a captured tensor, or on a tensor
positively known by the dispatch-origin ledger or storage identity to derive from captured tensors,
permanently ceilings the artifact's replay proof to `unverifiable` with numeric attestation
`not_applicable`. The rule does not depend on the owner thread's per-dispatch logging toggle and does
not require the thread to have been created by `threading.Thread`; raw `_thread` and pre-existing
worker threads are covered when they touch captured tensors.

The rule keys on the tensor, not on thread existence. A background thread that never touches a
captured tensor, or that only reads/stringifies tensors it created independently of the capture,
records nothing and does not downgrade a deterministic artifact. Tensor helper work on a non-owner
thread remains outside the ceiling ONLY when its operands are independent of captured storage.

**Cross-thread captured-operand consumption (r45 hon2_1, r47 hon2_1).** The ceiling additionally
fires when a non-owner thread runs ANY torch op that CONSUMES a captured tensor as an operand -- not
only a direct value / pointer / string / metadata escape of a captured object. No single
Python-visible net is universal, so coverage of the op surface is the UNION of process-wide observers:
(1) the global torch-function wrapper (`torch` / `torch.Tensor` / `functional` / `linalg` / `fft` /
`_VF` / `special` spellings); (2) a class-level observer on every `torch._ops` class that defines its
own `__call__` (`OpOverloadPacket` / `OpOverload` / `TorchBindOpOverload` / `HigherOrderOperator`,
feature-detected by a structural scan so the set self-updates across torch versions; installed for the
armed window because a `TorchDispatchMode` / aten census is thread-local and cannot see a non-owner
thread); (3) the r43 Tensor host-escape method belt (`.item` / `.tolist` / `.numpy` / storage
reads, which do not route through `torch._ops.__call__`); and (4) an armed-window module-function belt
over the patchable private-C free-function modules `torch._C._{nn,special,fft,linalg,sparse,nested}`,
structurally enumerated from the canonical forward-op module authority (`_ALLOWED_FORWARD_OP_MODULES`
via `private_c_forward_op_module_names`) filtered to module-typed `torch._C._*` submodules, so a future
private-C op module added to that set is auto-covered and a torch lacking one degrades gracefully -- a
private-C free function bypasses BOTH the global torch-function wrapper (no `__torch_function__`) and the
`torch._ops.*` class patch (it dispatches its inner aten op in C++) (r49 hon2_1). All observers call the SAME storage-identity
captured-membership test and never log the worker op into the owner trace. The FIRST worker op
consuming a captured operand on any surface permanently ceilings the artifact to `unverifiable` /
`not_applicable`, regardless of derivation depth or the eventual escape spelling. A worker that
operates only on tensors it created independently of the capture consumes no captured operand and
stays `verified` (no over-trigger). This closes the worker-DERIVED escape sibling (a tensor freshly
derived on the worker from a captured input, whose new storage the owner-thread census never
registered). An inspection error inside the operand observer -- or an observer install / restore
failure -- during an armed capture fails closed (INCOMPLETE / ceiling), never silently passes. Every
observer is gated on a global belt-armed flag mirroring the forward window, so the disarmed steady
state and every plain (non-runnable) trace pay only a single bool read; an eager owner forward hits the
Python `torch._ops.*.__call__` path zero times (C++ dispatch), so the armed owner window adds ~no
overhead. **Accepted residuals (unclosable / narrow):** (a) the read-only, non-Python-patchable
private-C free-function CLASS `torch._C._VariableFunctions.<op>` -- its public alias `torch.<op>` is the
SAME object and IS wrapped, so only a worker calling the private CLASS spelling DIRECTLY on a captured
tensor from a pre-existing thread slips (same class as the `.__call__()` / `partial()`-mediated C-call
residual; C-level dispatch observation is only thread-local); the patchable private-C MODULES
`_nn/_special/_fft/_linalg/_sparse/_nested` are now CLOSED by observer (4) (r49 hon2_1); (b) a
`torch.ops.higher_order.*` HOP whose subclass OVERRIDES `__call__` and consumes a captured operand
PURELY in C++ without dispatching any Python-level `OpOverload` / `OpOverloadPacket` call -- the
overwhelmingly common inner-ATen dispatch IS observed on the worker thread, so only a
fully-C++-internal consumption is residual.

Owner-thread tensor escapes keep the ordinary precise witness ladder. Non-owner captured-tensor
touches are not promoted to precise `verified` proofs, even if a label or state origin is visible,
because concurrent host interaction is outside the replay model.

Op-logging is owner-thread-scoped: the global torch-function wrapper skips any thread other than the
capture owner, so a non-owner thread running torch ops during a capture (a worker formatting
`str(tensor)`, a DataLoader thread) is never recorded into the owner's Trace and never corrupts
owner-op attribution. The designated cross-thread observers -- independent of the owner op-logging
path -- are the mode-independent tensor-method belt, the global torch-function wrapper's non-owner
operand check, AND the `torch._ops` `__call__` class observer for direct `torch.ops.*` packet/overload
spellings. Captured activation storage identity is liveness-verified
(a freed-then-reused storage address never false-positives a benign own-tensor touch). The aten census
remains owner-thread-scoped (a `TorchDispatchMode` is thread-local).

### Input site-set exactness (r42 corr1)

Loaded sparse `.run(inputs=...)` must match the captured top-level model input site set exactly.
Extra positional arguments or keyword arguments not present in the captured site set are observed
input-tree contradictions and cannot be ignored or reported `verified`, including for captures of
Python variadic signatures whose recorded taken path contains only a finite concrete site set
(`INPUT_ARITY_EXTRA`; default policy raises `PathDivergenceError`, `return_diverged` returns
`diverged`). Dataclass model-input containers are traversed by declared fields using the same
tensor/non-tensor leaf vocabulary as tuples, mappings, and namedtuples; a tensor-only dataclass is
fully witnessable (`verified`, attestation-eligible), while a genuinely-opaque dataclass field still
surfaces and fails the run closed.

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
when both views share element size and byte starts congruent on the shared grid); a canonical
DENSE-INTERVAL proof (r39 corr2_6, the sole verdict-precision relaxation) -- when BOTH footprints
prove they cover one contiguous byte interval (drop singleton dims, reject zero stride, sort by
absolute element stride, require the recurrence `abs(stride) == running size product` starting at
1, which is sound for contiguous, transposed/permuted-dense, and signed-stride layouts and is
independently byte-oracle fuzzed), their already device-scoped byte intervals decide
`overlap`/`disjoint` exactly and numel-independently, WITHOUT enumeration and above the cap; it
never turns an `unknown` into a false `overlap` (expanded/zero-stride and genuinely sparse
over-cap geometry stay `unknown`, NOT the unsound `numel*esize == span`); and bounded exact
enumeration up to 65,536 logical elements per view, computed in PURE Python integers so the
verdict is identical under an implicit CPU default, a process-global meta default, and nested
`torch.device(...)` modes (r36 corr2-2). Distinct device address spaces are disjoint by
construction (cross-device aliasing is not constructible through public torch APIs); the device
key prevents spurious cross-device numerical collisions. A meta tensor (`data_ptr() == 0`) or
any unprovable footprint is `unknown`, never a proof. No bounding-interval overlap alone is an
overlap proof, and no complexity cap is a disjointness proof. Alias admission runs BELOW the
hard layout precondition choke point, so the engine's domain is admitted plain strided tensors
by construction.

### Execution-context equivalence and documented residuals

Replay equivalence is EXPLICIT, never ambient: the per-call `CallExecutionContext` and the
capture-scoped `AmbientExecutionContext` are recorded explicitly and restored transactionally
(section 4). The global autograd/inference mode (`grad_enabled`, `inference_mode`) is part of
the recorded ambient context (r53 hon_1): oracle 1 is a fresh run from declared state UNDER THE
RECORDED AMBIENT CONTEXT, including the global autograd/inference mode. A fresh run under a
different ambient mode (for example the same capture re-evaluated inside `torch.no_grad()` when
`grad_enabled=true` was recorded) is an out-of-declared-context comparison, not a divergence.
The consciously documented residual list -- contexts NOT captured, by decision --
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
Symmetrically at capture start (r55 corr_3), the lazily-imported callable-safety classifier probe
runs its import-time pure-view detection under a fully neutral context -- disabled torch-function,
`torch.inference_mode(False)`, and `torch.enable_grad()` -- so the FIRST capture inside an ambient
`torch.inference_mode()` no longer crashes on the probe's `_version` read; the neutralization is
scoped to the probe and never leaks into the caller's ambient grad/inference state.

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

### Sparse/live provider parity (r39 CLASS B)

The loaded-sparse and live-refresh providers may gather DIFFERENT evidence but never make a
different SETTLEMENT decision: both route through ONE finalizer (`_finalize_provider_run`), the
sole owner of the monotonic Trace-poison mark, divergence-policy enforcement, the single
`_run_report` constructor (which derives `poisoned` solely from the faithfulness lattice), and
`RunResult` construction. A source-scan meta-test forbids constructing `RunResult` or calling
`_run_report` anywhere else, so a future provider or payload family cannot fork the settlement
path. Parity means shared settlement machinery and typed handling for the same provider claim,
not identical verdicts from different evidence: loaded-sparse replays the recorded schedule and
may settle changed-input or witnessed-host-control runs as `diverged`/`unverifiable`, while
live-refresh executes a fresh model forward and may honestly report `verified` when that fresh
run completes and its output contract is lossless. Inexecutable input divergence is typed as
`PathDivergenceError` for both providers.

- **Live opaque output (corr2_5).** The live provider's bare-tensor fast path is gated on the
  FRESH refresh forward's output-losslessness proof (`bare_tensor_root` + `lossless`, one leaf, no
  fallback/duplicate) -- `save_new_outs` copies that proof onto the projected fork, because a
  changed input may select a different return-container kind than the original capture. An opaque
  `set`/`frozenset`/custom container (which yields the same "one leaf, no spec, no path" signature)
  is therefore NOT blessed a bare tensor: it returns the best-effort value with a failed
  `live_output_reconstruction` check -> `unverifiable` + poison, never `verified`. The sparse
  producer refuses the same outputs at save (`missing_output_container_contract`); the parity
  matrix asserts an identical verdict CLASS across both providers.
- **Descriptorless payload degradation (corr2_3).** ONE structural disposition governs every
  runnable payload family (weights, non-persistent buffers, archived activations, and any future
  execution-only family) and EVERY typed descriptor-parse refusal -- not just `context_field_invalid`
  or legacy v1. When a runnable descriptor was PRESENT but refused at parse, the trace loads
  ANALYSIS-ONLY (`provider == loaded_sparse`, `descriptor is None`, readiness `unavailable` with the
  typed diagnostic): its payload blobs stay unbound and the typed diagnostic SURVIVES -- the load
  never hard-fails on the payload binder (a tampered context field on an `include_weights` artifact
  no longer raises an untyped IO error pointing at intact weights). A genuine analysis artifact
  (`provider == loaded_analysis`) carrying STRAY runnable blobs still hard-fails.
- **Divergence-aware call raise (corr2_4).** Shape/dtype/device input checks stay SOFT so an
  EXECUTABLE divergent input (a changed batch dimension) returns `diverged` + poison under
  `return_diverged`. But an admitted-but-INEXECUTABLE divergent input (a wrong feature shape that
  fails the native call) surfaces as `PathDivergenceError` carrying the first-failed input check --
  NOT `RuntimeSignatureDriftError`, which stays reserved for genuine resolved-callable / torch-version
  drift with all input checks passing. Both policies roll back the transactional fork. The live
  provider classifies at native-failure time against pre-computed input-contract checks (the
  native error is chained as `__cause__`, and a divergent input whose forward fails for an
  unrelated reason is still classified as the divergence), while a native failure on a
  NON-divergent input re-raises unchanged -- a genuinely failing model is not a divergence.

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
