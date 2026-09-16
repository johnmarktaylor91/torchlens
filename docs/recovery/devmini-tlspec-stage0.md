# TLSPEC Stage 0 report

## Authoritative frozen contract

The single authoritative contract is:

`docs/reference/runnable_tlspec_contract.md`

It is mirrored by behavior-free types in `torchlens/runnable.py`. The contract explicitly states
that a prose/type disagreement is a release-blocking schema defect and requires a versioned
amendment.

## Frozen schema

- Capability/version: `sparse_recorded_taken_path_v1`
- Call recipe: `non_tensor_args_and_tensor_slots_v1`
- Callable-ref schema: integer `1`
- State binding: `module_path_role_v1`
- Input binding: `model_site_io_role_v1`
- Control witnesses: `scalar_bool_and_arm_entry_v1`
- Initializer policy: `torchlens_role_init_v1`
- Payload schemas: `state_dict_v1`, `selected_activation_v1`

`SparseRunDescriptor` fields are exactly: `capability: Literal[...]`, `backend: str`,
`call_recipe: Literal[...]`, `callable_ref_schema: Literal[1]`, `state_binding: Literal[...]`,
`input_binding: Literal[...]`, `control_witness: Literal[...]`,
`initializer_policy_version: Literal[...]`, `payload_layers: PayloadLayersDescriptor`,
`callable_registry: tuple[CallableRegistryEntry, ...]`,
`calls: tuple[RunnableCallDescriptor, ...]`, `tensor_slots: tuple[TensorSlotDescriptor, ...]`,
`control_witnesses: tuple[ControlWitness, ...]`,
`witness_completeness: WitnessCompleteness`, `compatibility: RunnableCompatibility`,
`preflight: ProducerPreflight`, and `unsupported_sites: tuple[RunnableDiagnostic, ...]`.

The contract freezes all nested registry, call, literal-tree, tensor-slot, input-binding,
state-binding, witness, compatibility, diagnostic, readiness, and report fields and types. It keeps
the existing `FunctionRegistryKey(namespace, qualname, dispatch_kind, version, import_path)` shape;
rung 1 rejects `custom` and non-null `import_path`. The Stage 2 capture coupling is contract-only:
future `FunctionCallRef.func_id` and cooked `Op.func_id` are `FunctionRegistryKey | None` and must
move through KEEP/FIELD_ORDER/schema gates together.

Frozen tensor roles: `model_input`, `parameter`, `buffer`, `intermediate`,
`constant_like_tensor`, `rng_source`, `output`.

Frozen state roles: `weight`, `bias`, `norm_scale`, `norm_offset`, `running_mean`, `running_var`,
`counter`, `generic_buffer`.

Frozen witness completeness: `complete`, `incomplete_scalar_escape`,
`incomplete_opaque_side_effect`, `incomplete_unobserved_predicate`.

Frozen faithfulness: `verified`, `diverged`, `unverifiable`. Frozen divergence policy: `raise`,
`return_diverged`. The contract also freezes provider, readiness, state-source, resolver-status,
numeric-attestation, control-witness, literal-kind, initializer-policy, and all 44 error-code values.

The literal grammar is tagged and recursive: exact-type None/bool/int/float/string atoms,
allowlisted non-callable torch symbols, list/tuple sequences, ordered mappings, and scalar/tuple
mapping keys. It excludes tensors, callables, imports, pickles, opaque objects, sets, bytes, complex
values, and arbitrary enums.

## Runtime/API contract

Frozen future signatures:

```python
Trace.run(
    inputs: Any,
    *,
    seed: int | None = None,
    on_divergence: DivergencePolicy = DivergencePolicy.RAISE,
) -> RunResult

Trace.load_state_dict(sd: Mapping[str, Any]) -> None
```

No methods were added to `Trace` in Stage 0. `run` is one verb with live and loaded-sparse providers;
analysis-only loaded traces raise typed capability-unavailable errors. Both executable providers
return `RunResult(output, trace, report)` and leave the source Trace unchanged.

Input binding is by persisted model-site/io-role/container path, never display order. State mapping
matches exact canonical state name, then verifies module path, role, shape, dtype, persistence,
trainability, and aliases. Validation/staging is strict and atomic. Explicit user state overrides
embedded capture state; otherwise embedded state precedes random fallback.

Readiness freezes provider/backend/capability, all resolver records, available state sources,
witness completeness, and complete diagnostics. `RunReport` freezes readiness, state source,
initializer version, seed, every random-filled slot ID, ordered contract checks, path faithfulness,
first mismatch, numeric-attestation status, and poison flag.

Default divergence raises on the first contradiction and rolls back. `return_diverged` is the sole
opt-in and permanently poison-marks diverged or unverifiable output/Trace. Faithfulness-consuming
validation/export/comparison/intervention APIs must refuse poison. Exact reproduction has two
independent oracles: a separate live-model comparison with original state/input, and optional
byte-exact selected-activation attestation with original input plus real equivalent state.

Non-torch sparse replay is frozen as unavailable with `unsupported_backend_replay`; analysis load
remains available.

## N1-a

`torchlens_role_init_v1` is frozen as:

- weight -> Kaiming normal (`std = sqrt(2 / fan_in)`)
- bias -> zeros
- norm scale -> ones
- norm offset -> zeros
- running mean -> zeros
- running variance -> ones
- counter -> integral zero
- generic buffer -> zeros

Alias groups initialize once in stable slot-ID order. A fixed seed controls isolated run-local state
and runtime random sources for the same descriptor/runtime/backend/device without changing global
RNG; null seed uses runtime entropy. Every random-filled slot, including alias members, is named in
the report. Random state is never described as original/recovered weights.

## Producer preflight and sparse-core invariant

Whole-graph preflight rejects missing/unsupported callable keys, custom imports, unsupported
literals, tensor argument leaves, incomplete call trees, incomplete tensor-slot metadata, opaque or
unreconstructable boundary containers, incomplete state bindings/aliases, uninitializable tensor
constants, missing control classifications/witnesses, insufficient optional mutation payloads, and
any forbidden content surviving final scrub.

The sparse core carries no tensor values, tensor blob files/references, weights, buffers,
activations, inputs, gradients, state dictionaries/snapshots, tensor RNG snapshots, saved tensor
arguments, live handles/models, executable callables/code, pickles, or import instructions. Later
optional payload blob families stay outside the sparse core; no payload-only runnable artifact is
valid.

## Typed scaffolding landed

- `torchlens/runnable.py`: frozen constants, enums, immutable dataclass schema/report shapes,
  canonical initializer mapping, and behavior-free `RunnableTraceProtocol`.
- `torchlens/errors/runnable.py`: runnable error hierarchy using existing TorchLens taxonomy/style.
- `torchlens/errors/__init__.py`: public error imports.
- `tests/test_runnable_contract_scaffolding.py`: version/value, error construction, and exact
  descriptor/readiness/report field-order tripwires.

No producer, resolver, state binder, executor, Trace method, capture change, bundle/save-level
change, oracle implementation, or validation change landed.

## Parked for JMT

The optional payload-layer public API spelling remains explicitly unfrozen: `include_weights=`,
`include_activations=`, `level="runnable"`/`save_level`, and placement on `tl.save`/`Trace.save` all
require JMT confirmation at the optional-payload sprint kickoff. Only sparse-default behavior,
independent layers, and reuse of capture-time `save=` selection are fixed.

The contract includes the documentation lockstep ledger. Public glossary/release-reference entries
for the error names and schema constant are needed before advertisement. `Trace.run`,
`Trace.load_state_dict`, `RunResult`, readiness/faithfulness/poison vocabulary, CLAUDE/AGENTS
examples, audit/example updates, and FIELD_ORDER/schema gates wait for the Stage 5 wiring change.

## Verification

- `ruff check . --fix`: PASS (`All checks passed!`; one pre-existing malformed-noqa warning was
  observed on an earlier run in `torchlens/backends/torch/prehook_provenance.py`).
- `mypy torchlens/`: PASS (`Success: no issues found in 321 source files`).
- `python -m pytest tests/test_runnable_contract_scaffolding.py -q -rA`: PASS, 3 passed.
- Error taxonomy plus contract tests: PASS, 62 passed.
- `PYTHONPATH="$PWD/tests:$PWD" python -m pytest tests/ -m smoke -x -q`: PASS,
  337 passed, 5 skipped, 4731 deselected.
- Public-boundary `not slow` gate: stopped after 667 passed on
  `tests/test_callbacks_lightning.py::test_maybe_profile_releases_activation_tensor_deterministically`.
  The mandated targeted rerun failed identically. The same test also fails identically on untouched
  local main at base `791c6371`, proving it is a pre-existing, out-of-scope baseline failure; no
  callback/capture cleanup code was changed.
- Commit hooks: PASS.

Commit: `00d5e2f3` (`docs(tlspec): freeze sparse runnable contracts`). No push performed.
