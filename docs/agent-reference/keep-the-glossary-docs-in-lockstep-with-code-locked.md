## Keep the glossary + docs in lockstep with code (LOCKED)

The glossary is the **canonical** API spec (maintained privately; `docs/reference/glossary.md` is the public copy); code conforms to it (spec-drives-code). A rename is not *done* until the docs match too:

- **Rename / add / remove any PUBLIC name** (dataclass field, `@property`, method, top-level `tl.*` name, kwarg) → in the SAME change, update: (1) the **glossary** entry (canonical), (2) the agent-reference examples (`docs/agent-reference/`, moved out of `AGENTS.md`), (3) the audit notebooks (`notebooks/audit/`) and `examples/` that use it.
- A change that touches code but leaves the glossary/docs stale is **INCOMPLETE.** This is exactly how the v7 `memory → activation_memory` gap and the stale `log_forward_pass`/`vis_opt` examples slipped through.
- After a rename/conformance sprint: re-file the updated canonical glossary (it supersedes the prior dated version), and confirm a `grep` of every old name is clean across `torchlens/`, `tests/`, `examples/`, `notebooks/`, AND the glossary itself.

### Trusted custom callable imports

Intervention-spec loads tolerate foreign `custom` callable keys for safe structural and metadata
analysis without importing their modules. Resolution for execution denies those foreign imports by
default because module imports execute top-level code. Trusted execution may opt in with
`trust_custom_callables=True`; prefer the narrower
`allowed_custom_callable_modules={"my_trusted_module"}`, which remains enforced even alongside broad
trust. TorchLens-owned `torchlens.*` custom callables and the fixed `torch`, `torch.Tensor`,
`torch.nn.functional`, and `operator` namespaces always resolve.

### Sparse runnable state binding

Loaded sparse runnable traces accept `trace.load_state_dict(sd)` to strictly validate and atomically
stage canonically named parameter and persistent-buffer tensors. The method never executes the DAG
or writes tensor payloads into the sparse descriptor. Run preflight selects explicitly staged user
state, then optional embedded capture state, then the versioned
`torchlens_role_init_v2` fallback (degenerate-total: empty slots consume zero RNG); random reports
name every initialized slot.

`tl.save(trace, path, level="runnable", include_weights=True)` bundles the full capture-time
`state_dict` (all named parameters plus persistent buffers) as the separate, schema-versioned
`state_dict_v1` blob family. The default is `include_weights=False`, so the sparse core stays
tensor-value-free. Load validates embedded state through the same strict binder; run reports
`embedded_capture_state`, never a reconstructed model, and a later `load_state_dict()` overrides it.
Used NON-persistent buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family
(declared state, not gated on either include flag; the save discloses it). State ALIAS topology is
declared (r37): repeated live object identity (tied weights, double-registered buffers) becomes a
shared alias group staged as ONE allocation; distinct-object overlapping or unprovable state
topology refuses at save with `state_alias_topology_unsupported`. Payload blobs keep their
`map_location` transport placement; readiness capability-checks recorded slot devices without
allocating, and one atomic run-preparation staging pass moves all state families to their recorded
devices (a CUDA artifact on a CPU-only host loads for analysis and refuses `.run()` typed).

`tl.save(trace, path, level="runnable", include_activations=True)` independently archives exactly
the activations already retained by the capture-time `save=` decision, including retained raw and
transformed outputs, as `selected_activation_v2` with physical `InputAttestationFingerprint`
eligibility records. Load exposes them through
`trace.archived_activations` for inspection. They never seed the sparse DAG. On original-input runs
with embedded or capture-equivalent staged state, recomputed saved raw slots are compared by exact
bytes and report `attested`; the first mismatch raises `numeric_attestation_failed` and rolls back.
Changed-input (logical or physical), random-state, non-equivalent-state, and
nondeterministic-capture-context runs report `not_applicable`; `attested` always implies
`verified`.

### Sparse runnable execution

`trace.run(inputs=x, seed=...)` is the provider-neutral execution spelling. A live Trace delegates
on a fork to the existing `save_new_outs` fast capture path. The live refresh projector's
buffer-sink refusal is TRAINING-MODE AWARE (D18, by design): eval-mode BatchNorm
(every buffer sink carries derived write evidence `buffer_value_changed=False` with agreeing
mode claims) is refresh-eligible and runnable on the DEFAULT path, while any value-changing
buffer write (train-mode running stats, counters), unproven (`None`) evidence, a mode claim
contradicting the evidence, or a value-changing write in the refreshed rerun's own journal
refuses with the typed `BufferSinkRoutingError` carrying
`RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE` (provisional spelling, documented-unstable;
pinned "computational graph changed" message term preserved). A loaded sparse Trace binds cloned
input leaves plus staged/random state and executes its resolved taken-path DAG under
`pause_logging()`. Both return `RunResult(output, trace, report)` and leave the source Trace
unchanged. Analysis-only loads raise typed `run_capability_unavailable`. Stage 5 populates
`report.path_faithfulness`; Stage 6 enforces the three-layer honesty transaction. Divergence raises
and rolls back by default; `return_diverged` is the sole opt-in and returns a monotonic poisoned
Trace refused by validation, export, faithful comparison, and path-assuming intervention chaining.
Incomplete witness coverage is `unverifiable`, never `verified`; numeric attestation is
`not_applicable` for sparse-only or ineligible activation-payload runs. Model outputs with ZERO
tensor leaves remain subject to the output-container rules below. `trace.run(inputs=x, fast=True)`
is the explicit guarded static-loop exception to source immutability: live traces execute native
`forward()` with targeted module hooks (and scoped functional collection only for an explicit
functional `save=` predicate); loaded traces perform one ordinary verified run, then reuse staged
state, compiled binders, and one result Trace. Every fast iteration retains input, output-shape/
dtype, call-path, and control-witness guards and always raises on divergence. Default `fast=False`
keeps the full transactional validation/attestation path. The fast-mode handle
`Trace._fast_run_session` is a session-time `FieldPolicy.DROP` field: it is ordered in
`MODEL_LOG_FIELD_ORDER` under a private name but never survives save/load, so it joins
`measure_python_peak_memory`, `save_budget`, and `distributed_witness` in the session-time class
(the private-named ordered DROP fields are ledgered with reasons in
`tests/test_schema_lockstep.py::PRIVATE_ORDERED_DROP_FIELDS`). Model outputs with ZERO
tensor leaves (all-literal trees, literal roots, empty containers) and namedtuple/mapping/
registered-container outputs carrying extra per-instance state refuse at save
(`missing_output_container_contract`; one per-kind capability table governs capture proof, save
refusal, and load-time recompute). Host nondeterminism beyond the two replayable global engines
-- RNG instances (incl. outside-held NumPy Generators, witnessed by a chained
`sys`/`threading.setprofile` receiver classifier + a cheap model-attribute state digest -- NO
process-wide gc scan; bare `_random.Random`; unseeded-construction entropy via the
`randbits` alias), `SystemRandom`/`secrets`, OS entropy, `uuid4`, the `default_rng` factory, and
the full clock family (`time.*` counters, `localtime`/`strftime`/`datetime.now`/`date.today`,
`os.times`/`getrusage`) -- ceilings every replay permanently (`unverifiable` + `not_applicable`);
monitor uncertainty (install/chain/restore/inventory failure) downgrades completeness, never reads
as no-consumption. A realistic pre-existing-thread draw from a persistent numpy generator is
witnessed by the setprofile classifier / model digest; only an externally-held generator drawn on
a pre-existing (non-hooked) thread is a documented residual (a benign background thread never
ceilings a capture). The loaded-sparse and live-refresh
providers settle through ONE finalizer (identical verdict class): a live opaque-container output
is `unverifiable` + poisoned (never a wrongly-blessed bare tensor), a parse-refused descriptor
degrades EVERY payload family to analysis-only with its typed diagnostic intact, and an
inexecutable divergent input raises `PathDivergenceError` (not `RuntimeSignatureDriftError`). Structseq
reconstruction trust keys on the RESOLUTION AUTHORITY (`spec.type_module == "torch.return_types"` +
identity re-resolution), never the spoofable `__module__` attribute; a namedtuple TYPE that can
carry instance state refuses at save even with an empty instance. Persisted execution-context values validate at parse
time against closed vocabularies (`context_field_invalid`); the recorded default device is entered
as a scoped `with torch.device(...)` context, never via `set_default_device`.

Runnable descriptors are `sparse_recorded_taken_path_v2` (call recipe
`non_tensor_args_tensor_slots_context_and_obligations_v3`): per-call `CallExecutionContext` and the
capture-scoped `AmbientExecutionContext` are REQUIRED and EXPLICIT, restored at replay or refused
typed; a legacy v1 artifact loads analysis-only with a typed readiness refusal (absent context is
never defaulted).

The witness-strip class is closed structurally (r71 A): `WITNESS_FAMILY_REGISTRY` v2
(`witness_family_registry_v2`) covers EVERY verdict-steering witness family (the four direct
control kinds + shape families + two claim-only families), each row naming its independent
replay-structural anchor. Every replay item that steers a verdict is a typed obligation stamped
on its OWNING replay record (`control_obligations`/`control_dependencies` on calls,
`host_escape`/`inert_sink` on slots, `captured_requires_grad`/`captured_grad_fn`/
`host_escape_disposition` on state bindings, the REQUIRED `input_boundary` record), discharged by
an exact witness XOR a typed `WitnessCoverageGap`; `witness_completeness` is DERIVED from the gap
ledger (redundant assertion, never authority) and the required-witness inventory is a redundant
mirror. No record deletion can improve a verdict; the ONE out-of-scope boundary is coherent
reauthoring -- byte-for-byte an honest capture of a weaker program, whose VERIFIED is TRUE against
that program's oracle 1 -- a documented threat-model scope statement, not an open residual.
Cluster C: instance-state inspection is fail-closed (`inspect_instance_state`; a
property/descriptor-shadowed `__dict__` or a custom `__getattribute__`/`__getattr__` on a
declared-schema container refuses `instance_state_uninspectable` without running the hook).
Cluster B: slice/composite-literal components are classifier-first (a semantic scalar can never
launder through a `slice` component). Cluster D: reserved input-path sentinels (the whole `\x00`
namespace) are escaped by the key codec so a real dict key equal to a marker round-trips.

The declared state model is the capture-time `state_dict` (named parameters plus persistent buffers)
PLUS the capture-time values of used non-persistent buffers (the required
`runnable_nonpersistent_buffer_v1` family), and the taken-path DAG. `verified` is faithfulness
against a *fresh* live-model run from that state on
the given inputs (oracle 1) — NOT reproduction of a specific already-run instance's later, differently
branched forwards. Hidden non-`state_dict` Python state mutated *across* forwards (an arbitrary
attribute, or a retained activation-derived handle — a kept `numpy()`/`untyped_storage()` view or a
detached tensor) is out of scope and stays `verified`; the "divergence" exists only against re-running
the same mutated instance, never oracle 1. In scope and witnessed identically for activations,
parameters, and buffers: a host write *within* the captured forward into captured storage — caught by
whole-storage byte comparison + per-consumption sampling, with the raw `data_ptr()` surface fail-closed
to `unverifiable`, and a read-only exposure staying `verified`. Full boundary in
`docs/reference/runnable_tlspec_contract.md` section 11.

The frozen runnable enums live in `torchlens.runnable`: `ReadinessStatus`, `RunProvider`,
`StateSource`, `PathFaithfulness`, `DivergencePolicy`, `NumericAttestationStatus`, and
`RunnableErrorCode`. Public code branches on these values or the structured report, not exception
text. The exhaustive error vocabulary and release threshold are maintained in
`docs/reference/runnable_tlspec_contract.md`.
