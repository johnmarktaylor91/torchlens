# Limitations and edge scenarios

This is the canonical user-facing catalog of TorchLens limitations. Each row says exactly when an
edge can occur, what you can observe, and what to do. Run
`tl.compat.report(model, inputs).to_markdown()` before capture for model-specific findings. For a
new or surprising case, include that report and a minimal reproducer in a bug report.

## Start here: wrap before references escape

The strongest detached-reference remedy is prevention:

```python
from torchlens.backends.torch.wrappers import wrap_torch

wrap_torch()  # do this before `from torch import ...`, closures, partials, or callable holders

from torch import relu
```

The first torch capture installs wrappers lazily, but a binding created before installation can
continue pointing at the raw callable. Wrapping early makes new aliases, closure cells, partials,
and object attributes capture the wrapper directly. It eliminates the stale-reference class and
avoids a rescue forward. The historical broad `sys.modules` crawler is deleted, along with
its `patch_policy=` / `patch_modules=` keywords.

A model built before the first capture is covered without wrapping early when it holds the raw
callable on the model itself: as a module attribute (transformers' `GELUActivation` keeps
`F.gelu`), a `functools.partial` (`GELUTanh`), a closure cell or default argument of a held
function or of the class `forward`, or as a value inside a plain `list`, `dict`, `tuple` or namedtuple
attribute. Each capture points those references at the wrappers before its forward and puts the
original objects back afterwards, so the model is left exactly as it was, and the capture takes one
forward with no warning. References held anywhere else (a module global, a custom object, a dict key,
a builtin subclass, a `forward` assigned on the instance) keep the provenance warning and the rescue
forward described below.

Held Python-level torch functions are rebound too, so they capture exactly like a direct call: a
held `F.relu`, such as the default activation every `nn.TransformerEncoderLayer` and
`nn.TransformerDecoderLayer` binds at torch import, now logs the outer `F.relu` call around the inner
`torch.relu`, as `F.relu(x)` written in `forward` does. Graphs are unchanged, but the `func_call_id`
values of later ops in such models shift by 2 per held call, compared with a capture in which the held
function bypassed its wrapper.

## Capture completeness and Python call routes

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| A stale pre-wrap callable produces a provenance, shadow-detector, or output-attribution signal on the owner thread. | TorchLens re-runs the forward once. `capture_verified=False`, `capture_verification_reason="mode_rescue_rerun"`, and `trace.rescue_rerun` names the trigger and recovered Ops. The result is not byte-attestable because the forward ran twice. | Wrap early. If a rescue is acceptable, treat its provenance label as part of the result and check that a second forward is safe for the model. |
| The capture streams payloads, uses `streaming.out_callback`, or halts before a complete re-runnable forward, and a stale reference is detected. | The escape is reported, but rescue is skipped; the trace cannot gain `mode_rescue_rerun` recovery. | Verify once with an ordinary, non-streaming full forward first. After the same model/configuration passes, stream subsequent captures; do not present the streamed capture itself as independently rescue-verified. |
| Tensor work or a stale callable runs on another thread, or Python thread count changes across the guarded forward. | Capture is owner-thread-qualified. Typical reasons are `escape_rescue_unrecovered` or `owner_thread_tripwire_changed`; `capture_verified=False`. | Keep model tensor work on the capture thread. Move preprocessing outside the forward, or capture each worker's result later on the owner thread. |
| A third-party `handle_torch_function` implementation pops the mode before running a composite interior. | The interior is de-moded and may report `capture_verification_reason="escape_rescue_unrecovered"`. | Prefer the ordinary eager tensor implementation, expose the interior as a module/callable outside the protocol handler, or accept the disclosed incomplete capture. |
| A stale reference targets a protocol-invisible constructor such as the build-derived `from_numpy`, `frombuffer`, or `Tensor.as_subclass` set. | No mode callback exists. TorchLens's small mechanical belt patches module-held references; no broad object crawl occurs. | Wrap early. Keep these constructors in an ordinary module attribute if a pre-wrap reference is unavoidable. |
| `escape_detector="shadow"` is enabled, even if it sees no raw-call report. | The diagnostic mode itself sets `capture_verified=False` with `shadow_diagnostic_mode`; a hit uses `callable_escape_shadow_report` and appears in `trace.escape_diagnostics`. Deferred backward reports `escape_detector_backward_coverage="not_armed"`. | Pair shadow mode with `completeness_witness=True` when you need a positive owner-thread dispatch claim; otherwise use it as a diagnostic, not a verification result. |
| A pre-wrap `torch.func`/functorch route (`vmap`, `grad`, `jacfwd`, ...) executes transform internals that cannot be expanded. | A boundary warning is emitted and `capture_verification_reason="transform_call_route_unverified"`; inner per-element Ops are absent. | Capture the untransformed function separately or treat the transform as an opaque boundary. |
| On torch <2.6, capture reaches a compiled plain callable/free function that cannot be unwrapped; on newer torch, `set_stance("force_eager")` is unavailable. | The region is skipped, a warning names it, and `capture_verification_reason="dynamo_region_not_logged"`. | Pass/call the eager source during capture, or compile an `nn.Module` child that TorchLens can unwrap. Use compiler tools to inspect fused execution. |
| A C `functools.partial` hides a C builtin from Python profiling, or a registered original recursively calls itself through a pre-wrap reference. | Shadow coverage may remain unverified or conservatively self-report after the one-shot wrapper-edge token is consumed. | Wrap early; replace the C partial with a small Python function when exact diagnostics matter. |
| A tensor is converted to Python with `item`, `bool`, `int`, `float`, `complex`, or `operator.index`. | `ScalarEscapeWarning` records the count and first user callsite. The scalar dependence is not a graph edge; runnable proof may be incomplete/unverifiable. | Keep the value as a tensor, or pass the Python value as an explicit model input. |
| A user hook mutates through `.data`, raw storage, a NumPy alias, a custom kernel, or an exotic subclass where version evidence is unavailable. | Hook attribution is marked incomplete rather than falsely unchanged. Private hook-registry bypass makes the affected scope sticky-incomplete. | Use public hook registration and ordinary tensor operations; make control inputs explicit. |
| A deliberately adversarial `__torch_dispatch__` body disables dispatch around a hidden mutation. | No dispatch-based tracer can observe the nested operation. This is outside the cooperative-model threat model. | Do not suppress dispatch around model computation you expect TorchLens to certify. |

A direct `torch.ops.*` call inside forward (a raw `torch.ops.aten.*` packet or overload, a `torch.library.custom_op`, or an operator a C++ extension registers with `TORCH_LIBRARY`) is recorded as an ordinary op: its parents are its tensor arguments, it is named by the operator's own name (`tanh`, the custom op's name), and validation replays it by calling the operator again, so a decomposing call such as `torch.ops.aten.linear` on a 3-d input is one op that owns all its dispatches. A C++ extension function bound only through pybind (no `torch.ops` registration) reaches the dispatcher with no Python call TorchLens can see. A module whose output such a call builds in several dispatches fails validation's `bfs_completeness` check, and `tl.validation.last_validation_failure()` names that check: TorchLens records the module's output as a synthesized boundary and credits it only to the dispatch that produced the boundary tensor and to pure aliases of it, so the other dispatches did real work that no captured Op accounts for. A single such call that is one dispatch passes that census, but its output still has no recorded origin: TorchLens adopts it at the module boundary as an internal source and warns, and validation fails its `source_provenance` check (below). To validate such a module, register the extension's functions with `TORCH_LIBRARY` and call them through `torch.ops`, call the public `torch` function, or validate the extension's computation separately.

A tensor argument with no recorded graph or source provenance fails forward validation's `source_provenance` check, and `tl.validation.last_validation_failure()` names that check, the first offending op, and every gap. Replay from saved arguments cannot catch these, because the source-less value sits in the saved arguments. The gaps are: a tensor the forward reads that TorchLens does not know as a source (a module-global or closure tensor, including one a submodule receives or returns directly and one whose only consumer is pruned from the graph, such as `G.sum().item()` or a branch predicate; module tensor attributes, inputs, parameters and buffers are known sources), an untagged tensor adopted at a module boundary (the opaque raw call above), and any `held_tensor_scan_truncated` cut of the module-held tensor scan, read or not. The one exemption is a genuine user intervention replacement (an output-replacement hook). To validate such a model, register the tensor with `register_buffer`, pass it as an input, or create it inside `forward`.

## Capture entry and execution contexts

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| Capture is called recursively from a hook, post-transform, or nested model call. | Inner entry raises `RuntimeError` instead of corrupting the outer Trace. | Capture the submodel in its own `tl.trace(...)` call after the outer run finishes; there is no public pause-logging context, so nested capture entry always refuses. |
| Capture runs in a `DataLoader`/spawn worker rather than the main process. | Entry raises `RuntimeError`. Initialized non-daemonic distributed rank processes are the explicit collective-capture exception. | Capture in the main process, or use the documented distributed rank workflow. |
| Input or model state includes a FakeTensor, FunctionalTensor, meta tensor, sparse tensor, or symbolic-shape tensor. | `UnsupportedTensorVariantError` at preflight. | Materialize dense, strided tensors with concrete shapes on a real device before capture. |
| The model is TorchScript (`torch.jit.script` / `torch.jit.trace`) or a `torch.export.ExportedProgram`. | Capture refuses at entry because execution is not ordinary eager Python. | Capture the source `nn.Module` before scripting/exporting. |
| `torch.compile(model)` wraps an `nn.Module`. | TorchLens captures the eager source, not compiled fusion/kernel semantics; a one-time note reports the unwrap. A wrapper identity guard may cause at most one later recompile. | Use TorchLens for eager provenance and a compiler/profiler tool for the compiled graph. |
| `nn.DataParallel` or DDP wraps the model. | The `.module` is unwrapped; threaded replicas are not captured as one concurrent Trace. | Capture the inner module directly when replica-local detail matters. |
| A tensor subclass implements custom `__torch_function__`. | Capture may run with reduced subclass-specific metadata fidelity. | Prefer plain tensors for the captured run. |
| Module nesting exceeds Python's recursion limit. | Traversal may raise `RecursionError`. | Flatten the hierarchy or deliberately raise `sys.setrecursionlimit`. |
| A buffer is reassigned through `buffer.data = value`. | End-of-capture reconciliation raises `RuntimeError`. | Use `self.buffer = value` or `self.buffer.copy_(value)`. |

## Structure-only capture

`structure_only=True` (DOCUMENTED-UNSTABLE surface; L7a wave 0, D8-default)
records the op graph, module hierarchy, parameter geometry, and per-op
shape/dtype as HYPOTHESES, never tensor values. The capability contract lives
in [structure_only_capabilities.md](structure_only_capabilities.md).

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| Your model branches on a tensor VALUE (`if x.sum() > 0:`, `.item()`, `bool(...)`, `tolist`/`numpy`/storage escapes) from user code — meta OR real tensor. | `ValueDependentBranchError` (`value_dependent_branch_unsupported`) naming the exact user source line and branch kind. A value branch would make every claim downstream a guess about WHICH graph exists. | Run a real capture (`tl.trace` without `structure_only`) to resolve the branch, or restructure the branch to be shape-derived. Shape/metadata reads (`shape`, `numel`, `dim`, `stride`) never refuse. |
| An op has no meta kernel on a meta-context tensor. | `MetaKernelUnavailableError` (`meta_kernel_unavailable`) at the failing callsite, with torch's original error chained. | Run a real capture, or upgrade torch for broader meta-kernel coverage. |
| `structure_only=True` combined with `raise_on_nan`, `intervention_ready`, or a not-provably-value-free `halt=`. | `structure_only_option_conflict` at entry. | Drop the conflicting option; use structured value-free selectors as `halt=`. |
| Any value-payload request (save selection, gradients, streaming sinks, raw input/output retention, output decoding). | `structure_only_values_unsupported` at entry. | Drop the payload request or run a real capture. |
| Replaying / validating / backward / runnable save on a structure-only trace. | Typed `structure_only_*_unsupported` refusals via the capability chokepoint. Analysis-level saves work as of tlspec v8: the artifact persists the `structure_only` marker plainly and every value claim stays a HYPOTHESIS on load. | Verify hypotheses with `trace.discharge_against(real_trace)`. |
| A meta-materialized model WITHOUT `structure_only=True`. | The entry gate refuses (`unsupported_tensor_variant`) and the teach names the admitted spelling: weights-free capture is admitted ONLY under the structure-only contract (D8 granted 2026-08-26). Mixed meta/real substrates refuse `structure_only_substrate_mismatch` in both directions. | Pass `capture=CaptureOptions(structure_only=True)` with an all-meta model and all-meta inputs (see docs/reference/weightsfree_capture.md), or materialize on a real device. |

Unenumerated REAL-value escapes in form (b) do not die and cannot be
intercepted; that residual is exactly why every value-bearing claim stays a
HYPOTHESIS until discharged against a real capture.

## Memory, payloads, and object lifetime

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| A normal capture runs with autograd enabled, even if the user never calls backward. | The live Trace can retain the captured autograd graph and its saved tensors until the Trace is cleaned up or released. This is trace-scoped, not a process leak. Measured on resnet18: ~1.48x the saved-payload bytes stay live at the `detach_saved_activations=False` default, vs ~1.01x with `True`. | For forward-only analysis use `inference_only=True` or pass `detach_saved_activations=True`. Otherwise call `trace.cleanup()` or drop the Trace promptly after use. |
| A Trace is dropped (`del` / rebinding) without `trace.cleanup()`. | The Trace object graph is cyclic and has no finalizer by contract, so a dropped capture stays fully alive until the next cyclic `gc.collect()` pass; after collection, most of its bytes remain in the glibc allocator arena, so process RSS does not recede on its own. | Call `trace.cleanup()` explicitly: it breaks the cycles immediately and best-effort returns freed arena memory to the OS (`malloc_trim`, glibc hosts). |
| `save_mode="view"` retains a value later mutated in-place. | An earlier saved activation visibly changes because the view intentionally aliases live storage. | Use the default `save_mode="copy"`, or `reference` only when live autograd identity is required. |
| `CaptureOptions(save_budget=...)` admits retained activations, then the user forward, a transform, or a cross-device temporary allocates more. | The budget may pass and the process can still OOM. On unmeasurable devices the first charge warns and automatic budgeting is disabled. | Treat `save_budget` as per-device admission control, not an OOM guarantee. Save fewer sites, stream selected disk-only payloads, and use an absolute ceiling on unmeasurable devices. |
| CPU/MPS capture is small or allocator/RSS samples do not move. | `Trace.forward_peak_memory` can legitimately be `0`; it is a cheap delta, not a high-water allocator proof. | Do not assert it is positive. Opt into `measure_python_peak_memory=True` when Python allocation peak justifies the extra cost. |
| Exhaustive saving (`capture=CaptureOptions(layers_to_save="all")`) is combined with disk storage. | Exhaustive payloads remain in RAM until postprocess and still count against the budget. | Use a selective `save=` predicate with streaming disk storage. |
| A lazy disk-backed activation appears in a text/HTML/JSON NaN/Inf report. | The report says disk-backed/unexamined and does not materialize it implicitly. | Call `op.materialize_out()` before the value-based report when that I/O is intended. |
| Whole-model `pickle`/`torch.save(model)` runs after tracing. | Persistent per-instance forward wrappers can produce `PicklingError`; a plain attribute holding a torch function from the other wrap epoch (`self.act = F.relu`) fails pickle's by-reference identity check in both directions; `state_dict()` is unaffected. | Call `tl.release_model(model)` immediately before whole-model serialization: it restores forwards AND normalizes held torch-function attributes (one level of exact builtin `list`/`tuple`/`dict`/`set`/`frozenset` containers — dict keys included — plus namedtuples, whose runtime type is preserved) to the live callables, and registers the model so any later `unwrap_torch()`/re-wrap re-normalizes it automatically. References inside closures/partials/custom objects or other builtin subclasses, and bare references held outside the model, stay unpicklable across wrap-state changes. |

## Runnable artifacts and repeated execution

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| `tl.save(..., level="runnable")` is called on a capture that omitted `intervention_ready=True`. | `RunnablePreflightError` with structured finding `MISSING_CALLABLE_REF`; no runnable artifact is guessed from incomplete templates. | Capture again with `capture=tl.options.CaptureOptions(intervention_ready=True)`. |
| A loaded runnable Trace is used repeatedly for activation collection. | The first ordinary `run()` performs the full transactional verification. `run(..., fast=True)` is refused until that run settles `verified`; later fast runs retain static-path and witness guards. | Verify once with ordinary `run(inputs=..., seed=...)`, then stream batches through `run(..., fast=True)`. Divergence always raises in fast mode. |
| The forward reads a private Python/NumPy RNG instance, constructs an unseeded generator, reads `secrets`/OS entropy/`uuid4`, or reads a current clock. | Capture records host nondeterminism. Every artifact run settles `path_faithfulness="unverifiable"`, `numeric_attestation="not_applicable"`; `RunReport.nondeterministic_sources` includes `host_rng`. | Pass randomness/time as explicit inputs, or use the replayable global engines with a fixed seed. Do not derive model control flow from wall time or fresh entropy. |
| Host-nondeterminism monitoring cannot install, chain, restore, inventory, or classify confidently, or its bounded inventory is exhausted. | Witness completeness is incomplete and replay cannot become `verified`; diagnostics include the relevant inventory/monitor gap. | Simplify opaque holders, keep generators reachable from the model, and avoid custom profile-hook interference during capture. |
| An externally held NumPy generator is drawn on an already-running, unhooked foreign thread and is unreachable from model/frame digest roots. | This is the documented residual: Python <=3.11 cannot retrofit `threading.setprofile` onto that thread, so the channel may be outside the named monitor vocabulary. | Route the draw through the owner/in-window thread, store the generator on an inspectable model object, or pass the sampled value explicitly. |
| CUDA capture uses `cudnn.benchmark=True` or a documented nondeterministic CUDA op without deterministic algorithms. | Path replay may verify, but numeric attestation is `not_applicable` via the positive `attestation_ineligible_context` marker. | Use `torch.use_deterministic_algorithms(True)` and deterministic kernels when byte attestation is required. |
| A caught in-forward exception changes control flow. | The current runnable contract conservatively ceilings the artifact at `unverifiable`; exception-handler replay proof is not shipped. | Refactor expected branching to explicit tensor/Python inputs instead of exception control flow. |
| Output uses an opaque/stateful container whose complete instance state cannot be reconstructed. | Runnable save refuses with `missing_output_container_contract`, or a live provider returns a poisoned `unverifiable` result. | Return tensors and supported stateless containers, or register a container with a complete state declaration. |
| A run diverges from the recorded path. | Default policy raises and rolls back. `on_divergence="return_diverged"` is the only opt-in result, and its Trace is permanently `poisoned=True`. | Treat divergence as a different program/input path and capture that path separately. |

## Distributed execution

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| Model/input state contains DTensor/ShardedTensor, active tensor-parallel hooks/styles, FSDP (`FullyShardedDataParallel`), or a pipeline stage. | `DistributedCaptureUnsupportedError`; structured `fields["findings"]` use kinds `dtensor`, `tensor_parallel`, or `pipeline_parallel` with sites and DTensor geometry. `tl.compat.report()` reports the same findings. | Capture an unsharded rank-local eager copy. A bare inert `DeviceMesh` is informational only. |
| The bounded distributed preflight scan hits an inspection boundary that could hide refusing state (an attribute read raises, a container walk is cut off). | `DistributedCaptureUnsupportedError` with a finding of kind `scan_incomplete`: the scan refuses rather than certifying absence it cannot prove. | Expose the model/input state through ordinary inspectable attributes, or remove the raising/opaque holder from the traced objects. |
| Explicit Python `torch.distributed` collectives execute in a rank-local forward. | They are captured only after `tl.distributed.arm()`. Async completion is marked `completion_binding="unobserved"`; wildcard receive refuses typed. Runnable save/forward replay refuses with `collective_boundary_runnable_unsupported`. | Arm at process start (`tl.distributed.arm()` is idempotent; `tl.distributed.is_armed()` reports the state and `tl.distributed.disarm()`, primarily for tests, removes the wraps -- ordinary programs stay armed for the process lifetime), use synchronous source-specific collectives, validate metadata, then combine ranks with `tl.merge_report` / `tl.merge_ranks`. |
| Rank cores disagree on membership, ordering, or collective correlation. | `tl.merge_report(...)` returns conflicted findings; `tl.merge_ranks(...)` refuses rather than inventing joins. | Fix process-group lifecycle and capture the same program on every rank. Digest witnesses can demote evidence but never rescue a conflict. |
| Point-to-point pipeline graphs or DTensor topologies are passed to rank merge. | Typed C3/C2 construction refusal; merged replay does not exist. | Keep rank-local traces for inspection or capture a supported dense explicit-collective SPMD program. |
| Distributed state is hidden behind descriptor-only/slots-only holders, opaque wrapped TP hooks, beyond the bounded walk, or created inside `forward`. | These are disclosed scan residuals; preflight cannot positively certify absence through opaque state. | Expose registered state/hooks through inspectable module attributes and run `tl.compat.report()` on the concrete inputs. |

## Preview backends

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| A JAX/MLX/tinygrad/Paddle/TensorFlow preview is asked for torch-only capabilities such as fastlog, halt, streaming, true backward, or unsupported intervention. | Backend-specific typed capability refusal. Static-label `save=` filters exposed results after full capture and may not reduce capture memory. | Use PyTorch eager for the full surface, or stay within the backend table in [Backends](../backends.md). |
| A loaded non-torch artifact is asked to replay-validate. | `trace.validation_replay_status` is unavailable, commonly reason `loaded_trace_runtime_capture_stripped` or `backend_validation_replay_unsupported`. | Validate while the original backend runtime capture is live; use the loaded artifact for analysis. |
| JAX importer-owned `scan`/`while_loop` regions exceed expansion caps or include forward `custom_vjp_call`; TensorFlow contains pure/effect regions outside exact replay. | Validation state is `unverified`; `bool(status)` raises so partial coverage cannot be mistaken for a pass. | Inspect the status counts/reason, reduce the region, or validate the opaque computation with its native framework. |
| tinygrad capture uses JIT, mutation/realization, or a runtime other than the pinned Python device; Paddle uses mutation, RNG, tensor-to-Python control, or active stochastic training composites. | The preview refuses or reports an unverified boundary rather than silently importing it. | Use the pinned runtime and deterministic eval-mode code, or capture the equivalent PyTorch model. |

## Numeric and structural interpretation

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| Quantized modules use uncommon kernels. | A warning is emitted; capture continues, but FLOPs are estimates and clone fallback may use CPU float32. | Treat performance metadata as approximate and validate outputs separately. |
| bf16/fp16 GPU reductions replay in a different legal order. | Validation can exceed its tolerance even when the model is semantically sound. | Re-run deterministically or validate in float32 before classifying the mismatch as capture failure. |
| Two unrelated repeated subgraphs have the same loop fingerprint. | Recurrence detection can group them into a layer with more passes than expected. | Capture with `recurrence_detection=False` / the current recurrence-disable option and inspect the ungrouped Ops. |
| An input-routed intervention targets an in-place or `out=` call. | Recognized in-place calls snapshot semantic inputs, but raw hooks may see live references; detected `out=` aliasing warns at hook fire time. | Prefer output interventions or non-mutating functional spellings when the original input value matters. |
| The rendered universe or predicted planner cost exceeds the smart-collapse admission budget (defensive constant `COLLAPSE_OPTIMIZER_MAX_OPS`, 2,000, on U; measured (U, W) estimator). | `draw(collapse=...)` warns (`collapse_budget_fallback`) and renders a deterministic compact fallback plan -- never an uncollapsed wall. Pathological inputs (raw ops above 20x the constant) still decline outright: `collapse_pathological_skip`, `collapse_plan()` refuses typed (`collapse_plan_unavailable`), one-step schedule. | Shrink the rendered graph to re-admit the quality planner: `module=` focus, `vis_call_depth`, or rolled mode. See `docs/reference/collapse.md`. |

## Visualization and rendering

| When it can occur | What you see | Remedy |
| --- | --- | --- |
| The rendered universe U exceeds the defensive constant `COLLAPSE_OPTIMIZER_MAX_OPS` (2000; `torchlens.visualization.collapse_optimizer`) or the measured (U, W) work estimator predicts over-budget planning cost. | A coded `TorchLensWarning` (`collapse_budget_fallback`) discloses U, W, and the prediction; the render is a deterministic compact fallback plan (`planner="linear_fallback"`), never an uncollapsed wall. Raw ops above 20x the constant decline outright (`collapse_pathological_skip`; `collapse_plan()` refuses `collapse_plan_unavailable`; one-step schedule). | Reduce the rendered graph to re-admit the quality planner: `module=` focus, `vis_call_depth`, or rolled mode. `fold_repeats` and manual `show_containers` styling still apply. |
| The resolved visible graph exceeds ~10k nodes / 40k edges. | A `TorchLensWarning` discloses that Graphviz layout beyond this size can take minutes and gigabytes; rendering proceeds. | Use `collapse="auto"`/`"max"`, `show_containers=False`, or draw a focused subgraph. |
| `draw(code_panel=True)` renders live-capture source panels. | The "Open source" link embeds the absolute local source path (an editor `vscode://file/...` URL, which can carry a username) so the panel is click-to-open on the capturing machine. The visible tooltip shows the file basename only. | Treat code-panel SVGs as local artifacts; strip or re-render before sharing. Saved `.tlspec` bundles are unaffected -- the artifact scrub stores basenames. |

## CI-attested platforms

Continuous integration executes the full tiered suite on Linux/CPU only
(Python 3.10–3.13, torch floor through newest-admitted). macOS and Windows
run a nightly import + capture + save/load canary, not the suite. CUDA and
MPS behavior — device peak memory reporting, `save_budget="auto"` device
measurement, `map_location` transport, and the CUDA-artifact-on-CPU-host
refusal — is implemented and documented but has **no CI execution**; treat
device-specific claims as attested by local testing only.

## Related contracts

- [Detached-reference handling](../migration/scoped_detached_patching.md)
- [Runnable artifact model](runnable_model.md)
- [Merged-trace contract](merged_trace_contract.md)
- [Capture outcomes](capture_outcomes.md)
- [Performance and memory controls](../performance.md)
