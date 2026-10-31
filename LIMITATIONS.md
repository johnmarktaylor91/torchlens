# TorchLens Limitations

> Canonical user-facing catalog: [docs/reference/limitations.md](docs/reference/limitations.md).
> This root file is the extended engineering companion; on any disagreement the
> canonical page wins.

TorchLens observes eager PyTorch execution by wrapping PyTorch call sites and recording what
actually runs. Its substrate identity is therefore "PyTorch eager execution plus TorchLens
metadata", not a separate static IR, compiler graph, or symbolic model format.

## Single-Threaded By Design

TorchLens is single-threaded by design. Capture depends on process-global logging state and
ordered operation counters, so running captures concurrently in multiple Python threads or child
processes can corrupt ordering assumptions. Use TorchLens from the main process.

## Remote Execution and Serving Engines

TorchLens records the forward that runs in its own Python process. It has no remote-execution
service, so a model running on another machine or behind a hosted-inference API is out of
capture scope; run TorchLens where the model runs and move experiments as `.tlspec/` recipes.
Serving engines such as vLLM and SGLang run fused kernels, paged KV caches and CUDA graphs outside
eager PyTorch; trace the eager Hugging Face twin of the same checkpoint instead. The full list of
jobs TorchLens leaves to other tools is [What TorchLens does not do](docs/non_goals.md).

## Dynamic Control Flow

Eager dynamic control flow is a feature, not a limitation. TorchLens records the branch, loop, and
module behavior that occurred for the concrete input you supplied. It does not claim to enumerate
branches that did not execute.

## torch.compile Regions and Tracing Tensors

Tensors inside a `torch.compile` (Dynamo) region are data-free `FakeTensor`s. Every TorchLens step
that reads a value -- `safe_copy`, `torch.equal`, `.item()`, `data_ptr()`, memory accounting -- is
meaningless or fatal on them, so TorchLens cannot record real activations there.

Compiled child `nn.Module`s are handled automatically: they are swapped for their eager source
module for the duration of capture (with a one-time note), so their interiors *are* logged. A
compiled **callable** cannot be swapped that way. On torch 2.6 and newer, TorchLens runs the capture
under `torch.compiler.set_stance("force_eager")`, so a compiled callable runs its original Python
and its interior is logged with full verification. On older torch, where `set_stance` does not
exist, reaching one degrades gracefully: operations inside the compiled region are not logged,
a one-per-forward `UserWarning` names the gap and the remedy, and the returned `Trace` contains
only what ran outside the region, marked `capture_verified=False` with
`capture_verification_reason="dynamo_region_not_logged"`. Call the eager function during capture
if you need its interior there.

On that older-torch path, for plain module attributes, TorchLens inventories Dynamo's original-callable marker before the
forward and temporarily invokes the callable with TorchLens logging paused. This prevents both cold
FakeTensor internals and warm-cache execution from entering capture wrappers, independently of
`torch.compiler.is_compiling()` timing. The inventory is conservative: merely holding such a
callable marks the capture incomplete even if a particular branch does not call it. A compiled
callable reached only through a module global/free-function reference still relies on the in-wrapper
boundary when Python is entered; opaque hot-cache execution there is a disclosed residual. Use an
eager callable for a complete claim.

Passing a `FakeTensor` or `FunctionalTensor` as a model input, or tracing a model whose parameters
were built under a fake mode, is refused at capture entry with `UnsupportedTensorVariantError`
alongside the other data-free variants. Previously this crashed mid-forward with a bare
`AssertionError: Please convert all Tensors to FakeTensors first` from torch's own fake machinery,
after TorchLens had already tripped torch's "almost definitely a bug in your code" warning by
reading a FakeTensor's `data_ptr()`.

Input preflight follows builtin containers and inspectable instance `__dict__` state to a bounded
12-level/4096-object limit, including exact-type tensors produced by
`torch._to_functional_tensor`. Descriptor-only or slots-only custom containers, and functional
tensors created inside `forward`, remain outside entry-time detection; the `vmap_functorch`
compatibility row discloses that residual.

## Retained Activation Footprint

`tl.trace(model, x)` retains every operation's output by default. That is the right default for
the models TorchLens was designed around and a footgun at frontier shapes, where it means an OOM
kill (or an allocator error from deep inside torch) rather than an explanation.

Capture therefore enforces a per-device ceiling on retained payload bytes, `save_budget`, which
defaults to half of each device's *available* memory measured at that device's first save. Crossing
it stops capture with `torchlens.errors.SaveBudgetExceededError`, naming the bytes accounted so far,
the budget and where it came from, the operation that tripped it, and the remedies.

The primary retained-copy path is admitted **before** `safe_copy`, using the source tensor's byte
size on the projected retention device. Physical retained storage is reconciled afterwards, so
aliased raw/transformed fields are charged once. The reported footprint is an explicitly-labelled
**lower bound** because the forward pass is incomplete.

This is not a general OOM-prevention guarantee. The model's own forward allocations happen before
TorchLens sees an output. A user activation transform's output size and alias behavior are unknowable
until that callable runs, and a cross-device move can require a temporary source copy; additional
transform storage is charged after it exists. The guard prevents the ordinary over-budget retained
copy that it can project, while these allocations remain disclosed residuals.

Only payloads retained in RAM are charged. Predicate-selected disk-only saves such as
`save=tl.func("relu"), storage=tl.to_disk(...)` are exempt. Exhaustive `capture=CaptureOptions(layers_to_save="all")` plus
`to_disk(...)` is **not** exempt: it keeps RAM copies until postprocess attaches disk refs and evicts
them, so a tiny budget refuses. `layers_to_save="none"` remains uncharged.

When an automatic fraction cannot measure device headroom (including MPS and unknown device types),
that device is left unbudgeted and a `UserWarning` fires on its first non-empty charge. Use an
absolute integer budget to enforce a ceiling there.

Tune it with `capture=tl.options.CaptureOptions(save_budget=...)`: a float in `(0, 1]` for another
fraction of available memory, an int for an absolute per-device byte cap, or `None` to disable the
guard. A malformed value raises rather than silently unguarding the capture.

## Distributed and Sharded Execution

TorchLens captures a **rank-local eager forward pass**. Sharded distributed execution is out of
scope for this release, and TorchLens now says so instead of returning a wrong trace.

`DTensor` and `ShardedTensor` are tensor subclasses whose real work happens under
`__torch_dispatch__`, *below* the `__torch_function__` layer TorchLens wraps. A capture of such a
model does not crash: it silently records rank-local view/reshape shims in place of the real ops
and reports zero parameters. Pipeline-parallel stages are a related case — a stage holds only its
own slice of the model and is driven by a schedule that runs microbatches outside the traced
forward, so a capture there is a fragment, not the model.

Because all-pass-then-silently-wrong is worse than a clear failure, both surfaces now report it:

- `tl.compat.report(model, x)` has `dtensor`, `device_mesh`, `tensor_parallel`, and
  `pipeline_parallel` rows.
- `tl.trace(...)` refuses at capture entry with
  `torchlens.errors.DistributedCaptureUnsupportedError`, naming the exact parameter, buffer, or
  input sites that carry the sharded state. The structured findings are on
  `exc.fields["findings"]`, so code branches on `finding.kind` rather than message text.

Refusal is deliberately narrower than reporting, but active tensor-parallel styles are refusing:
even with dense parameters, `PrepareModuleInput`/output hooks can run rank redistribution and
collectives below TorchLens' wrapped layer. Sharded tensor state (`dtensor`), active TP hooks/styles
(`tensor_parallel`), and pipeline stages (`pipeline_parallel`) refuse. Only a bare inert
`DeviceMesh` with ordinary dense state remains informational.

**What to do instead:** trace the undistributed module. Build and trace the model before wrapping
it with `parallelize_module` / `fully_shard` / pipeline stages, or materialize the logical tensors
(`DTensor.full_tensor()`) into a plain module first. Rank-local captures can be combined afterwards:
`tl.merge_report(...)` reports conflicts and `tl.merge_ranks(...)` merges, refusing rather than
inventing joins when ranks disagree (`docs/reference/merged_trace_contract.md`).

Detection is capability-probed, never version-parsed, and every degradation point is visible as a
named flag in `torchlens.report.doctor()` / `tl.compat.report()`: `HAS_DTENSOR`, `HAS_DEVICE_MESH`,
and `HAS_PIPELINING`. When one is absent, detection falls back from an exact `isinstance` check to
structural namespace matching, and the affected report row says so.

The entry scan covers registered parameters/buffers, builtin and inspectable user input containers,
plain module tensor attributes, direct TP-namespace forward-hook registries, and nested module
attributes up to 12 levels / 4096 objects. It never executes descriptors. Descriptor-only or
slots-only holders, user-wrapped/opaque TP hooks that hide their defining namespace, state beyond
the bound, and distributed tensors constructed inside `forward` remain residual classes; the clear
compatibility rows state this rather than issuing an unqualified clean bill of health.

## Disk-backed reporting

`repr`, `str`, notebook HTML, `report.explain(...)`, and `trace.to_agent_json()` never materialize
a saved activation solely to answer the NaN/Inf summary. An unmaterialized disk ref is counted as
"disk-backed and not examined by reporting" in clean text, HTML and JSON answers. Call `op.materialize_out()`
explicitly before asking for a value-based answer when the I/O cost is intended.

## Bundle Diff Rendering

`Bundle.show_diff()` is static for v1. Interactive bundle comparison belongs in
a future local inspection appliance.

## Compatibility Truth Table

`tl.compat.report(model, x)` reports the following rows at runtime. Rows marked
`known_broken` or `scope` here should be included when filing issues so maintainers can
separate unsupported contexts from new TorchLens bugs.

| Row | Status | Notes |
| --- | --- | --- |
| `hf_transformers` | `pass` when detected alone | Eager Hugging Face modules are supported; compiled, sharded, offloaded and quantized models follow their own rows. |
| `accelerate_device_map_auto` | `pass` on one execution device; `known_broken` across several | Single-device dispatch captures with hook infrastructure excluded from the op graph and offloaded weights re-attributed; maps across several execution devices are not yet capture-verified. |
| `accelerate_cpu_disk_offload` | `pass` when detected | Offload hooks run outside the op graph and materialized weights re-attribute to their records; with `offload_buffers=True`, offloaded-buffer reads may stay unattributed (severity `warning`). |
| `bitsandbytes_8bit_4bit` | `pass` when detected (severity `warning`) | Activation capture keeps value parity; quantization-state reads may appear unattributed, FLOPs are best-effort, and `tl.validate` metadata invariants may flag those reads. Log an unquantized reference when exact metadata is required. |
| `tied_parameters` | `pass` | Shared parameter objects are detected and reported. |
| `multi_gpu_rng` | `pass` | RNG snapshots now use all visible CUDA devices; include this row for multi-GPU bugs. |
| `data_parallel` | `known_broken` when detected | `nn.DataParallel` uses threaded replicas and conflicts with process-global capture state. |
| `distributed_data_parallel` | `pass` when detected | DDP unwraps to the rank-local `.module`. |
| `fsdp` | `scope` when detected | FSDP sharded materialization is not launch-scope support. |
| `dtensor` | `scope` when detected | DTensor/ShardedTensor state; capture refuses with `DistributedCaptureUnsupportedError`. |
| `device_mesh` | `scope` when detected | Informational: identifies the distributed topology. Does not refuse capture on its own. |
| `tensor_parallel` | `scope` when detected | Active TP state/hooks refuse; dense parameters do not make hidden redistribution/collectives capturable. |
| `pipeline_parallel` | `scope` when detected | A stage is a fragment driven by an external schedule; capture refuses. |
| `deepspeed` | `scope` when detected | DeepSpeed/ZeRO/offload execution is not launch-scope support. |
| `torch_compile` | `scope` when detected | Log the eager model; use `torchlens.bridge.depyf` for compiled-code context. |
| `fx_graph_module` | `scope` when detected | TorchLens targets eager execution, not FX IR parity. |
| `lightning_training_step` | `known_broken` in active training mode | Use the Lightning callback or log a plain forward outside the trainer loop. |
| `vmap_functorch` | `known_broken` when detected | TorchLens skips logging inside active functorch transforms and returns incomplete logs. |
| `quantized_tensor` | `known_broken` when detected | Quantized comparison no longer crashes, but full quantized activation metadata remains best-effort. |
| `device_context_factory` | `pass` | Factory functions honor active `torch.device(...)` contexts during logging. |
| `single_thread_design` | `known_broken` off main process/thread | Capture state is process-global and ordered for a single main-thread forward. |
| Static graph export parity | `known_broken` | TorchLens records eager execution, not static export IR. |
| Optional bridge without its extra | `known_broken` | Install the matching extra before using bridge adapters. |

See `ROADMAP.md` for planned follow-up work.
