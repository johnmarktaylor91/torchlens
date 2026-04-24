# TorchLens limitations and unsupported contexts

TorchLens works by wrapping every PyTorch callable with a toggle-gated Python
wrapper, then running a normal Python forward pass with the toggle enabled.
That design has clear consequences:

- If the forward pass is **not actually Python** (it ran a compiled,
  scripted, or exported graph), our wrappers don't fire and there's
  nothing to log.
- If the tensors involved aren't **standard dense strided tensors** on a
  real device, the metadata / copy paths the logger depends on misbehave
  in subtle, hard-to-debug ways.
- If the logging is **concurrent or re-entrant**, the single global
  ``_active_model_log`` state gets corrupted.

This page enumerates every context the authors currently know TorchLens is
not reliable in, what TorchLens does when it detects the situation, and
the recommended workaround. See the [robustness sprint
catalog](https://github.com/johnmarktaylor91/torchlens/pull/143) for the
original analysis.

If you hit a case we haven't listed, please
[open an issue](https://github.com/johnmarktaylor91/torchlens/issues/new).

---

## At-a-glance matrix

| Context | What TorchLens does today | Workaround |
|---|---|---|
| **Nested `log_forward_pass`** (hook/postfunc calls log again) | `RuntimeError` at inner entry | Use `pause_logging()` before the inner call, or run the inner log afterwards on the outer's sub-model |
| **`torch.compile(model)`** | `RuntimeError` with pointer to this page | Log the un-compiled model |
| **`torch.jit.script` / `torch.jit.trace`** | `RuntimeError` at entry | Log the un-scripted / un-traced Python module |
| **`torch.export.ExportedProgram`** | `RuntimeError` at entry | Log the source `nn.Module` before exporting |
| **`FullyShardedDataParallel` (FSDP)** | `RuntimeError` at entry | Log a rank-local unsharded copy of the inner module |
| **`DistributedDataParallel` (DDP)** | Automatically unwrapped via `.module` | — (just works) |
| **`nn.DataParallel`** | Automatically unwrapped via `.module` | — (just works) |
| **Meta tensor inputs / meta-init model** | `UnsupportedTensorVariantError` | Materialise the model on a real device (`model.to("cpu")`) first |
| **Sparse tensor inputs** | `UnsupportedTensorVariantError` | Pass dense tensors; densify sparse inputs with `.to_dense()` |
| **Symbolic-shape (`SymInt`) inputs** | `UnsupportedTensorVariantError` | Pass tensors with concrete integer shapes |
| **Quantized (`torch.ao.quantization`) model** | `UserWarning`; logging continues | Treat FLOPs in the log as approximate / wrong |
| **`torch.func.vmap` / `grad` / `jacfwd`** | `UserWarning` once per pass; inner ops skipped | Log the pre-vmap module separately, or accept an incomplete log |
| **Multi-process spawn / `DataLoader` workers** | `RuntimeError` if called from a worker | Log in the main process |
| **Tensor subclasses with custom `__torch_function__`** | May work; limited metadata fidelity | Log with a plain `torch.Tensor` input if possible |
| **Very deep module hierarchy (>1000 levels)** | May hit Python recursion limit | Flatten the hierarchy, or raise `sys.setrecursionlimit` |

---

## Details

### `torch.compile` — not supported (raises)

``torch.compile(model)`` returns a ``torch._dynamo.eval_frame.OptimizedModule``
that replaces the Python ``forward`` with a compiled graph. Depending on the
dynamo backend, our Python-level function wrappers are either inlined out of
existence (inductor) or called on flattened IR tensors whose metadata won't
match the user-visible graph.

``log_forward_pass`` detects ``OptimizedModule`` up front and raises a clear
error. **Workaround**: log the model before applying ``torch.compile``.

### `torch.jit.script` / `torch.jit.trace` — not supported (raises)

A ``torch.jit.ScriptModule`` runs its forward on the TorchScript interpreter
instead of Python. TorchLens's decorated wrappers are Python objects, so they
never see the calls.

``log_forward_pass`` detects ``torch.jit.ScriptModule`` and raises.
**Workaround**: log the Python ``nn.Module`` before scripting or tracing.

### `torch.export.ExportedProgram` — not supported (raises)

``torch.export`` produces a serialisable IR; it is not a callable
``nn.Module`` and cannot be re-executed in Python. ``log_forward_pass``
detects ``ExportedProgram`` and raises.

**Workaround**: log the source ``nn.Module`` before exporting.

### FullyShardedDataParallel (FSDP) — not supported (raises)

Parameters under FSDP are sharded across ranks as flat
``FlatParameter`` buffers; there is no single unsharded module whose
parameter tree matches what the user wrote. Silently unwrapping via
``.module`` would yield misleading layer metadata and break loop
detection's parameter-sharing heuristic.

**Workaround**: log a rank-local unsharded copy of the inner module
(before FSDP wrapping).

### Meta tensors, sparse tensors, symbolic shapes — pre-flight raise

Detected by ``torchlens._robustness.check_model_and_input_variants`` at
the top of ``log_forward_pass``. Each raises
``UnsupportedTensorVariantError`` with a bullet-listed message. Rationale:

- **Meta** tensors have no backing storage, so activation saving returns
  empty tensors instead of real values.
- **Sparse** tensors defeat ``safe_copy`` (``.clone()`` doesn't honour
  memory_format), ``print_override`` (no numpy fallback), and ``numel``
  semantics (counts are non-comparable).
- **Symbolic (``SymInt``) shapes** break counter alignment between the
  exhaustive and fast passes and FLOPs accounting that assumes concrete
  integer dims.

### Quantized models — warn, keep going

Detected by walking ``model.modules()`` and matching against
``torch.ao.quantization`` / ``torch.nn.quantized`` module name prefixes.
The log is generally usable but:

- FLOPs counts are **wrong** for quantized ops (the FLOPs table assumes
  standard floating dtypes).
- Activation dtype handling in ``safe_copy`` falls back to CPU + float32
  when the quantized clone rejects ``memory_format``.

### `torch.func.vmap` / `grad` / `jacfwd` — warn, ops inside the transform not logged

TorchLens detects an active functorch transform via
``torch._C._functorch.maybe_current_level`` and skips logging inside
the transform (internal ops like ``safe_copy`` would crash because they
have no vmap batching rules).

A ``UserWarning`` fires once per forward pass so users know the log omits
whatever ran inside the transform. Ops that ran **outside** the transform
are logged normally.

**Workaround**: log the non-vmap'd model separately if you need a full
log.

### Multi-process / `DataLoader` workers — raises

``log_forward_pass`` calls ``warn_parallel()`` early: if
``multiprocessing.current_process().name != "MainProcess"``, it raises.
TorchLens's global toggle state and ordered tensor counters are not
safe under concurrent access.

**Workaround**: run ``log_forward_pass`` in the main process only.

### Nested `log_forward_pass` — raises

``active_logging()`` raises ``RuntimeError`` on re-entry rather than
silently overwriting ``_active_model_log`` (which would corrupt the
outer log mid-pass). The realistic trigger is a user forward-hook or
activation callback that calls ``log_forward_pass`` on a sub-model.

**Workaround**: if you genuinely need to capture a sub-model's forward,
either (a) run ``log_forward_pass`` on the sub-model *outside* the outer
pass, or (b) wrap the inner call in ``torchlens.pause_logging()`` so the
outer's toggle is suspended.

### Partial-support contexts (no runtime guard)

These work but have known caveats. They are not detected automatically;
if your log looks wrong in one of these scenarios, suspect the caveat:

- **Tensor subclasses** (custom ``__torch_function__``): our wrappers call
  the original C function, bypassing the subclass dispatch; custom
  dtype / label handling in the subclass won't fire during logging.
- **Very deep module hierarchy** (>1000 levels): the submodule traversal
  is recursive and may hit Python's default recursion limit.
- **User forward hooks / pre-hooks**: TorchLens registers its own hooks
  at model-prep time. Ordering with user hooks depends on registration
  order; in particular, user pre-hooks that mutate inputs are seen by
  TorchLens as the new mutated input, not the pre-hook input.
- **bfloat16 / fp16 + non-deterministic GPU reductions**: validation
  replay compares activations to within ``3e-6`` absolute tolerance; on
  bf16/fp16 GPU atomics, small reordering differences can cross that
  tolerance and cause validation to fail even though the model is fine.
- **Loop detection on structurally unrelated repeat patterns**: the
  isomorphic-subgraph expansion in ``postprocess/loop_detection.py`` can
  merge two unrelated loops if they share the same operation
  fingerprint. Disable with ``detect_recurrent_patterns=False`` if you
  see a layer with more passes than you expect.

---

## Reporting a new failure

If you hit a crash or wrong result in a context that isn't listed above:

1. Run `python -c "import torch; print(torch.__version__)"`.
2. File an issue with: the torch / python version, the full traceback,
   and (if possible) a small reproducer.
3. If it's a silent-wrong-result, include the layer count in the resulting
   ``ModelLog`` vs. what you expected.
