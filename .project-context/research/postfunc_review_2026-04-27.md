## activation_postfunc Review (2026-04-27)

### Summary
Working in the core slow/replay path, but the implementation is under-specified and metadata/UI semantics drift when the callable changes shape, dtype, memory footprint, or autograd connectivity. `gradient_postfunc` is simpler and more internally consistent; `activation_postfunc` carries more integration risk because activations feed validation, summaries, streaming, and training-mode workflows.

### How it works (current state)
- Public API exposure is on `log_forward_pass(..., activation_postfunc=...)` and the internal runner passes it into `ModelLog` (`torchlens/user_funcs.py:341`, `torchlens/user_funcs.py:643`, `torchlens/user_funcs.py:671-679`).
- `ModelLog` stores both the callable and `repr(callable)` (`torchlens/data_classes/model_log.py:320-323`).
- Each `LayerPassLog` also stores the callable reference as metadata (`torchlens/capture/output_tensors.py:931-935`, `torchlens/capture/source_tensors.py:222-228`).
- The callable is invoked only when a layer/source tensor is selected for saving, not for metadata-only captured layers (`torchlens/capture/output_tensors.py:999-1009`, `torchlens/capture/source_tensors.py:400-406`).
- Invocation happens in `LayerPassLog.save_tensor_data`: `safe_copy()` first, optional device move second, `activation_postfunc` third, then sink/streaming/captured args (`torchlens/data_classes/layer_pass_log.py:760-811`).
- It runs inside `pause_logging()` so user tensor ops inside the callable do not appear as model ops (`torchlens/data_classes/layer_pass_log.py:765-769`).
- In two-pass mode, pass 1 is metadata-only with `layers_to_save=None`; the callable effectively runs only in pass 2 for requested layers (`torchlens/user_funcs.py:665-713`).
- Streaming writes happen after postfunc, so streamed blobs contain transformed activations (`torchlens/data_classes/layer_pass_log.py:797-805`).
- Fast replay uses the same `save_tensor_data` path for ordinary layers and source tensors (`torchlens/capture/output_tensors.py:712-723`, `torchlens/capture/source_tensors.py:400-406`).

### Findings (severity-ordered)

#### Finding 1
- **Severity:** HIGH
- **Location:** `torchlens/data_classes/layer_pass_log.py:760-781`; `torchlens/validation/invariants.py:370-384`; `torchlens/postprocess/labeling.py:432-433`
- **Issue:** `activation_postfunc` can change the saved activation’s shape, dtype, and byte size, but `tensor_shape`, `tensor_dtype`, `tensor_memory`, `saved_activation_memory`, summaries, and invariant checks continue to describe the pre-postfunc tensor.
- **Why it matters:** Dimensionality reduction and dtype casting are listed as core use cases. With `activation_postfunc=torch.mean`, tests assert scalar activations are saved, but metadata still describes the original tensor. This makes summaries/UI misleading and can make `check_metadata_invariants()` fail on valid user output.
- **Recommended fix:** Decide whether fields represent raw model outputs or stored activations. If stored activations, update shape/dtype/memory immediately after postfunc and compute saved-memory totals from transformed tensors. If raw outputs, add explicit stored-activation metadata fields and adjust summaries/invariants to distinguish raw vs stored.

#### Finding 2
- **Severity:** HIGH
- **Location:** `torchlens/data_classes/layer_pass_log.py:770-781`; `tests/test_train_mode/test_config_errors.py:79-88`
- **Issue:** `train_mode=True` only rejects transformed tensors with non-grad dtypes. It does not reject postfuncs that detach, move through non-differentiable conversions, or return non-tensors.
- **Why it matters:** `train_mode=True` promises saved activations stay graph-connected. A callable like `lambda t: t.detach()` or `lambda t: t.detach().cpu().numpy()` silently defeats that contract in non-streaming capture. Existing coverage only checks integer tensor output.
- **Recommended fix:** In train mode, validate that postfunc output is a `torch.Tensor`, has grad-compatible dtype, and remains autograd-connected when the input required grad. Document any allowed exceptions.

#### Finding 3
- **Severity:** MEDIUM
- **Location:** `torchlens/user_funcs.py:341-424`; `torchlens/fastlog/_record_one_shot.py:18-36`; `torchlens/fastlog/types.py:17-38`
- **Issue:** `activation_postfunc` is available on slow `log_forward_pass` but not on `tl.fastlog.record`/`Recorder`. Fastlog has `CaptureSpec(dtype=..., device=...)`, which covers some common postfunc uses but not arbitrary transforms.
- **Why it matters:** Users now have two activation-capture APIs with different transformation models. A power user who uses `activation_postfunc` for pooling, compression, or custom serialization cannot move that workflow to fastlog without rewriting it as a predicate/storage feature.
- **Recommended fix:** Either document this as an intentional API split or add a separate fastlog transform hook/CaptureSpec extension. This is probably a separate feature/PR, not a small cleanup.

#### Finding 4
- **Severity:** MEDIUM
- **Location:** `torchlens/user_funcs.py:341-424`; `torchlens/data_classes/layer_pass_log.py:737-755`; `torchlens/_io/streaming.py:151-159`; `torchlens/_io/bundle.py:793-804`
- **Issue:** The expected callable signature and return contract are not explicit. Type hints use bare `Callable`, docs say “function applied to each activation,” and one internal doc example says “to-numpy,” but streaming/portable save require tensor outputs.
- **Why it matters:** Users cannot tell whether the callable receives only a tensor or also a label/context, whether non-tensor returns are supported, or whether storage modes change the contract. Mypy cannot enforce the intended unary `(torch.Tensor) -> torch.Tensor` shape.
- **Recommended fix:** Define a type alias such as `ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]`, use it across public/internal APIs, and document storage-mode constraints. If non-tensor in-memory returns are intentionally supported, say that explicitly and isolate it from portable save.

#### Finding 5
- **Severity:** MEDIUM
- **Location:** `torchlens/data_classes/layer_pass_log.py:782-788`; `torchlens/capture/trace.py:510-527`; `tests/test_io_streaming.py:183-205`
- **Issue:** Non-streaming postfunc failures surface as the raw exception plus a generic “Feature extraction failed” print, without the layer label/op being processed. Streaming errors get contextual wrapping and partial-bundle marking.
- **Why it matters:** Postfuncs are user code and may fail on specific tensor shapes/dtypes. Without the layer label, users must instrument their callable manually to find the failing activation.
- **Recommended fix:** Wrap postfunc errors in a TorchLens-specific error that includes the best available label, raw label, function name, shape, dtype, and whether streaming was active. Keep the original exception as `__cause__`.

#### Finding 6
- **Severity:** LOW
- **Location:** `torchlens/data_classes/layer_pass_log.py:816-846`; `torchlens/data_classes/layer_pass_log.py:760-811`
- **Issue:** `activation_postfunc` and `gradient_postfunc` are parallel in naming but not fully symmetric in semantics. Activation postfunc runs after `safe_copy(..., detach_saved_tensor)` and before storage; gradient postfunc runs on the hook gradient and is always followed by `detach().clone()`.
- **Why it matters:** The asymmetry is mostly justified, but it should be documented. In particular, activation postfunc can preserve autograd in train mode, while gradient postfunc never stores graph-connected gradients.
- **Recommended fix:** Add one short public-doc paragraph comparing the two hooks: input, output, detach order, train-mode behavior, and persistence behavior.

#### Finding 7
- **Severity:** LOW
- **Location:** `torchlens/data_classes/model_log.py:143-168`; `torchlens/_io/scrub.py:166-170`; `torchlens/data_classes/model_log.py:534-546`
- **Issue:** Portable save intentionally drops the callable and keeps only `activation_postfunc_repr`; plain pickle may still attempt to pickle the callable. The user-visible persistence story is not documented.
- **Why it matters:** Users may expect a loaded `ModelLog` to know how activations were transformed or to reuse the transform in `save_new_activations`. Portable bundles preserve only a descriptive repr, not behavior.
- **Recommended fix:** Document that portable `torchlens.save/load` stores transformed tensors and `activation_postfunc_repr` only. Consider setting dropped callables to `None` consistently after portable load and exposing the repr in summaries/metadata.

#### Finding 8
- **Severity:** NIT
- **Location:** `README.md:252-256`; `notebooks/fastlog_tutorial.ipynb:162-183`; `torchlens/user_funcs.py:422-424`
- **Issue:** Discoverability is thin. README/fastlog docs cover `CaptureSpec` downcasting, but README does not appear to document `activation_postfunc`; the public docstring is only one line.
- **Why it matters:** This is a power-user feature with sharp interactions. Users are likely to discover it from signatures rather than guidance.
- **Recommended fix:** Add a compact example showing dtype cast or spatial pooling, plus warnings for metadata meaning, train mode, streaming, and persistence.

### Things that work well
- The callable is applied under `pause_logging()`, and tests cover that postfunc ops do not leak into the graph (`torchlens/data_classes/layer_pass_log.py:765-769`, `tests/test_decoration.py:756-762`).
- It runs after output-device placement and before streaming/sink callbacks, so persisted/callback payloads receive the transformed activation (`torchlens/data_classes/layer_pass_log.py:763-805`).
- Two-pass selective saves do not waste work applying the callable in the metadata-only pass (`torchlens/user_funcs.py:665-713`).
- Streaming has strict checks for non-tensor and unsupported tensor outputs and leaves a recoverable partial marker on failure (`torchlens/_io/streaming.py:151-173`, `tests/test_io_streaming.py:183-245`).
- `activation_postfunc_repr` is retained in portable scrubbed state while the callable is dropped, which is the right direction for safe persistence (`torchlens/data_classes/model_log.py:153-154`, `torchlens/_io/scrub.py:166-170`).
- Existing tests cover basic transform application, pause-logging behavior, streaming rejection, bundle-save rejection, and train-mode integer dtype rejection.

### Recommendations (prioritized)
- Fix immediately: clarify and repair stored-activation metadata semantics after postfunc, because current shape/dtype/memory/UI can be wrong for common use cases.
- Fix immediately: harden `train_mode=True` validation so postfunc cannot silently detach or return non-tensor activations.
- Fix next sprint: introduce explicit callable type aliases and fuller docstrings for both activation and gradient postfuncs.
- Fix next sprint: improve non-streaming error messages with layer/op context.
- Defer/live with: portable save dropping callables is reasonable; document it and keep `activation_postfunc_repr`.
- Separate feature/PR: decide whether fastlog should support arbitrary activation transforms or whether `CaptureSpec(dtype/device/keep_grad)` is the intended replacement.
