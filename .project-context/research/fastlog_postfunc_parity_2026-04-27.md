## Fastlog activation_postfunc Parity Research (2026-04-27)

### Summary
Fastlog should eventually accept arbitrary `activation_postfunc` callables, but it should land as a separate follow-up after the slow-path raw-vs-transformed refactor stabilizes. The fastlog implementation should be intentionally parallel, not shared through `LayerPassLog`: fastlog stores `ActivationRecord` objects, and its natural transform point is the central payload resolver after predicate selection and after `CaptureSpec` dtype/device transforms.

Do not ship `gradient_postfunc` for fastlog until fastlog has real predicate-selected gradient capture. `CaptureSpec(keep_grad=True)` currently keeps activation payloads graph-connected; it is not gradient recording.

### Fastlog architecture cliff notes
- Public one-shot entry is `torchlens.fastlog.record(...)`, which constructs a `Recorder`, calls `recorder.log(...)`, then returns `recorder.recording` (`torchlens/fastlog/_record_one_shot.py:18-36`, `torchlens/fastlog/_record_one_shot.py:66-85`).
- `Recorder` owns merged `RecordingOptions`, validates them, and creates a `RecordingState` in `__enter__` (`torchlens/fastlog/_recorder.py:168-182`, `torchlens/fastlog/_recorder.py:189-197`).
- Each `Recorder.log()` delegates to `_run_predicate_pass(...)` with the active state (`torchlens/fastlog/_recorder.py:221-258`).
- `_run_predicate_pass()` creates a lightweight `ModelLog` only to drive the decorated TorchLens wrappers in `logging_mode="predicate"`; the user-visible result is a `Recording`, not a `ModelLog` (`torchlens/fastlog/_orchestrator.py:137-143`, `torchlens/fastlog/_orchestrator.py:159-194`).
- Operation events are captured in `log_function_output_tensors_predicate(...)`: it assigns raw labels, builds a frozen `RecordContext`, evaluates `keep_op`, then records selected outputs (`torchlens/capture/output_tensors.py:165-227`).
- Source input/buffer events follow a sibling path in `log_source_tensor_predicate(...)` and only evaluate/store when `include_source_events=True` (`torchlens/capture/source_tensors.py:67-135`).
- Module enter/exit events are synthesized in the module forward decorator and can store metadata-only `ActivationRecord`s without tensor payloads (`torchlens/decoration/model_prep.py:763-836`).
- Stored fastlog data class is `ActivationRecord(ctx, spec, ram_payload, disk_payload, metadata, recorded_at)` inside a `Recording` (`torchlens/fastlog/types.py:112-121`, `torchlens/fastlog/types.py:220-239`).
- RAM/disk copy, detach, dtype, and device policy is centralized in `_storage_resolver._resolve_storage(...)` (`torchlens/fastlog/_storage_resolver.py:22-65`).
- Disk fastlog uses a separate directory-bundle format with `fastlog_index.jsonl`, not `ModelLog` portable scrubbing (`torchlens/fastlog/storage_disk.py:62-98`, `torchlens/fastlog/storage_disk.py:208-232`).

### Slow vs fastlog capture diff

| Area | Slow path | Fastlog |
|---|---|---|
| Public API | `log_forward_pass(..., activation_postfunc=..., gradient_postfunc=...)` (`torchlens/user_funcs.py:334-382`) | `tl.fastlog.record(...)` has predicates, streaming, train mode, but no postfunc args (`torchlens/fastlog/_record_one_shot.py:18-36`) |
| Capture unit | Every operation becomes a `LayerPassLog` via `_make_layer_log_entry(...)` (`torchlens/capture/output_tensors.py:1136-1178`) | Only selected events become `ActivationRecord`s (`torchlens/capture/output_tensors.py:141-162`) |
| Result object | Full `ModelLog` with `LayerPassLog` / `LayerLog` structures (`torchlens/data_classes/model_log.py:263-328`) | Sparse `Recording` with `ActivationRecord` list and indexes (`torchlens/fastlog/types.py:220-239`) |
| Activation storage | `LayerPassLog.save_tensor_data(...)` copies, moves, applies `activation_postfunc`, then stores/streams `activation` (`torchlens/data_classes/layer_pass_log.py:737-811`) | `_resolve_storage(...)` copies/detaches and applies only `CaptureSpec.dtype/device` (`torchlens/fastlog/_storage_resolver.py:22-65`) |
| Streaming | Slow path uses `BundleStreamWriter` from `LayerPassLog.save_tensor_data(...)` and portable `ModelLog` state (`torchlens/data_classes/layer_pass_log.py:798-811`) | Fastlog writes payload blobs and JSONL metadata directly in `DiskStorageBackend.append(...)` (`torchlens/fastlog/storage_disk.py:69-98`) |
| Train mode | Slow path stores graph-connected activations when configured and validates some postfunc dtype cases (`torchlens/data_classes/layer_pass_log.py:766-787`) | Fastlog `train_mode=True` promotes omitted defaults to `CaptureSpec(keep_grad=True)` and rejects conflicting defaults (`torchlens/fastlog/_recorder.py:98-119`) |

### Findings

#### A. Architecture diff
Slow-path capture builds a full `ModelLog`: `_run_model_and_save_specified_activations(...)` constructs `ModelLog` with `activation_postfunc` and `gradient_postfunc`, then runs the model with logging enabled (`torchlens/user_funcs.py:291-323`). Each logged op becomes a `LayerPassLog`, and saving happens through `_make_layer_log_entry(...)` into `LayerPassLog.save_tensor_data(...)` (`torchlens/capture/output_tensors.py:1136-1178`, `torchlens/data_classes/layer_pass_log.py:737-811`).

Fastlog runs the same decorated wrappers in `logging_mode="predicate"`, but it does not build full `LayerPassLog` entries for retained events. It builds `RecordContext` values, evaluates predicates, and stores selected outputs as `ActivationRecord`s in a `Recording` (`torchlens/fastlog/_orchestrator.py:137-159`, `torchlens/capture/output_tensors.py:193-227`, `torchlens/fastlog/types.py:112-121`). The main store sites are `_record_predicate_output(...)` for op outputs (`torchlens/capture/output_tensors.py:141-162`), `log_source_tensor_predicate(...)` for input/buffer payloads (`torchlens/capture/source_tensors.py:116-131`), module event metadata records (`torchlens/decoration/model_prep.py:798-831`), and backend append/write (`torchlens/fastlog/storage_ram.py:50-70`, `torchlens/fastlog/storage_disk.py:69-98`).

#### B. CaptureSpec coverage
`CaptureSpec` currently supports `save_activation`, `save_metadata`, `keep_grad`, `device`, and `dtype` (`torchlens/fastlog/types.py:16-38`). This declaratively covers metadata-only records, detached vs graph-connected RAM payloads, dtype downcasting/upcasting, and device placement. The tutorial documents dtype conversion as the power-user form (`notebooks/fastlog_tutorial.ipynb`, matched lines around the `CaptureSpec` section from `rg`; README also calls out downcasting with `CaptureSpec` at `README.md:254-257`).

It does not cover arbitrary tensor transforms such as pooling, slicing, compression, flattening, normalization, sparse/top-k extraction, custom quantization, or custom serialization. Today those require either changing the model output, post-processing `record.ram_payload` after capture, or using only dtype/device transforms in `CaptureSpec`.

#### C. Where would the postfunc be called?
The natural call site is `_storage_resolver._resolve_storage(...)`, not predicate evaluation. Both RAM and disk paths call this resolver through `RecordingState.resolve_storage(...)` (`torchlens/fastlog/_state.py:110-119`), and both storage backends delegate to it (`torchlens/fastlog/storage_ram.py:25-48`, `torchlens/fastlog/storage_disk.py:100-123`). It also covers op outputs and source events because `_record_predicate_output(...)` and `log_source_tensor_predicate(...)` both call `state.resolve_storage(...)` when `spec.save_activation` is true (`torchlens/capture/output_tensors.py:153-155`, `torchlens/capture/source_tensors.py:119-124`).

Place the callable after `safe_copy(...)` and after `_apply_payload_transforms(...)` so the order is: selected raw tensor -> safe copy/detach per storage target -> `CaptureSpec.dtype/device` -> `activation_postfunc` -> validation -> storage (`torchlens/fastlog/_storage_resolver.py:57-65`). This keeps predicates cheap and raw-metadata-based, while making the callable affect only retained payloads.

#### D. Shared infrastructure?
The new slow-path `transformed_activation` field cannot be reused directly by fastlog because fastlog does not write `LayerPassLog`; it writes `ActivationRecord` (`torchlens/fastlog/types.py:112-121`). Slow fields live in `LayerPassLog.PORTABLE_STATE_SPEC` and `LAYER_PASS_LOG_FIELD_ORDER` (`torchlens/data_classes/layer_pass_log.py:97-135`, `torchlens/constants.py:140-190`), while fastlog has separate frozen dataclasses and JSON serialization (`torchlens/fastlog/types.py:112-121`, `torchlens/fastlog/storage_disk.py:208-298`).

Recommendation: add parallel fastlog fields rather than a shared mixin. A shared “TransformableActivation” abstraction would bridge incompatible persistence models: `FieldPolicy`/portable scrubbing for slow path versus JSONL/index/blob metadata for fastlog (`torchlens/_io/scrub.py:139-190`, `torchlens/fastlog/storage_disk.py:208-232`). That abstraction would be premature unless a third capture backend appears.

#### E. Train mode interaction
Fastlog has a `train_mode` equivalent on both `record()` and `Recorder()` (`torchlens/fastlog/_record_one_shot.py:35-36`, `torchlens/fastlog/_recorder.py:125-140`). Its behavior is default promotion: omitted `default_op` / `default_module` become `CaptureSpec(keep_grad=True, save_activation=True, save_metadata=True)`, while explicit `True` or `CaptureSpec(keep_grad=False)` conflict (`torchlens/fastlog/_recorder.py:98-119`). The actual graph-connected payload comes from `_resolve_storage(...)`, where RAM uses `safe_copy(tensor, detach_tensor=not spec.keep_grad)` and disk always detaches (`torchlens/fastlog/_storage_resolver.py:57-65`).

Postfunc support must preserve this contract. If `spec.keep_grad=True`, the postfunc result for the RAM payload should be required to be a `torch.Tensor`, gradient-capable dtype, and graph-connected to the copied payload when the raw payload requires grad. Disk payloads are detached by design and should not be treated as trainable (`torchlens/fastlog/_storage_resolver.py:62-64`, `tests/test_fastlog/test_storage_modes.py:67-86`).

#### F. Streaming / disk persistence
Fastlog streams, but through a different path from slow `ModelLog` streaming. Slow streaming writes from `LayerPassLog.save_tensor_data(...)` via `BundleStreamWriter.write_blob(...)` (`torchlens/data_classes/layer_pass_log.py:798-811`). Fastlog disk mode constructs `DiskStorageBackend`, writes selected disk payloads before appending index lines, and finalizes fastlog-specific manifest/index files (`torchlens/fastlog/storage_disk.py:56-98`, `torchlens/fastlog/storage_disk.py:125-145`).

The new field would not flow through `FieldPolicy`; fastlog load/recover reconstructs `ActivationRecord` from JSON and metadata (`torchlens/fastlog/recover.py:89-128`, `torchlens/fastlog/storage_disk.py:219-232`). If fastlog stores both raw and transformed payloads, disk persistence needs either additional `ActivationRecord` fields plus JSON support, or explicit metadata keys for the transformed blob. Prefer explicit dataclass fields plus JSON keys because `ActivationRecord` is the public data model.

#### G. Predicate interaction
Postfunc should run after the predicate decides to store. Current predicate contexts are built from raw tensor metadata: `_build_record_context(...)` reads `tensor.shape`, `tensor.dtype`, `tensor.device`, and `tensor.requires_grad` from the raw tensor (`torchlens/fastlog/_record_context.py:114-123`). Predicate evaluation happens before storage resolution (`torchlens/capture/output_tensors.py:221-223`, `torchlens/capture/source_tensors.py:116-124`).

Defaulting to post-predicate execution is the right tradeoff: it runs only for retained events, avoids expensive transforms on every op, and keeps `RecordContext` semantics stable. A pre-predicate transform would force predicates to see transformed shape/dtype and would make `dry_run()` either expensive or misleading, since dry runs intentionally suppress payload capture (`torchlens/fastlog/dry_run.py:44-65`).

#### H. Existing fastlog gradient story
Fastlog does not currently record gradients. It supports `CaptureSpec.keep_grad=True`, which keeps RAM activation payloads attached so users can build losses from them (`torchlens/fastlog/types.py:26-38`, `tests/test_fastlog/test_storage_modes.py:49-64`). The project todo explicitly lists “Fastlog gradient support” as pending (`.project-context/todos.md:39-41`), and the backward research keeps it in the parking lot (`.project-context/research/backward_pass_sprint.md:160-166`).

Natural sequencing: ship fastlog gradient capture first, then `gradient_postfunc`. Landing them together is reasonable only if the gradient PR is already scoped to add a `GradientRecord`/gradient fields and disk persistence. Do not add `gradient_postfunc` as a dead option before fastlog has gradient payloads to transform.

#### I. Symmetry vs simplicity tradeoff
Fastlog should match slow path on user-visible transform contract: callable takes a `torch.Tensor`, returns a `torch.Tensor`, runs under `pause_logging()`, errors include event label/op/shape/dtype/storage context, and train-mode graph connectivity is validated. Slow path already applies postfunc under `pause_logging()` before streaming (`torchlens/data_classes/layer_pass_log.py:771-811`), and fastlog already uses `pause_logging()` around dtype/device transforms (`torchlens/fastlog/_storage_resolver.py:22-31`).

Fastlog should intentionally diverge on structure: use `ActivationRecord` fields, not `LayerPassLog.transformed_activation`; no function-argument capture; no `FieldPolicy`; no full metadata recomputation; no predicate-visible transformed metadata by default. That divergence follows the public README split: slow path is for full graph metadata/validation/visualization, fastlog is for predicate-selected activations across repeated rollouts (`README.md:234-257`).

#### J. One sprint or two?
Two. Slow-path raw-vs-transformed changes affect `LayerPassLog`, `LayerLog`, constants, portable IO, streaming, validation, and train-mode behavior (`torchlens/data_classes/layer_pass_log.py:97-135`, `torchlens/constants.py:140-190`, `torchlens/_io/bundle.py:773-806`). Fastlog requires a separate public API addition, `RecordingOptions` changes, `ActivationRecord` schema changes, JSON/load/recover updates, storage resolver changes, and tests across RAM, disk-only, RAM+disk mirror, dry-run, and train mode (`torchlens/fastlog/options.py:44-193`, `torchlens/fastlog/types.py:112-239`, `torchlens/fastlog/storage_disk.py:208-298`). Combining them would make review risk high and blur whether failures are slow-path data-model issues or fastlog storage/schema issues.

### Recommendation

Do this:

- Add `activation_postfunc` to `tl.fastlog.record(...)`, `Recorder(...)`, `RecordingOptions`, and metadata repr fields on `Recording` or `RecordingOptions` for persistence/debugging. The public surface mirrors `record(..., train_mode=...)` placement, not `CaptureSpec` initially (`torchlens/fastlog/_record_one_shot.py:18-36`, `torchlens/fastlog/_recorder.py:125-180`, `torchlens/fastlog/options.py:44-193`).
- Keep `CaptureSpec` declarative. Do not put arbitrary callables inside `CaptureSpec` in the first fastlog parity PR. Per-record callable serialization/repr/default inheritance would complicate predicate return semantics and JSON recovery (`torchlens/fastlog/_predicate.py:22-68`, `torchlens/fastlog/storage_disk.py:277-298`).
- Call the postfunc in `_storage_resolver._resolve_storage(...)`, after `safe_copy(...)` and `_apply_payload_transforms(...)`, before returning payloads to the storage backend (`torchlens/fastlog/_storage_resolver.py:57-65`).
- Apply it only after predicate selection. Predicates continue to see raw `RecordContext` metadata (`torchlens/fastlog/_record_context.py:114-123`, `torchlens/capture/output_tensors.py:221-223`).
- Add parallel `ActivationRecord` fields for transformed payloads. Conservative shape: keep existing `ram_payload` / `disk_payload` as raw payloads, add `transformed_ram_payload` / `transformed_disk_payload` defaulting to `None` (`torchlens/fastlog/types.py:112-121`). If no postfunc is set, existing payload fields behave as today.
- For disk fastlog, write transformed blobs with an explicit kind such as `"transformed_activation"` and store separate transformed blob metadata. Update `record_to_json(...)`, `record_from_json(...)`, manifest counts if desired, and recovery checks (`torchlens/fastlog/storage_disk.py:69-98`, `torchlens/fastlog/storage_disk.py:208-242`, `torchlens/fastlog/recover.py:144-167`).
- In train mode / `keep_grad=True`, validate transformed RAM payloads: must be `torch.Tensor`, must not be integer/bool dtype, and must remain graph-connected when the raw selected tensor requires grad. Disk transformed payloads remain detached inspection copies (`torchlens/fastlog/_recorder.py:98-119`, `torchlens/fastlog/_storage_resolver.py:52-65`).
- Reuse the new slow-path `TorchLensPostfuncError` conceptually, but make the message include fastlog context: `ctx.label`, `ctx.kind`, `ctx.func_name`, `ctx.tensor_shape`, `ctx.tensor_dtype`, storage target (`ram`, `disk`, or mirror), and whether `keep_grad=True`. Predicate errors already have separate accumulation/fail-fast semantics; postfunc errors should fail the pass and abort disk storage like storage errors do (`torchlens/fastlog/_state.py:158-180`, `torchlens/fastlog/storage_disk.py:146-156`).
- Ship fastlog activation postfunc as a separate follow-up PR after slow-path refactor. Ship fastlog `gradient_postfunc` only after, or inside, a real fastlog gradient-capture PR.

Not that:

- Do not reuse `LayerPassLog.transformed_activation` in fastlog.
- Do not make predicates see transformed tensors by default.
- Do not run postfuncs during `dry_run()`.
- Do not add `gradient_postfunc` as a no-op or future-reserved parameter.
- Do not route fastlog through `ModelLog` portable `FieldPolicy` just to share persistence semantics.

### Open questions for the architect

1. Should fastlog store both raw and transformed payloads by default when `activation_postfunc` is set, matching the planned slow path, even though sparse fastlog users may be especially memory-sensitive?
2. Should the raw opt-out flag be shared naming with slow path, or fastlog-specific because its fields are `ram_payload` / `disk_payload` rather than `activation`?
3. For disk fastlog, should transformed blobs use manifest kind `"transformed_activation"` or stay `"activation"` with role metadata?
4. Should fastlog expose `activation_postfunc_repr` on `Recording`, or is storing it only in disk `metadata.json` enough?
5. Should callable support apply to input/buffer source events when `include_source_events=True`, or only op events? My recommendation is yes for source events because they use the same `CaptureSpec.save_activation` pipeline.
