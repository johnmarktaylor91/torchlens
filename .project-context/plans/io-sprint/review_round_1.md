# Adversarial Review — I/O Sprint Plan Round 1

## Findings

1. **CRITICAL — The primary format still depends on pickling objects that do not round-trip today.**
Location: Fork A lines 13-18, Fork E lines 53-60, IO-S1 lines 145-154, IO-S4 lines 172-180.
Failure scenario: A user logs a normal `nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))` model and calls `torchlens.save()`. `metadata.pkl` still contains `func_applied` callables and other live Python objects. The current tree already throws `PicklingError` on common built-in ops like `linear`; the bundle dies before tensor sidecars matter. The plan claims "Pickle handles nested Python" when the audits already showed the opposite.
Recommendation: Rewrite IO-S1 and IO-S4. Stop treating raw pickle of live `ModelLog` state as the archival contract. Either strip/normalize every non-stable callable/object field before pickling, or replace `metadata.pkl` with a schema-owned state dict. Edit Fork A lines 13-18 and IO-S1 lines 148-153 to name the exact fields that must be scrubbed: `func_applied`, `activation_postfunc`, `_param_ref`, `FuncCallLocation` runtime objects, arbitrary `extra_attributes`, and any nested tensors in captured args/kwargs.

2. **CRITICAL — S5 breaks the capture pipeline by replacing activations with lazy refs too early.**
Location: Fork D lines 46-51, IO-S5 lines 184-190.
Failure scenario: User runs `log_forward_pass(..., save_activations_to=..., keep_activations_in_memory=False)`. `save_tensor_data()` swaps `self.activation` to `LazyActivationRef` during the forward pass. Postprocess immediately runs after forward and still expects real tensors for undecoration, output-node copying, buffer deduplication, equality checks, and validation-related metadata. The log crashes before it is returned.
Recommendation: Edit IO-S5 lines 187-190. Do not evict inside `save_tensor_data()`. Keep real tensors through postprocess, then evict in a postprocess step after all tensor-consuming logic is done. This needs either a new spec or an explicit S5/S6 hook point after `_set_pass_finished`.

3. **CRITICAL — Gradients are promised in the format, but the sprint never designs gradient streaming or backward-failure semantics.**
Location: Bundle layout lines 24-27, public API lines 68-70, IO-S4 lines 172-180, IO-S5 lines 184-190.
Failure scenario: User calls `torchlens.save(..., include_gradients=True)` or expects `<layer>.grad.safetensors` from streaming. Gradients do not exist until backward hooks fire. S5 only instruments `save_tensor_data()` in forward, not `log_tensor_grad()`. If backward fails halfway through, the plan has no temp-dir lifecycle, completeness rule, or manifest state for partial gradient capture.
Recommendation: Edit IO-S4 and IO-S5. Either cut gradient sidecars from Phase 4 entirely or add a separate gradient-write path in `log_tensor_grad()` with explicit bundle states for "forward complete / backward incomplete / gradients absent". The current text is lying.

4. **CRITICAL — `.tlens` as `.tar.gz` is incompatible with the plan’s lazy-load story.**
Location: Fork A line 31, Fork B lines 33-39, public API lines 68-78, IO-S4 lines 177-180.
Failure scenario: User saves `model.tlens` and later calls `torchlens.load("model.tlens", lazy=True)`. A gzipped tarball is not random-access friendly for thousands of tensor sidecars. Either the loader silently extracts the whole archive somewhere, defeating lazy loading, or it materializes everything eagerly, contradicting the default-lazy decision.
Recommendation: Edit Fork A line 31 and IO-S4 lines 177-180. Either drop `.tar.gz` from this sprint, or define hard semantics now: `.tlens` loads eagerly only, or `.tlens` is an uncompressed archive with a real index. Do not keep the current hand-wave.

5. **CRITICAL — The filename scheme is broken on Windows and fragile everywhere else.**
Location: Bundle layout lines 24-27, Fork D line 49, IO-S5 line 188.
Failure scenario: Pass-qualified labels use `:` today (`conv2d_1_1:2`). The plan literally names files `<layer_label_pass>.safetensors`. That is an invalid filename on Windows. Long labels, mixed casing, and any future non-filesystem-safe characters also create collisions or path-length failures.
Recommendation: Edit Fork A lines 20-29 and IO-S5 line 188. Manifest must own a stable opaque blob id per tensor. Filenames cannot be raw layer labels. Put the human-readable label in the manifest, not in the path.

6. **CRITICAL — `activation_postfunc` and other non-tensor activation payloads are ignored, but they blow up this design.**
Location: Fork D lines 49-51, IO-S5 line 188, IO-S6 lines 193-199, Risk R1 lines 226-226.
Failure scenario: User passes `activation_postfunc=lambda t: t.cpu().numpy()` or any custom object transform. Current TorchLens already allows this path to reach postprocess in dangerous ways. The proposed writer assumes `self.activation` is still a tensor after postfunc and tries to sidecar it. `LazyActivationRef.materialize()` also assumes tensor-only storage. Non-tensor activations become unsaveable, unloadable garbage.
Recommendation: Edit Fork D and IO-S5/IO-S6. Either explicitly forbid non-tensor `activation_postfunc` when save/load/streaming is enabled and enforce it early, or add a second serialization lane for non-tensor activation payloads. Right now the spec pretends this problem does not exist.

7. **HIGH — The lazy API is contradictory and will be miserable in real user code.**
Location: LazyActivationRef lines 113-128, LayerPassLog helpers lines 130-135, IO-S6 lines 193-199.
Failure scenario: The plan says `.activation` returns a `LazyActivationRef`, then says accessing `.activation` materializes on read, then says users must call `materialize()`. Pick one. Common workflows like `layer.activation.cpu().numpy()`, `plt.imshow(layer.activation[0])`, `torch.equal(layer.activation, other)`, and loops over activation values either break or materialize implicitly in invisible, expensive ways. Existing validation code also directly assumes `.activation` is a tensor.
Recommendation: Edit lines 121-125, 132-135, and IO-S6 lines 195-197. Make the API honest: `.activation` remains a tensor-or-None field, and lazy loads use a separate accessor like `.activation_ref` plus `.materialize_activation()`. Do not hide I/O behind magic attribute behavior.

8. **HIGH — Default-lazy loaded logs will break existing TorchLens methods that assume tensor activations.**
Location: Fork B lines 33-39, LayerPassLog helpers lines 130-135, IO-S6 lines 193-199.
Failure scenario: User loads a bundle and then calls `validate_saved_activations`, `validate_forward_pass`, string repr helpers, or anything else that reads `layer.activation.shape`, `.dtype`, `.numel()`, `.detach()`, or `torch.equal(...)`. Those call sites currently assume actual tensors. The plan does not audit or gate any of those methods against lazy refs.
Recommendation: Edit Fork B and IO-S6. Either make lazy loading opt-in for Phase 4 or add an explicit compatibility audit spec that enumerates which existing APIs still work on lazy-loaded logs and which raise structured errors. The default cannot change silently while the rest of the object graph still expects eager tensors.

9. **HIGH — The safetensors fallback story destroys the stated guarantees and creates a mixed-format mess the loader cannot reason about.**
Location: Fork A lines 13-18, public API lines 68-77, IO-S4 lines 172-180, Risk R1 lines 226-226, Risk R7 lines 232-232.
Failure scenario: Lean-install user without `safetensors` saves a bundle. Some blobs are written with `torch.save`, others maybe with safetensors, and `LazyActivationRef.materialize()` in S6 is specified only for safetensors. Device mapping semantics now differ by backend. Safety claims differ by backend. Version stability differs by backend. The manifest schema has nowhere to record this cleanly.
Recommendation: Edit IO-S4 and IO-S6. Either make `safetensors` required for the archival bundle path, or explicitly define a per-blob backend field plus load semantics for each backend. "Warn once and fall back" is not a format design.

10. **HIGH — The manifest is too weak to support integrity, lazy loading, or mixed backends.**
Location: Fork A lines 17-18 and 20-29, IO-S4 lines 175-180.
Failure scenario: One sidecar is truncated, another is missing, a third was written with `torch.save`, and the bundle was produced on a different torch build. The manifest only stores counts and versions. There is no per-blob index, checksum, storage backend, shape, dtype, layout, or byte-size entry. The loader cannot verify anything and lazy refs have no authoritative source of truth.
Recommendation: Rewrite IO-S4 line 175. Manifest v1 needs a `tensors` table keyed by stable blob id with label, kind, backend, relative path, shape, dtype, device-at-save, layout, checksum, and maybe storage size. Without that, corruption handling is fiction.

11. **HIGH — Atomicity is overstated and concurrent-writer hazards are unaddressed.**
Location: Fork D lines 51-51, IO-S5 lines 186-190, Risk R3 lines 228-228.
Failure scenario: Two processes save to the same path. Both use `<path>.tmp`. They stomp each other. On NFS or SMB the final rename is not the nice local-rename guarantee the plan assumes. On Windows, directory renames can fail if another process has a handle open. A crash after writing files but before manifest fsync exposes a "successful" rename with a corrupt bundle.
Recommendation: Edit Fork D and IO-S5. Use unique temp dirs per writer, lock or fail fast on existing final path, and stop calling the operation atomic across filesystems. If you keep the rename story, scope it to local same-filesystem best-effort semantics and add fsync requirements.

12. **HIGH — Selective-layer streaming inherits a known-bad fast path, and the plan pretends that is fine.**
Location: Fork D lines 47-49, IO-S5 lines 184-190, Execution Plan lines 243-246.
Failure scenario: User streams only selected layers on AlexNet or ResNet-like models. TorchLens already has tests documenting `save_new_activations()` failures on real models due to identity-op counter misalignment. The plan routes selective saves through exactly that machinery and calls it "single instrumentation point" as if the reliability issue vanished.
Recommendation: Edit Fork D and IO-S5. Either mark selective-layer streaming out of scope for this sprint or add a prerequisite spec to retire the known `save_new_activations()` failures before relying on that path for I/O.

13. **HIGH — The unsupported-tensor story is hand-waving, not design.**
Location: Risk R1 line 226, Risk R7 line 232, IO-S4 lines 174-180, IO-S6 lines 194-199.
Failure scenario: User captures sparse COO, sparse CSR, quantized, meta, nested tensor, tensor subclass, DTensor, FSDP-local shard, complex dtype, `bfloat16`, `float16`, or a non-contiguous tensor. Safetensors only handles dense contiguous tensors. `torch.save` fallback for a DTensor or shard preserves a local implementation detail, not a portable activation. Meta tensors have no data to store. CPU-only reload of CUDA-origin half/bfloat sidecars is not even specified for the mixed-backend case.
Recommendation: Edit Risk R1, IO-S4, and IO-S6. Add an explicit supported-tensor matrix now. For unsupported layouts, either fail fast with a precise error or define a portable coercion policy. "Fallback to torch.save" is not enough for distributed or layout-rich tensors.

14. **HIGH — Backward compatibility for existing `pickle.dump(model_log)` users is not actually planned.**
Location: Fork E lines 53-60, public API lines 68-85, IO-S1 lines 145-154.
Failure scenario: Existing users directly `pickle.dump(model_log)` and later `pickle.load(...)`. After this sprint, the canonical save path becomes a bundle, lazy refs enter the object graph, and `__setstate__` starts default-filling versioned fields. The plan never states whether raw pickle remains supported, deprecated, or explicitly limited. Users get silent behavior drift instead of a migration path.
Recommendation: Edit Fork E and IO-S1. State the compatibility contract for plain pickle in one sentence that is not evasive. Either keep it supported and test it, or deprecate it with warnings and a concrete migration note in S8.

15. **HIGH — Version skew is only recorded, not handled.**
Location: Fork E lines 53-60, IO-S4 line 175, Risk R6 lines 231-231.
Failure scenario: Bundle saved on Python 3.13 / torch 2.x / TorchLens N, then loaded on Python 3.10 / torch 1.x / TorchLens N+1. The manifest stores version strings, but the plan only promises a warning for newer TorchLens. There is no rule for incompatible torch majors, missing dtypes, changed RNG state formats, or Python pickle opcode incompatibility.
Recommendation: Edit Fork E and IO-S4. Define load policy: which version mismatches warn, which hard-fail before opening `metadata.pkl`, and which are unsupported. Without this, the manifest is decorative.

16. **HIGH — The default directory bundle is going to feel absurd for tiny models and punishing for large ones.**
Location: Bundle layout lines 20-31, Fork D lines 49-51, Risk R10 lines 235-235.
Failure scenario: Tiny model with three saved tensors produces a directory tree, manifest, pickle, and activation folder instead of one obvious file. Large model with tens of thousands of passes produces tens of thousands of tiny files and murders inode counts, stat overhead, backup tooling, and network filesystems. The plan chooses the worst operational shape at both ends.
Recommendation: Edit Fork A lines 20-31 and public API lines 68-72. Either make single-file the default for post-hoc save and directory mode opt-in for streaming, or shard activations into chunk files instead of one file per tensor.

17. **HIGH — Corruption handling is almost entirely missing from the specs.**
Location: IO-S4 tests line 179, IO-S7 lines 204-210, Risk R3 lines 228-228.
Failure scenario: `metadata.pkl` is corrupt, one activation file is missing, tar extraction is truncated, manifest counts do not match actual files, checksum mismatches, or disk fills after 70% of files are written. The test plan only mentions manifest mismatch and a monkeypatched disk-full case. That is nowhere near enough for a format that claims symmetric loading.
Recommendation: Expand IO-S4 and IO-S7. Add tests for missing blob files, truncated safetensors, corrupted pickle, stale manifest counts, unexpected extra files, and partial tar archives. Also specify the exact exception types.

18. **HIGH — `frames/` is in the bundle layout but nowhere in the sprint specs.**
Location: Bundle layout lines 28-29, IO-S4 lines 172-180, IO-S8 lines 212-218.
Failure scenario: User enables `save_source_context=True` and saves a bundle. The directory format advertises `frames/`, but no spec defines what goes there, how it maps back to `FuncCallLocation`, what gets redacted, or how it is loaded. This is a format boundary leak disguised as a comment.
Recommendation: Edit Fork A lines 20-29 and add a dedicated subtask under IO-S4 or a new spec. Either define `frames/` concretely or remove it from the layout until it exists.

19. **MEDIUM — The sprint goal says “comprehensive, frictionless I/O for ModelLog and its children,” but module-pass I/O is still missing.**
Location: Goal line 4, Fork C lines 41-44, accessors lines 100-110, IO-S2 lines 156-161.
Failure scenario: User wants one row per module pass, including `forward_args`/`forward_kwargs` or pass-level nesting structure. The plan only covers pass-level layer logs, aggregate layers, modules, params, and buffers. `ModulePassLog` remains a blind spot even though it is a first-class child object.
Recommendation: Edit Fork C and either explicitly exclude `ModulePassLog` from the sprint goal or add a spec for module-pass export. The current wording overpromises.

20. **MEDIUM — Several specs are not actually independent, and the LoC estimates are fantasy.**
Location: Group notes lines 141-141, IO-S1 lines 145-154, IO-S4 lines 172-180, IO-S5 lines 184-191, Execution Plan lines 243-246.
Failure scenario: S1 needs schema decisions from S4. S5 depends on lazy object semantics from S6 and on bundle manifest/index design from S4. S7 depends on real corruption semantics not defined anywhere else. Meanwhile the LoC estimates for fixing pickle hardening, designing a manifest, supporting tar, streaming, fallback backends, and tests are far below the actual blast radius.
Recommendation: Edit Section 3 and Section 5. Merge S4/S6 design-wise, split corruption/integrity into its own spec, and stop calling S1/S2 "parallel-safe" without acknowledging shared constants, accessors, and loader helpers.

21. **MEDIUM — The plan invents `compression=\"zstd\"` without a real backend contract.**
Location: public API lines 68-73.
Failure scenario: User sets `compression="zstd"` expecting tensor compression. Safetensors does not expose a tensor-compression knob in the documented torch API. Either the argument is ignored, silently reinterpreted as tar compression, or causes a backend-specific branch that is nowhere specified.
Recommendation: Edit public API lines 68-73. Remove `compression` from Phase 4 unless there is a concrete implementation and backend contract. Fake knobs are worse than no knobs.

22. **MEDIUM — The risk register mentions `torchlens.cleanup_tmp(path)`, but no spec adds it.**
Location: Risk R3 line 228, IO-S5 lines 184-190, IO-S8 lines 212-218.
Failure scenario: Disk fills mid-stream, user gets told to run `torchlens.cleanup_tmp(path)`, and that function does not exist because no spec owns it. This is sloppy plan leakage.
Recommendation: Edit IO-S5 or IO-S8 to add the utility explicitly, or delete it from Risk R3.

23. **MEDIUM — Re-saving a lazily loaded bundle is undefined and likely pathological.**
Location: public API lines 68-85, IO-S4 lines 172-180, IO-S6 lines 193-199.
Failure scenario: User loads a lazy bundle and immediately calls `model_log.save(new_path)`. Does TorchLens copy existing sidecars, materialize everything, or write new blobs only for materialized activations while silently dropping the rest? The plan does not say. That is not an obscure workflow; it is a standard archive-copy/edit/archive flow.
Recommendation: Edit IO-S4 and IO-S6. Define save-from-lazy semantics now, or reject it explicitly with a hard error in Phase 4.

## Verdict

**RED**

This plan is not ready. The core design still relies on broken pickle assumptions, the streaming eviction point is wrong, the gradient story is incomplete, the archive form contradicts lazy loading, and the manifest is too weak to support integrity or mixed backends. This is not a punch-list cleanup. It needs structural redesign before Phase 4.
