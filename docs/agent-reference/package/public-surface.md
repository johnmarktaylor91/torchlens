## Public Surface

`torchlens.__all__` is intentionally small and currently has 116 names. New user-facing
objects should usually live under submodules (`torchlens.io`, `torchlens.options`,
`torchlens.bridge`, `torchlens.errors`, etc.). Interim-phase policy is remove-and-rename,
never deprecation shims (tests/test_deprecation_inventory.py pins the package
deprecation-free).

Unified capture examples:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
paddle_trace = tl.trace(paddle_model, paddle_x, backend="paddle")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("linear"),
    intervene=tl.when(tl.func("linear"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
live_result = relu_trace.run(inputs=x, seed=42)
loaded_result = tl.load("architecture.tlspec").run(inputs=x, seed=42)
```

The unified `inputs=` run surface returns `RunResult(output, trace, report)` without mutating its
source. Live traces use the existing fast refresh projector; loaded runnable traces execute the
resolved sparse DAG with staged, embedded capture, or N1-a state. Analysis-only loads cannot run.
Runnable descriptors are `sparse_recorded_taken_path_v2`: every call carries a REQUIRED explicit
`CallExecutionContext` and the descriptor one `AmbientExecutionContext`, both restored at replay or
refused typed; legacy v1 artifacts load analysis-only with a typed readiness refusal.
Use `tl.save(trace, path, level="runnable", include_weights=True)` to opt into the full capture-time
`state_dict` (named parameters plus persistent buffers). It is a separate `state_dict_v1` blob
family, not part of the tensor-value-free sparse core or a reconstructed model. Used non-persistent
buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family (declared state, not
gated on either include flag; disclosed at save).
Use `include_activations=True` independently to archive exactly the existing capture-time `save=`
selection as `selected_activation_v2` (with physical `InputAttestationFingerprint` eligibility
records). Inspect it through `Trace.archived_activations`; never use those blobs as DAG inputs.
Eligible original-input/capture-equivalent-state runs byte-attest raw saved slots (`attested` or
transactional `numeric_attestation_failed`), while changed-input (logical or physical),
random/non-equivalent-state, and nondeterministic-capture-context runs are `not_applicable`;
`attested` always implies `verified`.
`trace.run(inputs=..., seed=..., on_divergence="raise")` reports readiness, state source,
`verified|diverged|unverifiable` path faithfulness, and numeric attestation in its `RunResult`.
Use `return_diverged` only when a permanently poisoned diagnostic result is intended. Match failures
through `RunnableErrorCode`; the complete frozen taxonomy is in
`docs/reference/runnable_tlspec_contract.md`. r37: overlapping/unprovable distinct-object state
alias topology and zero-tensor-leaf or instance-stateful container outputs refuse at save
(`state_alias_topology_unsupported` / `missing_output_container_contract`); tied live-identity
state stages as one alias-group allocation; persisted context values validate at parse
(`context_field_invalid`); non-global host RNG/entropy/clock touches permanently ceiling replay
(r39: numpy instances via a chained `sys`/`threading.setprofile` classifier + a cheap
model-attribute state digest -- NO process-wide gc scan; unseeded-construction `randbits` entropy;
`datetime`/`localtime` clocks; an externally-held generator on a pre-existing non-hooked thread is
a documented residual, and a benign background thread never ceilings a capture); tensor->host VALUE escapes are caught by dual observer routes (aten
census + a mode-independent method/predicate belt for `_disable_current_modes` regions, plus the
`__repr__`/`__str__` print interception); loaded-sparse and live providers settle through one
finalizer (a live opaque output is `unverifiable`+poisoned, a parse-refused descriptor degrades
every payload family analysis-only, an inexecutable divergent input raises `PathDivergenceError`);
structseq trust keys on the resolution authority, never `__module__`; CUDA state stages lazily at
run preparation behind a no-allocation readiness capability gate.

Provisional semantic I/O examples (review-day names):

```python
classifier_trace = tl.trace(
    model, x, capture=tl.options.CaptureOptions(output_style="classification", output_head="logits")
)
classifier_trace.output_table(top_n=5)

input_trace = tl.trace(
    model,
    raw_text,
    capture=tl.options.CaptureOptions(transform=text_to_tensor, save_raw_input="small"),
)
input_trace.draw(show_input_transform_summary=True)

mds_layers = tl.in_module("block1") | tl.in_module("block2")
image_trace = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
image_trace.model_profile
image_trace.output_table(top_n=5)
tl.repgeom.mds_evolution(image_trace, save=mds_layers, min_n=8)
tl.repgeom.rdm_evolution(image_trace, save=mds_layers)
tl.viz.feature_map_evolution(image_trace, save=mds_layers)
tl.repgeom.scree_evolution(image_trace, save=mds_layers)
image_trace.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
image_trace.draw(node_spec_fn=tl.repgeom.rdm_node_spec(max_stimuli=8))
image_trace.draw(node_spec_fn=tl.viz.feature_map_node_spec())
image_trace.draw(node_spec_fn=tl.repgeom.scree_node_spec())
```

Sprint B annotation/MDS names are provisional until review-day signoff. `Trace.model_profile`
is computed, not persisted. `tl.repgeom.mds_evolution(...)` requires the target batch
activations to have been saved at capture time; use a curated `save=` subset, not exhaustive
`layers_to_save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`record(keep_op=...)` and `record(keep_module=...)` are removed and raise `TypeError`.
`record(save=...)` is the only selective-capture spelling. `capture=CaptureOptions(
layers_to_save=[...])` remains the final-label selection door (the bare flat kwarg is
removed); it is NOT two-pass-only —
`_trace_selector_helpers.py` builds a live single-pass predicate whenever early labels
suffice, falling back to two-pass resolution otherwise. An
unqualified recurrent layer label saves all passes, while `"label:2"` saves only pass 2.

Current 2.x backend surface: torch eager is the stable default; MLX, JAX, tinygrad, Paddle, and
TensorFlow are technical previews behind `BackendSpec`. Paddle M3 is dygraph/eager only, uses
`tl.backends.paddle.GradOptions` for derived-gradient previews, materializes `.tlspec` array
payloads through the Paddle codec, and does not provide true backward capture.
TensorFlow preview targets Keras 3 on TF>=2.16 with
`keras.backend.backend() == "tensorflow"`; its shipped primary path is eager live capture via
`op_callbacks` with real values/control flow/op-level records/module stacks. The graph-only
FuncGraph static path is implemented for compiled/SavedModel entries (opaque regions stay
honestly unverified). Static-label `intervene=` SHIPS for eager entries (two-level writable
layer, fail-closed site reachability), and T1 derived gradients SHIP for eager entries via
`tl.backends.tf.GradOptions` (graph-only captures refuse `grad_options` typed; `intervene=`
cannot combine with `grad_options=`). Deferred: `halt=`/`recipes=`, true backward capture,
and value-dependent predicates.
