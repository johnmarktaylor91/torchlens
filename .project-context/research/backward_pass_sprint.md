# Backward-Pass First-Class Support -- Sprint Design

> Status: ACTIVE SPRINT. Architecture decisions agreed 2026-04-27.
> Phasing: 3 PRs. Pause + iMessage JMT after each one.

## Refactor 2026-04-27

PR #161 follow-up drops the `BackwardLog` wrapper after implementation review.
Backward data now follows the existing flat ModelLog collection pattern:

- `model_log.grad_fns` is the user-facing `GradFnLog` accessor, parallel to
  `model_log.layers`, `model_log.modules`, `model_log.params`, and
  `model_log.buffers`.
- Storage fields live directly on `ModelLog`: `grad_fn_logs`,
  `grad_fn_order`, `backward_root_grad_fn_id`, `backward_num_passes`,
  `backward_peak_memory_bytes`, `backward_memory_backend`, and
  `has_backward_log`.
- GradFn names are locked as
  `<normalized_grad_fn_type>_back_<type_idx>_<overall_idx>`, with pass labels
  as `<label>:<pass_num>`. Normalization strips a trailing `Backward<digit>`
  suffix from the autograd class name and lowercases the result.
- `model_log.grad_fns[...]` supports ordinal lookup, exact label lookup,
  first substring lookup, pass-qualified lookup, and iteration.
- Cross-links are bidirectional: `LayerLog.corresponding_grad_fn` and
  `GradFnLog.corresponding_layer`.

The original PR-1 plan below is preserved for history, but its `BackwardLog`
wrapper references are superseded by this flat ModelLog design.

## Goal

Promote backward-pass support from a hidden side-effect to a co-equal first-class
subsystem. Symmetric to forward in user mental model: every layer has activation
AND gradient; every forward op has a grad_fn AND there may be intervening
grad_fns; everything you can do with activations you can do with gradients.

Default UX unchanged -- invisible to users who don't care, full discovery for
users who do.

## What already exists (foundation)

- `LayerLog.gradient` / `LayerPassLog.gradient` -- gradient field exists
- `_add_backward_hook()` (capture/tensor_tracking.py) registers
  `tensor.register_hook(...)` on every captured forward output
- `grad_fn_name` captured per layer (string only; no object reference yet)
- `flops_backward`, `macs_backward` properties
- `gradient_ref: LazyActivationRef` scaffolding for disk
- `ParamLog` gradient metadata via the same hook path
- `train_mode=True` keeps saved tensors graph-connected
- Blue/purple gradient arrows in viz (`GRADIENT_ARROW_COLOR=#9197F6`)

## Central tension

**Forward graph topology != backward graph topology.**
- View / reshape / select ops add intervening grad_fns with no forward op
- save_for_backward adds dependencies invisible from forward DAG
- Custom autograd.Function adds arbitrary grad_fns
- Gradient checkpointing adds RecomputeBackward
- In-place ops bump version counters

Design honors the asymmetry: forward and backward live as parallel structures
with named cross-references, NOT one structure trying to do both jobs.

## Locked architectural decisions

1. **Data model: two graphs, loosely linked.** Superseded by Refactor
   2026-04-27 for storage shape. Original plan: ModelLog gets a `backward: BackwardLog`
   companion. BackwardLog has `GradFnLog` nodes (and `GradFnPassLog` for repeated
   calls). LayerLog gets `gradient` (already exists) plus a back-pointer
   `corresponding_grad_fn`. Cross-references navigate; no forced unification.

2. **API entry point: hybrid (1) + (2).**
   - Simple form: `model_log.log_backward(loss)` -- runs backward internally.
   - Advanced form: `with model_log.recording_backward(): loss.backward(...)`.
   - Implicit hook firing stays as today (gradient values per-layer for users
     not opting into the explicit path; backward graph topology requires explicit).

3. **Per-layer gradient selection: smart default + override.**
   `gradients_to_save=None` (default) means same set as `layers_to_save`.
   Independent override accepted.

4. **train_mode auto-enables when backward capture is opted in.** User does not
   need to set both flags.

5. **Validation: yes.** `validate_backward_pass()` parallel to forward's:
   rerun via stock autograd, element-wise compare captured gradients, perturb
   and verify divergence. Same bulletproofing standard.

6. **Visualization: separate backward graph view as opt-in.** Forward default
   unchanged. `model_log.show_backward_graph()` renders the backward DAG in
   its own visual idiom. Status-quo blue arrows on forward stay.

7. **Memory tracking: explicit in PR 1; auto-compute deferred.** PR 1 captures
   peak memory delta during backward. Per-grad_fn auto-computed cost (based on
   saved_for_backward sizes + grad_fn type) -> parking lot.

## Naming (locked)

- `BackwardLog` -- per ModelLog, exposed at `model_log.backward` (superseded:
  dropped in favor of flat `ModelLog` fields and `model_log.grad_fns`)
- `GradFnLog` -- one per grad_fn node
- `GradFnPassLog` -- per-pass entry when grad_fns called multiple times
- `model_log.log_backward(loss)` -- simple entry
- `model_log.recording_backward()` -- context manager entry
- `model_log.show_backward_graph(...)` -- viz entry
- `validate_backward_pass(...)` -- validation entry
- `gradients_to_save` -- per-layer selection arg
- `gradient_postfunc` -- analog of activation_postfunc
- LayerLog: `gradient` (exists) + `corresponding_grad_fn` (new)

## Phasing

### PR 1 -- Data model + capture (the core)

In scope:
- BackwardLog, GradFnLog, GradFnPassLog data classes (superseded:
  `GradFnLog` and `GradFnPassLog` remain; `BackwardLog` removed)
- Walk grad_fn DAG from loss to enumerate all reachable grad_fns
- Register grad_fn hooks for runtime capture (timing, gradient values)
- Link forward LayerLog to corresponding GradFnLog by object identity
- Identify intervening grad_fns (no forward correspondence) and flag them
- `model_log.log_backward(loss)` and `recording_backward()` context manager
- Per-layer gradient selection (`gradients_to_save` arg, default = same as `layers_to_save`)
- Auto-train_mode when backward capture is opted in
- `gradient_postfunc` per-layer
- ModuleLog gradient aggregation
- Inputs / leaf-tensor gradient access
- Custom autograd.Function: captured as GradFnLog with `is_custom: bool`
- Implicit hook path preserved (existing behavior unchanged)
- Multiple losses via context manager
- Higher-order gradients (`create_graph=True`) basic support
- Peak memory tracking during backward (cumulative for the sweep) on
  flat `ModelLog.backward_peak_memory_bytes`
- `validate_backward_pass()` in `validation/`
- New tests in `tests/test_backward.py` (or sibling files), all marked `@pytest.mark.smoke` for the smoke set

NOT in PR 1:
- Backward visualization (PR 2)
- Disk streaming for gradients (PR 3)
- Tutorial notebook (PR 3)
- Per-grad_fn auto-computed memory cost (parking lot)
- Fastlog gradient support (parking lot)

### PR 2 -- Backward visualization

- `model_log.show_backward_graph(...)` renders flat `ModelLog` backward fields in graphviz
- Visual idiom: blue/purple palette signaling backward; cross-references to
  forward where applicable
- Stays an opt-in view. `show_model_graph()` default = forward only, blue
  arrows overlay unchanged
- Tests + smoke

### PR 3 -- Disk streaming + tutorial

- Gradient streaming to disk parallel to activations (validate `gradient_ref`
  end-to-end if scaffolding incomplete)
- `notebooks/backward_tutorial.ipynb` walking through patterns:
  per-layer gradient access, backward graph topology, custom autograd, multi-loss

## Parking lot (NOT in this sprint)

- **Per-grad_fn auto-computed memory cost.** Once GradFnLog has saved_for_backward
  refs (PR 1), memory cost per grad_fn = sum of saved tensor sizes + output gradient
  shapes; type-specific contributions from grad_fn class. Add when asked.
- **Fastlog gradient support.** Predicate-selected gradient capture in fastlog.
  Slow path needs to settle first.
- **Backward-pass noise filter** (analog of `vis_buffer_layers="meaningful"` for
  filtering AsStridedBackward / SelectBackward / etc.). Wait until rendering
  shows whether it's actually noisy.
- **Higher-order grad full support** -- basic in PR 1; exhaustive coverage of
  create_graph=True scenarios deferred.

## Open during PR 1 (let codex make low-risk default calls)

- Exact internal helper layout (which file gets the walk, which gets the hook
  registration, etc.)
- Field order on dataclass FIELD_ORDER tuples (mirror existing patterns)
- Postprocess pipeline integration point (likely a new step similar to forward
  finalization)
