# Intervention Explainers

## Replay vs Rerun

`replay()` walks the saved TorchLens DAG and recomputes the affected downstream
cone without calling `model.forward` again. It is fast and useful for
graph-stable interventions where the saved op metadata is enough to recompute
children.

`rerun(model, x)` re-executes the model under the active intervention spec. Use
rerun when the intervention may affect Python control flow, module behavior,
randomness, or any operation that replay cannot faithfully reconstruct.

## Mutate In Place and Fork First

`set`, `attach_hooks`, and `do` mutate the `ModelLog` they are called on. A root
log warns once before this kind of mutation. For branched experiments, fork
first:

```python
candidate = clean_log.fork("candidate")
candidate.attach_hooks(tl.func("relu"), tl.zero_ablate()).replay()
```

Forked logs have independent intervention specs and intervention logs while
sharing immutable captured evidence where safe.

## Direct-Write Dirty

Writing directly to a pass activation, for example `pass_log.activation = value`,
bypasses the intervention recipe. TorchLens marks the owning log
`DIRECT_WRITE_DIRTY` and emits a warning once. A later replay overlays recipe
state; it does not treat the direct write as a portable, auditable intervention.
Use `set` or `attach_hooks` when the edit should be part of the recipe.

## Save Levels

`save_intervention(path, level=...)` writes a `.tlspec/` directory.

| Level | Intended use | Executable? |
| --- | --- | --- |
| `audit` | Record what was attempted, including non-portable pieces. | Not guaranteed. |
| `executable_with_callables` | Re-run in the same code environment when importable callables resolve. | Yes, unless opaque callables block execution. |
| `portable` | Share built-in-helper recipes and tensor sidecars with no opaque local callable dependency. | Yes for supported built-ins. |

Portable saves reject opaque local callables. Use built-in helpers such as
`zero_ablate`, `mean_ablate`, `steer`, or `scale` for reproducible publication.

## `.tlspec/` Directory

A `.tlspec/` directory contains:

- `spec.json`: serialized selector, hook, helper, function-registry, append, and
  target-manifest metadata.
- `manifest.json`: tensor sidecar manifest with hashes and storage metadata.
- `tensors/`: tensor payloads stored separately from JSON.
- `README.md`: a generated human-readable summary.

Load with `tl.load_intervention_spec(path)` and check a fresh capture with
`tl.check_spec_compat(spec, new_log)`.

## Append Constraints

Append mode is for memory-constrained evaluation over compatible chunks:

```python
log.rerun(model, first_chunk)
log.rerun(model, next_chunk, append=True)
```

The appended rerun must match the original graph shape, labels, dtypes, and
non-batch dimensions. Helpers must be batch-independent and append-compatible.
Batch-dependent helpers such as resampling from the current batch are rejected.
Training-mode batch normalization warns because cross-batch statistics can make
chunked evaluation semantically different from one full batch.

## Tier-1 Backward Hook

Backward helpers (`bwd_hook`, `gradient_zero`, `gradient_scale`) are
live/rerun-only in this release. They are not portable `.tlspec/` publication
units and are not replayed post hoc over a saved forward DAG. Use them when the
experiment explicitly re-executes the model and backward pass.

## Attribution-Patching Note

TorchLens currently exposes the ingredients for attribution-style patching:
capture clean/corrupted logs, choose a target site, attach a hook, and compute a
metric over a `Bundle`. Turnkey attribution-patching row/formula helpers are
deferred; keep those formulas in analysis code and record the exact selectors
and metric definitions next to saved specs.

When using TorchLens backward hooks for attribution patching, the recommended
local convention is the TransformerLens-style first-order estimate
`grad * (clean - corrupt)`, where `grad` is taken at the corrupted run and the
activation delta is clean minus corrupt. If an analysis instead reports
`(corrupt - clean) * patched`, treat that as a signed alternative and document
the sign convention next to the metric so higher/lower scores remain clear.
