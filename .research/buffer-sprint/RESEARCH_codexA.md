# TorchLens buffer write/update capture research - Codex A

Date: 2026-06-05
Repo: `/home/jtaylor/projects/torchlens`
Scope followed: research only; no tracked source modified; scratch/output in `/tmp`.

## Executive finding

The recurrent stateful buffer repro exposes two separate problems:

1. The observed `validate_forward_pass` failure is a validation false-negative caused by
   ground-truth tensor aliasing. `validate_forward_pass` stores references to model outputs,
   then restores `state_dict`. If the output is the same Tensor object as a reassigned buffer,
   `load_state_dict` mutates that saved "ground truth" reference back to the initial buffer
   value. The traced forward computes the correct final value, but Phase 0 compares it against
   a stale reset tensor and fails before replay starts.
2. The deeper buffer-write gap is real. TorchLens usually validates green because ordinary
   tensor lineage still captures the math, but buffer writes are not represented as buffer
   versions. Current traces expose only initial/read buffer nodes for most patterns; writes
   are ordinary compute ops, invisible assignments, or fused kernel side effects. There is no
   robust `read vN -> write vN+1 -> read vN+1` buffer state chain.

So the exact recurrent `validate_forward_pass` red result is not caused by loop detection,
dedup, or validation replay failing to thread buffer state. It is caused by the un-cloned
ground-truth output being mutated during validator reset. But the data model/replay promise
still needs a write journal and explicit buffer version nodes.

## File/line anchors

- `torchlens/user_funcs.py:2680-2696`: collects ground-truth output tensors and appends
  `entry[0]` directly, without clone/detach, before `model.load_state_dict(state_dict)`.
- `torchlens/validation/core.py:191-203`: Phase 0 compares captured output against those
  stored ground-truth references and returns the reported failure before parent-edge replay.
- `torchlens/backends/torch/wrappers.py:435-443`: buffers are logged lazily only when a
  tensor arg already has `_tl.address` and no `_tl.label_raw`.
- `torchlens/backends/torch/model_prep.py:724-744`: `_tag_untagged_buffers` only runs at
  module entry and can promote an already-logged tensor to `buffer_source`; same-module
  mid-forward assignments are not intercepted when they happen.
- `torchlens/backends/torch/sources.py:246-262`: buffer source nodes use
  `equivalence_class = f"buffer_{address}"` and read `_tl.buffer_source` if present.
- `torchlens/postprocess/control_flow.py:635-703`: `_fix_buffer_layers` can connect a buffer
  node to `buffer_source`, set identity, dedup same-value records, and assign `buffer_pass`.
  It does not detect writes; it only repairs buffer nodes that already exist.
- `torchlens/backends/torch/ops.py:1539-1567`: `out_versions_by_child` snapshots mutated
  parent values for validation replay, but this is per-parent/per-child replay metadata, not
  a persistent buffer version model.

## Root cause with measured evidence

### Repro model

```python
class StateCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("h", torch.zeros(3))
        self.lin = nn.Linear(3, 3)

    def forward(self, x):
        for _ in range(4):
            self.h = torch.tanh(self.lin(x) + self.h)
        return self.h
```

### The failing public path

Measured with fixed seed/input:

- Public `tl.validate_forward_pass(StateCell(), x)` prints:
  `The 0th output layer, output_1, does not match the ground truth output tensor.`
- Ground-truth output before reset:
  `[0.9113172293, -0.6403640509, 0.5779311061]`
- That ground-truth tensor is exactly `model.h`:
  `same object as m.h True`, `data_ptr same True`.
- After `model.load_state_dict(initial_state)`:
  saved ground-truth reference becomes `[0.0, 0.0, 0.0]`.
- In the traced run, output is the correct final recurrence value. In a manual validation
  sequence that clones ground truth before reset, `trace.validate_forward_pass([gt_clone])`
  returns `True`.

Why reassignment fails but in-place can pass:

- Reassignment path: ground-truth run returns Tensor object A as `h`; `load_state_dict` resets
  A to initial value; traced forward assigns a new Tensor object B to `h`. The validator still
  compares against A, which remains reset.
- In-place `copy_` path returning `self.h`: ground-truth reference and model buffer are the
  same object throughout. `load_state_dict` resets it to zero, but traced forward mutates that
  same object back to final before Phase 0 comparison, so it passes for the wrong aliasing
  reason.
- Returning `self.h + 0` or `self.h.clone()` makes the ground-truth output independent of the
  buffer object. Both validate green.

### Captured trace for recurrent reassignment

With deterministic identity linear weights and `x = [0.2, -0.4, 0.6]`:

- Cloned ground truth:
  `[0.6175236702, -0.8283466101, 0.9029963017]`
- Captured `output_1.out`:
  `[0.6175236702, -0.8283466101, 0.9029963017]`
- `trace.validate_forward_pass([gt_clone], validate_metadata=False)`:
  `True`

Buffer evidence:

- `buffer_layers = ['buffer_1']`
- `buffer_1`:
  `address='h'`, `out=[0.0, 0.0, 0.0]`, `buffer_source=None`,
  `parents=[]`, `children=['add_1_2:1']`, `buffer_pass=1`,
  `equivalence_class='buffer_h'`
- No `h:2`, `h:3`, `h:4`, or `h:5` buffer version node exists.

Recurrent chain evidence:

- `add_1_2:1` parents: `['linear_1_1:1', 'buffer_1']`
- `add_1_2:2` parents: `['linear_1_1:2', 'tanh_1_3:1']`
- `add_1_2:3` parents: `['linear_1_1:3', 'tanh_1_3:2']`
- `add_1_2:4` parents: `['linear_1_1:4', 'tanh_1_3:3']`
- `tanh_1_3:1..4` and `add_1_2:1..4` are loop-grouped with `num_passes=4`.
- `buffer_1` has `num_passes=1`; `op_equivalence_classes` only has
  `{'buffer_h': {'buffer_1:1'}}`.

Conclusion: the recurrence math is captured through normal tensor labels, not through buffer
version nodes. Validation replay can succeed when ground truth is cloned, but the buffer state
model is absent.

## Landscape table

All rows measured with `tl.validate_forward_pass(..., validate_metadata=False)` and trace
dumps from `_run_model_and_save_specified_outs(..., layers_to_save='all',
save_arg_values=True)`.

| Pattern | Public validation | Capture/replay behavior | Buffer data model exposed |
|---|---:|---|---|
| Static read in loop: `y = y + self.b` repeated | True | One initial buffer node feeds every add. | Good read-side model: one `buffer_1`, children are all loop adds, no overwrite. |
| BatchNorm train fused `batch_norm` updates | True | `batch_norm` output validates. `running_mean`/`running_var` final model buffers changed. | Only initial `bn.running_mean` and `bn.running_var` nodes. No write/version nodes. `num_batches_tracked` final becomes `1` but no buffer node was logged in my dump. |
| In-place `self.b.mul_(0.9); return x + self.b` | True | `mul_` is ordinary compute op: buffer initial -> `mul_` -> add. | Only initial buffer node. Final buffer value exists as `mul_` op output, not as `b:2`. |
| In-place `self.b.add_(x); return self.b + 1` | True | `add_` mutator is ordinary compute op. | Only initial buffer node. No buffer version node for post-add state. |
| In-place recurrent `self.h.copy_(tanh(...)); return self.h + 0` | True | Loop math validates, but `copy_` source edge is missing: `copy_` parent is destination/previous state, not tanh source. Saved literal args hide this in validation. | One initial buffer node only. `copy_` ops are compute ops; no `h:2..`. |
| In-place recurrent `copy_`; return `self.h` | True | Passes because traced forward mutates the same ground-truth buffer object back to final before Phase 0. | Same as above; pass is alias-dependent. |
| Mid-forward reassignment loop from brief: `self.b = y+1; y = self.b*2` | True | Math is captured as ordinary add/mul chain. | `buffer_layers=[]` because initial buffer is never read and assignments are not intercepted. Final `b` is invisible as buffer state. |
| Multi-overwrite reassignment in one forward | True | Math captured through ordinary tensor lineage. | No buffer nodes if initial buffer not read; overwrite history invisible. |
| Recurrent reassignment returning expression: `return self.h + 0` | True | Recurrence captured as `buffer_1` only for initial read, then previous `tanh` op feeds next add. | Initial buffer only; no buffer_source chaining or versions. |
| Recurrent reassignment returning buffer object: `return self.h` | False | Phase 0 false-negative: ground-truth ref reset to initial by `load_state_dict`; traced output is correct. Manual cloned-ground-truth validation is True. | Same initial-only buffer model. |
| Two loops overwriting same buffer | True for `return self.h + 0` | Compute ops across both loops group by ordinary op equivalence. | Still only initial `buffer_1`; no evidence for the SPEC's two buffer Layers or global version labels. |
| Shared alias buffers `a` and `b` registered to same Tensor | True | One buffer node feeds both reads. | Address is only first alias (`a`); `named_buffers`/tagging dedups aliases, so `b` is not represented as an address. |

## Reconciliation with round-1 scope note

The round-1 critique was right about the data model: current TorchLens does not capture
buffer writes as buffer versions. The brief's newer measurements were also right that many
mutation models validate green. These are not contradictory:

- Validation green often means ordinary compute tensor lineage is enough to reproduce the
  returned output.
- It does not mean buffer writes are captured as buffer state.
- `out_versions_by_child` helps validation recover mutated parent argument values, but it is
  child-specific replay metadata and cannot answer persistent buffer questions like
  `versions`, `value_at`, `value_after`, alias addresses, final write not read, or fused
  update provenance.

## Why loop detection is not the primary recurrent failure

Loop detection is not what causes the red `validate_forward_pass` result for the recurrent
return-buffer model. Evidence:

- Phase 0 fails before BFS parent replay and before perturbation checks.
- The same trace validates when the ground-truth tensor is cloned before `load_state_dict`.
- The traced recurrent compute ops are loop-grouped correctly (`linear`, `add`, `tanh` each
  `num_passes=4`).

Loop detection is still a future data-model risk. If real buffer version nodes are added with
the current `equivalence_class = buffer_<address>`, all versions at the same address are prone
to collapsing into one equivalence bucket, including versions from separate loops. Replay
correctness must not depend on rolled buffer layers; use global version ids from the write
journal as the authoritative ordering.

## Robust fix proposals

### 1. Immediate validator fix: clone ground-truth outputs before restoring model state

Change `torchlens/user_funcs.py:2694` behavior so `ground_truth_output_tensors` stores safe
detached clones, not live references into the model:

- For tensors: `safe_copy(..., detach_tensor=True)` or equivalent detached clone preserving
  dtype/device enough for comparison.
- For nested outputs: clone after dedupe by structural address.

This directly fixes the measured recurrent `validate_forward_pass` false-negative. It also
removes the in-place return-buffer false-positive alias dependence. It does not solve buffer
write capture.

Cost/risk: low. The only risk is memory for large outputs, but validation already saves all
outs and exists for correctness checking. If memory matters, clone to current comparison
device or CPU behind an option, but correctness should win.

Rank: mandatory first patch.

### 2. Primary write-capture fix: active buffer write journal plus explicit version nodes

Implement a per-capture BufferWriteJournal keyed by persistent buffer identity:

- Discover aliases using `named_buffers(remove_duplicate=False)` and store all addresses per
  storage/object identity.
- Assign a persistent buffer id per address group; keep current version counter and current
  version node label.
- On every buffer read, route the consumer parent edge through the current buffer version
  node, not just through whatever producer label happens to be on the Tensor.
- On every write, append a journal event:
  buffer id, address(es), previous version, new version, writer op label if known,
  pre/post values when saving, mutation kind (`setattr`, `inplace`, `fused`, `register`),
  tensor object id/data ptr/version counter, module stack, and call id.

Write detection sources:

- Patch active `nn.Module.__setattr__` and `register_buffer` during logging. When assigning a
  Tensor to an existing or new buffer slot, propagate address metadata immediately and record
  a write event. If the assigned Tensor has a producer label, that producer is the
  `buffer_source`. This is the direct fix for recurrent reassignment.
- In wrapped torch calls, inspect buffer-tagged tensor args before and after calls. For known
  in-place mutators and any Tensor arg whose `_version` or value changes, record a write event
  with the just-created mutator op as writer. This fixes `mul_`, `add_`, `copy_`, etc.
- For fused kernels, maintain a small registry of functions whose buffer args are mutated
  (`batch_norm` running stats are the obvious first target). Snapshot only those buffer args,
  compare after call, and record version nodes with the fused op as writer.
- Fix `copy_` parent extraction so the source tensor is a real parent edge of `copy_`, not
  only a saved literal arg.

Materialization:

- Create buffer version nodes as plain `Op` with `is_buffer=True`.
- Initial version has no writer parent.
- Written version has parent = writer op and `func=identity`/passthrough semantics.
- Consumers read from the latest version node. The recurrent reassignment graph should become:
  `h:1 -> add1 -> tanh1 -> h:2 -> add2 -> tanh2 -> h:3 ...`
- The journal remains authoritative even for final writes that are not read. If the public
  graph keeps "node only if read", then final unread writes cannot produce full history.
  Prefer revising the spec: journal every write; materialize graph nodes for read-visible
  versions and for saved/history modes. The persistent `Buffer` entity can still expose full
  write history from the journal.

Replay correctness:

- Validation can replay version nodes as identity from writer output and verify read consumers
  use the correct version.
- Intervention replay can propagate changed writer outputs into later buffer versions instead
  of relying on stale module state.
- Rerun/stateful validation can restore both initial and final buffer state from journal
  snapshots.

Cost/risk:

- Moderate to high. This touches wrapper hot path and module attribute setting.
- Keep hot-path cost bounded by checking only tensor args with buffer metadata or known fused
  mutators, not all model buffers after every op.
- Use `torch.Tensor._version` where available as a cheap mutation indicator, falling back to
  value snapshot only when saving or for known fused kernels.

Rank: highest-confidence robust architecture.

### 3. Module-level buffer snapshot/diff fallback

At module entry/exit, snapshot registered buffers and diff them at exit. This catches hidden
updates not visible as torch function args.

Pros:

- Broad safety net for fused/module-internal state mutation.
- Can catch final writes that were never read.

Cons:

- Harder to assign precise writer op. Parent may be only the module call or unknown.
- Expensive for large buffers if done universally.
- Too coarse for recurrent loops inside one forward: exit diff only gives initial/final, not
  per-iteration versions.

Rank: useful fallback, not sufficient as the main design.

### 4. Minimal assignment metadata propagation

Patch only `__setattr__` so reassigned tensors get `_tl.address`, promote existing producer
label to `buffer_source`, and let the next read create a buffer node.

Pros:

- Smallest change that would make recurrent reassignment produce some versions.
- Directly aligns with `_fix_buffer_layers`' existing `buffer_source` support.

Cons:

- Does not handle fused kernels, in-place mutations, final unread writes, alias addresses, or
  `copy_` source edges.
- Same-module assignment followed by no read still has no version node.
- Replay remains dependent on incidental tensor labels.

Rank: acceptable incremental stepping stone only.

## SPEC.md revisions needed

- Keep the persistent `Buffer` entity and plain `Op` version-node direction. That model is
  right.
- Drop the claim that write detection already exists. It does not.
- Replace "node exists only if the buffer is read" or qualify it. Full history and final
  values require a write journal even for unread writes.
- Treat address-global `buffer_version` from the journal as authoritative. Op-label/Layers are
  presentation/grouping, not the source of truth.
- Dual labels are only real after write capture and alias discovery exist. Today they are
  fictional for overwrites.
- Add alias discovery via `named_buffers(remove_duplicate=False)`.
- Do not promise two separate buffer Layers for two loops until loop grouping semantics are
  updated. Version order should be correct even if rolled display grouping is conservative.

## Tests to add

- Validator regression: model returns a reassigned buffer Tensor; public
  `validate_forward_pass` must pass after cloning ground truth.
- Negative alias regression: in-place return-buffer should pass for cloned-ground-truth
  reasons, not because the traced forward mutates the ground-truth reference.
- Recurrent reassignment: assert version chain `h:1..h:N+1`, with `buffer_source` parents
  `tanh:1..N`, and consumers read the expected version.
- In-place `mul_`/`add_`/`copy_`: assert read-vN -> mutator -> write-vN+1; `copy_` source is
  a parent edge.
- BatchNorm train: assert running_mean/running_var and num_batches_tracked writes are journaled
  and final values match module state.
- Final unread write: entity history records it even if no graph consumer reads it.
- Shared alias buffers: one persistent Buffer entity has all addresses; reads through both
  aliases resolve to the same current version.
- Two-loop overwrite: global versions are continuous; display layer grouping is tested only
  to the extent the loop algorithm formally supports it.

## Controversial choices

- I would not make loop detection responsible for replay correctness. Buffer replay should be
  driven by journal version order and explicit parent edges. Rolled layers are display and
  metadata grouping.
- I would revise the "node only if read" rule. It is elegant for graph minimality but conflicts
  with the stated "full overwrite history" and final-state promises. A write journal lets the
  graph stay sparse while the Buffer entity remains honest.
- I would fix `validate_forward_pass` ground-truth cloning immediately, even before the buffer
  sprint. It is a general stateful-output bug, not only a buffer bug.

## RECOMMENDED APPROACH

First, patch `validate_forward_pass` to clone/detach ground-truth output tensors before any
`load_state_dict` call. This is the single highest-confidence fix for the current recurrent
red validation result.

Then implement buffer write capture as an active BufferWriteJournal with explicit version
nodes. Patch `nn.Module.__setattr__`/`register_buffer` during active logging for reassignment,
detect in-place and fused-kernel mutations from wrapped call pre/post buffer snapshots, fix
`copy_` source edges, and route future reads through the latest buffer version node. Use the
journal's address-global version sequence as the source of truth for the persistent `Buffer`
entity; let loop detection group only the presentation layer. This is the robust path that
fixes recurrent replay, fused updates, in-place updates, aliases, final-state history, and the
entity/version data model together.
