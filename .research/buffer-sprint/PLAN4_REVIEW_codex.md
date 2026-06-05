# OpenAI Codex critic findings on Buffer PLAN v4

Repo: `/home/jtaylor/projects/torchlens`
Role: adversarial implementation critic. I did not modify tracked source. Scratch/prototypes:
`/tmp/plan4_codex_probe.py`.

## Empirical bottom line

PLAN v4 is materially better than v3. A per-assignment hook can catch the exact top-level
recurrent repro, and a storage-indexed value snapshot can catch BatchNorm running_mean /
running_var despite no `_version` bump. However, the plan is not bulletproof as written. It
still has four blocking issues:

1. The promised "instance-scoped `__setattr__`" hook is not a real Python mechanism unless it
   changes `type(self)` or becomes a scoped class patch.
2. The fused live-state replay-validation idea is real in pure PyTorch, but it does not fit the
   current TorchLens validation/replay architecture without a new state-isolated live-buffer
   replay path.
3. `.data = new_tensor` buffer writes bypass both proposed hooks and currently make TorchLens log
   the post-write value as the initial buffer.
4. The locked event-history semantics are incompatible with value-change-only fused detection for
   idempotent/native writes.

## Confirmed feasible pieces

### Top-level recurrent reassignment can be caught

Prototype: temporary hook around a root module running:

```python
for _ in range(4):
    self.h = torch.tanh(self.h + x)
```

Measured:

```text
A.instance_attr___setattr___calls []
A.recurrent_event_count 4
A.recurrent_events [
  ('setattr', 'h', [0.0996679961681366, -0.1973753273487091, 0.29131263494491577]),
  ('setattr', 'h', [0.19705621898174286, -0.3777009844779968, 0.5308390259742737]),
  ('setattr', 'h', [0.2886163592338562, -0.5209924578666687, 0.6809262633323669]),
  ('setattr', 'h', [0.3701667785644531, -0.6175236701965332, 0.7534666061401367])
]
A.class_restored RecurrentTop dict
A.no_leak_after_context_event_count 4
```

So v4 closes v3 BLOCKING-A if the hook is implemented with the corrected mechanism below.

### Direct `_buffers[name] = t` is catchable, but only via a proxy

Measured with a `_buffers` proxy:

```text
A.direct_buffers_event_count 1
A.direct_buffers_events [('buffers_dict', 'h', [0.7615941762924194, 0.9640275835990906, 0.9950547814369202])]
```

The proxy must suppress events while normal `nn.Module.__setattr__` is executing, or
`self.h = t` can double-log through both the setattr wrapper and `_buffers.__setitem__`.

### BatchNorm's three writes can be detected

Prototype: storage-indexed registry + wrapped `Tensor.add_` + wrapped `F.batch_norm`.

Measured:

```text
B.bn_versions
  before {'running_mean': 1, 'running_var': 1, 'num_batches_tracked': 1}
  after  {'running_mean': 1, 'running_var': 1, 'num_batches_tracked': 2}
B.bn_event_count 3
B.bn_events [
  ('add_', 'num_batches_tracked', [1], 2),
  ('batch_norm', 'running_mean', [...], 1),
  ('batch_norm', 'running_var', [...], 1)
]
```

This confirms v4 closes the `_version` false-negative as a detector for changed BatchNorm stats.

### Pure live-state fused validation is non-vacuous

Measured standalone:

```text
C.live_state_validation_passes True
C.live_state_validation_rejects_corrupt True
```

Restoring the BN initial state, rerunning `F.batch_norm` with the actual live
`bn.running_mean` / `bn.running_var`, and comparing the post-op live buffer against the captured
version value is a real check. A corrupted captured value fails.

## Blocking issue 1: "Instance-scoped `__setattr__`" is impossible as stated

SEVERITY: blocking

Construction + measured evidence:

Python assignment syntax does not consult an instance attribute named `__setattr__`; special
method lookup goes through the class.

```text
A.instance_attr___setattr___calls []
```

The only prototype that truly patched one instance used a temporary dynamic subclass via
`module.__class__ = Subclass`. That catches assignments, but it changes `type(self)` during
forward. Measured repro:

```text
dynamic_subclass_forward_error RuntimeError type changed to _TLScopedExact
dynamic_subclass_events []
```

A model that checks `type(self) is MyModel` breaks before the hook can record anything. That is
not acceptable for a transparent capture library.

A safer empirical alternative works: patch each prepared module class's `__setattr__` inside the
capture window, gate by a `WeakSet` of prepared instances, and restore in `finally`.

```text
class_patch_m1_out [2.0] type_exact True
class_patch_events [('class_patch', <id>, [2.0])]
```

Fix:

Replace "install on prepared module instances" with a precise mechanism:

- group prepared modules by concrete class;
- patch each class's `__setattr__` in a scoped context;
- gate every event by `self in prepared_weakset`;
- maintain a per-class stack for nested/reentrant traces;
- restore in the outer capture `finally`;
- separately proxy each prepared instance's `_buffers` dict.

Do not use dynamic subclassing unless exact-type changes are explicitly accepted and tested.

## Blocking issue 2: Fused live-state validation does not fit current TorchLens replay

SEVERITY: blocking

Construction + measured evidence:

The live-state idea is real only if replay passes the actual model buffer object into
`batch_norm`. Current validation does not do that. `validation/core.py` builds cloned saved args,
then swaps in cloned parent outs:

```text
_prepare_input_args_for_validating_layer:
  input_args = saved_args/saved_kwargs
  _copy_validation_args(...)
  parent_values = parent_layer.out or out_versions_by_child[child]
  parent_values = parent_values.detach().clone()
```

Measured pure replay shape:

```text
C.clone_replay_live_model_unchanged True
C.clone_replay_clone_changed True
```

Rerunning `F.batch_norm` on cloned running stats mutates the clone, not the model. A fused
version node that tries to read "the live buffer after replay" will read unchanged model state
unless validation is substantially rewritten.

Worse, using actual live buffers naively is stateful. Validation re-executes layers for normal
replay and perturbation checks. Three independent live replays advanced BN state three times:

```text
C.three_independent_live_replays_mean [0.2693171, 0.1836920, 0.1755572, 0.2620685]
C.captured_once_mean [0.0993790, 0.0677830, 0.0647812, 0.0967042]
```

That means fused validation can corrupt the model state during validation unless every check is
isolated by state restore.

Fix:

Specify a new fused-buffer validation path, not a small tweak:

- store enough metadata to map a fused version node to `(module, buffer address, fused op label)`;
- before validating that node, restore the model/buffer state to the captured pre-op state;
- execute the fused op with actual live buffer tensors, not cloned saved args;
- compare the live post-op buffer to the captured version value;
- restore state after the check, including after failures and perturbation attempts;
- do not run perturbation checks through live buffers unless they are separately isolated.

Until this exists, v4 has not really closed confirm-round BLOCKING-B inside TorchLens.

## Blocking issue 3: `.data = new_tensor` bypasses both hooks

SEVERITY: blocking

Construction + measured evidence:

The plan names view/slice/`.data` writes as in scope. Operation-style alias writes are catchable
by storage:

```text
D.alias view      events [('add_', 'b', [1, 1, 1, 1, 1, 1])]
D.alias slice     events [('add_', 'b', [2, 2, 2, 0, 0, 0])]
D.alias data_copy events [('copy_', 'b', [7, 7, 7, 7, 7, 7])]
D.alias setitem   events [('__setitem__', 'b', [7, 7, 7, 0, 0, 0])]
```

But tensor `.data` assignment is neither `nn.Module.__setattr__` nor a wrapped torch op:

```text
D.alias data_setter out [7, 7, 7, 7, 7, 7] events []
```

Current TorchLens behavior is worse than "missed write": it logs the mutated value as the initial
buffer source.

```text
TRACE DataSetter
buffer_layers ['buffer_1'] {'b': 1}
clone_1_1 func clone parents ['input_1'] children []
buffer_1 func none parents [] children ['mul_1_2'] addr b
model buffers [('b', [1.0, 1.0, 1.0, 1.0])]
```

The producing `clone` is orphaned, the write is not represented, and `buffer_1.out` is already
post-write.

Fix:

Either explicitly document `.data` attribute assignment as unsupported and add a hard diagnostic
when end-of-capture reconciliation sees an unjournaled storage/object change, or add a broader
state reconciliation path that can recover/report it. Do not claim `.data` writes are covered by
the two v4 hooks.

## Blocking issue 4: Event-history semantics contradict value-change fused detection

SEVERITY: blocking

PLAN v4 locks "event-history, not value-history": every write event should count, including a
no-op write. The detector text says fused writes are recorded only when pre/post value comparison
changes.

Measured:

```text
momentum 0.0 events [('add_', 'num_batches_tracked', [1])]
buffers running_mean [0,0,0,0], running_var [1,1,1,1]

momentum 1.0 with zero input events [
  ('add_', 'num_batches_tracked', [1]),
  ('batch_norm', 'running_var', [0,0,0,0])
]
running_mean stayed [0,0,0,0] and produced no event
```

For Python in-place no-ops, `_version` can supply an event signal. For native fused mutators like
BatchNorm running stats, `_version` does not bump. If the final value equals the prior value,
value comparison cannot distinguish "no write" from "write of same value." That is value-history,
not event-history.

Fix:

Choose one:

- Change the locked semantics to value-history for version-blind fused/native writes.
- Or, for known fused mutators, emit a write event for each mutable buffer argument whenever the
  mutating op executes in training/update mode, with a `value_changed` flag. Validation can still
  compare live post-state, but it cannot prove that an idempotent native kernel physically wrote
  memory; it can only prove the state transition.

The current plan says both "event-history" and "if changed -> record"; both cannot be true.

## Major issue 5: Storage aliasing works, but overlapping aliases need range semantics

SEVERITY: major

Storage-key detection catches view/slice/`.data.copy_` writes. But a storage key alone is not a
buffer address. Overlapping registered buffers produce multiple changed buffer records:

```text
D.shared_overlap_out [5.0, 0.0]
D.shared_overlap_events [
  ('add_', 'a', [5.0, 5.0]),
  ('add_', 'b', [5.0, 0.0])
]
```

That may be correct if both registered buffers are independent entities whose values changed, but
the plan must define it. Identical aliases, overlapping views, and non-overlapping views sharing
one storage need deterministic address/range handling, or version counts and `Buffer.value_after`
will be surprising.

Fix:

Index buffers by `(storage data_ptr, device, dtype, storage_offset, shape, stride)` and record
overlap ranges. For a mutating arg, compare only candidate buffers whose storage ranges overlap
the mutated arg's range, then emit events in canonical `(module address, buffer name)` order.

## Major issue 6: The cost claim needs hard gates

SEVERITY: major

Prototype cost on `torchvision.models.resnet50(weights=None).train()`, CPU, batch 2,
64x64 input, `torch.set_num_threads(1)`, three timed forwards:

```text
B.resnet50_baseline_3iters_sec 0.1269
B.resnet50_detector_3iters_sec 0.1349
B.resnet50_overhead_pct 6.32
B.resnet50_events_per_iter 159.0
```

This is not catastrophic, but it is only the detector prototype. It does not include full
TorchLens wrapper overhead, version-node construction, storage-range indexing, or live validation.
The plan's "few buffer ops" claim is directionally true for ResNet, but it needs a budget.

Fix:

Add explicit performance gates for:

- resnet50 train/eval;
- transformer with no buffers;
- buffer-heavy synthetic model;
- view/slice alias-heavy model.

Measure detector-only and full `tl.trace` overhead separately.

## Major issue 7: The silent fallback exists in more places than v4 names

SEVERITY: major

PLAN v4 says to convert the fallback at `validation/core.py:~1000` to a raise for buffer-version
parents. There are at least three relevant fallback sites:

- `validation/core.py` parent argument replay around `_prepare_input_args_for_validating_layer`;
- `validation/core.py` argument consistency around `_check_arglocs_correct_for_arg`;
- `intervention/replay.py` `ParentRef` resolution falls back to `parent.out`.

If version-node rewiring moves a child from the old buffer parent to a new buffer-version parent,
all replay/validation paths must share the same "missing child-specific value is an error for
buffer-version parents" rule. Fixing only one line leaves intervention replay and metadata
validation with the old behavior.

Fix:

Centralize parent-value resolution and make the buffer-version-parent miss policy explicit:
ordinary parents may fall back to `out`; buffer-version parents must either have a valid
producer/live-state validation source or raise.

## Concerns and assumptions

- I assumed registered `nn.Module` buffers are the required surface. Plain tensor attrs and list
  attrs remain out of scope in v4 unless restated.
- I did not run the full TorchLens quality gates because the task explicitly prohibited tracked
  source modifications and requested prototype evidence in `/tmp`.
- No dead tracked code was created or removed.

## Knowledge worth carrying forward

- v4's assignment-time capture is the right family of fix for top-level recurrent reassignment,
  but implement it as scoped class patches with membership guards, not instance `__setattr__`
  attributes or dynamic subclasses.
- BatchNorm running_mean/running_var changed values without `_version` bumps in torch
  `2.8.0+cu128`; num_batches_tracked bumped via `add_`.
- Current TorchLens validation uses cloned saved parent values, so live buffer mutation is not
  available unless a new validation mode is added.
- Storage alias detection can catch view/slice/data-copy/setitem writes, but `.data` attribute
  assignment is invisible to both proposed hooks.

VERDICT: NOT BULLETPROOF -- 4 blocking issues
