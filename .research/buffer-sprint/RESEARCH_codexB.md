# TorchLens buffer write/update capture research - Codex agent B

Repo: `/home/jtaylor/projects/torchlens`
Date: 2026-06-05
Scope: research/design only. I did not modify tracked source.

## Executive findings

1. The observed public `validate_forward_pass` failure for the recurrent buffer repro is
   primarily a validation ground-truth aliasing bug, not a failure of the traced recurrent
   compute graph.
   - In `torchlens/user_funcs.py:2680-2695`, ground-truth output tensors are appended by
     reference.
   - In `torchlens/user_funcs.py:2696`, `model.load_state_dict(state_dict)` restores the
     model before tracing.
   - For `return self.h`, the returned ground-truth tensor is the live registered buffer
     object. `load_state_dict` copies the initial state back into that same object, so the
     stored "ground truth" is mutated from the final recurrent state back to zeros before
     comparison.
   - Returning `self.h.clone()` makes public validation pass.

2. Current capture/replay is better than the round-1 scope note claimed for validation,
   because replay has a value-variation patch:
   - `torchlens/backends/torch/ops.py:1542-1566` records
     `out_versions_by_child` when a parent tensor's value at child-call time differs from
     the parent's saved `out`.
   - `torchlens/validation/core.py:987-1001` uses that child-specific parent value during
     validation replay.
   - This explains why BatchNorm train, in-place `mul_`, and some post-mutation reads can
     validate even though explicit buffer version nodes are missing.

3. Current capture still does not provide a robust buffer write/version data model.
   Validation can pass via ordinary op parent edges and `out_versions_by_child`, but
   persistent `Buffer` entity + explicit version chain does not fall out.
   - Reassignment loop: validation passes, but `trace.buffer_layers == []`; writes are not
     exposed as buffers.
   - Recurrent read-modify-write: only the initial `h` buffer is a buffer node; later states
     are ordinary `tanh` ops.
   - In-place `mul_`: the mutator op is captured, but no `b:v2` buffer node is created.
   - `copy_`: graph dependency is incomplete; the source tensor of `copy_` was not recorded
     as a parent in my repro.
   - BatchNorm fused running-stat updates are not explicit write-version nodes; a later
     read can validate through `out_versions_by_child`, not through a buffer version chain.

## Root cause of the failing recurrent repro

Repro shape:

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

Public validation result:

```text
The 0th output layer, output_1, does not match the ground truth output tensor.
recurrent-state-buffer: validate_forward_pass -> False
```

Measured aliasing proof:

```text
ReturnBuffer
gt before restore tensor([ 0.9025, -0.9474,  0.6133], grad_fn=<TanhBackward0>)
gt after restore tensor([0., 0., 0.], grad_fn=<TanhBackward0>)
model.h after restore tensor([0., 0., 0.], grad_fn=<TanhBackward0>)
public validate False

ReturnClone
gt before restore tensor([ 0.9082, -0.9110,  0.9138], grad_fn=<CloneBackward0>)
gt after restore tensor([ 0.9082, -0.9110,  0.9138], grad_fn=<CloneBackward0>)
model.h after restore tensor([0., 0., 0.], grad_fn=<TanhBackward0>)
public validate True
```

When I manually cloned the ground-truth output before restoring model state, the exact
recurrent graph validated:

```text
gt tensor([-0.8052, -0.9500,  0.4712])
model.h after load tensor([0., 0., 0.], grad_fn=<TanhBackward0>)
trace output tensor([-0.8052, -0.9500,  0.4712], grad_fn=<CloneBackward0>)
equal? True
trace.validate True
```

The traced recurrent graph is computationally correct because after the first iteration the
assigned tensor already has the producing op's label. Subsequent reads of `self.h` become
ordinary parents from the previous `tanh`, not buffer-version parents:

```text
buffer_1 -> add_1_2:1
tanh_1_3:1 -> add_1_2:2
tanh_1_3:2 -> add_1_2:3
tanh_1_3:3 -> add_1_2:4
tanh_1_3:4 -> output_1
```

So the recurrent failure is not loop detection collapsing versions, not stale saved trace
output, and not replay failing to thread the recurrent value. It is the validator mutating
the ground-truth output reference. However, this computationally correct graph still lacks
the explicit `h:1 -> h:2 -> ...` buffer version chain required by the buffer entity design.

Immediate validator fix: clone/detach every ground-truth output tensor before
`model.load_state_dict(state_dict)` in `torchlens/user_funcs.py:2694`, preserving device/dtype
and structural de-dup semantics. This fix is necessary regardless of the broader buffer
write-capture sprint.

## Current mechanisms

### Read-side buffer capture

Wrappers lazily log buffer source tensors on first use:

- `torchlens/backends/torch/wrappers.py:438-443`: if an argument tensor has a buffer address
  and no tensor label, `log_source_tensor(trace, t, "buffer", address)` is called.
- `torchlens/backends/torch/sources.py:248-262`: buffer source nodes are internal sources
  with `equivalence_class = f"buffer_{extra_addr}"`.

### Buffer postprocess

`torchlens/postprocess/control_flow.py:635-704`:

- Connects a buffer to `buffer_source` if set.
- Makes written buffers identity passthroughs.
- Deduplicates by `modules + buffer_source + address` plus tensor equality.
- Assigns `buffer_pass` sequentially per address.

This only works when some earlier mechanism sets `TensorMeta.buffer_source`.

### Dynamic buffer tagging

`torchlens/backends/torch/model_prep.py:724-744` tags untagged buffers during module entry.
If a buffer already has a raw tensor label, it promotes that label into
`buffer_source` and clears the label. This catches some dynamic cases at module boundaries,
but not mid-forward reassignment inside the same active module call.

### In-place replay patch

`torchlens/backends/torch/wrappers.py:490-544` detects same-object in-place returns, logs a
safe copy, and propagates the new label back to the original tensor. This is good for
ordinary op chaining, but it is not a buffer write journal.

`out_versions_by_child` is a child-specific replay repair. It can preserve validation
correctness for mutated parent values, but it is keyed by child op and does not model
buffer writes as first-class state transitions.

## Empirical landscape

All cases were run with `torchlens.validation.validate_forward_pass(..., validate_metadata=False)`
and `tl.trace(..., layers_to_save="all", save_arg_values=True)`.

| Pattern | validate_forward_pass | Captured compute correctness | Buffer data model exposed today |
| --- | ---: | --- | --- |
| BatchNorm train, returns BN output | Pass | Yes | `running_mean` and `running_var` initial read nodes only; no write versions for fused updates. |
| BatchNorm train, returns `running_mean` after update | Pass | Yes, via `out_versions_by_child` for `output_1` | Still one initial buffer node; post-update value is child-specific variation, not `running_mean:2`. |
| In-place `self.b.mul_(0.9); return x + self.b` | Pass | Yes | Initial buffer node plus `mul_` compute op; no `b:2` version node. |
| `self.b.copy_(x + 2); return self.b * 2` | Pass | Partly, but graph is incomplete | Initial buffer node plus `copy_` compute op. Source `x + 2` had no child edge to `copy_` in my trace. No `b:2`. |
| Reassignment loop `self.b = y+1; y = self.b*2` | Pass | Yes via normal producer labels | No buffer nodes at all when initial `b` is unused. All writes are ordinary `add` ops. |
| Recurrent `self.h = tanh(lin(x)+self.h); return self.h` | Public validator fails due ground-truth alias; cloned-ground-truth trace validates | Yes | One initial `h` buffer node only. Later states are ordinary `tanh` ops. |
| Recurrent, returns `self.h.clone()` | Pass | Yes | Same as above: only initial `h` buffer node. |
| Multi-overwrite read-last `self.b=x+1; self.b=x+2; return self.b*3` | Pass | Yes | No buffer write history; first overwrite is dead in graph, second is ordinary `add`. |
| Two loops with reassignment | Pass | Yes | No buffer write history, so no way to split per-loop buffer Layers. |
| Shared alias static buffers `a` and `b` same tensor | Pass | Yes | One buffer node with address `a`; alias `b` is not represented as an address. |

Important detail from BatchNorm return-running-mean:

```text
model running_mean after trace tensor([0.0039, 0.0213, 0.0088])
buffer out tensor([0., 0., 0.])
buffer out_versions {'output_1': tensor([0.0039, 0.0213, 0.0088])}
output out tensor([0.0039, 0.0213, 0.0088]) parents ['buffer_1']
```

This validates because child-specific value variation is used during replay. It does not
mean the graph has a post-BatchNorm `running_mean:2` node.

## Design goals for a robust fix

The robust target should satisfy both validation and data-model requirements:

1. Preserve the immediate validator guarantee by cloning ground-truth outputs before state
   restoration.
2. Capture every buffer write event in execution order.
3. Represent each persistent buffer address as one entity with a flat version chain:
   `address:1`, `address:2`, ...
4. Use version nodes as the authoritative replay parents for reads after writes.
5. Handle all write mechanisms:
   - `nn.Module.__setattr__` reassignment.
   - `register_buffer` during forward.
   - Tensor in-place methods such as `mul_`, `add_`, `copy_`, `zero_`, `__setitem__`.
   - Fused/native ops mutating buffer arguments, especially `batch_norm`.
   - Shared aliases and plain tensor attributes currently treated as buffers.
6. Keep wrapper hot-path overhead bounded.
7. Avoid using loop-detection labels as the source of truth for buffer version identity.

## Ranked approaches

### Rank 1 - Hybrid write journal + explicit buffer version nodes (recommended)

Build a capture-time buffer write journal and synthesize version nodes from it. This is the
only approach that cleanly covers all mutation patterns.

Mechanism:

1. At capture start, build a `BufferRegistry`.
   - Enumerate `model.named_buffers(remove_duplicate=False)` to preserve aliases.
   - Include existing plain tensor attributes if TorchLens continues treating them as
     buffers.
   - Track `address`, all aliases, owning module address, live tensor object id, storage
     pointer where available, current PyTorch `_version`, current TorchLens version number,
     and current version-node label.

2. Use an active-capture patch of `nn.Module.__setattr__`.
   - If the assigned name is a registered buffer name, or a Tensor is assigned to a
     buffer-tracked address, record a write event after PyTorch's original `__setattr__`
     succeeds.
   - If the assigned tensor already has a producer label, writer parent is that op.
   - If it is a literal/external tensor with no producer label, create an internal-source
     write parent or a sourceless version with provenance marked external.
   - Update registry mapping from address to the new tensor object and set buffer metadata
     on that tensor without destroying its producer label.

3. Patch `nn.Module.register_buffer` under the same active-capture gate.
   - New dynamic buffers get address metadata and an initial version event.
   - Re-registering/replacing an existing buffer records a write event.

4. Extend torch wrapper mutation handling.
   - Before each wrapped call, identify buffer-tracked tensor args.
   - Record cheap pre-call state for those args only: object id, `_version`, storage pointer,
     maybe shape/dtype.
   - After the call and after the compute op label exists, detect changed buffer args.
   - For true in-place returns (`mul_`, `add_`, `zero_`, `__setitem__`), create a
     write-version node whose parent is the mutator op.
   - For `copy_`, fix parent detection so the source tensor is a parent of `copy_`, then
     write `b:vN+1` from the `copy_` op.
   - For fused/native ops such as `batch_norm`, if buffer args changed across the call,
     create write-version nodes from the fused op.

5. Route future reads through current version nodes.
   - Current wrappers only log a buffer source if the tensor has address and no raw label
     (`wrappers.py:438-443`). That is insufficient after reassignment because the tensor has
     a producer label.
   - Add separate metadata for "this tensor is the current value of buffer address X at
     version V". Parent resolution should use the current buffer version node for future
     reads. If the same tensor is also held in a local variable, routing via an identity
     version node is semantically acceptable and makes state explicit.

6. Version nodes are plain `Op`s with `is_buffer=True`.
   - Initial version: source node, no writer parent.
   - Written version: identity passthrough node, one parent writer op.
   - Reads are children of the version node.
   - The persistent `Buffer` entity indexes these nodes by address-global version.

How it fixes the recurrent case:

For `self.h = tanh(lin(x) + self.h)`:

```text
h:1 -> add iter1 -> tanh iter1 -> h:2 -> add iter2 -> tanh iter2 -> h:3 -> ...
```

The validator alias fix makes public validation compare against the real ground truth.
The write journal then makes the buffer state chain explicit for replay, visualization, and
the Buffer entity.

Replay correctness:

- For each child op, validation no longer needs to guess with `out_versions_by_child`; the
  child parent is the exact version node it read.
- `out_versions_by_child` should remain as a defensive mechanism for non-buffer in-place
  aliasing, but buffer replay should be driven by version nodes.

Hot-path cost:

- Attribute interception is only active while `_logging_enabled` is true.
- Wrapper pre/post checks are limited to tensor args that are buffer-tracked.
- Use `_version`/identity checks first; clone full buffer values only when saving that
  version is required or a mutation is detected and equality confirmation is necessary.
- Optional module-entry/module-exit full snapshots can be debug/audit fallback, not the
  primary path.

Loop detection interaction:

- Do not make loop detection assign version identity. `buffer_version` is a registry/journal
  sequence number.
- Let loop detection group buffer version nodes for display only.
- Current `sources.py:256-258` uses `equivalence_class = f"buffer_{address}"`, which would
  collapse versions across unrelated loops. Revise buffer version equivalence to include
  write/read site context, e.g. address + writer op equivalence + surrounding module/call
  context. The persistent entity still keeps a flat version list.
- Validate version-chain invariants before/after loop detection: address versions are
  strictly increasing in raw execution order; every read uses the latest preceding version;
  every written version has exactly one writer parent unless externally sourced.

Migration:

- Phase 0: clone ground-truth tensors in `validate_forward_pass`.
- Phase 1: add journal data structures and registry, but keep existing Buffer(Op) API.
- Phase 2: create version nodes from journal while preserving old accessors.
- Phase 3: introduce persistent `Buffer` entity and make `trace.buffers[address].versions`
  point to version-node Ops.
- Phase 4: remove `Buffer(Op)` subclass and update docs/tests.

Risk:

- Medium/high implementation scope, but best correctness profile.
- Needs careful metadata separation: tensor producer label and current-buffer-version label
  must not clobber each other.

### Rank 2 - Attribute interception + targeted mutator detection, no general fused diff

This covers reassignment and common in-place methods but punts on fused/native kernels.

Pros:

- Lower scope than Rank 1.
- Fixes the recurrent explicit reassignment/version-chain case cleanly.
- Handles user-written RNN-style state updates.

Cons:

- BatchNorm and other native ops that mutate buffer args remain implicit.
- Still cannot honestly claim all buffer writes are captured.
- The Buffer entity would be correct only for Python-level assignment and known in-place
  methods.

Verdict: acceptable as an intermediate milestone, not as the universal solution requested.

### Rank 3 - Post-op snapshot/diff of all buffers

After each bottom-level op, compare every registered buffer against a previous snapshot and
synthesize writes when values differ.

Pros:

- Captures fused kernels and arbitrary side effects.
- Does not require as much special-casing of mutator names.

Cons:

- Hot-path cost is poor for models with large buffers or many ops.
- Attribution is ambiguous: a diff after an op means the op likely wrote the buffer, but
  nested wrapped calls and Python assignments can blur causality.
- It misses multiple writes between two checks unless checks happen extremely often.
- Full snapshots of all buffers per op are not viable.

Verdict: useful as an optional audit fallback or debug assertion, not as the primary design.

### Rank 4 - Module-entry/module-exit buffer snapshot/diff

Snapshot buffers at module entry and diff at module exit.

Pros:

- Cheap relative to per-op full diff.
- Good for final state summaries.

Cons:

- Cannot capture intermediate overwrites inside loops.
- Cannot build read-vN/write-vN+1 chaining.
- Cannot fix recurrent version modeling.

Verdict: insufficient for this task. Useful only as a coarse safety net to detect missed
writes.

### Rank 5 - Postprocess-only reconstruction from existing op graph

Infer buffer versions after capture by looking at assigned op chains, `out_versions_by_child`,
and repeated reads.

Pros:

- Minimal capture changes.

Cons:

- Cannot see Python `__setattr__` writes when the initial buffer is unused.
- Cannot reliably distinguish local tensor reuse from buffer state.
- Cannot recover dead writes.
- Cannot fix missing `copy_` source edges.

Verdict: not robust.

## Data model revisions to SPEC.md

Keep:

- Persistent `Buffer` entity is the right noun.
- Buffer graph nodes should be plain `Op`s with `is_buffer=True`.
- Version nodes as identity passthroughs are the right graph representation.
- Flat address-global version list should be authoritative.

Revise:

- "Detection is largely built" is false. Replay has partial mutation patches, not full write
  capture.
- Do not rely on current `buffer_source` promotion alone; it misses mid-forward reassignment.
- Do not rely on loop-detection pass labels for global buffer version numbers.
- "Node exists only if read" conflicts with "full overwrite history" for dead writes. For a
  complete persistent Buffer history, write events must produce versions even if never read,
  at least when buffer history capture is enabled. If visualization wants to hide unread
  versions, that should be a view policy, not a capture omission.
- Shared aliases require `named_buffers(remove_duplicate=False)` and alias-aware entity
  fields. Current static shared alias captured only address `a` in my test, not `b`.

## Validation and invariant plan

Add tests in two groups.

Immediate validator tests:

- Recurrent buffer returns live `self.h`: public `validate_forward_pass` should pass after
  cloning ground-truth outputs.
- Same model returning `self.h.clone()` should continue to pass.
- BatchNorm returning `running_mean` should compare against the actual post-update value, not
  a state-restored alias.

Write-capture tests:

- BatchNorm train: `running_mean` and `running_var` get written versions with writer parent
  `batch_norm`.
- `mul_`, `add_`, `zero_`, `copy_`, `__setitem__`: read vN, mutator op, write vN+1.
- `copy_` source tensor is a parent of `copy_`.
- Reassignment loop: one persistent `Buffer` entity, versions for each assignment, reads use
  preceding version.
- Recurrent buffer: version chain exactly threads iterations and validates.
- Multi-overwrite with dead write: entity history includes dead write, graph/view can mark it
  no-output-descendant.
- Two loops: one entity, versions globally ordered; display grouping does not conflate global
  version identity.
- Shared alias: one entity with all addresses.
- Integer buffers such as `num_batches_tracked`: dtype-preserving versions and equality.

Invariant checks:

- For each `Buffer` entity, `versions[i].buffer_version == i + 1`.
- Version raw indices are strictly increasing.
- A read child of `address:k` occurs after version `k` and before version `k+1`, unless the
  child is the writer producing `k+1`.
- Written version nodes have one writer parent; initial/external versions have explicit
  provenance.
- Loop detection may alter layer labels/pass indices but must not alter address-global
  version order or read/write edges.

## Controversial choices

1. Clone ground-truth outputs immediately. This is not optional. It fixes the measured
   recurrent failure and protects all models that return live buffers, parameters, or mutated
   input aliases.

2. Keep `out_versions_by_child` but do not build the buffer model on it. It is valuable for
   validation of generic in-place aliasing, but it is child-keyed replay metadata, not a
   persistent state model.

3. Create version nodes for dead writes if the promise is full buffer update capture. A model
   can update a buffer for use in the next forward pass without reading it again in the same
   forward; omitting that write makes `final_value` and `num_overwrites` lie.

4. Separate tensor producer identity from buffer-state identity. A tensor can be both "the
   result of `tanh_3`" and "the current value of buffer `h:4`". Overloading one
   `label_raw` field for both is the source of many current limitations.

## Concerns and risks

- Patching `nn.Module.__setattr__` must be toggle-gated and exception-safe. It should not
  alter user semantics outside active capture.
- Tensor aliasing can produce multiple addresses for one storage. The registry needs
  address-level and storage/object-level indexes.
- PyTorch `_version` is a practical fast signal but should be treated as an optimization, not
  the only source of truth.
- Fused ops with multiple mutated buffers need write ordering. The op-level write events can
  share the same writer raw index but must still receive deterministic version event order.
- Backward-ready traces must not detach user tensors just to save buffer versions. Save policy
  should mirror existing activation saving controls.

## Knowledge worth keeping

- The current recurrent compute trace is already correct when the ground truth is cloned.
- The current public validation failure is caused by aliasing of the returned live buffer
  across `load_state_dict`.
- BatchNorm post-update reads validate because `out_versions_by_child` stores the post-update
  value for `output_1`.
- Reassignment writes are invisible to `trace.buffer_layers` unless a module-entry retagging
  opportunity occurs later.
- In-place ops are represented as compute ops and can propagate labels, but no buffer version
  node is created.
- `copy_` currently validated in the tested case despite a missing source edge, because saved
  raw args can replay the function. That is a graph correctness gap.

## RECOMMENDED APPROACH

First, fix the measured validation bug by cloning/detaching ground-truth output tensors in
`validate_forward_pass` before restoring `state_dict`.

Then implement Rank 1: a hybrid capture-time buffer write journal with explicit
address-global version nodes. Intercept `nn.Module.__setattr__` and `register_buffer` for
Python-level writes, extend wrapper mutation detection for buffer-tagged in-place methods,
add targeted pre/post mutation checks for buffer args to catch fused kernels like
`batch_norm`, and route all subsequent buffer reads through the current version node.

This is the single highest-confidence robust design because it fixes the recurrent case,
preserves replay correctness, keeps hot-path cost bounded to buffer-touching operations,
handles fused/in-place/reassignment mutation families, and makes the persistent `Buffer`
entity + flat version chain fall out naturally instead of being inferred after the fact.
