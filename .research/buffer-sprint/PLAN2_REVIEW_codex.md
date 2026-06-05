# Codex critic findings on PLAN_PHASE2 buffer write capture

Repo: `/home/jtaylor/projects/torchlens`
Role: adversarial critic. No tracked source modified. Scratch probes used under `/tmp`.

## Executive verdict

The plan is not bulletproof. Its core detector is under-specified in exactly the places where
PyTorch makes mutation detection hostile: fused kernels that mutate without `_version` bumps,
alias/view/data writes that do not carry TorchLens buffer metadata, non-`__setattr__`
rebinding paths, and graph/replay semantics that already have a child-specific mutation repair
mechanism. The plan can be made viable, but only if it is reframed as a write journal keyed by
storage/object alias identity plus an explicit replay migration, not as a narrow
buffer-tagged-arg `_version` diff bolted onto the hot path.

## Blocking issue 1: `_version` is not a reliable write signal for fused BatchNorm writes

SEVERITY: blocking

Concrete construction/evidence:

Probe: `/tmp/probe_version.py`

```text
bn before 1 1 1 0
bn after  1 1 2 1
bn changed mean tensor([-0.0909,  0.0266, -0.0062,  0.0132]) var tensor([1.0135, 0.9811, 1.0249, 1.0581])
```

For `nn.BatchNorm1d(...).train()`, `running_mean` and `running_var` values changed but their
autograd `_version` counters did not change at all. Only `num_batches_tracked` bumped from
version 1 to 2. This directly breaks the plan's headline "post-op autograd `_version`-diff
detector" for one of its named target cases: fused BatchNorm updating multiple buffers.

The plan says "`_version` is an optimization signal, NOT sole source of truth" but never
specifies which calls require mandatory value snapshots. If implementation starts with
`_version` and only value-confirms after a version bump, BatchNorm running stats are missed.
If implementation value-snapshots every buffer-tagged arg on every op, the cost model changes
substantially and is no longer the advertised cheap `_version` diff.

Proposed fix:

Build a mutation registry, not a generic `_version` detector:

- Mandatory pre/post value snapshots for known version-blind fused mutators, starting with
  `batch_norm` running stats.
- `_version` diff only as a cheap positive signal for ordinary tensor in-place ops.
- Tests must assert `running_mean`, `running_var`, and `num_batches_tracked` each produce
  deterministic write events from one BatchNorm call.

## Blocking issue 2: buffer-tagged-arg scanning misses writes through views, slices, detach aliases, and `.data`

SEVERITY: blocking

Concrete construction/evidence:

Probe: `/tmp/probe_alias_metadata.py`

After calling TorchLens `_tl.set_buffer_address(b, "b")` and
`_tl.set_tensor_label(b, "buffer_1_raw")`, derived aliases did not inherit either address or
label:

```text
view addr None label None is_view True version 0 storage True
  add_ mutated base True base version 0 -> 1 view version 0 -> 1
reshape addr None label None is_view True version 0 storage True
  add_ mutated base True base version 0 -> 1 view version 0 -> 1
transpose addr None label None is_view True version 0 storage True
  add_ mutated base True base version 0 -> 1 view version 0 -> 1
slice addr None label None is_view True version 0 storage True
  add_ mutated base True base version 0 -> 1 view version 0 -> 1
detach addr None label None is_view False version 0 storage True
  add_ mutated base True base version 0 -> 1 view version 0 -> 1
data addr None label None is_view False version 0 storage True
  add_ mutated base True base version 0 -> 0 view version 0 -> 1
```

So the proposed detector condition "any argument tensor that is buffer-tagged
(`_tl.address` present)" is not enough. These aliases mutate the underlying buffer storage, but
the mutating op argument is not buffer-tagged. `.data` is worse: it changes the base value while
the base tensor's `_version` remains unchanged.

Current TorchLens traces confirm the problem shape:

Probe: `/tmp/probe_trace_current.py`

```text
== ViewMut ==
buffer_1 ... children ['view_1_1', 'add_2_3'] out tensor([[0., 0., 0.], [0., 0., 0.]])
  out_versions {'add_2_3': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
view_1_1 func view ... parents ['buffer_1'] children ['add_1_2']
add_1_2 func add_ ... parents ['view_1_1'] children [] out tensor([1., 1., 1., 1., 1., 1.])
validate True

== SliceSet ==
getitem_1_1 func __getitem__ ... parents ['buffer_1'] children ['add_1_2']
add_1_2 func add_ ... parents ['getitem_1_1'] children [] out tensor([2., 2.])
```

Validation passes only because `out_versions_by_child` repairs the later read. The mutator
itself (`add_1_2`) is on an unaddressed alias, so a detector restricted to buffer-tagged args
does not know it was a buffer write.

For `.data` mutation:

```text
== DataMut ==
buffer_layers ['buffer_1']
buffer_1 ... addr b ... out tensor([[1., 1., 1.], [1., 1., 1.]])
```

The first logged buffer node is already the mutated value. The original zero state and the
write are both lost as buffer history.

Relevant code:

- `wrappers.py:438-443` logs buffers only when the actual arg has `_tl.address`.
- `wrappers.py:539-544` propagates labels for in-place ops to the mutated arg object, not to
  storage aliases or base buffers.

Proposed fix:

Maintain an active `BufferRegistry` keyed by storage identity plus object id/version, and
propagate buffer identity through view-like ops. At minimum, any tensor whose storage aliases a
registered buffer must be considered a buffer alias for mutation detection. `.data` writes need
explicit detection or an honest documented unsupported class with tests proving the limitation.

## Blocking issue 3: the reassignment hook does not cover all buffer writes TorchLens currently treats as buffers

SEVERITY: blocking

Concrete construction/evidence:

TorchLens currently tags registered buffers, plain tensor attributes, and list/tuple tensor
attributes as buffers in `model_prep.py:661-714`. The plan's reassignment coverage is only
`nn.Module.__setattr__` plus `register_buffer`.

Probe: `/tmp/probe_reassign_paths.py`

```text
== DirectBuffers ==
class forward: self._buffers["b"] = x + 1; return self.b * 2
buffer_layers []
validate True

== ListAttr ==
class forward: self.state[0] = x + 1; return self.state[0] * 2
buffer_layers []
validate True

== ListInplace ==
class forward: self.state[0].add_(1); return self.state[0] + x
buffer_layers ['buffer_1']
...
The 0th output layer, output_1, does not match the ground truth output tensor.
validate False
```

Direct `_buffers` replacement bypasses `__setattr__`. List/tuple element replacement bypasses
`__setattr__`. Yet TorchLens already treats list-held tensors as buffer-like state. A Phase-2
`Buffer` entity that ignores these paths will be internally inconsistent with current
buffer tagging.

The `SetNone` case is also dangerous:

```text
== SetNone ==
class forward: self.b = None; return x + 1
buffer_layers []
RuntimeError: Unexpected key(s) in state_dict: "b".
```

Buffer removal/replacement with `None` is a state write, not a tensor write. The plan only says
"when a registered-buffer name is reassigned, tag the new tensor"; it does not define remove,
delete, or non-tensor versions.

Proposed fix:

Define the supported state surface before implementation:

- Either narrow TorchLens Phase 2 to `nn.Module` registered tensor buffers only and stop
  advertising plain/list tensor attrs as persistent `Buffer` entities, or hook/track those
  containers with a journal.
- Treat `_buffers` mutation and `None`/delete as explicit unsupported cases with tests, or
  capture them through a registry consistency check at module exit.
- Add a module-exit reconciliation pass comparing registered buffer object/storage identities
  and current values against the registry so bypassed reassignments are caught or reported.

## Blocking issue 4: the hot-path cost claim is false as written

SEVERITY: blocking

Concrete construction/evidence:

The plan claims cost is "bounded to ops that actually touch a buffer-tagged tensor." The
current wrapper already scans every tensor arg on every logged op:

- `wrappers.py:431-443`: `_collect_tensor_args(args, kwargs)` then `get_buffer_address(t)` for
  every tensor arg.

Instrumentation of `wrappers.get_buffer_address` during real traces:

Probe: `/tmp/probe_scan_cost.py`

```text
TinyTransformer trace_ops 74 buffer_layers 0 get_buffer_address calls 152 hits 0 seconds 0.6375
NoBufferManyOps trace_ops 5 buffer_layers 0 get_buffer_address calls 1000 hits 0 seconds 1.2987
200k get_buffer_address non-buffer seconds 0.0715
```

Probe: `/tmp/probe_resnet_scan.py`

```text
resnet18 eval trace_ops 111 buffer_layers 40 get_buffer_address calls 421 hits 160 seconds 2.315
```

Even with zero buffer hits, the wrapper did hundreds or thousands of address probes. A
post-op detector either duplicates this scan after each op or must store pre-call buffer arg
state from the pre-call scan. Once value snapshots are required for fused version-blind ops
and alias detection is required for views, the cost is not just a cheap check on
buffer-touching ops.

Proposed fix:

Specify and measure the actual implementation:

- Reuse the existing pre-call scan; do not add a second post-call full arg crawl.
- Cache a per-call list of candidate buffer aliases.
- Snapshot values only for registry-marked mutator ops or storage aliases whose `_version`
  cannot be trusted.
- Add an explicit performance gate: no-buffer Transformer, ResNet train/eval, and a
  buffer-heavy mutation microbenchmark with max acceptable overhead.

## Blocking issue 5: identity write-version nodes are not integrated with existing replay metadata

SEVERITY: blocking

Concrete construction/evidence:

Current validation correctness for mutation often comes from `out_versions_by_child`, not from
graph version nodes:

- `ops.py:1542-1566` records a parent value variation keyed by child op.
- `validation/core.py` uses that child-specific parent value during replay.
- `validation/invariants.py:572-580` requires `out_versions_by_child` keys to be a subset of
  `children`.

Current view/slice mutation traces show this repair in action:

```text
buffer_1 out tensor([[0., 0., 0.], [0., 0., 0.]])
out_versions {'add_2_3': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
children ['view_1_1', 'add_2_3']
```

If Phase 2 inserts `b:v2` identity nodes and reroutes later reads from `buffer_1 -> add_2_3`
to `buffer_1 -> writer -> b:v2 -> add_2_3`, it must also migrate or delete existing
`out_versions_by_child` entries. Otherwise replay may use stale child-specific values, and
metadata invariants can fail if keys remain for children no longer directly attached.

The plan says "don't double-handle mutation" as a risk, but does not define a migration rule.
This is not optional; the current validation mechanism is already active for exactly the cases
Phase 2 wants to materialize as graph nodes.

Proposed fix:

Before minting public buffer version nodes, define an invariant:

- Either version nodes replace `out_versions_by_child` for buffer mutations, with variation
  entries removed/rewired during insertion, or version nodes are purely data-model nodes and
  validation explicitly ignores them.
- Add regression tests for `BatchNorm returns running_mean`, `view().add_ then read base`,
  `slice.add_ then read base`, and `copy_ then read base` with metadata validation enabled.

## Blocking issue 6: "full history" conflicts with orphan pruning and unreachable final writes

SEVERITY: blocking

Concrete construction/evidence:

The plan reopens "node only if read vs full history" and leans toward "mint on write" so an
unread final write can be a node. Current postprocess removes nodes unreachable from inputs and
outputs:

- `graph_traversal.py:223-262` starts from input/output nodes, records internally terminated
  tensors, and batch-removes orphan nodes unless `keep_orphans` is set.

A final buffer write that is not read and does not feed the model output is output-unreachable
by construction. Merely minting `b:vN` at write time does not guarantee it survives Step 3 or
appears in `trace.buffers`. The plan does not specify whether buffer history nodes bypass
orphan removal, live in `Trace.orphans`, or are stored only in a journal.

Proposed fix:

Choose one of these explicitly:

- Persistent buffer history is journal-backed and not dependent on graph reachability.
- Buffer write-version nodes are exempt from orphan pruning and marked as state outputs.
- Full history requires `keep_orphans=True`, in which case the public API must say so.

Add a test where `forward` mutates `self.b` after computing the returned output:

```python
y = x + 1
self.b.add_(1)
return y
```

The final write must appear in `buffer.versions` under the chosen semantics.

## Major issue 7: existing in-place label propagation will fight version-node routing unless redesigned

SEVERITY: major

Concrete construction/evidence:

For in-place ops, current wrapper behavior is:

- Safe-copy the output so logging does not overwrite the original tensor label
  (`wrappers.py:490-503`).
- Log the mutator as an ordinary op.
- Propagate the mutator's label back to the original tensor
  (`wrappers.py:539-544`).

The plan says "route subsequent reads of that buffer through the new version node." For an
in-place buffer tensor, subsequent reads currently see the mutator op label, not a buffer
version label. If Phase 2 does not replace or layer this label, consumers will continue to
attach to the mutator op and the version node will have no children. If it does replace the
label with `b:v2`, the mutator op's ordinary compute lineage and `out_versions_by_child`
behavior must be preserved deliberately.

Proposed fix:

Document the live tensor label state machine:

- After a buffer write, should `_tl.label_raw` be the writer op, the buffer version op, or a
  separate `current_buffer_version` field?
- How do downstream parent extraction and replay resolve this?
- Add tests asserting exact parents for `b.add_(x); return b * 2` and
  `b.view(-1).add_(1); return b * 2`.

## Major issue 8: no-op in-place writes create semantically noisy versions unless policy is explicit

SEVERITY: major

Concrete construction/evidence:

Probe: `/tmp/probe_version.py`

```text
add_(0) no-op value
  version 0 -> 1 delta=1
  value_equal=True
mul_(1) no-op value
  version 0 -> 1 delta=1
  value_equal=True
copy_(same)
  version 0 -> 1 delta=1
  value_equal=True
masked_fill_ no hits
  version 0 -> 1 delta=1
  value_equal=True
__setitem__ same
  version 0 -> 1 delta=1
  value_equal=True
```

Autograd `_version` records mutation attempts, not value changes. The plan says "if changed ->
a write occurred" but does not define whether a no-op write should produce a new buffer
version. Both policies are defensible, but they yield different public semantics:

- Event history: every mutation attempt is a version, even if equal.
- Value history: only value/metadata changes are versions.

Current `_fix_buffer_layers` deduplicates same-value buffers with the same
`modules + buffer_source + address` (`control_flow.py:672-689`), which can erase event history
unless disabled for write-version nodes.

Proposed fix:

Choose event-history or value-history semantics and encode it in tests. If event history wins,
disable same-value dedup for write-version nodes or include write event id in the dedup key.

## Major issue 9: `__setattr__` hook needs a concrete global patch lifecycle and failure model

SEVERITY: major

Concrete construction/evidence:

The hook is described as "active-capture-gated" and "exception-safe," but no implementation
boundary is specified. A global patch of `nn.Module.__setattr__` has several failure modes:

- Record-before-original-success can journal writes that PyTorch rejects.
- Record-after-original-success can miss provenance if the assigned tensor's TorchLens label is
  cleared by registration/tagging.
- Nested traces or exceptions inside forward must restore the exact previous method.
- Non-buffer attrs are frequent in user models; hook must be fast and not touch them.
- `load_state_dict` and TorchLens internal module metadata changes must not be mistaken for
  user forward writes.

Proposed fix:

Implement as a reentrant context manager with a token/refcount, always call original
`__setattr__` first, and journal only after success. Add tests for:

- exception during forward restores original `nn.Module.__setattr__`;
- assigning a `Parameter` or `Module` to a buffer name follows PyTorch semantics exactly;
- nested trace/validation does not leak the hook;
- non-buffer attr writes inside forward are ignored and measured.

## Major issue 10: fallback "ship read-side + document limitation" is not clean

SEVERITY: major

Concrete construction/evidence:

The fallback says to ship "truthful-accessor/entity read-side" if write capture fails. That is
only clean for static registered buffers that are actually read. Measured cases show otherwise:

```text
DirectBuffers: buffer_layers []
ListAttr: buffer_layers []
DataMut: buffer_1 out is already mutated final value
```

A read-side `Buffer` entity can be actively misleading if it exposes:

- no entity for a reassigned buffer whose initial value was never read;
- one entity with initial value equal to a post-`.data` mutation;
- no alias information for direct/list rewrites;
- no final state for unread writes.

Proposed fix:

If fallback ships, its public contract must be narrow:

"Read-side entity covers buffer tensors that are tagged and consumed as tensor args during
captured computation. It is not a state-history API and does not promise final buffer state,
write counts, or assignment history."

Anything stronger is half-built.

## Minor issue 11: copy-source edge is acknowledged but should be a P2a prerequisite

SEVERITY: minor

The plan lists `copy_` source edge missing as P2c, after detector/version chain work. That is
the wrong order. `copy_` is a canonical buffer write, and the write-version node's parent must
represent the copied value source. If `copy_` source extraction remains wrong during P2a, the
new version node will encode wrong provenance and tests may pass only because saved literal args
hide the dependency.

Proposed fix:

Move `copy_` source parent extraction into P2a before public version nodes are created.

## Test/probe commands run

All commands were run from repo root with `python` in the active py311 environment.

```bash
python /tmp/probe_version.py
python /tmp/probe_alias_metadata.py
python /tmp/probe_trace_current.py
python /tmp/probe_scan_cost.py
python /tmp/probe_resnet_scan.py
python /tmp/probe_reassign_paths.py
```

No tracked source files were modified.

## Final blocking count

Blocking issues:

1. `_version` misses fused BatchNorm running-stat writes.
2. Buffer-tagged-arg scanning misses alias/view/slice/detach/`.data` writes.
3. `__setattr__`/`register_buffer` hook misses current TorchLens buffer-like reassignment paths.
4. Hot-path cost claim is false as written.
5. Identity write-version nodes are not integrated with `out_versions_by_child` replay metadata.
6. Full-history write nodes conflict with current orphan pruning.

VERDICT: NOT BULLETPROOF -- 6 blocking issues
