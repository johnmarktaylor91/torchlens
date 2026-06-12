# S0.G tinygrad Findings

## Round 1 - 2026-06-12

Exact pin: `tinygrad==0.13.0` from PyPI. The isolated environment is
`/tmp/jaxtg/tinygrad-venv`. Probe execution uses `DEV=PYTHON` because the host
does not have `clang`, and tinygrad's default CPU runtime attempts to compile C.

### Deliverable 1 - Interception Inventory

Evidence:

- Tensor API events are visible by wrapping `Tensor._apply_uop`; the toy
  `(Tensor([1, 2]) + 1).relu()` expression emits `ADD` and `WHERE`.
- UOp graph snapshots are available from `Tensor.uop.toposort()` before
  realization; the same toy expression has a durable root `WHERE` with `ADD` and
  `CMPLT` in lineage.
- Scheduler/realize hooks are visible by wrapping `tinygrad.tensor.run_linear`,
  which is the symbol used by `Tensor.realize()`.

Current source-of-truth lean: UOp graph snapshots should be the semantic source
of truth for capture. Tensor API wrapping is useful for event timing but misses
lower-level rewrites and fused semantics. Scheduler/realize hooks are best
treated as execution-boundary evidence, not operation semantics.

### Deliverable 2 - Realization Policy Evidence

`probe_interception_realization.py` asserts that reading a tiny payload through
`tolist()` leaves the source tensor's UOp signature unchanged. The same
non-interference assertion passes with an observational `run_linear` hook
installed. This supports an initial policy of observing realization boundaries
without rewriting tensor lineage.

### Deliverable 3 - Identity Invariant + Mutation Probes

`probe_identity_autograd_payload.py` currently covers:

- `assign`: preserves Python object identity but replaces `tensor.uop` with an
  `AFTER` root.
- `replace`: preserves Python object identity but swaps `tensor.uop` directly to
  the replacement's root.
- view assignment: rewrites both base and view lineage; base root becomes
  `AFTER`.
- `setitem`: rewrites lineage through the assignment path; root becomes `AFTER`.
- repeated backward: accumulates into the existing `grad` tensor object.
- `TinyJit`: first call ignores, second captures, third executes captured state.
- GC: tinygrad's weak `all_tensors` registry drops collected tensors.

Current invariant lean: TorchLens cannot use Python `Tensor` object identity as
the backend tensor identity. The minimum viable identity key must track UOp
identity/lineage and treat mutation operations as identity-rebinding events.

### Deliverable 4 - Autograd Lifecycle Probes

Initial probes show `Tensor.gradient(...)` returns gradient tensors without
populating `.grad`, while `Tensor.backward()` populates `.grad` on in-scope
floating tensors. Repeated `backward()` accumulates into the existing grad tensor
via assignment. Non-scalar `backward()` without an explicit gradient raises.

### Deliverable 5 - Payload Capability Evidence

Initial payload evidence is positive but not complete. For small `DEV=PYTHON`
tensors, `tolist()` and `clone()` produce host-readable payloads without
mutating the source UOp graph. This is enough to keep investigating sanctioned
realized-buffer copies, but not enough yet for a full-save decision across
devices/JIT/autograd payloads.

### Probe Results

| Probe | Command | Result | Evidence |
|---|---|---|---|
| interception + realization | `DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python .research/spikes/tinygrad/probe_interception_realization.py` | PASS | Tensor API events, UOp lineage, and `run_linear` hook all asserted. |
| identity + autograd + payload | `DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python .research/spikes/tinygrad/probe_identity_autograd_payload.py` | PASS | Mutation, repeated backward, TinyJit, GC, gradient lifecycle, and payload non-interference asserted. |

### Remaining For Later S0.G Rounds

- Expand scheduler evidence around `create_linear_with_vars`, compiled linear
  calls, and JIT-captured linears.
- Add device/backend variation if available without heavy compute or host
  interference.
- Probe explicit-gradient backward, grad payload reads, stale graph reads after
  mutation, and mutation during/after JIT capture.
- Decide whether payload copies are sanctioned for v1 or whether tinygrad must
  ship metadata/audit-only.

### M2 Lean

Conditional go. The UOp graph gives a credible source of truth and initial
payload reads are non-interfering, but mutation identity and grad accumulation
are sharp enough that M2 should stay gated until later rounds prove robust
identity rebinding and payload behavior around JIT/autograd.

## Round 2 - 2026-06-12

Exact pin remains `tinygrad==0.13.0`; `pip index versions tinygrad` reported
`LATEST: 0.13.0` and `pip show tinygrad` reported `Version: 0.13.0`. The
isolated environment remains `/tmp/jaxtg/tinygrad-venv`. Probe execution still
uses `DEV=PYTHON` because this host has no `clang`.

### Deliverable 1 - Interception Inventory

Additional evidence:

- `Tensor.realize()` calls `Tensor.linear_with_vars(*to_realize)`, which returns
  exactly `(linear: UOp, var_vals: dict[str, int])` in this pin, then calls
  `tinygrad.tensor.run_linear(linear, var_vals, update_stats=...)`.
- The new probe wraps both `Tensor.linear_with_vars` and `run_linear` and asserts
  that explicit realization of `(Tensor([1, 2]) + 1).relu()` emits one LINEAR
  planning event and one non-JIT run event.
- Explicit `realize()` collapses the expression's semantic root from `WHERE` to
  `BUFFER`, while the source-of-truth semantic lineage is available before that
  boundary.

Source-of-truth decision strengthened: use UOp snapshots before realization as
the semantic capture source of truth. Scheduler/realize hooks should mark
execution boundaries and prove timing, but they are too late to be the only
semantic source because realization can rebase a tensor to a buffer.

### Deliverable 2 - Realization Policy Evidence

`probe_scheduler_autograd_payload_round2.py` adds a split assertion:

- `tolist()` returns `[2.0, 3.0]` without changing the source expression's UOp
  signature.
- explicit `realize()` returns the same Tensor object but rewrites its UOp
  signature to a `BUFFER` root.

Policy implication: a capture adapter must snapshot semantic UOps before
explicit realization or mutation boundaries. Host payload reads through `tolist`
remain a promising non-interfering copy mechanism for toy `DEV=PYTHON` tensors.

### Deliverable 3 - Identity Invariant + Mutation Probes

Round 2 adds an aliasing tripwire: a held view can keep the same UOp operation
signature while its observed values change after `view.assign(...)` mutates the
base buffer. That means identity cannot be "Python object id + current graph op
names." It needs a buffer/UOp identity story plus mutation-generation tracking
for aliases.

TinyJit evidence was sharpened: the first call has no captured state, the second
call captures, and the third call executes captured state. With a `run_linear`
hook installed, the first two calls emit `[(False, 1), (False, 0)]`; the third
captured execution does not go back through the same normal `run_linear` hook.

### Deliverable 4 - Autograd Lifecycle Probes

Round 2 adds explicit-gradient coverage:

- scalar `backward(gradient=Tensor(2.0))` populates `x.grad` with `[6.0, 6.0]`
  for `((x * 3).sum())`.
- non-scalar `gradient(x, gradient=Tensor([1.0, 10.0]))` returns `[3.0, 30.0]`
  and does not mutate existing `x.grad`.

Lifecycle implication: `gradient(...)` remains read-only with respect to `.grad`,
while `backward(...)` is stateful and can accumulate through `.grad.assign`.

### Deliverable 5 - Payload Capability Evidence

Additional positive evidence:

- A realized clone of a lazy expression keeps `[2.0, 3.0]` after the source is
  assigned to `[7.0, 8.0]`.
- A cloned and realized gradient snapshot keeps `[2.0, 4.0]` after the primal is
  assigned to `[10.0, 20.0]`.

Current payload lean: sanctioned realized-buffer copies are plausible for
`DEV=PYTHON` toy tensors, including gradient payloads, but the decision should
stay open until device variation and TinyJit captured-execution payload reads
are covered.

### Probe Results

| Probe | Command | Result | Evidence |
|---|---|---|---|
| interception + realization | `DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python .research/spikes/tinygrad/probe_interception_realization.py` | PASS | Existing Tensor API, UOp lineage, and `run_linear` hook assertions still pass. |
| identity + autograd + payload | `DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python .research/spikes/tinygrad/probe_identity_autograd_payload.py` | PASS | Existing mutation, repeated backward, TinyJit, GC, gradient lifecycle, and payload assertions still pass. |
| scheduler + autograd + payload round 2 | `DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python .research/spikes/tinygrad/probe_scheduler_autograd_payload_round2.py` | PASS | Explicit realization rebasing, tolist non-interference, JIT hook shape, view aliasing, explicit-gradient lifecycle, and realized payload snapshots asserted. |
| ruff spike lint | `nice -n 19 ionice -c3 ruff check .research/spikes/tinygrad --fix` | PASS | `All checks passed!` |

### Remaining For Later S0.G Rounds

- Confirm whether any available non-default device backend can be probed safely
  without compiling or interfering with the host benchmark.
- Inspect TinyJit captured execution more deeply; the third call bypasses the
  normal `run_linear` hook used here.
- Test payload reads from tensors produced inside JIT-captured execution.
- Convert the source-of-truth lean into a final S0.G decision after device/JIT
  payload behavior is known.

### M2 Lean

Still conditional go, slightly stronger. UOp snapshots before realization look
like the right semantic source of truth, and sanctioned payload copies look
viable for toy CPU/PYTHON tensors. The remaining blockers are TinyJit captured
execution visibility, device coverage, and a precise mutation-generation
identity invariant for aliases.
