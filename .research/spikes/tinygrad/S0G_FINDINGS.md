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
