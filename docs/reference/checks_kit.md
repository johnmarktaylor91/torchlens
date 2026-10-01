# Training sanity checks (the checks kit)

> Every spelling on this page is DOCUMENTED-UNSTABLE pending naming-session
> ratification. The import spelling is `import torchlens.checks`; the one-shot
> parameter audit is also served as `tl.debug.audit_params`. The kit is
> STANDALONE and CAPTURE-FREE: it constructs and runs with TorchLens capture
> entirely off, and the step-check family runs on `torch.compile`'d models.

TorchLens's training checks occupy the abandoned `torcheck` niche -- params
changing, nonfinite anywhere, output ranges, gradient flow, weight-update
ratios -- with each check honestly redefined by measurement rather than
transplanted naively (the naive versions of the two flagship checks are
provably broken; see "What the measurements broke" below).

Start with torch's own tripwire: `clip_grad_norm_(error_if_nonfinite=True)`.
It is upstream, free, and OFF by default -- and without it the canonical clip
recipe SILENTLY ZEROES every healthy gradient in the model and NaN-poisons the
culprit parameter when a single Inf appears (measured: 202/203 healthy
gradients zeroed on electra-small, fp32, no AMP involved). The checks kit's
pre-write raise exists because almost nobody passes that flag; the remedy
strings point back at it.

## Quick start: hook-only mode (zero loop edits)

```python
import torch
import torch.nn as nn
import torchlens.checks as checks

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
optimizer = torch.optim.AdamW(model.parameters())

session = checks.ChecksSession(model, optimizer, scaler=None)
session.register_change_check()            # gradient-fact detector, M-of-N warn
session.register_nonfinite_gradient_check()  # collect at S-A, raise at S-B pre-write
session.register_update_ratio_check()      # raw + lr-normalized columns, collect

with session:                               # attach() installs the hooks; detach() on exit
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).square().mean()
        loss.backward()
        optimizer.step()

report = session.report()
print(report)                               # house-style summary (matches TraceAudit)
```

Construction is silent and does no device work: no hooks, no syncs, no stdout.
`session.estimated_snapshot_bytes` is pure numel x dtype-size metadata, and
`session.profile()` runs the measured scan on demand. `scaler=None` is a
first-class DECLARATION that the loop runs no GradScaler -- pass the real
handle when there is one, so skip accounting can read scale decrements.

## The four sites (where evidence lives)

| Site | Handle | What it sees |
|---|---|---|
| S-A | per-parameter post-accumulate-grad hooks | every backward INCLUDING scaler-skipped attempts; values SCALED, PRE-clip |
| S-B | optimizer step pre-hook | accepted steps only; values already UNSCALED, POST-clip; the last moment before weights change |
| S-C | optimizer step post-hook | after the write: exact deltas, ratios, scheduled scans |
| S-D | the shipped activation path | `raise_on_nan`, `track_nonfinite`, `capture.hooks` |

Two standing laws fall out of this table. Gradients at S-B are already
unscaled -- dividing by `scaler.get_scale()` there under-reports by exactly the
loss scale (measured: 17.2% of a real model mislabeled vanishing). And
absolute magnitude evidence lives ONLY at S-A: the step site is post-clip,
where clipping pins the total norm to `max_norm` (measured 6/6 steps on a real
fine-tune), so an "exploding gradient" check mounted there is structurally
incapable of firing. Requesting magnitude where clipping censored the evidence
returns a typed `unavailable` entry in the report, never a silent pass.

## What the measurements broke (why the checks look like this)

- **"Params changing" is vacuous at torch defaults.** Under default AdamW,
  weight decay rescales every parameter of a DEAD network every step. The
  movement statistic also lags: Adam momentum keeps a dead parameter moving
  for 22 to >32 steps after death. The shipped primary detector is therefore
  the GRADIENT FACT (`grad is None` / `grad_norm == 0`) on a sustained M-of-N
  window (`param_received_no_gradient` -- single zero-grad steps are
  legitimate: MoE routing, untaken branches, embedding coverage). Movement
  statistics are demoted to corroborating facts carrying their measured
  latency, and no finding ever says "learning" or "dead".
- **The finding names a cone; the frontier names the culprit.** A dead layer
  zeroes the gradient of everything upstream, so the detector names many
  parameters; each finding's `follow_up` points at `gradient_flow_audit`'s
  module / `is_frontier` column, which resolves the exact dead layer.
- **Nonfinite policy is site-conditional (the AMP hole, closed both ways).**
  At S-A nonfinite gradients are COLLECTED, never raised -- under fp16 +
  GradScaler they are the mechanism working. At S-B they RAISE before the
  weights are written. Under a scaler S-B is structurally unreachable by a
  nonfinite (the skip fires first; measured 0/3 seeded legs), while WITHOUT
  one (fp32, and bf16, which by design has no scaler) one seeded NaN corrupts
  every parameter in the model. The raise is free exactly where AMP protects
  and load-bearing exactly where nothing else does. Weights stay bitwise clean
  and the batch is re-runnable; the finding reports the culprit AND the
  clipping collateral count, and the NaN-smear case sets `attribution=smeared`
  rather than naming an innocent.

## AMP accounting: the skip ledger and the watchdog

With a `scaler=` handle, the session counts GradScaler scale DECREMENTS --
each decrement is exactly one skipped attempt (measured exact in 4/4
configurations including consecutive skips and growth events, with the
accumulation factor K recovered as an exact integer). Naive fire-count
subtraction (S-A fires minus S-B fires) is wrong under gradient accumulation
by construction (measured 19 reported vs 1 true skip at K=4) and is locked out
by tests. Hook-only evidence labels attempt grouping
`step_provenance="inferred"`; caller `global_step` and exact grouping are the
explicit boundary's unique property (below).

The death-spiral watchdog rides the free counting tier: N backwards since the
last accepted step exceeding K x threshold warns (`check_no_accepted_steps`)
even when no successful step ever arrives -- the case where every
retrospective method stays silent forever.

## Boundary mode (one `with` line per optimizer attempt)

```python
session2 = checks.ChecksSession(model, optimizer, scaler=None)
session2.register_change_check()
with session2:
    for step_idx in range(2):
        with session2.step(global_step=step_idx):
            optimizer.zero_grad()
            model(torch.randn(4, 8)).square().mean().backward()
            optimizer.step()
report2 = session2.report()
print(report2.to_dict()["counters"])        # explicit attempts / accepted / skipped
print(report2.to_dict()["ledgers"]["scale"])  # the skip ledger (decrement counting)
```

Boundary mode adds what hook-only mode honestly cannot know: caller
`global_step`, declared accumulation grouping (`micro_batches=`), and exact
accepted/skipped attribution per attempt. Both modes are first-class; unknown
or inferred evidence never produces an exact pass.

## One-shot parameter auditing: `audit_params`

```python
import torchlens.checks as checks

audit = checks.audit_params(model)          # also: tl.debug.audit_params(model)
print(audit)                                # notebook-style summary
assert not audit.findings                    # healthy fresh model
```

`audit_params` takes an `nn.Module` or ANY name->tensor `Mapping` -- a
`state_dict`, flattened optimizer state, EMA shadow weights, a loaded
checkpoint: same door, same batched scan kernel the session's scheduled scan
uses. Findings cover signed nonfinite counts, caller-declared bounds, dtype
headroom (the fp16 overflow-next-step signature), subnormal-heavy tensors, and
suspicious all-same/all-zero fills, with honest per-tensor coverage and
skipped-with-reason rows (meta/sparse/quantized). There is no `grads=` and no
`optimizer=` by design: transient gradients are phase-sensitive (a checkpoint
verdict must not depend on WHEN you called it), and the mapping input already
serves those populations:

```python
# Recipe: optimizer-state scan (Adam moments, step counters -- six lines).
optimizer_state = {
    f"{index}.{key}": value
    for index, slot in enumerate(optimizer.state.values())
    for key, value in slot.items()
    if isinstance(value, torch.Tensor)
}
if optimizer_state:
    print(checks.audit_params(optimizer_state))
```

```python
# Recipe: one-shot gradient triage (label the phase yourself: post-backward).
optimizer.zero_grad()
model(torch.randn(4, 8)).square().mean().backward()
grads = {f"grad.{n}": p.grad for n, p in model.named_parameters() if p.grad is not None}
grad_audit = checks.audit_params(grads)
print(f"{len(grad_audit.rows)} gradient tensors scanned, {len(grad_audit.findings)} findings")
optimizer.zero_grad()
```

## Loss-finiteness guard (a one-liner, not an API)

```python
# Recipe: guard the loss scalar before backward -- torch already has the check.
loss = model(torch.randn(4, 8)).square().mean()
if not torch.isfinite(loss):
    raise RuntimeError(f"nonfinite loss {loss.item()!r}: skip or debug this batch")
loss.backward()
optimizer.zero_grad()
```

Under fp16 + GradScaler, prefer letting the scaler handle nonfinite GRADIENTS
(that is the mechanism working); the loss guard catches the upstream case
where the forward itself produced a nonfinite before any scaling.

## Live activation ranges and liveness (the `capture.hooks` seam)

The range and liveness probes ride the public `capture.hooks` value-probing
path, which adds ZERO ops to a trace (measured 13/13/13, BatchNorm model
included). The in-tree implementation uses only the public seam -- it doubles
as the seam's conformance test.

```python
import torchlens as tl

probe = checks.RangeProbe({tl.func("relu"): (0.0, 6.0)})
trace = tl.trace(model, torch.randn(4, 8),
                 capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))
print([f.code for f in probe.findings])    # activation_range_violated rows, if any
```

Bounds are the caller's semantic declaration; findings carry
`activation_range_violated` and point at `tl.threshold(within=...)` for
post-hoc forensics. The `LivenessProbe` records per-site zero-fraction and
running-max STATISTICS only -- its findings never contain the word "dead":

```python
# Recipe: dead-unit triage. The liveness statistic tells you WHERE to point
# tl.dead; tl.dead keeps the only dead-unit VERDICT (>=2 samples, upper-bound
# epistemics -- one batch can never prove a unit dead).
liveness = checks.LivenessProbe([tl.func("relu")])
t1 = tl.trace(model, torch.randn(4, 8),
              capture=tl.options.CaptureOptions(hooks=liveness.hook_plan()))
t2 = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
t3 = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
suspects = tl.dead([t2, t3])                # the verdict door, multi-sample
```

## Severity and action (two axes, never conflated)

Severity (`critical/warning/info`) describes evidence; action
(`raise/warn/collect`) describes control flow and is overridable per
registration. Raises fire only on exact evidence at safe pre/post-mutation
points (never mid-`step()`), as `CheckViolationError` carrying the finding,
a stable `fields["code"]`, step ids, names, remedy, and the partial report.
Warns fire only on sustained M-of-N windows with one emission per
(check, site), re-armed only after a clean window. Everything else collects
into the report, which finalizes on scope exit including exceptional exit.
`disable()` / `enable()` / `with session.paused():` ship day one.

Every gradient/parameter scalar that crosses any surface carries `stage`
(`pre_clip_scaled` / `pre_clip_unscaled` / `post_clip_applied`) and
`scale_provenance` (`unscaled` / `gradscaler` / `explicit` / `unknown`).
Unknown provenance yields UNKNOWN verdicts, never a silent pass-as-1.0.

## Reports and export

`session.report()` returns an immutable `CheckReport` (findings, checks run,
ledgers, disclosures, mandatory `unavailable` entries). `to_dict()` /
`to_json()` carry `schema_version=1` -- the public export from day one.
CheckReports are declared NON-PERSISTENT in `.tlspec` bundles for v1; the
schema version and immutable record shape are the forward-compatibility hooks
for a later sidecar.

## Costs (absolute numbers at named shapes; no adjectives)

Measured on the panel's CPU runner (gpt2-124M-class step, 203 parameters,
min-of-N with load recorded): the always-on S-A counting tier measured
0.910/0.995/1.019x step time over 3 interleaved runs; counting plus ONE
batched `torch._foreach_norm` + 1 sync measured 0.971/0.969x over 2 runs; the
`register_multi_grad_hook(mode="all")` fallback measured 1.117-1.198x over 3
runs and stays in-tree behind the `HAS_MULTI_GRAD_HOOK` compat flag for graphs
whose expected parameter set never closes. Per-parameter `.item()` in any S-A
callback is forbidden on mechanism (203 device syncs on CUDA) and gated by a
static test. The same scan kernel measured 18.1%/2.2%/0.1% of step time as
only the batch shape changed -- percentages belong to the caller's shape, so
CI gates milliseconds and bytes, never percentages.

The S-A magnitude pass arms ONLY on explicit `register_magnitude_check()`:
the default stays OFF until the canonical gate set passes (idle-box + CUDA
fp16 + multi-rank DDP `no_sync` + at-scale compile -- all four are owed and
unrun; every scaler measurement to date is the CPU scaler).
