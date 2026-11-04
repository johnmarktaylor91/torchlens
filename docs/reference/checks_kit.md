# Training sanity checks (the checks kit)

> Every spelling on this page is DOCUMENTED-UNSTABLE pending naming-session
> ratification. The public names live on `tl.debug` (`tl.debug.ChecksSession`, `tl.debug.RangeProbe`, and so on); the one-shot
> parameter audit is also served as `tl.debug.param_health`. The kit is
> STANDALONE and CAPTURE-FREE: it constructs and runs with TorchLens capture
> entirely off, and the step-check family runs on `torch.compile`'d models.

TorchLens's training checks occupy the abandoned `torcheck` niche -- params
changing, nonfinite anywhere, output ranges, gradient flow, weight-update
ratios -- with each check honestly redefined by measurement rather than
transplanted naively (the naive versions of the two flagship checks are
provably broken; see "What the measurements broke" below).

### Coming from torcheck

torcheck's nine registrar calls and its four controls, mapped to the checks kit. The
checks are redefined by measurement, so a row maps the intent, not identical
semantics (see "What the measurements broke" below).

| torcheck | Checks kit | Notes |
| --- | --- | --- |
| `torcheck.register(optimizer)` | `session = tl.debug.ChecksSession(model, optimizer)`, then `with session:` (or `session.attach()`) | Hooks install on attach, not on construction. |
| `add_module_changing_check(module)` | `session.register_change_check(within=[...parameter names])` | Detects parameters that stop receiving gradient over a window of steps. |
| `add_tensor_changing_check(tensor)` | `session.register_change_check(within=["name"])` | Name the parameter. |
| `add_module_unchanging_check(module)` | `session.register_frozen_check([...parameter names])` | Declared-frozen parameters; the default evidence keeps exact snapshots and shows the delta. |
| `add_tensor_unchanging_check(tensor)` | `session.register_frozen_check(["name"])` | |
| `add_module_output_range_check(module, output_range=(low, high))` | `tl.debug.RangeProbe({site: (low, high)})`, passed to a trace as `capture=tl.options.CaptureOptions(hooks=probe.hook_plan())` | Sites are TorchLens selectors such as `tl.func("softmax")`. |
| `add_module_nan_check(module)` | `session.register_nonfinite_param_check()` and `session.register_nonfinite_grad_check()`; for activations, `tl.trace(..., capture=tl.options.CaptureOptions(on_nonfinite='raise'))` | Parameters, gradients and activations are checked at the points where the evidence is exact. |
| `add_module_inf_check(module)` | The same three checks | They cover NaN and Inf together. |
| `add_tensor_nan_check(tensor)` | `session.register_nonfinite_param_check()` | Scans every parameter and, by default, every buffer. |
| `add_tensor_inf_check(tensor)` | `session.register_nonfinite_param_check()` | |
| `torcheck.disable()` | `session.disable()` | |
| `torcheck.enable()` | `session.enable()` | `with session.paused():` suspends checks for a block. |
| `torcheck.verbose_on()` / `verbose_off()` | `session.report()` | Findings collect into the report; `action="raise"`, `"warn"` or `"collect"` on each registration sets what interrupts the loop. |

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
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
optimizer = torch.optim.AdamW(model.parameters())

session = tl.debug.ChecksSession(model, optimizer, scaler=None)
session.register_change_check()            # gradient-fact detector, M-of-N warn
session.register_nonfinite_grad_check()  # collect at S-A, raise at S-B pre-write
session.register_update_ratio_check()      # raw + lr-normalized columns, collect

with session:                               # attach() installs the hooks; detach() on exit
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).square().mean()
        loss.backward()
        optimizer.step()

report = session.report()
print(report)                               # house-style summary (matches HealthReport)
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
| S-D | the shipped activation path | `on_nonfinite='raise'`, `on_nonfinite='track'`, `capture.hooks` |

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
  parameters; each finding's `follow_up` points at `grad_health`'s
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
session2 = tl.debug.ChecksSession(model, optimizer, scaler=None)
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

## One-shot parameter auditing: `param_health`

```python
import torchlens as tl

audit = tl.debug.param_health(model)          # also: tl.debug.param_health(model)
print(audit)                                # notebook-style summary
assert not audit.findings                    # healthy fresh model
```

`param_health` takes an `nn.Module` or ANY name->tensor `Mapping` -- a
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
    print(tl.debug.param_health(optimizer_state))
```

```python
# Recipe: one-shot gradient triage (label the phase yourself: post-backward).
optimizer.zero_grad()
model(torch.randn(4, 8)).square().mean().backward()
grads = {f"grad.{n}": p.grad for n, p in model.named_parameters() if p.grad is not None}
grad_audit = tl.debug.param_health(grads)
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

probe = tl.debug.RangeProbe({tl.func("relu"): (0.0, 6.0)})
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
liveness = tl.debug.LivenessProbe([tl.func("relu")])
t1 = tl.trace(model, torch.randn(4, 8),
              capture=tl.options.CaptureOptions(hooks=liveness.hook_plan()))
t2 = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
t3 = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
suspects = tl.dead([t2, t3])                # the verdict door, multi-sample
```

## Attention sinks and outlier residual dimensions (the `capture.hooks` seam)

> Pending spelling: the sink probe is implemented privately as
> `torchlens.debug._checks._sinks.SinkProbe` until its public name, check names and
> finding codes are ruled. It reports statistics and flags, never a `CheckFinding`.

Two facts most trained transformers share, measured on the same zero-op seam as the
range probe:

- **Attention sinks**: heads whose queries put most of their attention mass on one key
  position, usually the first token.
- **Outlier residual dimensions**: residual-stream features whose mean absolute value is
  many times the typical feature's.

```python
from torchlens.debug._checks._sinks import SinkProbe


class TinyLayer(nn.Module):
    """One causal attention head over an 8-wide residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.query, self.key, self.write = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = (self.query(x) @ self.key(x).transpose(-1, -2)).unsqueeze(1) / 8**0.5
        causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
        probs = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        return x + self.write((probs @ x.unsqueeze(1)).squeeze(1))


tiny = nn.Sequential()
tiny.add_module("layer", TinyLayer())
probe = SinkProbe(attention=[tl.func("softmax")], residual=[tl.module("layer")])
trace = tl.trace(tiny, torch.randn(2, 6, 8),
                 capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))
probe.sink_heads      # SinkHead(label, head, position, mass) per flagged head
probe.outlier_dims    # OutlierDim(label, dim, factor) per flagged dimension
print(probe.facts())  # per-site statistics; "skipped" says why a site was not scored
```

On a language model the sites are its attention probabilities and its block outputs, for
GPT-2 small `tl.func("softmax")` and `tl.module(f"transformer.h.{i}")` for `i` in 0 to 11.

Conventions, stated rather than guessed:

| Convention | Default | Meaning |
| --- | --- | --- |
| Attention layout | `(..., heads, queries, keys)` | the site's value must be probabilities: non-negative rows summing to 1 within 1e-3, else the site is reported as skipped |
| `mass_threshold` | `0.5` | a head is a sink when its largest mean mass on one key, averaged over the batch and the kept queries, reaches this |
| `skip_positions` | `1` | leading query positions left out of the sink average (under a causal mask query 0 sees only itself) and leading positions left out of the outlier statistic (the first token carries the massive activations) |
| `query_mask` | none | `batch x queries` booleans; padded queries are left out of the sink average. Masked keys already carry no probability |
| Residual layout | `(..., positions, features)`, `feature_axis=-1` | each feature's mean absolute value over the kept rows, divided by the median feature's |
| `outlier_factor` | `10.0` | a feature is an outlier at or above this ratio |

Sinks and outlier dimensions are normal in trained language models, so these are
statistics to look at, not failures. A sink head's attention pattern says little about
what the head reads; an outlier dimension dominates norms, cosine similarities and
naive quantization ranges.

**GPT-2 small** (pinned revision, float32, eager attention; the 14-token prompt "When
Mary and John went to the store, John gave a drink to"; `tl.func("softmax")` and the
12 block outputs; defaults):

| Measure | Result |
| --- | --- |
| Sink heads per layer, layers 0 to 11 | 0, 5, 3, 6, 8, 12, 10, 12, 12, 11, 12, 9 (100 of 144) |
| Sink position | 0 for every flagged head |
| Sink heads at `mass_threshold` 0.3 / 0.7 | 127 / 65 of 144 |
| Outlier dimensions per block, blocks 0 to 11 | 7, 6, 6, 5, 4, 4, 3, 4, 4, 3, 4, 11 |
| Dimension present in every block | 447 (26 to 43 times the median feature) |
| Other recurring dimensions | 373, 64, 266, 481, 496 |
| Outlier dimensions per block at `outlier_factor` 20 | 4, 4, 3, 3, 1, 1, 1, 1, 1, 3, 3, 8 |
| Trace ops, bare and probed | 623 and 623 |

The `s_aux` sink logits some models pass to attention are a different thing: a learned
extra column of the softmax. `tl.bridge.transformers.record_attention(check_sdpa=True)`
skips its SDPA cross-check for those calls; this probe measures the emergent sinks.

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
