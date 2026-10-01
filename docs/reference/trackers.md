# Trackers: feeding TensorBoard and wandb honestly

> Every spelling on this page is DOCUMENTED-UNSTABLE pending the naming
> sprint. Access is `import torchlens.trackers` for now.

TorchLens never builds dashboards. It computes more and better statistics
than the trackers' built-in watchers and feeds their dashboards: one
collection engine (the `torchlens.observability` per-step records), one
conversion layer, and first-party sinks.

```python
import torch
import torchlens.trackers as trk

model, opt = build_model_and_optimizer()
sink = trk.TensorBoardSink("runs/exp1")   # or an existing SummaryWriter
with trk.watch(model, to=sink, signals=("gradients", "updates"),
               optimizer=opt, every=50, hist_every=200) as watch:
    for global_step, batch in enumerate(loader, start=start_step):
        with watch.step(global_step, scaler=scaler):
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
print(watch.close().format())
```

## The three cost tiers, disclosed

| Tier | Signals | Mechanism | Cost class | Status |
|---|---|---|---|---|
| P (default) | parameters, gradients, updates | optimizer-boundary hooks; NO forward is wrapped or run | cheap (whole-step numbers publish only after the quiet-GPU gate) | shipped |
| M | module-output activations | ordinary forward hooks; one real discovery forward at attach (`example_input=`), run in the model's CURRENT mode | ~1.1x class (forward-only ratio) | shipped |
| O | op-grain activations | scheduled fastlog capture | ~34x class on CUDA, WRAP-dominated | deferred: typed refusal |

The tier-O refusal carries the sentence no user could guess: narrowing
`save=` will NOT reduce op-capture time (op wrapping dominates the bill);
**`halt=` is the lever that does** (10.4x vs 35.0x on the repo's own gpt2
CUDA rows). Until the tier ships, `tl.record(model, inputs,
save=<predicate>)` on the steps you care about is the explicit spelling.

## AMP truth: observed, derived, or unknown -- never asserted

Pass the scaler into the step scope (`watch.step(n, scaler=scaler)`) and
the engine reads TWO facts at the optimizer boundary: the scale
(`torchlens/run/amp_scale`) and whether the gradients it reduced were
already unscaled. `scaler.step(optimizer)` unscales in place before the
boundary, so the record says `unscaled="yes"` and the numbers are
observed. A plain `optimizer.step()` after `scaler.scale(loss).backward()`
never unscales: the record says `unscaled="no"`, the emitted gradient
statistics are corrected in closed form (bit-exact for the scaler's
power-of-two scales) and the derivation is disclosed on the run-health
series `torchlens/run/amp_unscale_derived`. A scaler whose stage cannot be
read (a duck-typed object without the per-optimizer stage record) stamps
`unknown` and ships the numbers as measured -- unknown never masquerades
as a factor of one.

## The step law

Every record carries the CALLER's `global_step`. There is no hidden
counter: a checkpoint resume with an internal counter restarting at 0
silently misaligns every panel (and the wandb relay ships this exact
failure -- see the table below). Sources: `with watch.step(n):`
(authoritative; also the home of AMP scale stamping and skip truth),
`step=callable` at attach for unchanged loops, or the framework callbacks
(`HFTrainerWatchCallback`, `LightningWatchCallback`). Duplicate or
decreasing steps refuse unless `new_segment=True` declares a resume.

Under `step=callable` the engine schedules tier-M activation observation
at forward entry (one root forward pre-hook, installed only when both
`step=` and `activations` are requested) and carries the forward's
observations across the implicit block open at the optimizer boundary, so
the two step spellings observe the same activation statistics. The HF
callback lands rows on the Trainer's own log axis: `state.global_step` is
incremented AFTER `optimizer.step()`, so the callback reads it `+ 1`.

The manifest and the `torchlens/run/attached` heartbeat ride the FIRST
step the session sees (and the last seen step, or 0, at close when no
step ran): a resumed wandb run drops every `log(step=0)` call, and those
were exactly the rows that vanished.

## Attach-time refusals (nothing latches silently after step 0)

- Two sinks of one class get two ledger rows (`JSONLSink`, `JSONLSink#2`):
  rows are keyed by sink OBJECT, so one failing file never starves its
  sibling. The same sink object listed twice refuses
  (`tracker_sink_duplicate`).
- A second `watch()` on a model that already has an open session under the
  same `name=`/`namespace=` refuses (`tracker_namespace_collision`); a
  distinct `name=` keeps both series distinguishable.
- Tag safety is checked for every catalogued site at attach
  (`tracker_tag_unsafe`), never first inside a step's `finally`; a typed
  drain refusal while the loop's own exception unwinds is recorded as a
  named skip and never masks that exception.
- Gradient watching over `sparse=True` modules refuses at attach
  (`watch_sparse_grad_unsupported`) instead of crashing inside
  `optimizer.step()`.
- Sinks may implement `preflight_histograms(descriptor)`; `WandbSink`
  (512-bucket cap) and `TensorBoardSink` (relay detected) refuse a
  `hist_every=` request at attach rather than latching on the first
  sampled step.
- `watch_close_empty` counts DATA points only: heartbeat, manifest, and
  check rows never make an empty run look populated. A run whose every
  step is explained by a named skip (AMP-skipped, demoted missing phase)
  is not empty.
- `JSONLSink` writes each row unbuffered and truncates a torn row on
  failure, so rows already on disk are always whole JSONL; `flush()` and
  `close()` are separate attempts at teardown.

## What each route costs you (the measured fidelity table)

Measured by the trackers panel (wandb 0.28.2, clearml 2.1.12, TB 2.21.0,
real distilgpt2 tensors, offline; ClearML measured independently twice):

| | TB native | wandb native sink | wandb relay | ClearML relay |
|---|---|---|---|---|
| Scalars | faithful | faithful | faithful | faithful |
| Caller step on scalars | preserved | preserved | REWRITTEN 0,1,2 | preserved |
| Histogram counts | exact | exact | exact (uniform bins only) | INTERPOLATED, ~75% of mass |
| Histogram edges | exact | exact | exact only for uniform bins | 2-decimal strings |
| Summary fields (min/max/num/sum/sumsq) | preserved | dropped by wandb.Histogram | DESTROYED | DESTROYED |
| Caller step on histograms | preserved | preserved | rewritten | LOST (iter=0) |
| Bucket cap | none | 512 | >512 silently re-binned | resampled to ~48 regardless |
| Graph | renders | n/a | ignored | not captured |

Consequences, wired into the code:

- **Anything you alert on or compare across runs must be a scalar.** Every
  histogram ships its scalar summary series (mean/std/norm/min/max/count/
  zero fraction, plus nonfinite counts) as first-class tags.
- **Relay ordering matters:** call `wandb.init(sync_tensorboard=True)`
  BEFORE constructing the TB writer, or nothing appears in wandb at all.
  `TensorBoardSink` detects both relays and reports `relay_configured` in
  the close report -- runtime reports never claim downstream presence
  beyond their evidence tier.
- **Histograms over a detected relay refuse typed**
  (`tracker_relay_histogram_unsupported`): the signed-log2 edges are
  non-uniform and both relays reconstruct them wrong. Use the native sink.

## Coming from `wandb.watch`

Organized as their open issues with our answers:

| wandb issue class | Stock `wandb.watch` | `torchlens.trackers.watch` |
|---|---|---|
| activation watching (#5218) | absent | module or op grain, by explicit selector |
| exact site/parameter selection (#6945) | absent | `select=` prefixes + hard budgets |
| update magnitudes (#5218) | absent | measured accepted optimizer deltas, never `lr*grad` |
| silent empty panels (#1639/#3701/#2625) | known class | typed refusals + close report + heartbeat |
| silent FSDP no-op (#9866) | silent | typed teaching refusal (deferred distributed emission) |
| relay step rewrite | measured above | caller axis preserved in both first-party sinks |
| multi-model naming (#5135/#2284) | weak | `name=` namespace component |
| unexplained slowdown | log_freq folklore | plan table printed at attach; preflight budget FACTS |

Bytes per logged step (arithmetic, not measurement): a full-tensor watcher
moves ~655 MB per step on real distilgpt2 shapes; counts+edges move
~84.5 KB -- and the transfer is independent of tensor numel (the
bounded-transfer law, tested).

## The TB graph

`build_graph_ir(trace)` turns the executed TorchLens DAG into a graph IR
(never `torch.jit.trace` -- dict inputs and data-dependent branches are
fine because the DAG is what actually ran);
`graph_ir_to_tensorboard(ir, logdir)` serializes GraphDef + per-node
StepStats through torch's own writer path (151/151 real resnet18 stats rows
bound in TB 2.21, measured). Truthful fields only: CPU host-bracket timing
with its evidence label; no CUDA timing and no allocator-memory coloring
until real sources exist -- missing evidence omits a field, never writes
zero.

## Framework callbacks

`HFTrainerWatchCallback(to=...)` and `LightningWatchCallback(to=...)` are
explicit wrappers over the same engine: the framework's global step is the
step source and its logging cadence derives the watch cadence. No
environment variable ever activates instrumentation; the off-only kill
switch `TORCHLENS_WATCH_DISABLE=1` disables collection and still writes a
`torchlens/run/disabled` row saying so.

**Migration note:** the shipped
`torchlens.callbacks.lightning.LayerProfilerCallback` is a different tool
(a per-epoch structure profiler). It re-traces under `eval()`/`no_grad()`
-- the wrong mode for dropout/BatchNorm statistics -- cannot see training
gradients, and pays an extra forward per profile. For training-time
watching, use `LightningWatchCallback`; keep `LayerProfilerCallback` only
for structure snapshots.

## Writing your own sink

The sink protocol is documented and closed over capability words
(`scalar`, `raw_histogram`, `text_manifest`, `image`, `graph`,
`run_metadata`, `structured_event`, `embedding`): implement
`capabilities()`, the `emit_*` methods for what you advertise, `flush`,
and `close`. A requested capability you do not advertise refuses by name
before any partial emission; a raise in your sink latches it failed in the
close report and can never corrupt collection. `JSONLSink` is the
reference implementation (one JSON object per line, header row first, a
`closed` footer marking complete files).
