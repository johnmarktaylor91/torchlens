# Capture speed, by kind of use

What each way of using TorchLens costs against a plain forward pass of the same model, measured
on 2026-11-03 (the `tl.extract` row on 2026-11-08). TorchLens records every operation the model runs, so most kinds of capture cost
far more than a forward hook; this page says how much, for each kind, including where TorchLens
is slow. Every number is a median wall time from the harness in `benchmarks/capture/`, and every
one comes from a result file in [`data/`](data/capture_2026-11-03.json) (one file per run; see
[Hosts, versions and method](#hosts-versions-and-method)).


The machines were shared and loaded, so absolute times are slower than on an idle machine.
Compare rows within a workload, not across workloads or hosts.

## Summary

Each cell gives the ratio to a plain forward of the same workload on the same host, then the
absolute median in brackets. For the LSTM tagger and ResNet-18 the reference is the harness's
plain-forward arm (nothing attached). The GPT-2 runs had no plain-forward arm, so their reference
is the forward-hook arm of the same run, which on the other models timed within a quarter of a
plain forward. "Not measured" means no run covered that cell; [Not measured yet](#not-measured-yet)
lists them.

| Kind of use | LSTM tagger, CPU | ResNet-18, CPU | ResNet-18, GPU | GPT-2 small, CPU (against forward hooks) | GPT-2 small, GPU (against forward hooks) |
| --- | --- | --- | --- | --- | --- |
| A forward hook on one module, no TorchLens (for reference) | 1.23x (1.75 ms) | 1.00x (13.77 ms) | 1.01x (2.32 ms) | 1.00x (the reference) | 1.00x (the reference) |
| A plain forward after a trace: TorchLens's torch wrappers stay installed, nothing is recorded | 1.02x (1.50 ms) | 1.47x (22.11 ms) | 1.04x (2.39 ms) | not measured | not measured |
| A live edit bound to the model, `tl.when(...).bind(model)`, one edit, nothing recorded | 1.81x (2.78 ms) | 1.39x (26.78 ms) | 1.57x (3.70 ms) | not measured | not measured |
| The same bound live edit over a sweep of edits | 2.1x (25.60 ms) | 1.30x (195.36 ms) | 2.2x (33.63 ms) | 1.02x (27.15 s) | not measured |
| The fast recorder, `tl.record(model, x, save=...)`: selected sites, no graph | not measured | 11x (150.06 ms) | 43x (98.55 ms) | 1.34x (39.17 s) | 50x (8.35 s) |
| A trace saving selected sites, `tl.trace(model, x, save=...)`: the full graph, few tensors kept | 55x (77.96 ms) | 37x (505.21 ms) | 169x (389.71 ms) | 3.4x (99.88 s) | 296x (49.67 s) |
| A trace saving every activation, `tl.trace(model, x)` | 55x (81.38 ms) | 47x (710.60 ms) | 186x (427.04 ms) | not measured | not measured |
| One module's output by name, `tl.extract` (remeasured on 2026-11-08 with its hooked fast path; see [its section](#one-modules-output-by-name-tlextract)) | 35x (37.57 ms): the LSTM returns a tuple, so a full capture | 1.10x (14.18 ms) | 1.21x (2.70 ms), second GPU machine | not measured | not measured |
| A sweep of edits, one recording per edit, `tl.record(..., intervene=...)` | not measured | 12x (1.79 s) | 70x (1.06 s) | 1.20x (32.32 s) | 53x (6.30 s) |
| A sweep of edits, one trace per edit, `tl.trace(..., intervene=...)` | 52x (640.23 ms) | 35x (5.26 s) | 238x (3.61 s) | 1.85x (49.97 s) | 199x (23.84 s) |
| Features over a stimulus set, `tl.extract_dataset` | 37x (356.00 ms) | 21x (2.69 s) | 241x (1.99 s), second GPU machine | not measured | not measured |
| A gradient at one module: `tl.trace` with gradients saved, then backward | 34x (117.50 ms) | 12x (980.27 ms) | 62x (551.37 ms) | not measured | not measured |
| A forward replay check, `tl.validate(model, x, scope="forward")` | not measured | 361x (5.42 s) | 1107x (2.55 s) | not measured | not measured |
| Stopping the forward early, `stop_after=` | not measured | not measured | not measured | not measured | not measured |
| A fast rerun of a trace on new inputs, `trace.run_transaction(inputs=..., fast=True)` (measured on 2026-11-05 on other workloads; see [Reruns of a finished trace](#reruns-of-a-finished-trace)) | not measured | not measured | not measured | not measured | not measured |
| A trace of metadata only, no activations saved | not measured | not measured | not measured | not measured | not measured |
| Recording a generation (measured on Pythia-70M only; see [its section](#recording-a-generation)) | not measured | not measured | not measured | not measured | not measured |

The workloads behind the cells. LSTM tagger and ResNet-18: one input; a sweep is eight steering
strengths; `tl.extract_dataset` reads one module over 16 stimuli in batches of 4; the gradient is
taken at one module output. GPT-2 small: the recorder and trace rows cache every block's output
for 256 sequences of 32 tokens in batches of 16 (16 forwards); a sweep is 12 steered forwards (3
layers by 4 strengths) on a batch of 16 sequences of 32 tokens. The ResNet-18 GPU column ran on a
workstation with an RTX 2080 Ti; its `tl.extract_dataset` cell ran on a second RTX 2080 Ti machine
after a fix (see [`tl.extract_dataset`](#features-over-a-stimulus-set-tlextract_dataset)).

Two patterns run through the table:

- **TorchLens adds a roughly fixed amount of time per recorded operation, so the ratio depends on
  how fast the plain forward is.** A full trace of ResNet-18 adds about 696 ms on the CPU worker
  (710.60 ms against 15.01 ms) and about 425 ms on the GPU workstation (427.04 ms against
  2.30 ms); the GPU forward is about 6.5 times faster, so the GPU ratio is about four times
  larger. GPT-2 small on one loaded CPU thread takes about 1.8 s per forward, so the trace's
  added cost is a small multiple there and about 300 times the forward on a GPU.
- **Editing without recording is cheap; recording is not.** The bound live edit stays within about
  two times a plain forward on every model measured. Every kind that records lands between 11 and
  241 times a plain forward on the LSTM tagger and ResNet-18, and the replay check costs more.

## A plain forward after a trace

What it measures: a forward pass with nothing recorded, run after a trace in the same process.
`import torchlens` leaves torch untouched; the first capture installs TorchLens's torch wrappers,
and they stay installed until `torchlens.backends.torch.wrappers.unwrap_torch()` (or a capture
with `CaptureOptions(unwrap_when_done=True)`) removes them. Every forward
your process runs afterwards, including code that never calls TorchLens, goes through them. This
row says what that costs.

Numbers: 1.02x on the LSTM tagger on CPU (1.50 ms against 1.47 ms), 1.04x on ResNet-18 on GPU
(2.39 ms against 2.30 ms) and 1.03x on the LSTM tagger on GPU (0.36 ms against 0.35 ms). On
ResNet-18 on the loaded CPU worker the median was 1.47x (22.11 ms against 15.01 ms), but the
lowest process medians were 15.41 ms against 14.32 ms, and two identical plain-forward arms on
that worker differed by 54%, so this cell is mostly the worker's noise.

Known costs: the one-time cost of the first capture in a process (installing the wrappers) is not
in any row here; every arm ran 3 warmup calls first.

## Live edits at hook speed

What it does: `tl.when(site, edit).bind(model)` applies an edit inside the model's own forward and
records nothing. Each timed call builds the bound edit and runs it, the same way the hook arm
registers its hook, runs, and removes it.

When to use it: steering or patching over many forwards, or through generation, when you do not
need a record of each forward.

Numbers: within about two times a plain forward on every model measured (see the summary). The
first run's rows (LSTM tagger on CPU, and GPT-2 small on CPU):

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| GPT-2 small, CPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 26.70 s |
| GPT-2 small, CPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.when(...).bind(model)` | 27.15 s |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | plain forward, nothing attached | 1.54 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | forward hook | 1.70 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | `tl.when(...).bind(model)` | 2.78 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | nnsight 0.8.1 | 2.06 ms |
| LSTM tagger, CPU, plain forward after a trace with TorchLens's torch wrappers still installed, n=2 processes of 20 samples | plain forward after a trace, nothing recorded | 1.50 ms |
| LSTM tagger, CPU, plain forward after a trace with TorchLens's torch wrappers still installed, n=2 processes of 20 samples | plain forward | 1.47 ms |

On GPT-2 small the bound edit and the forward hook land 0.46 s apart on a 27 s sweep, inside the
spread of three repeats (hooks 25.36 to 28.00 s, bound edit 26.80 to 27.74 s), and the steered
logits are identical (largest absolute difference 0.0). On the LSTM tagger, whose plain forward
takes about 1.5 ms, the bound edit's fixed setup cost per call is visible. A plain forward after a
trace, with TorchLens's wrappers still installed and nothing recorded, runs at plain-forward speed.

Known costs: a bound edit records nothing, so it cannot give you activations or gradients; pair it
with a hook or a capture for those.

## The fast recorder, `tl.record`

What it does: `tl.record(model, x, save=...)` runs the forward through TorchLens's wrapped
operations and keeps events and payloads for the selected sites, with no graph postprocessing. It
returns a `Recording`; call `Recording.to_trace()` later if you need the graph.

When to use it: pulling activations from many forwards in a loop, when you know the sites.

Numbers: 11x a plain forward on ResNet-18 on CPU (150.06 ms), 43x on GPU (98.55 ms), 50x on the
LSTM tagger on GPU (18.16 ms against 0.36 ms), and 50x on GPT-2 small on GPU (8.35 s against
168 ms for forward hooks). It costs a fraction of a trace of the same sites: 3.4 times less on
ResNet-18 on CPU (150.06 ms against 505.21 ms), 4.0 times less on GPU (98.55 ms against
389.71 ms) and about 6 times less on GPT-2 small on GPU (8.35 s against 49.67 s).

Known costs: it still runs every operation through the wrappers, so it stays well above a forward
hook. Backward capture needs a trace; these runs did not time `tl.record` with gradients.

## Traces: selected sites or every activation

What it does: `tl.trace(model, x)` records every operation's metadata and graph edges, checks
them, and saves every activation. `tl.trace(model, x, save=...)` records the same graph and keeps
only the selected payloads.

When to use it: you need operations no module returns, you do not yet know the site, you want the
graph, or you want the recording checked against the forward.

Numbers: 37x to 55x a plain forward on the LSTM tagger and ResNet-18 on CPU, 169x to 186x on
ResNet-18 on GPU, and 296x on GPT-2 small on GPU; on GPT-2 small on one loaded CPU thread, where
each forward takes about 1.8 s, 3.4x (see the summary).

Saving only selected sites keeps fewer tensors but takes about the same time, because the
structure of every operation is still recorded: on ResNet-18 on GPU a full trace took 427.04 ms
and a trace keeping one module 381.29 ms; on the LSTM tagger on CPU 81.38 ms and 83.49 ms.

Known costs: the trace's fixed cost dominates on small models (the LSTM tagger's 1.5 ms forward
becomes about 80 ms).

## One module's output by name, `tl.extract`

What it does: `tl.extract(model, x, ["layer2"])` returns the outputs of the named modules. When
every name is an exact module address, the model is in eval mode, the input is one tensor (or text)
on the model's device and each named module runs once and returns one tensor, it runs one plain
forward with forward hooks on those modules and records nothing. Otherwise, for example for a
module that returns a tuple, it runs a full capture and reads the outputs from it.

When to use it: reading a few module outputs by name, without writing hooks.

Numbers (remeasured on 2026-11-08 on the CPU worker, after the hooked fast path landed): 1.10x a
plain forward on ResNet-18 (14.18 ms against 12.90 ms), with outputs bit-identical to the forward
hook's. The LSTM tagger's module returns a tuple (its output and its hidden state), so it takes the
full capture: 35x (37.57 ms against 1.06 ms). On ResNet-18 on GPU (remeasured on 2026-11-08 on the
second RTX 2080 Ti machine) it is 1.21x a plain forward (2.70 ms against 2.24 ms; the forward hook
took 2.30 ms), again bit-identical to the hook's output. The 3 Nov GPU figure (376.36 ms, 164x) was
a full capture and stays in [All
measurements](#resnet-18-on-gpu-second-run-rtx-2080-ti-workstation) as history.

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, CPU, one module's output (`layer2`), 8 Nov, n=3 processes | plain forward, nothing attached | 12.90 ms (12.78 ms to 13.74 ms) |
| ResNet-18, CPU, one module's output (`layer2`), 8 Nov, n=3 processes | forward hook | 12.96 ms (12.79 ms to 13.02 ms) |
| ResNet-18, CPU, one module's output (`layer2`), 8 Nov, n=3 processes | `tl.extract` | 14.18 ms (13.95 ms to 14.18 ms) |
| ResNet-18, GPU (second RTX 2080 Ti machine), one module's output (`layer2`), 8 Nov, n=3 processes | plain forward, nothing attached | 2.24 ms (2.24 ms to 2.83 ms) |
| ResNet-18, GPU (second RTX 2080 Ti machine), one module's output (`layer2`), 8 Nov, n=3 processes | forward hook | 2.30 ms (2.29 ms to 2.33 ms) |
| ResNet-18, GPU (second RTX 2080 Ti machine), one module's output (`layer2`), 8 Nov, n=3 processes | `tl.extract` | 2.70 ms (2.70 ms to 2.71 ms) |
| LSTM tagger, CPU, one module's output (`lstm`), 8 Nov, n=3 processes | plain forward, nothing attached | 1.06 ms (1.06 ms to 1.06 ms) |
| LSTM tagger, CPU, one module's output (`lstm`), 8 Nov, n=3 processes | forward hook | 1.07 ms (1.06 ms to 1.07 ms) |
| LSTM tagger, CPU, one module's output (`lstm`), 8 Nov, n=3 processes | `tl.extract` (a full capture: the module returns a tuple) | 37.57 ms (37.16 ms to 38.96 ms) |

Known costs: a module that returns a tuple or a dict, runs twice, or never runs takes the full
capture, as does a model in training mode or an input on another device.

## Reruns of a finished trace

Measured on 2026-11-05, on a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average 14 to 19. Python 3.10, torch 2.13.0 (CPU build), transformers 5.18.0,
and a development build of TorchLens (2.38.1) from that day. Raw data:
[`data/capture_2026-11-05-rerun.json`](data/capture_2026-11-05-rerun.json). Each LSTM tagger and
ResNet-18 row is three processes (GPT-2 two); each process times up to 15 samples after 3 warmup
calls (GPT-2: up to 5 after 1) and reports its median, and the table gives the median of the
process medians with the lowest and highest in brackets. The GPT-2 small rows use the GPT-2 small
architecture with random weights (no download). The timing script is not yet part of
`benchmarks/`. The forward-hook rows ran
in a second job on the same worker; each one names that job's own plain forward.

What it does: `trace.run(model, new_x)` reruns the model on new input and returns a new trace with
the same saved sites. On a trace that saved a selection of sites (`save=tl.module(...)` or
`tl.func(...)`) it takes the guarded fast rerun, the model's native forward with the staged hooks, if
any, writing the saved sites into a new trace, with or without a staged edit; a trace that saved
every activation recaptures, and when a guard refuses (a changed call structure, a value
replacement, a non-module target) it falls back to recapturing the whole forward, with the reason in
`last_run["fast_refused"]`. The tables below were measured on the earlier build, in which an
unedited `trace.run` recaptured and then copied each operation's state into the new trace, costing
1.69 times a fresh `tl.trace` on ResNet-18, 1.57 times on GPT-2 small and 1.53 times on the LSTM
tagger; the fast engine's timings are in the next subsection. `trace.run_transaction(inputs=new_x,
fast=True)` takes the same engine whether or not an edit is staged, as long as the capture saved a
selection, and raises instead of falling back.

When to use it: to read the same sites on many new inputs, or to rerun a staged edit,
`run_transaction(inputs=new_x, fast=True)` costs 1.3 to 3 times a plain forward on these models.

Correctness: the fast rerun with a staged edit matched a plain forward with the same edit applied
by a forward hook (largest absolute difference 0 on GPT-2 small and ResNet-18, 1.9e-08 on the LSTM
tagger), and the unedited fast rerun's saved site matched a fresh `tl.trace` of the new input
exactly on all three models.

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, CPU, one 64x64 image, n=3 processes | plain forward, nothing attached | 14.20 ms (13.30 ms to 14.26 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | forward hook on the saved module (second job; its plain forward: 14.39 ms) | 14.56 ms (14.20 ms to 18.52 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 405.28 ms (395.31 ms to 408.93 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | fresh `tl.trace` saving every listed module | 417.94 ms (402.45 ms to 665.18 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run(model, new_x)` on a finished trace (earlier build: recaptures) | 684.49 ms (622.50 ms to 693.68 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run_transaction(inputs=new_x)` | 855.95 ms (849.77 ms to 898.79 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 18.85 ms (18.19 ms to 19.18 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | fresh `tl.trace(..., intervene=tl.when(...))` | 452.97 ms (444.02 ms to 521.99 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run(model, new_x)` on a trace with a staged edit | 49.66 ms (48.54 ms to 56.45 ms) |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)` with a staged edit | 19.36 ms (19.35 ms to 22.06 ms) |

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | plain forward, nothing attached | 1.64 s (1.63 s to 1.65 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | forward hook on the saved module (second job; its plain forward: 1.63 s) | 1.34 s (1.33 s to 1.35 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 2.99 s (2.91 s to 3.07 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | fresh `tl.trace` saving every listed module | 4.28 s (4.21 s to 4.34 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run(model, new_x)` on a finished trace (earlier build: recaptures) | 4.71 s (4.65 s to 4.77 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run_transaction(inputs=new_x)` | 7.86 s (7.71 s to 8.01 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 1.33 s (1.33 s to 1.34 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | fresh `tl.trace(..., intervene=tl.when(...))` | 3.17 s (3.05 s to 3.29 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run(model, new_x)` on a trace with a staged edit | 1.41 s (1.38 s to 1.45 s) |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run_transaction(inputs=new_x, fast=True)` with a staged edit | 1.32 s (1.28 s to 1.37 s) |

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | plain forward, nothing attached | 1.08 ms (1.03 ms to 1.46 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | forward hook on the saved module (second job; its plain forward: 1.07 ms) | 1.11 ms (1.08 ms to 1.13 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 43.16 ms (42.24 ms to 44.64 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | fresh `tl.trace` saving every listed module | 43.95 ms (43.04 ms to 45.28 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run(model, new_x)` on a finished trace (earlier build: recaptures) | 66.08 ms (65.84 ms to 66.79 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run_transaction(inputs=new_x)` | 82.67 ms (80.72 ms to 82.98 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 3.11 ms (3.09 ms to 3.23 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | fresh `tl.trace(..., intervene=tl.when(...))` | 51.34 ms (47.44 ms to 53.41 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run(model, new_x)` on a trace with a staged edit | 8.89 ms (8.74 ms to 14.64 ms) |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)` with a staged edit | 3.32 ms (3.30 ms to 3.75 ms) |

Saving more sites on the rerun, `trace.run_transaction(inputs=new_x, save=[...module addresses])`,
took about as long as the same saves, measured on a second worker (two processes each; plain
forward 10.61 and 10.73 ms on ResNet-18): ResNet-18
637.45 and 731.43 ms against
692.01 and 719.65 ms; LSTM tagger
58.84 and 58.65 ms against
60.71 and 60.09 ms. `trace.run` takes no `save=`.

Known costs: on GPT-2 small the plain-forward arm timed slower than the forward-hook and fast
arms in both jobs, and the cause was not found; compare GPT-2 rows with the fresh trace rather
than the plain forward. A `save=` selector such as `tl.module(...)` on `run_transaction` is
refused in this build (module addresses work). The first capture in a process took 1.7 to 2.4 s
on ResNet-18, 1.3 to 2.7 s on the LSTM tagger and 4.9 to 5.5 s on GPT-2 small (one call each,
installing the torch wrappers).

### `trace.run` on new inputs with the fast engine (development build)

A development build of TorchLens leading to 3.0 sends `trace.run(model, new_x)` on an
unedited trace that saved a selection through the guarded fast engine instead of a full recapture.
Both builds were timed in two jobs that ran at the same time on the same shared 32-core Intel Xeon
2.80 GHz Linux CPU worker (one torch thread per process, load average 6 to 15), with the same
models, inputs, repeats and saved site as the tables above. The plain forward timed differently in
the two jobs, so compare the "times plain forward" columns, not the milliseconds across columns.

Correctness: the run reported engine `rerun` before and `guarded_fast` after on every call, with no
fallback. Before and after, the saved site of the new trace equalled a fresh `tl.trace` of the new
input exactly (largest absolute difference 0) on all three models, and the new trace's output
matched a plain forward (0 on ResNet-18 and GPT-2 small, 3.0e-08 on the LSTM tagger).

After the change, `trace.run` costs less than a fresh `tl.trace` on every model (0.11 times on
ResNet-18, 0.51 on GPT-2 small, 0.16 on the LSTM tagger) but still more than
`run_transaction(inputs=new_x, fast=True)` on the two small models (about 28 ms more on ResNet-18
and 7 ms more on the LSTM tagger per call).

| Workload (model, device, repeats) | Tool | Before: median (lowest to highest) | Before: times plain forward | After: median (lowest to highest) | After: times plain forward |
| --- | --- | --- | --- | --- | --- |
| ResNet-18, CPU, one 64x64 image, n=3 processes | plain forward, nothing attached | 22.68 ms (20.90 ms to 23.02 ms) | 1.00x | 14.42 ms (14.28 ms to 15.26 ms) | 1.00x |
| ResNet-18, CPU, one 64x64 image, n=3 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 508.54 ms (490.37 ms to 709.94 ms) | 22.4x | 442.97 ms (423.87 ms to 442.99 ms) | 30.7x |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run(model, new_x)`, no edit staged | 651.95 ms (649.85 ms to 1.08 s) | 28.7x | 47.48 ms (46.99 ms to 48.89 ms) | 3.29x |
| ResNet-18, CPU, one 64x64 image, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 19.70 ms (18.89 ms to 29.97 ms) | 0.87x | 19.18 ms (19.15 ms to 19.80 ms) | 1.33x |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | plain forward, nothing attached | 1.29 s (1.28 s to 1.30 s) | 1.00x | 1.32 s (1.31 s to 1.34 s) | 1.00x |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 2.91 s (2.86 s to 2.96 s) | 2.26x | 2.95 s (2.87 s to 3.03 s) | 2.23x |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run(model, new_x)`, no edit staged | 4.28 s (4.28 s to 4.29 s) | 3.32x | 1.50 s (1.47 s to 1.54 s) | 1.13x |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, n=2 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 1.45 s (1.45 s to 1.45 s) | 1.12x | 1.47 s (1.47 s to 1.47 s) | 1.11x |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | plain forward, nothing attached | 1.04 ms (1.04 ms to 1.07 ms) | 1.00x | 1.67 ms (1.62 ms to 1.67 ms) | 1.00x |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | fresh `tl.trace(model, x, save=tl.module(site))` | 42.45 ms (42.11 ms to 43.35 ms) | 40.7x | 70.56 ms (67.46 ms to 71.20 ms) | 42.4x |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run(model, new_x)`, no edit staged | 64.86 ms (64.10 ms to 90.98 ms) | 62.2x | 11.58 ms (10.93 ms to 12.07 ms) | 6.96x |
| LSTM tagger, CPU, one sequence of 32 tokens, n=3 processes | `trace.run_transaction(inputs=new_x, fast=True)`, no edit staged | 3.00 ms (2.97 ms to 3.04 ms) | 2.88x | 4.66 ms (4.45 ms to 4.85 ms) | 2.80x |

### Stopping a rerun early

`trace.run_transaction(inputs=new_x, fast=True, until=<saved module>)` stops the fast rerun once
that module has run. The stopped module's output equalled a fresh `tl.trace` of the new input
exactly on all three models. Without `fast=True` the same `until=` still costs about a full trace.
Same worker and build as above; each row is the median of up to 15 samples (GPT-2: 5) in one
process.

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| ResNet-18, CPU, one 64x64 image, one process | plain forward, nothing attached | 13.65 ms |
| ResNet-18, CPU, one 64x64 image, one process | `trace.run_transaction(inputs=new_x, fast=True)`, whole forward | 20.30 ms |
| ResNet-18, CPU, one 64x64 image, one process | the same with `until="layer1"` | 6.16 ms |
| ResNet-18, CPU, one 64x64 image, one process | the same with `until="layer2"` | 8.43 ms |
| ResNet-18, CPU, one 64x64 image, one process | `trace.run_transaction(inputs=new_x, until="layer1")`, without `fast=True` | 434.31 ms |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, one process | plain forward, nothing attached | 1.62 s |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, one process | `trace.run_transaction(inputs=new_x, fast=True)`, whole forward | 1.51 s |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, one process | the same with `until="transformer.h.1"` | 161.36 ms |
| GPT-2 small (random weights), CPU, one forward of 16 sequences of 32 tokens, one process | `trace.run_transaction(inputs=new_x, until="transformer.h.1")`, without `fast=True` | 3.40 s |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | plain forward, nothing attached | 1.01 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | `trace.run_transaction(inputs=new_x, fast=True)`, whole forward | 3.36 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | the same with `until="embed"` | 0.77 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | `trace.run_transaction(inputs=new_x, until="embed")`, without `fast=True` | 51.97 ms |

### Replaying a saved trace without the model

A trace captured with `CaptureOptions(replayable=True)` and saved with
`trace.save(path, level="runnable", include_weights=True)` can be loaded with `tl.load(path)` and
run on new input with `loaded.run_transaction(inputs=new_x)`, with no model object. Without
`include_weights=True` the file holds no weights and the replay runs on a fresh random
initialization (TorchLens warns `runnable_random_init_run`). With the weights, the replay's
output equalled `model(new_x)` (largest absolute difference 0 on ResNet-18 and GPT-2 small, 3e-08
on the LSTM tagger), and every run reported path faithfulness `verified` with the state taken from
the file. Measured on 2026-11-05 on a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average 14 to 21, same build as above, in one process per model (median
of up to 15 samples, GPT-2 up to 5). The worker was busier than for the rows above, so compare
these rows with each other only.

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| ResNet-18, CPU, one 64x64 image, one process | plain forward, nothing attached | 20.47 ms |
| ResNet-18, CPU, one 64x64 image, one process | fresh `tl.trace(model, x)` | 694.92 ms |
| ResNet-18, CPU, one 64x64 image, one process | fresh capture with `CaptureOptions(replayable=True)` | 3.37 s |
| ResNet-18, CPU, one 64x64 image, one process | replay of the loaded runnable file on new input, `loaded.run_transaction(inputs=new_x)` | 583.57 ms |
| GPT-2 small (random weights, no key-value cache), CPU, one forward of 16 sequences of 32 tokens, one process | plain forward, nothing attached | 2.34 s |
| GPT-2 small (random weights, no key-value cache), CPU, one forward of 16 sequences of 32 tokens, one process | fresh `tl.trace(model, x)` | 5.27 s |
| GPT-2 small (random weights, no key-value cache), CPU, one forward of 16 sequences of 32 tokens, one process | fresh capture with `CaptureOptions(replayable=True)` | 17.56 s |
| GPT-2 small (random weights, no key-value cache), CPU, one forward of 16 sequences of 32 tokens, one process | replay of the loaded runnable file on new input, `loaded.run_transaction(inputs=new_x)` | 10.40 s |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | plain forward, nothing attached | 1.59 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | fresh `tl.trace(model, x)` | 66.93 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | fresh capture with `CaptureOptions(replayable=True)` | 290.84 ms |
| LSTM tagger, CPU, one sequence of 32 tokens, one process | replay of the loaded runnable file on new input, `loaded.run_transaction(inputs=new_x)` | 58.86 ms |

One-time costs per model: ResNet-18 save with weights 5.08 s (a 46.3 MiB file) and load 3.31 s;
GPT-2 small 26.85 s (627.8 MiB) and 14.42 s; LSTM tagger 1.15 s (1.7 MiB) and 498 ms. The replay
costs about what a fresh trace does on ResNet-18 and the LSTM tagger (0.84 and 0.88 times) and
about twice a fresh trace on GPT-2 small; the replayable capture itself is 4 to 5 times a default
trace. GPT-2 small saves only with `use_cache=False`: with the default key-value cache the save is
refused (`output_container_contract_missing`).

## Edit sweeps with a recording per edit

What it does: one `tl.record(..., intervene=tl.when(...))` or one
`tl.trace(..., intervene=tl.when(...))` per edit, each returning the edited output.

When to use it: you need a record of each edited forward (activations, a graph, a check). If you
only need the edited outputs, the bound live edit is 9 to 107 times faster on the LSTM tagger and
ResNet-18. On GPT-2 small on CPU, where each forward takes seconds, the gap is under two times.

Numbers: on ResNet-18 on CPU, eight strengths took 195.36 ms bound, 1.79 s with a recording per
strength and 5.26 s with a trace per strength; on GPU 33.63 ms, 1.06 s and 3.61 s.

## Features over a stimulus set, `tl.extract_dataset`

What it does: runs a trace per batch and gathers the requested module's output across batches,
in stimulus order.

Numbers: 37x a plain loop over the batches on the LSTM tagger on CPU (356.00 ms against 9.69 ms),
21x on ResNet-18 on CPU (2.69 s against 129.86 ms) and 241x on ResNet-18 on GPU (1.99 s against
8.25 ms). On the same GPU machine one `tl.record` per batch took 428.00 ms, about a fifth.

Known costs: the capture cost is paid per batch, so large batches spread it. On GPU the first run
failed with inputs on two devices; the cause was fixed the same day and the GPU cell above is the
fixed run, on a second RTX 2080 Ti machine.

## Gradients

What it does: `tl.trace(..., capture=CaptureOptions(save_grads=...))` then `trace.backward(...)`
records the forward graph and the gradient at the selected module.

Numbers: 34x a plain forward and backward on the LSTM tagger on CPU (117.50 ms against 3.49 ms),
12x on ResNet-18 on CPU (980.27 ms against 80.91 ms) and 62x on GPU (551.37 ms against 8.88 ms).
A forward hook plus a tensor hook took 70.31 ms on ResNet-18 on CPU, and Captum's layer gradient
39.27 ms.

Known costs: the LSTM tagger's gradient cells could not run on GPU for any tool (cuDNN's RNN
backward needs training mode, and the harness runs in eval mode).

## Operations on a finished trace

The same runs timed what you do with a trace after capture. Medians of three processes, from the
same result files:

| Operation | ResNet-18, CPU | ResNet-18, GPU | LSTM tagger, GPU |
| --- | --- | --- | --- |
| Draw the whole graph to an SVG file, `trace.draw(...)` | 256.97 ms | 234.26 ms | 67.78 ms |
| Save a portable trace file and load it back | 3.22 s | 3.40 s | 187.84 ms |
| Forward replay check, `tl.validate(model, x, scope="forward")` | 5.42 s | 2.55 s | 372.44 ms |
| Trace clean and ablated forwards, then align them in a `tl.Bundle` supergraph | 1.33 s | 986.80 ms | 79.72 ms |

The one-call `tl.summary(model, x)` is measured separately, with its own tool, on
[Summary performance](summary_performance.md).

## Recording a generation

Recording a generation keeps a record of every decode step. Measured on 2026-11-03 on
the same 32-core Intel Xeon worker, four torch threads, one process per token count, on a
development build of TorchLens from that day, with a 70-million-parameter GPT-NeoX-style model
(`EleutherAI/pythia-70m`, float32), greedy decoding from a 7-token prompt, and the output of
every block's MLP and of the final layer norm saved at every step. Each row is one call of
`tl.record(model, ids, save=sites, generate_kwargs={"max_new_tokens": n, "min_new_tokens": n,
"do_sample": False, "use_cache": True})`, where `sites` joins `tl.module(...)` for each saved
module with `|`. One process per row, no repeats. Peak process memory is the operating system's
maximum resident set size; saved activations are the summed saved bytes of every step's record.
No run timed a plain `model.generate(...)` for comparison. Raw data:
[`data/capture_2026-11-03-generation.json`](data/capture_2026-11-03-generation.json) and
[`data/capture_2026-11-03-gpu2.json`](data/capture_2026-11-03-gpu2.json).

Current behaviour: process memory grows linearly, about 2.7 MiB per token, and the time per token
stays flat (the 500- and 1,000-token rows from the fix's own check, the 2,000-token row from the
third run):

| Generated tokens | Wall time per token | Peak process memory (RSS) | Saved activations at the end |
| --- | --- | --- | --- |
| 500 | 356 ms | 2.1 GiB | 6.9 MiB |
| 1,000 | 395 ms | 3.4 GiB | 13.8 MiB |
| 2,000 | 397 ms | 6.2 GiB | 27.4 MiB |

The same 2,000-token run on the RTX 2080 Ti node (four torch threads) took 269 ms per token, peaked
at 6.8 GiB of process memory and 458 MiB of GPU memory, and saved the same 27.4 MiB of activations.

Before a fix on 2026-11-03 the recorder kept every decode pass's whole output, including the
model's key-value cache, so retained memory grew with the square of the generated tokens and the
time per token grew with the run:

| Generated tokens | Wall time per token | Peak process memory (RSS) | Saved activations at the end |
| --- | --- | --- | --- |
| 500 | 857 ms | 5.3 GiB | 6.9 MiB |
| 1,000 | 1,336 ms | 15.7 GiB | 13.8 MiB |
| 2,000 | not reached | over 17 GiB (stopped after 17 min) | not reached |

The saved activations were a small part of that growth: 6.9 MiB of saved tensors at 500 tokens
against 5.3 GiB of process memory; 13.8 MiB of saved tensors at 1,000 tokens against 15.7 GiB.

For long generations, record a shorter run or save fewer sites.

## Where TorchLens is slower

A capture records every operation's metadata and checks it, so it costs much more than a hook
that reads one tensor. On a small model the capture's fixed cost dominates. The first run's rows
(LSTM tagger on CPU):

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| LSTM tagger, CPU, full trace of every operation, n=2 processes of 20 samples | `tl.trace(model, x)` | 81.38 ms |
| LSTM tagger, CPU, full trace of every operation, n=2 processes of 20 samples | forward hooks on every module | 1.61 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | `tl.trace(..., save=tl.module(site))` | 77.96 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | `tl.extract` | 82.93 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | forward hook | 1.75 ms |
| LSTM tagger, CPU, eight steering strengths, n=3 processes of 20 samples | `tl.trace(..., intervene=...)` per strength | 640.23 ms |
| LSTM tagger, CPU, eight steering strengths, n=3 processes of 20 samples | `tl.when(...).bind(model)` | 25.60 ms |
| LSTM tagger, CPU, eight steering strengths, n=3 processes of 20 samples | forward hooks | 12.52 ms |
| LSTM tagger, CPU, one module's output over 16 stimuli in batches of 4, n=2 processes of 20 samples | `tl.extract_dataset` | 356.00 ms |
| LSTM tagger, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes of 20 samples | forward hook | 10.21 ms |
| LSTM tagger, CPU, gradient at one module output, n=3 processes of 20 samples | `tl.trace` with gradients saved | 117.50 ms |
| LSTM tagger, CPU, gradient at one module output, n=3 processes of 20 samples | tensor hook | 4.40 ms |

What this means in practice:

- To read one module's output, `tl.extract` costs about a forward hook when the module returns
  one tensor (1.10x a plain forward on ResNet-18 on CPU and 1.21x on GPU, 8 Nov). A module that returns a tuple,
  like the LSTM, takes a full capture, and a forward hook is the faster tool there. A capture is
  worth its cost when you need operations no module returns, do not yet know the site, or want
  the recording checked against the forward.
- To sweep many edits, use the bound edit (`tl.when(...).bind(model)`), not a trace per edit.
- To pull selected activations in a loop, use `tl.record`, not `tl.trace`.
- Batch feature extraction pays the capture cost per batch. Large batches spread it.
- Saving only selected sites (`save=`) keeps fewer tensors, not less time: the structure of every
  operation is still recorded, which is why the one-module trace costs about what the full trace
  does.
- On GPU the ratios are larger, because the plain forward is faster and TorchLens's added time
  is not.
- torchvision's fx feature extraction is faster than a plain forward of ResNet-18 for one module,
  because it builds a graph that stops at that module.
- To time a model itself, locate the slow part with TorchLens, then measure it clean with
  `torch.utils.benchmark.Timer`.

## Not measured yet

- Stopping a fresh capture early with `stop_after=` (stopping a fast rerun early is in
  [Stopping a rerun early](#stopping-a-rerun-early)).
- A fast rerun of a trace on new inputs on the 3 Nov workloads (the 5 Nov reruns are in
  [Reruns of a finished trace](#reruns-of-a-finished-trace)).
- A trace of metadata only, saving no activations.
- `tl.record` on the LSTM tagger on CPU, and every kind except the recorder, the trace and the
  sweeps on GPT-2 small (the GPT-2 runs used their own workloads and had no plain-forward arm).
- The one-time cost of the first capture in a process.
- A plain `model.generate(...)` as the reference for recording a generation, and generation past
  2,000 tokens.
- fx on the LSTM tagger: its data-dependent control flow cannot be traced by fx.
- Gradient arms on the LSTM tagger on GPU: cuDNN's RNN backward needs training mode.

## Hosts, versions and method

Each run's result file names its host and versions; the page's tables quote the files' medians.

- First run (LSTM tagger on CPU; GPT-2 small live edits on CPU): a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average about 40.
  Python 3.10, torch 2.13.0 (CPU build), transformers 5.18.0, TorchLens 2.36.3, nnsight 0.8.1.
  Raw data: [`data/capture_2026-11-03.json`](data/capture_2026-11-03.json).
- Second run, CPU rows (ResNet-18 and GPT-2 small): a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average 7 to 28.
  Versions as in the first run, with torchvision 0.28.0 and Captum 0.9.0. Raw data:
  [`data/capture_2026-11-03-run2.json`](data/capture_2026-11-03-run2.json).
- Second run, GPU rows (ResNet-18): a 20-core Intel Core i9-9900X 3.50 GHz Linux workstation with an NVIDIA GeForce RTX 2080 Ti, one torch thread per process, load average about 10.
  Versions as above except torch 2.13.0 and torchvision 0.28.0 built for CUDA 12.6. Raw data:
  [`data/capture_2026-11-03-gpu.json`](data/capture_2026-11-03-gpu.json).
- Third run, GPT-2 small and LSTM tagger on GPU: a 256-core AMD EPYC 7662 Linux GPU node with an NVIDIA L40 (46 GiB), one torch thread per process, load average 56 to 74 (a shared node).
  Raw data: [`data/capture_2026-11-03-gpu3.json`](data/capture_2026-11-03-gpu3.json).
- Third run, `tl.extract_dataset` on GPU and the GPU generation run: a 48-core Intel Xeon Gold 6126 2.60 GHz Linux GPU node with an NVIDIA GeForce RTX 2080 Ti (11 GiB), one torch thread per process, load average 1 to 3.
  Raw data: [`data/capture_2026-11-03-gpu2.json`](data/capture_2026-11-03-gpu2.json).
- Third run versions on both GPU nodes: Python 3.10, torch 2.13.0 built for CUDA 12.6,
  transformers 5.18.0, nnsight 0.8.1, TransformerLens 4.3.0, and a development build of TorchLens
  from that day.
- Reruns of a finished trace (2026-11-05): see [their section](#reruns-of-a-finished-trace).
- `tl.extract` remeasured (2026-11-08): a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average 6 to 7.
  Python 3.10, torch 2.13.0 (CPU build), torchvision 0.28.0 and a development build of TorchLens
  from that day, with the hooked fast path. Raw data:
  [`data/capture_2026-11-08-extract.json`](data/capture_2026-11-08-extract.json).
- `tl.extract` on GPU remeasured (2026-11-08): a 48-core Intel Xeon Gold 6126 2.60 GHz Linux GPU node with an NVIDIA GeForce RTX 2080 Ti (11 GiB), the third run's `tl.extract_dataset` machine, one torch thread per process, load average about 4.
  Python 3.11, torch 2.14.1 and torchvision 0.29.1 built for CUDA 12.6, TorchLens 3.0.3. Raw data:
  [`data/capture_2026-11-08-extract-gpu.json`](data/capture_2026-11-08-extract-gpu.json).
- Recording a generation: see [its section](#recording-a-generation).

Method. Each LSTM tagger and ResNet-18 row is three separate processes (two for some first-run
rows, as each row says); each process times up to 20 samples after 3 warmup calls (as few as 3 for
the slowest TorchLens cells, which reach the 20 s budget first) and reports its median. The tables
give the median of the process medians, and in brackets the lowest and highest. Each GPT-2 row is
one process that runs the whole workload three times; the tables give the median of the three
wall times and, in brackets, the fastest and slowest. The CPU GPT-2 rows come from two runs at
different times on the shared worker, and its load changed between them, so each run carries its
own forward-hook row: compare a tool with the hook row of the same run. Every GPU arm synchronises
the GPU before it stops its clock, and each GPU job had one GPU to itself.

Each tool's output was compared with the forward-hook arm's output and matched: bit for bit for
every TorchLens and nnsight arm on ResNet-18, and with a largest absolute difference of 0.0 on
GPT-2. The CPU worker was loaded while the second run's rows ran: two identical plain-forward arms
timed on it differed by 54%, against 5% on the GPU workstation. TransformerLens runs its own GPT-2
implementation loaded with the same weights, so its rows time a different model code path with
hooks on `hook_resid_post`; every other GPT-2 row runs the transformers model.

## All measurements

Every row the summary draws on, by run, with the peer tools measured beside TorchLens.

### GPT-2 small on CPU (second run)

| Workload (model, device, repeats) | Tool | Median (fastest to slowest) |
| --- | --- | --- |
| GPT-2 small, CPU, first run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 26.44 s (26.24 to 26.59 s) |
| GPT-2 small, CPU, first run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | nnsight 0.8.1 | 26.92 s (26.43 to 27.40 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 29.30 s (28.76 to 30.95 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.record` | 39.17 s (39.05 to 40.18 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 99.88 s (93.72 to 106.37 s) |
| GPT-2 small, CPU, first run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 20.50 s (19.86 to 20.67 s) |
| GPT-2 small, CPU, first run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | nnsight 0.8.1 | 20.73 s (20.20 to 22.00 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 26.96 s (26.48 to 27.50 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.record` | 32.32 s (27.23 to 35.98 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 49.97 s (49.80 to 59.79 s) |
| GPT-2 small, CPU, first run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 7.07 s (6.96 to 7.24 s) |
| GPT-2 small, CPU, first run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | nnsight 0.8.1 | 7.24 s (7.16 to 7.36 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 6.62 s (6.39 to 6.80 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.record` | 43.85 s (40.75 to 47.59 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 143.45 s (116.07 to 165.04 s) |

### ResNet-18 on CPU (second run)

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | plain forward, nothing attached | 13.74 ms (13.69 ms to 19.94 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | forward hook | 13.77 ms (13.51 ms to 20.05 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | nnsight 0.8.1 | 15.24 ms (14.81 ms to 20.51 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | torchvision fx `create_feature_extractor` | 4.79 ms (4.62 ms to 7.86 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | `tl.record(..., save=tl.module(site))` | 150.06 ms (141.98 ms to 154.78 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | `tl.trace(..., save=tl.module(site))` | 505.21 ms (482.66 ms to 776.78 ms) |
| ResNet-18, CPU, one module's output (`layer2`), n=3 processes | `tl.extract` | 562.88 ms (519.58 ms to 758.70 ms) |
| ResNet-18, CPU, one module replaced by zeros, n=3 processes | forward hook | 14.21 ms (13.30 ms to 21.36 ms) |
| ResNet-18, CPU, one module replaced by zeros, n=3 processes | nnsight 0.8.1 | 15.83 ms (15.66 ms to 16.31 ms) |
| ResNet-18, CPU, one module replaced by zeros, n=3 processes | `tl.when(...).bind(model)` | 26.78 ms (18.16 ms to 27.01 ms) |
| ResNet-18, CPU, one module replaced by zeros, n=3 processes | `tl.record(..., intervene=tl.when(...))` | 167.86 ms (159.73 ms to 246.03 ms) |
| ResNet-18, CPU, eight steering strengths, n=3 processes | forward hooks | 152.54 ms (150.13 ms to 168.64 ms) |
| ResNet-18, CPU, eight steering strengths, n=3 processes | nnsight 0.8.1 | 130.62 ms (120.35 ms to 182.87 ms) |
| ResNet-18, CPU, eight steering strengths, n=3 processes | `tl.when(...).bind(model)` | 195.36 ms (144.42 ms to 196.22 ms) |
| ResNet-18, CPU, eight steering strengths, n=3 processes | `tl.record(..., intervene=...)` per strength | 1.79 s (1.52 s to 2.31 s) |
| ResNet-18, CPU, eight steering strengths, n=3 processes | `tl.trace(..., intervene=...)` per strength | 5.26 s (4.58 s to 5.57 s) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 130.10 ms (127.56 ms to 143.35 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 143.95 ms (128.83 ms to 168.10 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 58.35 ms (47.24 ms to 70.77 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 965.86 ms (681.98 ms to 1.09 s) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 2.69 s (2.60 s to 4.12 s) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | forward hook plus tensor hook | 70.31 ms (67.09 ms to 97.64 ms) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | Captum 0.9.0 layer gradient | 39.27 ms (31.06 ms to 42.61 ms) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | `tl.trace` with gradients saved | 980.27 ms (740.80 ms to 1.06 s) |
| ResNet-18, CPU, full forward, n=3 processes | plain forward | 15.01 ms (14.32 ms to 15.64 ms) |
| ResNet-18, CPU, full forward, n=3 processes | plain forward after a trace (torch wrappers still installed, nothing recorded) | 22.11 ms (15.41 ms to 22.90 ms) |
| ResNet-18, CPU, full forward, n=3 processes | forward hooks on every module | 17.80 ms (15.83 ms to 23.05 ms) |
| ResNet-18, CPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 236.13 ms (219.81 ms to 237.66 ms) |
| ResNet-18, CPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 710.60 ms (494.10 ms to 745.69 ms) |

### ResNet-18 on GPU (second run, RTX 2080 Ti workstation)

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | plain forward, nothing attached | 2.30 ms (2.29 ms to 2.34 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | forward hook | 2.32 ms (2.31 ms to 2.32 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | nnsight 0.8.1 | 2.61 ms (2.56 ms to 2.72 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | torchvision fx `create_feature_extractor` | 0.76 ms (0.75 ms to 0.82 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | `tl.record(..., save=tl.module(site))` | 98.55 ms (95.98 ms to 158.48 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | `tl.trace(..., save=tl.module(site))` | 389.71 ms (376.99 ms to 403.93 ms) |
| ResNet-18, GPU, one module's output (`layer2`), n=3 processes | `tl.extract` | 376.36 ms (372.56 ms to 393.48 ms) |
| ResNet-18, GPU, one module replaced by zeros, n=3 processes | forward hook | 2.34 ms (2.34 ms to 2.36 ms) |
| ResNet-18, GPU, one module replaced by zeros, n=3 processes | nnsight 0.8.1 | 2.65 ms (2.61 ms to 2.94 ms) |
| ResNet-18, GPU, one module replaced by zeros, n=3 processes | `tl.when(...).bind(model)` | 3.70 ms (3.58 ms to 4.06 ms) |
| ResNet-18, GPU, one module replaced by zeros, n=3 processes | `tl.record(..., intervene=tl.when(...))` | 103.33 ms (102.42 ms to 115.96 ms) |
| ResNet-18, GPU, eight steering strengths, n=3 processes | forward hooks | 13.10 ms (13.08 ms to 15.33 ms) |
| ResNet-18, GPU, eight steering strengths, n=3 processes | nnsight 0.8.1 | 17.01 ms (16.59 ms to 17.29 ms) |
| ResNet-18, GPU, eight steering strengths, n=3 processes | `tl.when(...).bind(model)` | 33.63 ms (29.87 ms to 49.86 ms) |
| ResNet-18, GPU, eight steering strengths, n=3 processes | `tl.record(..., intervene=...)` per strength | 1.06 s (1.05 s to 1.19 s) |
| ResNet-18, GPU, eight steering strengths, n=3 processes | `tl.trace(..., intervene=...)` per strength | 3.61 s (3.59 s to 4.20 s) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 8.20 ms (7.59 ms to 8.25 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 8.48 ms (8.47 ms to 9.35 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 3.02 ms (3.00 ms to 3.13 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 393.03 ms (387.17 ms to 421.00 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4 | `tl.extract_dataset` | fails: inputs end up on two devices (cuda:0 and cpu) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | forward hook plus tensor hook | 9.27 ms (8.69 ms to 10.54 ms) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | Captum 0.9.0 layer gradient | 7.76 ms (7.32 ms to 8.92 ms) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | `tl.trace` with gradients saved | 551.37 ms (466.61 ms to 576.62 ms) |
| ResNet-18, GPU, full forward, n=3 processes | plain forward | 2.30 ms (2.27 ms to 2.33 ms) |
| ResNet-18, GPU, full forward, n=3 processes | plain forward after a trace (torch wrappers still installed, nothing recorded) | 2.39 ms (2.36 ms to 3.11 ms) |
| ResNet-18, GPU, full forward, n=3 processes | forward hooks on every module | 3.10 ms (3.08 ms to 3.44 ms) |
| ResNet-18, GPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 99.82 ms (95.38 ms to 109.57 ms) |
| ResNet-18, GPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 427.04 ms (391.87 ms to 435.26 ms) |

The fx rows are faster than the plain forward because `create_feature_extractor` builds a graph
that stops at the requested module, so it runs only part of ResNet-18. The other arms run the
whole network. On GPU the `tl.extract_dataset` arm failed in this run with "Expected all tensors
to be on the same device" (cuda:0 and cpu): in-memory results were copied to the host while the
model stayed on the GPU. It was fixed the same day; the last table in this section measured the
fixed arm.

### GPT-2 small on GPU (third run, L40 node)

| Workload (model, device, repeats) | Tool | Median (fastest to slowest) |
| --- | --- | --- |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 168 ms (167 ms to 173 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | nnsight 0.8.1 | 200 ms (194 ms to 200 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 369 ms (368 ms to 376 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.record` | 8.35 s (8.07 s to 9.84 s) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 49.67 s (48.81 s to 49.85 s) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 120 ms (119 ms to 120 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | nnsight 0.8.1 | 135 ms (134 ms to 139 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 259 ms (250 ms to 260 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.record` | 6.30 s (6.28 s to 6.39 s) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 23.84 s (23.22 s to 24.57 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 447 ms (447 ms to 454 ms) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | nnsight 0.8.1 | 524 ms (504 ms to 536 ms) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 1.11 s (1.08 s to 1.14 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.record` | 26.29 s (26.03 s to 26.74 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.trace(..., save=...)` (full graph, selected outputs saved) | 92.10 s (91.33 s to 93.01 s) |

### LSTM tagger on GPU (third run, L40 node)

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| LSTM tagger, GPU, one module's output, n=3 processes | plain forward, nothing attached | 0.36 ms (0.35 ms to 0.36 ms) |
| LSTM tagger, GPU, one module's output, n=3 processes | forward hook | 0.37 ms (0.37 ms to 0.38 ms) |
| LSTM tagger, GPU, one module's output, n=3 processes | nnsight 0.8.1 | 0.44 ms (0.44 ms to 0.46 ms) |
| LSTM tagger, GPU, one module's output, n=3 processes | `tl.record(..., save=tl.module(site))` | 18.16 ms (17.96 ms to 19.11 ms) |
| LSTM tagger, GPU, one module's output, n=3 processes | `tl.trace(..., save=tl.module(site))` | 35.26 ms (35.17 ms to 37.08 ms) |
| LSTM tagger, GPU, one module's output, n=3 processes | `tl.extract` | 36.25 ms (35.57 ms to 38.38 ms) |
| LSTM tagger, GPU, one module replaced by zeros, n=3 processes | forward hook | 0.39 ms (0.39 ms to 0.41 ms) |
| LSTM tagger, GPU, one module replaced by zeros, n=3 processes | nnsight 0.8.1 | 0.49 ms (0.48 ms to 0.50 ms) |
| LSTM tagger, GPU, one module replaced by zeros, n=3 processes | `tl.when(...).bind(model)` | 0.77 ms (0.75 ms to 0.79 ms) |
| LSTM tagger, GPU, one module replaced by zeros, n=3 processes | `tl.record(..., intervene=tl.when(...))` | 18.61 ms (18.60 ms to 19.79 ms) |
| LSTM tagger, GPU, eight steering strengths, n=3 processes | forward hooks | 2.55 ms (2.49 ms to 2.67 ms) |
| LSTM tagger, GPU, eight steering strengths, n=3 processes | nnsight 0.8.1 | 3.62 ms (3.49 ms to 3.73 ms) |
| LSTM tagger, GPU, eight steering strengths, n=3 processes | `tl.when(...).bind(model)` | 7.85 ms (7.45 ms to 8.21 ms) |
| LSTM tagger, GPU, eight steering strengths, n=3 processes | `tl.record(..., intervene=...)` per strength | 150.25 ms (148.79 ms to 157.70 ms) |
| LSTM tagger, GPU, eight steering strengths, n=3 processes | `tl.trace(..., intervene=...)` per strength | 307.09 ms (306.43 ms to 324.52 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 1.46 ms (1.37 ms to 1.48 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 1.86 ms (1.76 ms to 1.91 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 72.40 ms (71.18 ms to 75.21 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 159.69 ms (149.83 ms to 202.53 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | plain forward | 0.35 ms (0.35 ms to 0.36 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | plain forward after a trace (torch wrappers still installed, nothing recorded) | 0.36 ms (0.36 ms to 0.38 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | forward hooks on every module | 0.39 ms (0.39 ms to 0.39 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 18.20 ms (18.06 ms to 19.07 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 36.66 ms (36.39 ms to 38.00 ms) |

The LSTM tagger's gradient arms could not run on GPU for any tool (cuDNN's RNN backward is only
available in training mode, and the harness runs the model in eval mode), and fx refuses the
model's data-dependent control flow. Of the peer tools, nnsight ran; Captum is a gradient arm;
TransformerLens has no LSTM.

### ResNet-18 on GPU after the `tl.extract_dataset` fix (third run, RTX 2080 Ti node)

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 8.29 ms (8.28 ms to 8.33 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 3.67 ms (3.62 ms to 3.68 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 428.00 ms (416.55 ms to 450.06 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 1.99 s (1.83 s to 2.02 s) |

## Reproduce

```bash
pip install -e ".[dev,test]" nnsight==0.8.1 captum==0.9.0
OMP_NUM_THREADS=1 python benchmarks/capture/driver.py --smoke --warmups 3 --samples 20 \
    --budget-s 20 --processes 3 --no-sweep --subjects lstm-tagger --out capture-lstm.json
OMP_NUM_THREADS=1 python -m benchmarks.capture.mi_workloads.matrix --model gpt2 \
    --tools hooks,tl_bind --workloads steer_sweep --out mi-gpt2.jsonl --resdir mi-res
# second run: ResNet-18 (add --device cuda for the GPU rows) and GPT-2 small full forwards
OMP_NUM_THREADS=1 python benchmarks/capture/driver.py --smoke --warmups 3 --samples 20 \
    --budget-s 20 --threads 1 --processes 3 --no-sweep --subjects resnet18 --out capture-resnet18.json
OMP_NUM_THREADS=1 python -m benchmarks.capture.mi_workloads.matrix --model gpt2 --timeout 1500 \
    --tools hooks,nnsight,tl_record,tl_trace --workloads cache,steer_sweep,patch_sweep \
    --out mi-gpt2.jsonl --resdir mi-res -- --slow-rep-s 0
# third run: the same on a GPU (TL_BENCH_DEVICE moves the GPT-2 workloads; the driver takes --device)
pip install transformer-lens==4.3.0
TL_BENCH_DEVICE=cuda OMP_NUM_THREADS=1 python -m benchmarks.capture.mi_workloads.matrix --model gpt2 \
    --timeout 1500 --tools hooks,nnsight,tlens,tl_record,tl_trace --workloads cache,steer_sweep,patch_sweep \
    --out mi-gpu-gpt2.jsonl --resdir mi-res -- --slow-rep-s 0
OMP_NUM_THREADS=1 python benchmarks/capture/driver.py --device cuda --smoke --warmups 3 --samples 20 \
    --budget-s 20 --threads 1 --processes 3 --no-sweep --subjects lstm-tagger,resnet18 --out capture-gpu.json
```
