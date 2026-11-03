# Capture benchmark

What a live edit and a capture cost against a plain forward hook, measured on 2026-11-03 on
a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average about 40. Versions: Python 3.10, torch 2.13.0 (CPU build), transformers 5.18.0,
TorchLens 2.36.3, nnsight 0.8.1. Every number below is a median wall time from the harness in
`benchmarks/capture/`; the raw medians are in
[`data/capture_2026-11-03.json`](data/capture_2026-11-03.json).

<!-- JMT review: plain factual prose, written for review. -->

The worker was shared and loaded, so absolute times are slower than on an idle machine. Compare
rows within a workload, not across workloads.

## Live edits at hook speed

`tl.when(site, edit).bind(model)` applies an edit inside the model's own forward and records
nothing. Each timed call builds the bound edit and runs it, the same way the hook arm registers
its hook, runs, and removes it.

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| GPT-2 small, CPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 26.70 s |
| GPT-2 small, CPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.when(...).bind(model)` | 27.15 s |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | plain forward, nothing attached | 1.54 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | forward hook | 1.70 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | `tl.when(...).bind(model)` | 2.78 ms |
| LSTM tagger, CPU, one module replaced by zeros, n=3 processes of 20 samples | nnsight 0.8.1 | 2.06 ms |
| LSTM tagger, CPU, plain forward with TorchLens imported and torch wrapped, n=2 processes of 20 samples | wrapped, not capturing | 1.50 ms |
| LSTM tagger, CPU, plain forward with TorchLens imported and torch wrapped, n=2 processes of 20 samples | plain forward | 1.47 ms |

On GPT-2 small the bound edit and the forward hook land 0.46 s apart on a 27 s sweep, inside the
spread of three repeats (hooks 25.36 to 28.00 s, bound edit 26.80 to 27.74 s), and the steered
logits are identical (largest absolute difference 0.0). On the LSTM tagger, whose plain forward
takes about 1.5 ms, the bound edit's fixed setup cost per call is visible. With TorchLens
imported and torch wrapped, a forward that captures nothing runs at plain-forward speed.

## Where TorchLens is slower

A capture records every operation's metadata and checks it, so it costs much more than a hook
that reads one tensor. On a small model the capture's fixed cost dominates.

| Workload (model, device, repeats) | Tool | Median |
| --- | --- | --- |
| LSTM tagger, CPU, full trace of every operation, n=2 processes of 20 samples | `tl.trace(model, x)` | 81.38 ms |
| LSTM tagger, CPU, full trace of every operation, n=2 processes of 20 samples | forward hooks on every module | 1.61 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | `tl.trace(..., save=tl.module(site))` | 77.96 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | `tl.extract` | 82.93 ms |
| LSTM tagger, CPU, one module's output, n=3 processes of 20 samples | forward hook | 1.75 ms |
| LSTM tagger, CPU, eight steering strengths through a trace and rerun, n=3 processes of 20 samples | `tl.trace` then `fork().do(...)` | 640.23 ms |
| LSTM tagger, CPU, eight steering strengths, n=3 processes of 20 samples | `tl.when(...).bind(model)` | 25.60 ms |
| LSTM tagger, CPU, eight steering strengths, n=3 processes of 20 samples | forward hooks | 12.52 ms |
| LSTM tagger, CPU, one module's output over 16 stimuli in batches of 4, n=2 processes of 20 samples | `tl.extract_dataset` | 356.00 ms |
| LSTM tagger, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes of 20 samples | forward hook | 10.21 ms |
| LSTM tagger, CPU, gradient at one module output, n=3 processes of 20 samples | `tl.trace` with gradients saved | 117.50 ms |
| LSTM tagger, CPU, gradient at one module output, n=3 processes of 20 samples | tensor hook | 4.40 ms |

What this means in practice:

- To read one module's output, a forward hook is the faster tool. A capture is worth its cost
  when you need operations no module returns, do not yet know the site, or want the recording
  checked against the forward.
- To sweep many edits, use the bound edit (`tl.when(...).bind(model)`), not a trace and rerun
  per edit.
- Batch feature extraction pays the capture cost per batch. Large batches spread it.
- Saving only selected sites (`save=`) saves memory, not time: the structure of every operation
  is still recorded, which is why the one-module trace costs about what the full trace does.
- To time a model itself, locate the slow part with TorchLens, then measure it clean with
  `torch.utils.benchmark.Timer`.

## Not measured here

- GPU timings. Comparisons against torchvision's fx feature extraction wait for a GPU run;
  the LSTM tagger has data-dependent control flow that fx cannot trace.
- Full traces of GPT-2 and the other comparison tools on GPT-2 (TransformerLens, nnsight) were
  queued on 2026-11-03 and did not run on the shared worker that day.
- `tl.record`, the sparse recorder, is not in these tables: its harness arm used an argument
  the current `tl.record` no longer accepts. The arm is fixed and will be measured in the next
  run.

## Reproduce

```bash
pip install -e ".[dev,test]" nnsight==0.8.1 captum==0.9.0
OMP_NUM_THREADS=1 python benchmarks/capture/driver.py --smoke --warmups 3 --samples 20 \
    --budget-s 20 --processes 3 --no-sweep --subjects lstm-tagger --out capture-lstm.json
OMP_NUM_THREADS=1 python -m benchmarks.capture.mi_workloads.matrix --model gpt2 \
    --tools hooks,tl_bind --workloads steer_sweep --out mi-gpt2.jsonl --resdir mi-res
```
