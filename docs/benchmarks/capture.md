# Capture benchmark

What a live edit and a capture cost against a plain forward hook, measured on 2026-11-03 on
a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average about 40. Versions: Python 3.10, torch 2.13.0 (CPU build), transformers 5.18.0,
TorchLens 2.36.3, nnsight 0.8.1. Every number below is a median wall time from the harness in
`benchmarks/capture/`; the raw medians are in
[`data/capture_2026-11-03.json`](data/capture_2026-11-03.json). A second run the same day added
GPT-2 small, ResNet-18 on CPU and GPU, fx, nnsight and `tl.record`, on the hosts named in its
section, and the memory cost of recording a generation (Known costs). A third run the same day
measured GPT-2 small and the LSTM tagger on a GPU, and the generation cost again after a fix to
the recorder (Third run; Known costs).

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

## Second run: GPT-2 small, ResNet-18, CPU and GPU

A second run on 2026-11-03 filled the rows the first run left out: full forwards of GPT-2 small,
ResNet-18 on CPU and on GPU, torchvision's fx feature extraction, nnsight and `tl.record`.

- CPU rows ran on a shared 32-core Intel Xeon 2.80 GHz Linux CPU worker, one torch thread per process, load average 7 to 28.
  Versions as above. Raw data:
  [`data/capture_2026-11-03-run2.json`](data/capture_2026-11-03-run2.json).
- GPU rows ran on a 20-core Intel Core i9-9900X 3.50 GHz Linux workstation with an NVIDIA GeForce RTX 2080 Ti, one torch thread per process, load average about 10.
  Versions as above except torch 2.13.0 and torchvision 0.28.0 built for CUDA 12.6. Raw data:
  [`data/capture_2026-11-03-gpu.json`](data/capture_2026-11-03-gpu.json).

Method. Each ResNet-18 row is three separate processes; each process times up to 20 samples after 3
warmup calls (as few as 3 for the slowest TorchLens cells, which reach the 20 s budget first) and reports its median. The table gives the median of
the three process medians, and in brackets the lowest and highest process median. Each GPT-2 row
is one process that runs the whole workload three times; the table gives the median of the three
wall times and, in brackets, the fastest and slowest. The GPT-2 rows come from two runs at
different times on the shared worker (the first measured forward hooks and nnsight, the second
forward hooks and TorchLens), and the worker's load changed between them, so each run carries its
own forward-hook row: compare a tool with the hook row of the same run. Each tool's output was compared with the
forward-hook arm's output and matched: bit for bit for every TorchLens and nnsight arm on
ResNet-18, and with a largest absolute difference of 0.0 on GPT-2. The CPU worker was loaded while these rows ran: two identical plain-forward
arms timed on it differed by 54%, against 5% on the GPU workstation. Compare rows within a
workload, not across workloads or hosts.

### GPT-2 small, full forward

| Workload (model, device, repeats) | Tool | Median (fastest to slowest) |
| --- | --- | --- |
| GPT-2 small, CPU, first run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 26.44 s (26.24 to 26.59 s) |
| GPT-2 small, CPU, first run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | nnsight 0.8.1 | 26.92 s (26.43 to 27.40 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 29.30 s (28.76 to 30.95 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.record` | 39.17 s (39.05 to 40.18 s) |
| GPT-2 small, CPU, second run, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.trace` (full trace) | 99.88 s (93.72 to 106.37 s) |
| GPT-2 small, CPU, first run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 20.50 s (19.86 to 20.67 s) |
| GPT-2 small, CPU, first run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | nnsight 0.8.1 | 20.73 s (20.20 to 22.00 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 26.96 s (26.48 to 27.50 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.record` | 32.32 s (27.23 to 35.98 s) |
| GPT-2 small, CPU, second run, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.trace` (full trace) | 49.97 s (49.80 to 59.79 s) |
| GPT-2 small, CPU, first run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 7.07 s (6.96 to 7.24 s) |
| GPT-2 small, CPU, first run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | nnsight 0.8.1 | 7.24 s (7.16 to 7.36 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 6.62 s (6.39 to 6.80 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.record` | 43.85 s (40.75 to 47.59 s) |
| GPT-2 small, CPU, second run, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.trace` (full trace) | 143.45 s (116.07 to 165.04 s) |

### ResNet-18 on CPU

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
| ResNet-18, CPU, eight steering strengths, n=3 processes | `tl.trace` then `fork().do(...)` | 5.26 s (4.58 s to 5.57 s) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 130.10 ms (127.56 ms to 143.35 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 143.95 ms (128.83 ms to 168.10 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 58.35 ms (47.24 ms to 70.77 ms) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 965.86 ms (681.98 ms to 1.09 s) |
| ResNet-18, CPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 2.69 s (2.60 s to 4.12 s) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | forward hook plus tensor hook | 70.31 ms (67.09 ms to 97.64 ms) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | Captum 0.9.0 layer gradient | 39.27 ms (31.06 ms to 42.61 ms) |
| ResNet-18, CPU, gradient at one module output (`layer2`), n=3 processes | `tl.trace` with gradients saved | 980.27 ms (740.80 ms to 1.06 s) |
| ResNet-18, CPU, full forward, n=3 processes | plain forward | 15.01 ms (14.32 ms to 15.64 ms) |
| ResNet-18, CPU, full forward, n=3 processes | plain forward, TorchLens imported and torch wrapped | 22.11 ms (15.41 ms to 22.90 ms) |
| ResNet-18, CPU, full forward, n=3 processes | forward hooks on every module | 17.80 ms (15.83 ms to 23.05 ms) |
| ResNet-18, CPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 236.13 ms (219.81 ms to 237.66 ms) |
| ResNet-18, CPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 710.60 ms (494.10 ms to 745.69 ms) |

### ResNet-18 on GPU

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
| ResNet-18, GPU, eight steering strengths, n=3 processes | `tl.trace` then `fork().do(...)` | 3.61 s (3.59 s to 4.20 s) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 8.20 ms (7.59 ms to 8.25 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 8.48 ms (8.47 ms to 9.35 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 3.02 ms (3.00 ms to 3.13 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 393.03 ms (387.17 ms to 421.00 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4 | `tl.extract_dataset` | fails: inputs end up on two devices (cuda:0 and cpu) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | forward hook plus tensor hook | 9.27 ms (8.69 ms to 10.54 ms) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | Captum 0.9.0 layer gradient | 7.76 ms (7.32 ms to 8.92 ms) |
| ResNet-18, GPU, gradient at one module output (`layer2`), n=3 processes | `tl.trace` with gradients saved | 551.37 ms (466.61 ms to 576.62 ms) |
| ResNet-18, GPU, full forward, n=3 processes | plain forward | 2.30 ms (2.27 ms to 2.33 ms) |
| ResNet-18, GPU, full forward, n=3 processes | plain forward, TorchLens imported and torch wrapped | 2.39 ms (2.36 ms to 3.11 ms) |
| ResNet-18, GPU, full forward, n=3 processes | forward hooks on every module | 3.10 ms (3.08 ms to 3.44 ms) |
| ResNet-18, GPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 99.82 ms (95.38 ms to 109.57 ms) |
| ResNet-18, GPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 427.04 ms (391.87 ms to 435.26 ms) |

Notes on these rows:

- The fx rows are faster than the plain forward because `create_feature_extractor` builds a graph
  that stops at the requested module, so it runs only part of ResNet-18. The other arms run the
  whole network.
- On GPU, the `tl.extract_dataset` arm failed in this run: with the model and the dataset both on
  the GPU, the run stopped with "Expected all tensors to be on the same device" (cuda:0 and cpu).
  The cause (in-memory results were copied to the host while the model stayed on the GPU) was
  fixed the same day; the third run below measured the fixed arm.
- `tl.record` records only the sites it is asked for and costs a fraction of a full trace, but
  it still runs through TorchLens's wrapped operations, so it stays well above a forward hook.

## Third run: GPT-2 small and the LSTM tagger on GPU

A third run on 2026-11-03 measured on a GPU the two models the second run had left on CPU,
and `tl.extract_dataset` on GPU after its fix. Two GPU nodes of one cluster were used; each table
below names its node.

- The GPT-2 small and LSTM tagger rows ran on a 256-core AMD EPYC 7662 Linux GPU node with an NVIDIA L40 (46 GiB), one torch thread per process, load average 56 to 74 (a shared node).
  Raw data: [`data/capture_2026-11-03-gpu3.json`](data/capture_2026-11-03-gpu3.json).
- The `tl.extract_dataset` rows and the GPU generation run (Known costs) ran on a 48-core Intel Xeon Gold 6126 2.60 GHz Linux GPU node with an NVIDIA GeForce RTX 2080 Ti (11 GiB), one torch thread per process, load average 1 to 3.
  Raw data: [`data/capture_2026-11-03-gpu2.json`](data/capture_2026-11-03-gpu2.json).
- Versions on both: Python 3.10, torch 2.13.0 built for CUDA 12.6, transformers 5.18.0, nnsight 0.8.1,
  TransformerLens 4.3.0, and a development build of TorchLens from that day.

Method as in the second run: each LSTM tagger row is three separate processes, each timing up to
20 samples after 3 warmup calls, and the table gives the median of the three process medians with the
lowest and highest in brackets; each GPT-2 row is one process that runs the whole workload three
times, and the table gives the median wall time with the fastest and slowest in brackets. Every
arm synchronises the GPU before it stops its clock. Every TorchLens arm's output was compared with the
forward-hook arm's output and matched, with a largest absolute difference of 0.0 on GPT-2.
Each job had one GPU to itself. The L40 node is shared and its CPUs were busy (load average 56 to 74
on 256 cores); the RTX 2080 Ti node's CPUs carried a load average of 1 to 3. TransformerLens runs its
own GPT-2 implementation loaded with the same weights, so its rows time a different model code path
with hooks on `hook_resid_post`; every other GPT-2 row runs the transformers model.

### GPT-2 small on GPU (L40 node)

| Workload (model, device, repeats) | Tool | Median (fastest to slowest) |
| --- | --- | --- |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | forward hooks | 168 ms (167 ms to 173 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | nnsight 0.8.1 | 200 ms (194 ms to 200 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 369 ms (368 ms to 376 ms) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.record` | 8.35 s (8.07 s to 9.84 s) |
| GPT-2 small, GPU, cache every block's output for 256 sequences of 32 tokens in batches of 16 (16 forwards), n=3 | `tl.trace` (full trace) | 49.67 s (48.81 s to 49.85 s) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | forward hooks | 120 ms (119 ms to 120 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | nnsight 0.8.1 | 135 ms (134 ms to 139 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 259 ms (250 ms to 260 ms) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.record` | 6.30 s (6.28 s to 6.39 s) |
| GPT-2 small, GPU, 12 steered forwards (3 layers by 4 strengths) on a batch of 16 sequences of 32 tokens, n=3 | `tl.trace` (full trace) | 23.84 s (23.22 s to 24.57 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | forward hooks | 447 ms (447 ms to 454 ms) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | nnsight 0.8.1 | 524 ms (504 ms to 536 ms) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | TransformerLens 4.3.0 (its own GPT-2, hooks on `hook_resid_post`) | 1.11 s (1.08 s to 1.14 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.record` | 26.29 s (26.03 s to 26.74 s) |
| GPT-2 small, GPU, activation patching sweep, 50 patched forwards of 16 tokens, n=3 | `tl.trace` (full trace) | 92.10 s (91.33 s to 93.01 s) |

### LSTM tagger on GPU (L40 node)

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
| LSTM tagger, GPU, eight steering strengths, n=3 processes | `tl.trace` then `fork().do(...)` | 307.09 ms (306.43 ms to 324.52 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 1.46 ms (1.37 ms to 1.48 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | nnsight 0.8.1 | 1.86 ms (1.76 ms to 1.91 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 72.40 ms (71.18 ms to 75.21 ms) |
| LSTM tagger, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 159.69 ms (149.83 ms to 202.53 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | plain forward | 0.35 ms (0.35 ms to 0.36 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | plain forward, TorchLens imported and torch wrapped | 0.36 ms (0.36 ms to 0.38 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | forward hooks on every module | 0.39 ms (0.39 ms to 0.39 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | `tl.record` (sparse recorder) | 18.20 ms (18.06 ms to 19.07 ms) |
| LSTM tagger, GPU, full forward, n=3 processes | `tl.trace(model, x)` (full trace) | 36.66 ms (36.39 ms to 38.00 ms) |

### ResNet-18 on GPU (RTX 2080 Ti node), `tl.extract_dataset` after its fix

| Workload (model, device, repeats) | Tool | Median (lowest to highest process median) |
| --- | --- | --- |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | forward hook | 8.29 ms (8.28 ms to 8.33 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | torchvision fx `create_feature_extractor` | 3.67 ms (3.62 ms to 3.68 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.record` per batch | 428.00 ms (416.55 ms to 450.06 ms) |
| ResNet-18, GPU, one module's output over 16 stimuli in batches of 4, n=3 processes | `tl.extract_dataset` | 1.99 s (1.83 s to 2.02 s) |

Notes on these rows:

- On the LSTM tagger the gradient arms (forward hook plus tensor hook, Captum, `tl.trace` with
  gradients saved) could not run on GPU for any tool: cuDNN's RNN backward is only available in
  training mode, and the harness runs the model in eval mode. Those rows are not given. fx is not
  given either: symbolic tracing refuses the model's data-dependent control flow.
- `tl.extract_dataset` on GPU runs a trace per batch and gathers the requested module's output
  across batches; on ResNet-18 it costs about five times `tl.record` per batch on the same node.
- The peer tools on the LSTM tagger: nnsight ran; Captum is a gradient arm (see above); TransformerLens
  has no LSTM.

## Known costs

Recording a generation keeps every decode step's record in memory. Process memory grows faster
than the number of generated tokens, and the time per token grows as the run gets longer. Measured on
2026-11-03 on the same 32-core Intel Xeon worker, four torch threads, one process per token count, on a development
build of TorchLens from that day, with a 70-million-parameter GPT-NeoX-style model
(`EleutherAI/pythia-70m`, float32), greedy decoding from a 7-token prompt, and the output of
every block's MLP and of the final layer norm saved at every step. Each row is one call of
`tl.record(model, ids, save=sites, generate_kwargs={"max_new_tokens": n, "min_new_tokens": n,
"do_sample": False, "use_cache": True})`, where `sites` joins `tl.module(...)` for each saved
module with `|`.

One process per row, no repeats. Peak process memory is the operating system's maximum resident
set size; saved activations are the summed saved bytes of every step's record. Raw data:
[`data/capture_2026-11-03-generation.json`](data/capture_2026-11-03-generation.json).

| Generated tokens | Wall time per token | Peak process memory (RSS) | Saved activations at the end |
| --- | --- | --- | --- |
| 500 | 857 ms | 5.3 GiB | 6.9 MiB |
| 1,000 | 1,336 ms | 15.7 GiB | 13.8 MiB |
| 2,000 | not reached | over 17 GiB (stopped after 17 min) | not reached |

The saved activations are a small part of the growth: 6.9 MiB of saved tensors at 500 tokens against 5.3 GiB of process memory; 13.8 MiB of saved tensors at 1,000 tokens against 15.7 GiB of process memory. The rest of the process memory
grows with each recorded decode step and is not accounted for by the saved tensors.

That growth was a bug: the recorder kept every decode pass's whole output, including the model's
key-value cache, so retained memory grew with the square of the generated tokens. It was fixed on
2026-11-03. Measured again after the fix, on the same 32-core Intel Xeon worker with four torch
threads and the same model, prompt and saved sites (the 500- and 1,000-token rows from the fix's own
check, the 2,000-token row from the third run), process memory grows linearly, about 2.7 MiB per
token, and the time per token stays flat:

| Generated tokens | Wall time per token | Peak process memory (RSS) | Saved activations at the end |
| --- | --- | --- | --- |
| 500 | 356 ms | 2.1 GiB | 6.9 MiB |
| 1,000 | 395 ms | 3.4 GiB | 13.8 MiB |
| 2,000 | 397 ms | 6.2 GiB | 27.4 MiB |

The same 2,000-token run on the RTX 2080 Ti node above (four torch threads) took 269 ms per token, peaked at 6.8 GiB of process memory and 458 MiB of GPU memory, and saved the same 27.4 MiB of activations.

For long generations, record a shorter run or save fewer sites.

## Not measured here

- fx on the LSTM tagger: its data-dependent control flow cannot be traced by fx.
- Gradient arms on the LSTM tagger on GPU: cuDNN's RNN backward needs training mode (see the third
  run's notes).
- Generation cost past 2,000 tokens.

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
