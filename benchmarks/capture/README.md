# Capture benchmarks

Matched-contract speed and memory measurements for TorchLens capture, the costs of what a
full trace keeps, operation fixtures, a real-model panel, and the "what hooks miss"
census. Every runner writes one JSON document in the format of `schema.py`
(`torchlens.bench.capture.v1`), so tables and charts are generated from JSON and
unsupported cells stay visible as rows with a reason.

## Suites

**Suite A, matched contracts** (`arms.py`). Every arm must produce the same output as the
plain-hooks reference; each non-reference cell records the comparison (`exact`), and a
differing output is a `mismatch` row, never a timing.

| Contract | What every arm returns |
|---|---|
| `single_module_output` | one module's output |
| `matched_retained` | the outputs of a fixed set of modules |
| `replacement_edit` | the model output with one module's output replaced by zeros |
| `repeated_steering` | the model outputs for eight added multiples of one direction |
| `same_site_gradient` | the gradient of the summed model output at one module output |
| `ordered_dataset_features` | one module's outputs over 16 stimuli, in order |

Arms: `raw` (the forward passes a contract needs, nothing attached: the floor), `hooks`
(reference), TorchLens entry points (`tl_extract`, `tl_trace_sparse`, `tl_record`,
`tl_bind`, `tl_extract_dataset`), `nnsight`, `transformer_lens` (decoders only, through the
mech-interp workloads), `tv_fx` (torchvision `create_feature_extractor`, symbolically
traceable models only), `thingsvision` (its extraction task only) and `captum` (the
gradient contract only). Arms that do not use TorchLens run in processes that never import
it, so its torch wrappers cannot slow them; each row records `torchlens_imported`.

**Suite B, what overhead buys** (`suite_b.py`). Each row is one TorchLens operation with a
`keeps` statement: full trace with default and with declared retention, the recorder,
validation, portable save and load, backward capture, drawing, and a two-member bundle
comparison, beside a plain forward, a plain forward with TorchLens's wrappers installed,
and hooks retaining every module output. A hook row is not a full-trace substitute: it
keeps module outputs only, and its `keeps` text says so.

**Operation fixtures** (`fixtures.py`). Stacks of one operation kind (pointwise chains,
matmul, convolution, normalization, fused and decomposed attention, views and in-place
mutation, reductions, dynamic branches, custom autograd) at 100, 1,000 and 10,000
operations and three tensor sizes, so per-operation host cost separates from device
compute.

**Real-model panel** (`panel.py`). CNN, recurrent net, encoders, 124M and 1.4B decoders,
a 7 to 9B decoder on GPU, a vision transformer, a diffusion U-Net and a dynamic custom
model. Decoder generation cells at 1, 32, 128 and 1,000 steps each name their entry point
or are marked refused. Entries with `smoke=True` build without downloads.

**Mech-interp workloads** (`mi_workloads/`). Caching, steering sweeps, activation and
attribution patching and steered generation on public decoders, across hooks, TorchLens
entry points, nnsight and TransformerLens, with exactness against hooks.

**Hooks census** (`census.py`). For each classics-corpus model, the share of operations no
module hook can reach (not a module output, not a model output, not fed to a module call),
the forward-replay validation verdict, and the fused or opaque calls the capture records
as single operations. A validation failure is kept as a finding with its summary.

## Method

One fresh subprocess per cell and process index; 5 warmups and 50 samples by default,
fewer when one call is slow (`--budget-s`, at least 3 samples; the row records what ran);
3 processes, outermost so host drift spreads across arms; a separate memory pass with
peak RSS, USS and device memory (`benchmarks/perf_runner.py` helpers); one torch thread by
default. An A/A pair (the same raw forward in two processes, first and last) sets the
noise floor, and a cell whose spread across processes exceeds twice that floor (or 10%)
is listed as noisy in `groups`. Peer packages live only in the benchmark environment
(`peers.txt`), never in TorchLens's dependencies.

```bash
bash benchmarks/capture/run_smoke.sh                                   # CPU smoke end to end
python benchmarks/capture/driver.py --out capture.json                 # full CPU matrix
python benchmarks/capture/census.py --jobs 4 --out census.json         # hooks census
python -m benchmarks.capture.mi_workloads.matrix --model gpt2 --out mi.jsonl --resdir res/
```
