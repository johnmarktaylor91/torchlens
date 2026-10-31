# Streaming statistics: `tl.stats` and `tl.stats.aggregate`

`tl.stats` computes dataset-level statistics of activations or gradients in
constant memory: each accumulator folds one batch at a time and never retains
per-batch tensors. `tl.stats.aggregate` is the one-call door that drives those
accumulators across a whole dataloader. Both are shipped, tested surface --
promoted into `torchlens.__all__` by the completeness work (they were
previously reachable but undeclared). Spellings are DOCUMENTED-UNSTABLE
pending naming-session ratification.

## The accumulators

Every accumulator implements the `tl.stats.StreamingStat` protocol --
`update(value)` folds one batch, `compute()` finalizes (`tl.stats.Histogram` finalizes with `result()`):

| Accumulator | Result | Notes |
|---|---|---|
| `tl.stats.Mean()` | float | Count-weighted elementwise mean |
| `tl.stats.Norm(p=2.0)` | float | Streaming p-norm over all elements |
| `tl.stats.Quantile(quantiles=...)` | dict | Reservoir-backed quantile estimates; per unit with `unit_axis=` |
| `tl.stats.TopK(k=10)` | list, or `UnitTopK` per unit | Largest k elements seen; per unit with example ids under `unit_axis=` |
| `tl.stats.Covariance()` | Tensor | Feature-space covariance (rows = observations) |
| `tl.stats.CrossCovariance()` | Tensor | Paired `update(a, b)` |
| `tl.stats.CKA()` | `SimilarityScore` (a float) | Paired `update(a, b)`; exact full-data linear CKA; the result names its variant in `similarity_repository_id` |
| `tl.stats.PCA(n_components)` | dict | Streaming PCA; `.fitted()` freezes a transform |
| `tl.stats.Histogram(...)` | histogram | Fixed signed-log2 bins (observability kernel) |
| `tl.stats.RunningStats(...)` | scalar spine | Always-on scalar summary (observability kernel) |
| `tl.stats.Aggregator(*stats)` | dict | Fans one stream out to several accumulators |

`tl.stats.CKA` is exact: it retains only feature-sized covariance terms, so
the finalized value equals the full-data linear CKA of Kornblith et al.
(2019) computed in one shot -- it is not a minibatch approximation that
depends on batch boundaries. `tl.stats.cka(a, b)` is the one-shot form.

## `tl.stats.aggregate`: statistics across a dataloader

Map layer selectors (any `log[...]` spelling, or `"output"` for the model
output) to accumulators; `tl.stats.aggregate` runs the captures and streams every
batch through them:

```python
# `model` is an application-supplied nn.Module (here: conv -> relu -> conv).
import torch
import torchlens as tl

results = tl.stats.aggregate(
    model,
    [torch.rand(2, 3, 16, 16) for _ in range(4)],   # any iterable of inputs
    metrics={
        "relu_1_2": tl.stats.Aggregator(tl.stats.Mean(), tl.stats.Norm()),
        "output": tl.stats.Quantile(),
    },
)
print(results["relu_1_2"]["Mean"], sorted(results["output"]))
```

Gradient statistics stream the same way: pass `target="grad"` and a
`loss_fn=` used to build the backward pass from `(output, *batch_tail)`:

```python
grad_results = tl.stats.aggregate(
    model,
    [(torch.rand(2, 3, 16, 16),) for _ in range(2)],
    metrics={"relu_1_2": tl.stats.Norm()},
    target="grad",
    loss_fn=lambda output: output.square().mean(),
)
print(grad_results["relu_1_2"] >= 0.0)
```

## Per-unit statistics: neurons, channels and SAE latents

`tl.stats.Quantile` and `tl.stats.TopK` pool every value by default. Name the
unit axis with the keyword-only `unit_axis=` and they keep one reservoir, or
one top-k list, per unit. Axis 0 of every batch is the example axis; every
axis other than the example and unit axes is a position axis (a token
position, a spatial location).

- `Quantile(..., unit_axis=-1).compute()` maps each quantile to a `float64
  [units]` tensor: a quantile vector per unit. It is exact while each unit has
  seen at most `reservoir_size` values; beyond that the rank-space error is
  about `sqrt(q * (1 - q) / reservoir_size)`, from a private seeded stream, so
  the same seed gives the same result.
- `TopK(k, unit_axis=-1).compute()` is a `tl.stats.UnitTopK`: `values`
  (`float64 [units, k]`, descending), `example_ids` (`int64 [units, k]`) and
  `positions` (`int64 [units, k, position_axes]`, one column per position
  axis). Under `tl.stats.aggregate` an example id is the example's index in the
  dataloader's iteration order, which is the dataset index for an unshuffled
  loader; outside it, `update(batch, example_ids=...)` names the examples.
  Ties break toward the lowest example id, then the lowest position, so the
  answer does not depend on batch size. NaN values are ignored, as in the
  pooled tracker; a unit with fewer than `k` values has NaN values and `-1`
  ids in its empty slots.
- `encoder=` applies a callable to each batch before it is folded, so SAE
  latents use the same tool as neurons: pass the SAE's encoder and the site's
  activations become latents inside the loop. `tl.stats.Aggregator` fans one
  site out to a neuron tracker and a latent tracker.

Memory: per-unit state is `8 * reservoir_size` bytes per unit for `Quantile`
(64 KiB at the default 8192) and `k * (16 + 8 * position_axes)` bytes per unit
for `TopK` (160 B at the default `k=10` with no position axis, 240 B with one).
Past 256 MiB of state, a `stats_per_unit_state_large` warning fires once per
session and names the cheaper spelling: a smaller `reservoir_size=` or `k=`,
or fewer units through `encoder=lambda v: v.narrow(axis, 0, n)`. No unit is
ever dropped.

```python
import torch
import torchlens as tl
from torch import nn

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
sae_encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU()).eval()  # a tiny SAE encoder
stimuli = [torch.randn(3, 4) for _ in range(3)]  # 9 examples, ragged batches welcome

relu_label = next(
    op.layer_label for op in tl.trace(mlp, stimuli[0]).layer_list if op.layer_type == "relu"
)
per_unit = tl.stats.aggregate(
    mlp,
    stimuli,
    metrics={
        relu_label: tl.stats.Aggregator(
            tl.stats.TopK(k=3, unit_axis=-1, name="neurons"),
            tl.stats.TopK(k=3, unit_axis=-1, encoder=sae_encoder, name="latents"),
            tl.stats.Quantile((0.5, 0.9), unit_axis=-1, name="neuron_quantiles"),
        )
    },
)[relu_label]
neurons, latents = per_unit["neurons"], per_unit["latents"]
print(neurons.values.shape, latents.values.shape)  # (8, 3) and (16, 3)
print("top stimuli of neuron 0:", neurons.example_ids[0].tolist())
print("top stimuli of latent 0:", latents.example_ids[0].tolist())
print(per_unit["neuron_quantiles"][0.5].shape)  # one median per neuron
```

## Paired statistics (CKA between two layers)

Paired accumulators take `update(a, b)`, so drive them from one capture per
batch:

```python
similarity = tl.stats.CKA()
for batch in [torch.rand(2, 3, 16, 16) for _ in range(3)]:
    log = tl.trace(model, batch)
    similarity.update(
        log["conv2d_1_1"].out.flatten(1),
        log["relu_1_2"].out.flatten(1),
    )
print(0.0 <= similarity.compute() <= 1.0)
```

## Fitted PCA as a persisted transform

`tl.stats.PCA(...).fitted(...)` freezes a fit into a `FittedPCA` payload;
`tl.stats.save_fitted` / `tl.stats.load_fitted` persist it, and the
activation-transform kernel `pca_apply(fitted)` consumes it at extraction
time (see [docs/reference/transforms.md](transforms.md)). Fit-during-write is
refused -- early and late shards would get different coordinates.

## Save, reload and cache

`tl.stats.save_fitted(stat, path)` saves an accumulator's whole state: its
configuration, its folded tensors and its private random stream (schema
`tl_stats_state_v1`, a safetensors file with a `sha256:` digest). It takes
every accumulator on this page, pooled or per unit, and an `Aggregator` of
them. `tl.stats.load_fitted(path)` rebuilds an object that keeps folding as
if it had never stopped: folding the rest of the data gives the same values,
example ids, positions and quantiles as one uninterrupted run. The reload
keeps the unit layout, so a batch with a changed unit count still refuses
(`stats_unit_count_changed`). An `encoder=` callable is code, not state, so
it is not stored: pass it back as `tl.stats.load_fitted(path,
encoder=sae.encode)`. Loading refuses a file whose contents no longer match
its digest (`stats_state_digest_mismatch`), a format version this torchlens
does not read (`stats_state_version_unknown`) and an `encoder=` that does not
match how the state was saved (`stats_state_encoder_mismatch`). Nothing in
the file is unpickled or imported.

`tl.stats.aggregate(..., cache=True)` keys the run and, when the same run was
cached before, restores each accumulator's saved final state in place and
returns its results without running the model. The key covers the torchlens
and torch versions; the model's tensor contents, `requires_grad` and
`training` flags and its implementation (forward code, plain attributes,
registered hooks: the same fingerprints as `tl.trace(cache=True)`); every
batch the dataloader yields, in order, by value; each metric's name and
starting state, configuration included; the code and weights of any
`encoder=`; `target=` and the code of `loss_fn=`. Keying reads the data once
before the loop (the model does not run for it) and restores the global
random state afterwards. A run that cannot be keyed by value refuses with
`stats_aggregate_cache_unkeyable` instead of risking a stale result: a
one-shot iterator, a batch leaf of an opaque type, `Quantile(seed=None)`, a
model with modules in training mode, a metric that is not a `tl.stats`
accumulator, or an `encoder=` or `loss_fn=` whose state the key cannot read.
A loader that yields different batches when keyed and when folded returns
its result, stores nothing and warns once
(`stats_aggregate_cache_data_unstable`). Entries live in the `aggregate`
folder of `cache_dir=` (default `TORCHLENS_CACHE_DIR` or
`~/.cache/torchlens`) and are not evicted; delete the folder to clear them.
A cache hit does not advance the global random state the way running the
loop would.

```python
import tempfile
from pathlib import Path

acts = [torch.rand(2, 4, 5) for _ in range(5)]   # [examples, positions, units]
whole = tl.stats.TopK(3, unit_axis=-1)
for batch in acts:
    whole.update(batch)

with tempfile.TemporaryDirectory() as folder:
    first_half = tl.stats.TopK(3, unit_axis=-1)
    for batch in acts[:2]:
        first_half.update(batch)
    path = tl.stats.save_fitted(first_half, Path(folder) / "top.safetensors")
    resumed = tl.stats.load_fitted(path)
    for batch in acts[2:]:
        resumed.update(batch)
    print(torch.equal(resumed.compute().example_ids, whole.compute().example_ids))

    # A rerun of the same run reads the cache and skips the forward passes.
    batches = [torch.rand(2, 3, 16, 16) for _ in range(3)]
    first = tl.stats.aggregate(
        model, batches, {"relu_1_2": tl.stats.TopK(3, unit_axis=1)},
        cache=True, cache_dir=folder,
    )
    again = tl.stats.aggregate(
        model, batches, {"relu_1_2": tl.stats.TopK(3, unit_axis=1)},
        cache=True, cache_dir=folder,
    )
    print(torch.equal(first["relu_1_2"].values, again["relu_1_2"].values))
```

## Reduce and discard: `save="stats"`

`tl.trace(model, x, save="stats")` is a save level beside `"all"` and
`"none"`: each op's out is folded into a `tl.stats.RunningStats` while the forward
runs, and no activation payload is kept. The capture logs the same graph as a
full capture (the reducer's own torch calls never enter it), makes no raw
copy of an out, and keeps one fixed 16-number float64 vector (128 bytes) per
op whatever the batch size. `tl.stats.site_stats(trace)` returns
`{op label: RunningStats}` in capture order. For a float64 out the counts, the
extrema, the sum and the mean equal `RunningStats().update(out)` on the full
capture's out exactly. The reducer folds each out in chunks of 2^16 values,
so its largest working tensor is 512 KiB whatever the out's size (where
`RunningStats.update` holds full-size float64 copies); the sum of squares, the sum
of absolute values and M2 are therefore bit-equal for outs up to one chunk
and otherwise equal within float64 reduction-order rounding. A float32 out's sum
accumulates in float64 without a converted copy, so it can differ from
`RunningStats.update` in the last bits. The returned spines are live accumulators:
`merge` folds another batch's spines in and `tl.stats.save_fitted` saves
them.

A stats trace keeps nothing to read back or replay from, so it refuses
instead of answering with an empty value: reading an op's `out` refuses
`activation_reduced_to_stats` (the remedy names `site_stats` and
`save="all"`), and replay or validation on the trace (`run`,
`validate_forward_pass`, `save_new_outs`, `push`, `push_from`) refuses
`stats_capture_not_replayable`; `tl.validate(model, x, scope="forward")`
still works because it runs its own full capture. `site_stats` on a trace captured any other
way refuses `stats_capture_not_reduced`.

```python
x = torch.rand(2, 3, 16, 16)
reduced = tl.trace(model, x, save="stats")
spines = tl.stats.site_stats(reduced)
label = next(iter(spines))
print(label, spines[label].compute().numel, spines[label].compute().mean)

full = tl.trace(model, x)
reference = tl.stats.RunningStats()
reference.update(full[label].out)
print(reference.compute() == spines[label].compute())
```

## Trajectory statistics: `tl.mechinterp.measure_trajectory`

`tl.mechinterp.measure_trajectory(model, data)` measures the logit-lens
trajectory: each block's hidden state projected through the model's own
final norm and head (the lens `torchlens.semantic.logit_lens` builds and
validates, or a user `lens=` such as a tuned lens), one distribution per
layer and position. It traces each batch, projects one layer at a time,
reduces it to the named statistics and drops it, so only one
`[batch, positions, vocab]` projection is alive at once. The statistics
(natural log) are `entropy`, `kl_to_final` (KL from the final output
distribution to the layer's), `max_prob`, and, when a batch is
`(input_ids, target_ids)`, `cross_entropy` and `target_rank` of the target
token (rank 1 is the top token; tied logits share the smallest rank). The
result is a `tl.mechinterp.TrajectoryStats`: `position_means[statistic]` is a
float64 `[layers, positions]` tensor of means over every example, and
`running_stats[statistic][layer]` is a `RunningStats` pooled over examples and positions
(`summary(statistic)` returns their results). `mask_input=True` bans each
row's own input tokens before every reduction, final output included; the
same knob on `logit_lens_predictions` masks before its top-k and
`logsumexp`, which cannot be undone afterwards. Refusals carry
`stats_trajectory_*` codes (targets missing for a targeted statistic, an
unknown statistic name, batches of different lengths, `mask_input=True` on
a model fed embeddings, no batches).

```python
from torch import nn
from torchlens.facets import FacetSpec
from torchlens.semantic.recipes._helpers import parameter_spec

class DocBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
    def forward(self, h):
        return h + torch.tanh(self.lin(h))

class DocLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed, self.ln_f = nn.Embedding(20, 8), nn.LayerNorm(8)
        self.blocks = nn.Sequential(DocBlock(), DocBlock())
        self.lm_head = nn.Linear(8, 20, bias=False)
    def forward(self, ids):
        return self.lm_head(self.ln_f(self.blocks(self.embed(ids))))

@tl.facets.register(class_name="DocBlock", target_scope="module", facets=("resid_post",))
def doc_block(module):
    return {"resid_post": FacetSpec.from_home(module.trace.ops[module.calls[0].output_ops[0]])}

@tl.facets.register(class_name="DocLM", target_scope="module", facets=(
    "logits", "unembed_weight", "final_norm_kind", "final_norm_eps",
    "final_norm_gamma", "final_norm_beta"))
def doc_head(module):
    head, norm = module.trace.modules["lm_head"], module.trace.modules["ln_f"]
    return {
        "logits": FacetSpec.from_home(module.trace.ops[head.calls[0].output_ops[0]]),
        "unembed_weight": parameter_spec(head, "weight", "doc_head"),
        "final_norm_kind": "layer_norm", "final_norm_eps": 1e-5,
        "final_norm_gamma": parameter_spec(norm, "weight", "doc_head"),
        "final_norm_beta": parameter_spec(norm, "bias", "doc_head"),
    }

lm = DocLM().eval()
ids = torch.randint(0, 20, (4, 6))
data = [(ids, ids.roll(-1, dims=1))]          # (input_ids, next-token targets)
traj = tl.mechinterp.measure_trajectory(lm, data, mask_input=False)
print(traj)
print(traj.position_means["entropy"].shape)    # [layers, positions]
print(traj.compute_stats("target_rank")["blocks.1"].mean)
```

## Where it sits

- Install both: lovely-tensors for any tensor at a REPL, TorchLens for the
  numbers of a whole model's run (this page across a dataset, the stats line on
  each op of a trace).
- `tl.stats.aggregate` captures with the ordinary verified `tl.trace` engine; for
  tight training loops that only need a few sites, `tl.fastlog.Recorder`
  (built once, `.log()` per step) is the cheaper repeated-capture door.
- The observability kernels (`RunningStats`, `Histogram`) have one implementation
  home shared with the `torchlens.trackers.watch` tiers; see
  [docs/reference/observability_substrate.md](observability_substrate.md).
