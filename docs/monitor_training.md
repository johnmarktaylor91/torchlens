# Monitor training

Watch how the distribution at every site of your model drifts over training,
and find where it broke -- as a portable, exactly-mergeable history artifact
first, and paper-ready static figures second. (Spellings on this page are
DOCUMENTED-UNSTABLE pending the naming session.)

The design credits [torchexplorer](https://github.com/spfrommer/torchexplorer)
(Samuel Pfrommer, Apache-2.0) for the question and the instrument-once
ergonomic, and deliberately refuses its mechanisms: no live web app, no
value-space rebinning, no averaged history columns, no per-site CPU syncs.
Counts merge by integer addition across micro-batches, ranks, time, and runs;
missing is never zero; every approximation is labelled in the record AND on
the figure.

## 1. Watch a training loop

```python
import torch
from torchlens.observability import HistoryCollector, StepTruth

collector = HistoryCollector(model, streams=("activation", "param", "param_delta"))
collector.discover(x)              # one REAL forward; returns your output
print(collector.plan.format_table())  # elements/step AND bytes/step, per site
collector.attach(optimizer=optimizer)

for step, batch in enumerate(loader):
    with collector.step(step, truth=StepTruth()):
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

collector.detach()
```

`sites=` narrows by module-path prefix; a zero-match selection refuses at
plan time, never silently at step N. The plan table prints REDUCED ELEMENTS
per logged step alongside bytes -- elements predict time, bytes predict
storage, and they differ by orders of magnitude (a giant embedding is the
element budget of the whole rest of the model). Cadence is per-stream.

## 2. Read the contact sheet

```python
from torchlens.observability import HistoryView, render_contact_sheet

view = HistoryView.from_collector(collector)
svg = render_contact_sheet(view)          # page 0
```

Tiles are ranked by the named score `nonfinite_first_drift_v1`: sites with
NaN/inf events first, then by mean drift. The "N of M -- page k of P" footer
is mandatory and pagination is deterministic and complete -- there is no
hidden top-N truncation. The first real gpt2 sheet this repo produced is
`docs/images/watch_gpt2_contact_sheet.svg`.

## 3. Drill into a site

```python
from torchlens.observability import render_fan, render_waterfall, render_detail

fan = render_fan(view, site_id)        # the default per-site view
wf = render_waterfall(view, site_id)   # step x signed-log2-bin density
sheet = render_detail(view, site_id)   # every stream on one step axis
```

The fan's median and p25-75/p5-95 bands are DERIVED from the sketch
(approximate, grid resolution disclosed in the caption); the dotted min/max
envelope is EXACT from the always-on spine, so true extremes are never lost
and no outlier-rejection knob exists. Presence gaps, nonfinite events,
optimizer skips, segment boundaries, and coarsened spans are visible marks
on the figure, not footnotes. The waterfall uses fixed signed log2 bins
(bpo=4, window [2^-48, 2^16] per sign), never rebinned; zero, nonfinite,
and out-of-range counts stay exact in separately visible annotated bands.

## 4. Parameters and updates

The five v1 streams are `activation`, `activation_grad`, `param`,
`param_grad`, `param_delta`. The update channel (`param_delta`) uses a
deterministic seeded fp32 subvector of min(numel, 4096) indices per
parameter -- never a lower-precision previous copy (which inflates reported
update norms exactly where you consult them) -- and degrades to EXACT with
`estimated=False` when the sample covers the whole tensor. A skipped
optimizer step records `skipped` and NO fake update.

## 5. AMP, accumulation, and clipping truth

`StepTruth(applied=, scale=, unscaled=, clipped=, micro_batches=)` carries
what your loop can attest; the collector never guesses -- `unknown` is a
legal, honest value that never masquerades as `scaled` or `unscaled`.
Gradient-stream observations always state their scale basis. Pre-clip and
post-clip are distinct phases; reading a series that spans phases without
naming one refuses (`watch_phase_ambiguous`).

## 6. Op and facet grain (the tier module tools cannot see)

```python
import torchlens as tl
from torchlens.observability import OpTierCollector, split_heads

collector = OpTierCollector(model, save=tl.func("softmax"))
collector.discover(x)                      # metadata-only pass; prices the plan
with collector.step(step):
    output = collector.observe_step(x)     # the REAL forward; graph intact
    output.loss.backward(); optimizer.step()
```

Route B rides fastlog: every selected OP becomes a site (567 on gpt2 vs
~125 leaf modules), reduced at the save point by a declared summary
transform. The plumbing that makes this affordable is public:
`torchlens.observability.summary(fn)` declares a NON-DIFFERENTIABLE reducer
(integer outputs legal even under `backward_ready=True`), and with
`save_raw_activations=False` the raw clone is skipped entirely -- reduce,
never copy. `facets={label: split_heads(n, dim=...)}` gives per-head series
with stable identity. A reducer failure on one site records
`capture_failed` for that site and your training step survives.

## 7. Export: dashboards, pandas, resume

```python
from torchlens.observability import histogram_payload

payload = histogram_payload(observation)   # add_histogram_raw's fields,
                                           # canonical bucket_limits/counts
frame = view.to_pandas()                   # long-form rows (needs pandas)
```

NaN/inf counts are excluded from dashboard buckets and disclosed exactly in
`excluded_nonfinite`. Disk artifacts (`WatchSettings(output_dir=...)`) are
versioned, checksummed, crash-readable to the last committed chunk, and load
without torch or the model. A resume declares a NEW segment
(`collector.step(step, new_segment=True)`); the boundary is preserved and
rendered -- never a plausible-but-false continuous history.

## Cost

Every published number comes from the pinned interleaved A/B harness with a
measured same-workload noise band (`docs/_watch_perf_numbers.md` is
generated; numbers are never hand-quoted). Current CPU status, both rows on
a real gpt2 training loop: the always-on SPINE tier measured within the
box's noise band (a disclosed non-result -- not distinguishable from noise
even at cadence 1), while every-site full SKETCHES at cadence 1 on a
deliberately tiny forward measured a published 2.28x -- which is exactly
why sketch cadence defaults sparser than the spine (10-100). The CUDA gate
rows (<=15% module-tier spine, <=30% sketch tier at shipped cadence, <=10%
peak memory) run on a dedicated GPU via the C-EXPLORER row and no GPU claim
is made until they pass.

`torchlens.stats` note: the spine and Histogram kernels implement the
`StreamingStat` protocol, so `tl.aggregate` gets dataset-mode histograms
with the same grid for free -- `aggregate` is the dataset-mode sibling of
the watch (it drives its own dataloader once; the watch rides your live
training loop).
