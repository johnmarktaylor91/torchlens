# torchlens.neuro -- the treaty desk

torchlens owns everything that touches the network; rsatoolbox and Brain-Score
own everything that touches the brain or the hypothesis. `torchlens.neuro`
hands representations across that seam with provenance that cannot lie, and
teaches where everything else lives. Every spelling on this page is
DOCUMENTED-UNSTABLE pending the naming sprint.

Install: `pip install "torchlens[neuro]"` (rsatoolbox and xarray, every supported
Python). The Brain-Score seam is a separate extra:
`pip install "torchlens[brainscore]"` (brainscore-vision, Python >= 3.11).
Keep the two in separate environments: brainscore-vision 2.3 pins
scikit-learn<1.6 while rsatoolbox 0.3 needs scikit-learn>=1.6, so
`torchlens[neuro,brainscore]` resolves but `import rsatoolbox` fails
(measured 2026-09-02; an upstream pin, not a torchlens choice).
`__all__`/`dir()` advertise only the names whose dependency is present;
accessing an unadvertised name teaches the exact install command.

## The map (mirrors `help(torchlens.neuro)`)

1. **Extract** (needs no extra): `tl.trace(model, x, save=...)`,
   `tl.extract_dataset(..., output_dir=..., stimulus_ids=...)`,
   `tl.load(path)`.
2. **Geometry** (needs no extra): `tl.repgeom.rdm` / `classical_mds` /
   `scree` / `effective_dimensionality` / `procrustes_align`,
   `tl.stats.cka`. Nothing is re-exported into neuro.
3. **Handoff** (`neuro` extra): `neuro.datasets(...)`, `neuro.rdms(...)`.
4. **Score** (`brainscore` extra): `neuro.activations_extractor(...)`,
   `neuro.get_activations_fn(...)` -- live-gate verified (below).
5. **Compare** (needs no extra): `neuro.compare(source, target,
   analyses=("rsa", "ridge", "cka"))` -- descriptive RSA, cross-validated
   ridge and linear CKA on one extraction (below).
6. **Refusals**: noise ceilings, crossnobis/mahalanobis, searchlight,
   inferential bootstrap/permutation/model fitting (descriptive intervals on
   TorchLens's own scores are `tl.repgeom.rdm_compare_interval` /
   `cka_interval`), encoding-model inference, CCA/SVCCA/PWCCA,
   non-metric MDS, model zoos, brain data. Each attempted name raises a
   teaching refusal naming the owner, the missing ingredient, and the
   torchlens spelling that feeds it.

## neuro.datasets

```python
import torchlens as tl

log = tl.trace(model, images, save=tl.in_module("layer1"))
site_datasets = tl.neuro.datasets(log, stimulus_ids=ids)   # per-site rsatoolbox Datasets
site_datasets.ledger                                       # per-site computed/skipped report
```

Returns an insertion-ordered mapping of canonical site key to
`rsatoolbox.data.Dataset`. `source` is a Trace, a `LoadedExtraction`, or an
extraction-directory path. `sites=None` sweeps every STIMULUS-INDEXED saved
site through the core eligibility gate (buffer overwrites are skipped with
one summarized disclosure -- never fabricated into stimulus-space results);
an explicit lookup refuses ineligible sites actionably, and a bare
multi-pass layer lookup names the pass-qualified alternatives.

Descriptors are the product -- rsatoolbox promotes Dataset descriptors into
RDM descriptors automatically (measured at 0.1.5 and 0.3.2), so everything
written here survives into figures: site, requested lookup, structural
`site_key`, layer label, `pool` (ALWAYS present; `"flatten"` when nothing
was applied -- the readout is scientific provenance, never sugar), source
kind, per-stimulus shape, dtype and any handoff cast (`float16`/`bfloat16`
widen to `float32`, never `float64`), torchlens and rsatoolbox versions,
model identity, and an intervention marker on edited traces.

Observation rows always carry `tl_presentation_index`: rsatoolbox's
`calc_rdm(descriptor=...)` returns rows in SORTED descriptor order, and this
always-written original-row index is measured (both versions) to be permuted
along with the data, so presentation order is always recoverable. Stimulus
identity follows the authority rule: extraction-artifact ids are
authoritative (an explicit `stimulus_ids=` acts as validation, never silent
replacement); explicit ids on a bare Trace validate length; absent both, a
positional index is DISCLOSED as synthetic. `obs=` columns (conditions,
runs, repeats) are length-checked before rsatoolbox receives them. Channel
rows use a neutral `feature_index` (never "neuroid") plus factual unravelled
unit coordinates.

## neuro.rdms

```python
rdms = tl.neuro.rdms(log)                          # source mode: computed, correlation default
rdms = tl.neuro.rdms(log, metric="euclidean")      # any tl.repgeom metric
external = tl.neuro.rdms({"V4": matrix}, dissimilarity_measure="crossnobis")  # matrix mode
```

ONE name, two type-distinct modes, one shared condensed converter
(`square[np.triu_indices(n, k=1)]`, measured to match rsatoolbox's own
convention at both versions -- no scipy dependency):

- **Source mode** (the taught path): computes through the canonical
  `tl.repgeom` arithmetic and records the metric ACTUALLY used
  (`measure_source="computed"`) -- when the function computes, the label
  cannot lie. Defaults to `metric="correlation"` (field canon for the RSA
  audience, and the one metric measured to match rsatoolbox exactly);
  this deliberately diverges from `tl.repgeom.rdm`'s euclidean default, and
  both docstrings cross-reference it. A pinned anti-divergence test holds
  site selection and exception classes identical to calling
  `tl.repgeom.rdm_evolution` directly.
- **Matrix mode** (the escape hatch): converts already-computed square
  matrices only; `dissimilarity_measure=` is REQUIRED (a bare matrix has
  forgotten its metric) and `measure_source="declared"`. Validates square
  shape, symmetry, zero diagonal, finite values, shared pattern counts, and
  descriptor lengths BEFORE rsatoolbox sees anything. It never invents
  pattern identity: supply `pattern_descriptors=`, or rows carry a
  positional index disclosed as positional.

The modes reject mixed arguments. `rdm_descriptors` additionally carry
`measure_convention` -- the formula in words -- because rsatoolbox's
`"euclidean"` equals torchlens's euclidean SQUARED and divided by
`n_features` (a Pearson-visible discrepancy produced by nothing but a
convention), and rsatoolbox has NO cosine RDM method (ours is real and
tested; their `calc_rdm(method="cosine")` raises `NotImplementedError`, and
a pinned test fails loudly if that ever changes). Source mode also records
`similarity_repository_id`, the RDM variant's identifier in the
similarity-repository catalogue (for example `rdm/*/correlation`; absent for
a metric the catalogue does not list; DOCUMENTED-UNSTABLE spelling). A `chunked=` execution
seam is reserved for streaming-scale RDMs; launch is eager and any value
refuses typed.

`neuro.rdms` implements no comparison, fitting, ceiling, or inference --
that vocabulary belongs to rsatoolbox, and the refusal rows point there.

## The Brain-Score adapters (live-gate verified)

`neuro.activations_extractor(model, preprocessing, ...)` builds a Brain-Score
activations model over TorchLens capture, shaped like Brain-Score's own
`PytorchWrapper` (its `_extractor` is a real `brainscore_vision`
`ActivationsExtractorHelper`), so it goes straight into a model plugin's
`ModelCommitment(activations_model=...)` or is called with stimuli and
layers directly; `neuro.get_activations_fn(model, ...)` is the underlying per-batch callable
(standalone-testable; benchmark layer lists may address functional ops, not
just modules). Exposed in neuro because the end-to-end live gate PASSED
(2026-08-29): a real resnet18 IMAGENET1K_V1 checkpoint through a real
`ActivationsExtractorHelper` and `StimulusSet` against a live
brainscore-vision 2.3.22 install on Python 3.12 -- values, presentation
order, layer coordinates, logits, dotted module paths, a functional-op
mapping, a short final batch, and CPU behavior matched direct torchlens
extraction, and default sites ran on a PARTIALLY saved trace. CUDA is not
claimed (no covered runner). End to end through a plugin (2026-10-14):
a Brain-Score model plugin registering `ModelCommitment(activations_model=
neuro.activations_extractor(resnet18, ...))` was scored by
`brainscore_vision.score` on one neural benchmark
(`MajajHong2015public.IT-pls`) and one behavioral benchmark
(`Geirhos2021colour-top1`, which reads the `"logits"` layer) against
brainscore-vision 2.4.1 / brainscore-core 2.5.0 on Python 3.11, CPU; on 24
stimuli every committed layer and the logits matched Brain-Score's
hook-based `PytorchWrapper` exactly. Brainscore-vision 2.4.1 pins xarray
2022.3.0, whose backend discovery fails on Python 3.12, so score on Python
3.11. The offline `per_layer` recipe keeps its
bridge-only spelling: a generic score-a-callable loop must not borrow a
benchmark's authority.

## neuro.compare: RSA, ridge and CKA on one extraction

```python
import torchlens as tl

trace = tl.trace(model, images, save=...)
result = tl.neuro.compare(trace, brain, analyses=("rsa", "ridge", "cka"))
print(result.table())                  # one row per site, one column per analysis
print(result.disagreement().table())   # where the analyses rank sites differently
result = tl.neuro.compare(trace, brain, analyses=("rsa", "ridge", "cka"), num_resamples=1000)
```

`neuro.compare` needs no extra (it is numpy-only). It reads each site's
activations once from what the source already holds -- a Trace with saved
activations, a `LoadedExtraction` or extraction directory, or a
`{site: [n_stimuli, ...]}` mapping -- and never re-runs the model. `brain`
is any `[n_stimuli, ...]` target for the same stimuli in the same order
(recordings, or another model's features). Every score is a descriptive
similarity on these stimuli:

- `"rsa"`: `tl.repgeom.rdm_compare(rdm(site, rdm_metric), rdm(target,
  rdm_metric), rsa_method)`; defaults `rdm_metric="correlation"`,
  `rsa_method="spearman"`.
- `"ridge"`: a cross-validated ridge score. The stimuli are split into
  `n_folds` folds (default 5) by a permutation seeded with `seed`. On each
  training fold, features and target are centred by the training means and
  one penalty for all target columns is chosen from `ridge_alphas` (absolute,
  on unscaled features; default 1e-2 to 1e8) by the exact leave-one-out
  squared error of the training fold. The held-out fold is predicted, and the
  score is the mean over target columns of the Pearson r between out-of-fold
  predictions and the target (constant columns are left out of the mean).
  `result.ridge_alphas_chosen[site]` lists each fold's penalty.
- `"cka"`: linear CKA with the biased Gretton HSIC, the variant
  `tl.stats.cka` computes.

With `num_resamples=`, each score gets a stimulus-bootstrap `ScoreInterval`
(the ridge interval resamples the evaluation of the out-of-fold predictions
and does not refit). `result.similarity_repository_ids` names the catalogued
variant of each analysis (ridge has none). Inference, voxel maps, variance
partitioning and stacked fits stay with rsatoolbox, himalaya, scikit-learn
and Net2Brain; this call makes no claim of matching any of them.
The refused analyses have recipe pages: [TorchLens to netrep](../neuroai/netrep.md)
(Procrustes, CCA and the shape-metric family) and
[TorchLens to himalaya](../neuroai/himalaya.md) (banded ridge over a multi-site
feature bank).

## Uncertainty on TorchLens's own scores

`tl.repgeom.rdm_compare_interval(rdm_a, rdm_b, method, num_resamples=1000,
seed=0, confidence=0.95)` and `tl.repgeom.cka_interval(a, b, ...)` return a
`ScoreInterval`: the unchanged point score (exactly `rdm_compare`; linear CKA
as `tl.stats.cka` computes it) with a percentile bootstrap interval over
stimuli beside it, plus the settings that made it (`num_resamples`, `num_valid_resamples`,
`seed`, `confidence`, `basis`). RDM resamples drop the entries that pair a
stimulus with a copy of itself; CKA resamples reindex the stimulus Gram
matrices, so each resample costs O(n^2) whatever the feature count. The same
seed gives the same interval, and identical inputs give a zero-width
interval at 1.0.

These intervals are descriptive: they say how much TorchLens's score moves
when the stimulus sample changes. They are not hypothesis tests, they do not
model subjects or measurement noise, and they make no claim of parity with
rsatoolbox's bootstrap. Inferential RSA (corrected bootstraps, model
comparison, noise ceilings) stays with `rsatoolbox.inference`; asking
`tl.neuro` for `bootstrap` still refuses and points here for the descriptive
interval. The stimulus bootstrap is known to be conservative for RDM
comparisons (it tends to cover more often than its nominal rate).

## Raw beside ceiling-normalised scores

TorchLens computes no ceilings. Where the Brain-Score bridge reports a
ceiling-normalised score, the raw score sits beside it under its own name:
`tl.bridge.brain_score.per_layer(...)` returns, per layer, a `ScoreParts`
record (`raw`, `ceiling_normalised`, `ceiling`, `error`, `note`), and
`tl.bridge.brain_score.score_parts(score)` reads one Brain-Score `Score` the
same way. A result that carries a ceiling attribute is ceiling-normalised:
its value goes to `ceiling_normalised` and its `"raw"` attribute to `raw`. A
result without one (a behavioral engineering score, or a plain number from
an offline callable) is raw, and `ceiling_normalised` is `None`. Nothing is
recomputed; Brain-Score's normalisation is its own.

## Where metrics disagree

`tl.repgeom.metric_disagreement(scores)` takes `scores[metric][site]` (floats
or `ScoreInterval`s from one extraction) and returns a `MetricDisagreement`:
per-metric ranks (1 is the highest score), the pairs of sites that two
metrics order oppositely, `table()` for a plain-text report and `to_dict()`
for data. When the scores carry intervals, each flagged pair says whether the
opposite orders are separated by the intervals or lie within their overlap.

```python
report = tl.repgeom.metric_disagreement(
    {"rsa": {"layer1": 0.31, "layer2": 0.52}, "cka": {"layer1": 0.74, "layer2": 0.69}}
)
print(report.table())
```

## The CKA method contract (published)

| Function | Estimator | Precision | Inputs | Degenerate case |
|---|---|---|---|---|
| `tl.stats.cka(a, b)` | Linear CKA, BIASED estimator (Kornblith et al. 2019 gram-matrix form; no unbiased-HSIC correction) | CPU `float64` | matched `[n_obs x n_features]` pairs (rows are observations; mismatched row counts refuse; NaN/Inf input refuses `cka_input_nonfinite`, as `tl.repgeom.rdm` does) | zero-variance input returns `NaN` with a `cka_zero_variance` warning |

Reference values are pinned in `tests/test_neuro_pkg_geometry.py`; an
estimator change fails the literals. RBF/debiased estimators are
approved-with-conditions and deliberately DECOUPLED from the neuro launch;
no `device=` flag ships without a designed dtype/memory contract, and no
`svcca` kernel value is reserved (SVCCA is a different statistic -- future
CCA-family functions land as siblings over the same matched pairs).

## Version matrix

The provenance claim rests on a foreign package's behaviour, so the
convention/ordering/averaging suite (`tests/test_neuro_pkg_matrix.py` and
friends) is pinned at BOTH rsatoolbox versions: 0.1.5 (the declared floor)
and current 0.3.x. Verified rows include descriptor promotion, condensed
ordering, the euclidean convention RELATION, correlation equality, the
sorted-output recovery via `tl_presentation_index`, and mean-then-compute
condition averaging.

## Credit

rsatoolbox (the Dataset/RDMs descriptor model, condensed convention,
comparison and inference vocabulary this page points at); Brain-Score (the
`ActivationsExtractorHelper` contract); thingsvision (the Gaussian RDM
bandwidth convention behind `tl.repgeom.rdm(metric="gaussian")` and the
rank-scaled RDM display behind `tl.repgeom.rank_transform_rdm`); Nili et
al. 2014 (the percentile-ranked RDM display convention); netrep and
himalaya (the shape-metric and encoding-model homes the refusals name).
