# torchlens.neuro -- the treaty desk

torchlens owns everything that touches the network; rsatoolbox and Brain-Score
own everything that touches the brain or the hypothesis. `torchlens.neuro`
hands representations across that seam with provenance that cannot lie, and
teaches where everything else lives. Every spelling on this page is
DOCUMENTED-UNSTABLE pending the naming sprint.

Install: `pip install "torchlens[neuro]"` (rsatoolbox, every supported
Python). The Brain-Score seam is a separate extra:
`pip install "torchlens[brainscore]"` (brainscore-vision, Python >= 3.11).
`__all__`/`dir()` advertise only the names whose dependency is present;
accessing an unadvertised name teaches the exact install command.

## The map (mirrors `help(torchlens.neuro)`)

1. **Extract** (needs no extra): `tl.trace(model, x, save=...)`,
   `tl.extract_dataset(..., output_dir=..., stimulus_ids=...)`,
   `tl.load_extraction(path)`.
2. **Geometry** (needs no extra): `tl.repgeom.rdm` / `classical_mds` /
   `scree` / `effective_dimensionality` / `procrustes_align`,
   `tl.stats.cka`. Nothing is re-exported into neuro.
3. **Handoff** (`neuro` extra): `neuro.datasets(...)`, `neuro.rdms(...)`.
4. **Score** (`brainscore` extra): `neuro.activations_extractor(...)`,
   `neuro.get_activations_fn(...)` -- live-gate verified (below).
5. **Refusals**: noise ceilings, crossnobis/mahalanobis, searchlight,
   bootstrap/permutation/model fitting, encoding models, CCA/SVCCA/PWCCA,
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
a pinned test fails loudly if that ever changes). A `chunked=` execution
seam is reserved for streaming-scale RDMs; launch is eager and any value
refuses typed.

`neuro.rdms` implements no comparison, fitting, ceiling, or inference --
that vocabulary belongs to rsatoolbox, and the refusal rows point there.

## The Brain-Score adapters (live-gate verified)

`neuro.activations_extractor(model, preprocessing, ...)` builds a real
`brainscore_vision` `ActivationsExtractorHelper` over TorchLens capture;
`neuro.get_activations_fn(model, ...)` is the underlying per-batch callable
(standalone-testable; benchmark layer lists may address functional ops, not
just modules). Exposed in neuro because the end-to-end live gate PASSED
(2026-08-29): a real resnet18 IMAGENET1K_V1 checkpoint through a real
`ActivationsExtractorHelper` and `StimulusSet` against a live
brainscore-vision 2.3.22 install on Python 3.12 -- values, presentation
order, layer coordinates, logits, dotted module paths, a functional-op
mapping, a short final batch, and CPU behavior matched direct torchlens
extraction, and default sites ran on a PARTIALLY saved trace. CUDA is not
claimed (no covered runner). The offline `per_layer` recipe keeps its
bridge-only spelling: a generic score-a-callable loop must not borrow a
benchmark's authority.

## The CKA method contract (published)

| Function | Estimator | Precision | Inputs | Degenerate case |
|---|---|---|---|---|
| `tl.stats.cka(a, b)` | Linear CKA, BIASED estimator (Kornblith et al. 2019 gram-matrix form; no unbiased-HSIC correction) | CPU `float64` | matched `[n_obs x n_features]` pairs (rows are observations; mismatched row counts refuse) | zero-variance input returns `NaN` |

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
