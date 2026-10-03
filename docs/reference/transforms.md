# Built-in activation transforms (`tl.transforms`)

Every spelling on this page is DOCUMENTED-UNSTABLE pending the naming
sprint; `tl.transforms` is the placeholder home (its collision with the
input-side `transform=` slot is a named naming-sprint item).

TorchLens ships a curated library of declared transforms for the common
activation reductions — pooling, flatten, cast, normalize, sparse random
projection (SRP), PCA-apply — so they are one word instead of a lambda:
seeded, provenance-recorded, axis-aware, composable. A built-in is a frozen
`TransformSpec` with a validated `plan` and a pure `apply`; the canonical
JSON of a chain is its resume identity, and every numerics-visible change
mismatches. Raw callables keep working, recorded as identification-only.

```python
import torchlens as tl
from torchlens import transforms as T

paths = tl.extract_dataset(
    model, stimuli,
    layers=["layer3", "layer4", "avgpool"],
    output_dir="run/",
    transform={
        "layer3": T.chain(T.flatten(), T.srp(1024, seed=0)),
        "layer4": T.srp(4096, seed=0),
        T.DEFAULT: T.chain(T.flatten(), T.unit_norm(), T.cast("bfloat16")),
    },
)
```

## Six things worth knowing before a harvest

Every number below names the corpus that produced it, and yours will differ
(contract clause T-C12). Two corpora appear: the design panel's **128 varied
COCO test2017 images** and this repo's **vendored fixture** (128 rows
augmented from 32 CC0/public-domain photographs,
`tests/transforms_corpus/`), both measured on real resnet50 IMAGENET1K_V2
features with the shipped `very_sparse_fixed` construction (v1).

1. **SRP's guarantee is Euclidean, and it delivers.** L2-RDM Spearman
   0.989-0.9995 on resnet50 layer4/avgpool over the 128 COCO images at
   k >= 1024 (the lowest measured k), and 0.947-0.991 across the panel's
   corpus ladder at k=1024. On the vendored fixture: 0.977-0.978 at k=1024,
   0.995 at k=4096 (layer4). No k=256 L2 number exists for the COCO corpus;
   the fixture measures 0.927.
2. **If your analysis is a correlation-distance RDM — most RSA is — expect
   real loss.** COCO corpus, resnet50 layer4: corr-RDM Spearman 0.28 at
   k=256, 0.49 at k=1024, 0.71 at k=4096, 0.84 at k=16384. Vendored
   fixture: 0.43-0.48 at k=256, 0.69-0.73 at k=1024, 0.87 at k=4096.
3. **The reason is your RDM's dynamic range, not k alone.** The measured
   spread BETWEEN corpora at the same site and k is larger than the spread
   across k — run `T.srp_fidelity_probe(pilot_batch, n_components=(...))`
   on one batch of your own stimuli before committing a harvest. The
   probe's report carries its scope fields, so pasted numbers stay honest.
4. **Pool before you project; normalize AFTER you project.** Pool-first
   measured 0.73 vs 0.49 at matched k (COCO corpus); the
   center+normalize-FIRST recipe was refuted in all 8 measured cells (worse
   by 0.06-0.22) and is deliberately not shipped as a preset.
5. **SRP is for strong compression.** Near k ~ D/2, plain coordinate
   subsampling wins (by 0.18, COCO corpus); at strong compression
   (k <= D/8) SRP earns its keep — +0.09-0.14 on the COCO corpus and
   +0.36-0.40 on the vendored fixture (the self-baselining O1d gate).
6. **Seeds share matrices under the default policy.** With
   `share="by_extent"`, five runs with the SAME seed (default or explicit)
   share ONE projection; independence needs DISTINCT seeds under either
   policy. The realized seed and its source (`"explicit"` /
   `"library_default"`) are always recorded, so the manifest reconstructs
   every projection.

A retraction note, kept on purpose: the panel's first fidelity table
(r = 0.914-0.996) was measured on 128 augmentations of ONE photograph and
collapsed to 0.28-0.84 on diverse images. It was deleted, not supplemented,
and the corpus-admissibility gate (O14: `n_distinct >= 16`, effective rank
>= 40, corr-distance CoV <= 0.12, asserted on the fixture ITSELF) exists so
a future degenerate fixture fails loudly instead of silently republishing a
flattering table.

## The roster

| Name | What it does | Contract highlights |
|---|---|---|
| `srp(n_components, seed=0, density="auto", construction=..., mode=..., share=...)` | Sparse random projection | `n_components` REQUIRED; seed + seed_source recorded; two named constructions; binds one extent per (chain, site) and REFUSES drift |
| `pool_tokens(op, assume_no_padding=False)` | Token mean/max/first/last | Three-valued mask policy; `first`/`last` gather VALID indices, never physical endpoints |
| `cls_token(assume_no_padding=False)` | The CLS gather | DISTINCT semantic name gated on recorded `SpecialTokenFacts` — gpt2 REFUSES a fictional CLS |
| `pool_spatial(op="mean")`, `channel_mean()` | Spatial/channel reductions | Resolve DECLARED roles or refuse teaching both remedies; rank is never evidence |
| `reduce(op, axis)`, `take_index(axis, index)`, `flatten(start_axis=1)`, `unit_norm(axis=-1)`, `magnitude()`, `cast(dtype)` | Role-free kernels | Explicit axes; the stimulus axis is inviolable; float-only cast; fp32 accumulation for half inputs |
| `project(basis, center=None)` | Frozen linear projection | Digest-addressed: records carry the digest, never values; unstaged digests refuse with restaging remedies |
| `pca_apply(fitted)` | Apply a fitted PCA | Consumes `tl.stats.PCA(...).fitted(...)` / `tl.stats.load_fitted(path)`; centered; fit scope recorded. Fit-during-write is REFUSED (early and late shards would get different coordinates) |
| `srp_dims_for(n, eps)` | Sizing helper | The DENSE-JL L2 heuristic, honestly labeled: a poor guide for corr-RDMs, and it says so |
| `srp_fidelity_probe(pilot, n_components)` | Fidelity probe | Measures L2-RDM and corr-RDM fidelity per candidate k on YOUR pilot batch; report carries scope |

Chains compose left to right (`T.chain(a, b)` or `[a, b]` — one object, one
record); per-site `Mapping` dispatch (with `T.DEFAULT`) resolves before
coercion; bare strings name only zero-parameter deterministic presets.
Custom transforms register versioned via `register_transform` and cannot
shadow builtins; artifact loading never imports or executes code.

## The SRP contract, briefly

The default construction `very_sparse_fixed` places EXACTLY
`m = round(density * D)` distinct nonzeros per output column (deterministic
counter-hash rejection; realized nnz equals `k * m`; signs keyed to
positions; values `+/- 1/sqrt(density * k)`). It is NOT the i.i.d.
Bernoulli matrix of the JL literature — the fixed per-column count has
variance 0 where the i.i.d. hypothesis requires 44-316 — so it records
`distance_claim="empirical_only"` at every density: the oracle suite is its
entire warrant, and it is geometrically indistinguishable from the i.i.d.
construction on real activations (the O1e gate). `iid_bernoulli` IS the
i.i.d. matrix (the O1f distribution-identity gate proves it) and ALSO
records `empirical_only` until the literature task verifies Achlioptas 2003
against the paper itself; its O(D*k) generation cost is disclosed at plan
time. Generation is block-independent and bit-identical under any column
chunking, the canonical matrix is digested before any layout conversion
(`srp_verify_matrix` re-derives and checks), and the realized per-site
facts (`srp_realized_facts`: extent, effective seed, construction, realized
density/nnz, digest, multiply path, share key, distance claim) are what the
extraction manifest embeds.

GPU/MPS: no on-device number is published anywhere in this page — the
machine-gated suite (`tests/test_transforms_lib_gpu_gate.py`, the C-XFORM
cluster row) must run green first. On CPU, the `dense` / `sparse_csr` /
`dense_chunked` multiply paths all run one row-local fold (elementwise
multiplies and adds in canonical slot order), so they are bit-identical and a
projected row never depends on which rows share its batch (T-C10); byte
reduction on the
measured chains was 92-184x — while taking ZERO off the retained forward
peak (that win belongs to emission-time streaming; every built-in declares
`stream_safe` and aliasing facts for it, and nobody may cite this library
for a peak-memory claim).

## Credits

scikit-learn (`SparseRandomProjection` and the
`johnson_lindenstrauss_min_dim` practice of shipping a sizing helper, kept
with its scope relabeled); Achlioptas 2003 and Li/Hastie/Church 2006
(constructions implemented and selected deliberately — cited as lineage,
never as a guarantee until the literature task verifies them);
Dasgupta/Kumar/Sarlos 2010 and Kane/Nelson 2014 (the fixed-sparsity
sparse-JL literature the open task must check); DeepJuice (on-device
reduction precedent); Net2Brain (reduction-before-PCA placement);
thingsvision (pooling/flatten vocabulary); sentence-transformers and
Hugging Face (masked-mean reference semantics); timm/torchvision (pooling
precedent); baukit (running statistics); COCO (the diverse-stimulus
corpus, credited WITH the licensing caveat that keeps it un-vendorable —
the repo's precision path pins URL + sha256 and downloads on demand).
