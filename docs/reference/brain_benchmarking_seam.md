# The brain-benchmarking seam: nine methods, one boundary

TorchLens draws one explicit identity boundary for brain-benchmarking work
(the seam): **geometry ours, inferential statistics theirs**. For each of
Net2Brain's nine evaluation methods, TorchLens supplies every geometric
prerequisite -- features, RDMs, alignment, interchange -- and refuses every
inferential step with a typed error naming the tool that owns it. This page
is the ledger of that boundary (brainpipe memo D-24). Every TorchLens
spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

The working surfaces behind the "we supply" column:

- `torchlens.brainpipe.extraction_plan` / `ExtractionPlan.run` -- the
  memory-planned whole-model sweep: two measured probes, a printable plan
  table (measured bytes, peak pairs with backend named, exact pass counts),
  and a run door onto the atomic, resumable extraction artifact.
- `torchlens.brainpipe.export_npz` -- npz interchange, including
  Net2Brain's per-stimulus layout with their lexicographic file-naming
  contract and a name-to-stimulus-id sidecar.
- `tl.repgeom.rdm` -- euclidean / manhattan / cosine / correlation RDMs,
  GPU compute device, row-blocked memory-bounded computation, condensed
  output, explicit batched `[B, N, D]` input.
- `tl.repgeom.rdm_compare` -- DESCRIPTIVE Pearson/Spearman/Kendall tau-a
  over aligned strict upper triangles (diagonal excluded). No p-values, no
  noise ceilings: asking for them refuses naming rsatoolbox.
- `tl.stats.CKA` / `tl.stats.cka` -- one-shot and streaming linear CKA
  (memory-bounded over arbitrarily many stimuli), `device=`/`dtype=`.
- `tl.bridge.rsatoolbox.dataset` -- the per-site rsatoolbox handoff.
- `tl.bridge.brain_score` -- the Brain-Score `ActivationsExtractorHelper`
  seam (live verification is CI-gated; the docstring's UNVERIFIED marker
  leaves only when the L1/L2 job passes).

| # | Net2Brain method | We supply | We refuse and point |
|---|---|---|---|
| 1 | RSA | Per-site/per-time RDMs at scale, stimulus alignment, square/condensed npz + rsatoolbox export, descriptive `rdm_compare` | Brain-RDM acquisition, subject aggregation, t-tests, correction, noise ceilings, %R2, ranking -> Net2Brain `RSA.evaluate` / rsatoolbox |
| 2 | Weighted RSA | Ordered multi-site RDM stacks with stable site identity | Weight fitting, regularization, folds, ceilings -> rsatoolbox `ModelWeighted` / Net2Brain `WRSA` |
| 3 | Cross-temporal RSA | Pass-qualified per-timestep RDMs from ANY recurrent model with zero per-model configuration (Net2Brain needs hand-written per-network cleaners); frame/time pooling | Cross-temporal orchestration and inference -> Net2Brain CT-RSA |
| 4 | Searchlight RSA | Model RDMs with stimulus identity; npz interchange | Brain-space neighborhoods, maps, tests -> Net2Brain `Searchlight` / rsatoolbox / nilearn |
| 5 | CKA | One-shot + streaming linear CKA (memory-bounded; DeepJuice advertises no equivalent), `device=`/`dtype=`, convention recorded | Kernel CKA (documented non-offer), permutation/subject inference, rankings -> Net2Brain `CKA.run` / statistics tools |
| 6 | Distributional comparison | Aligned features; streaming marginals when that accumulator lands | Binning/KDE/optimal-transport estimator choices, JSD/Wasserstein aggregation -> Net2Brain / scipy / POT |
| 7 | Linear/ridge encoding | Aligned, ordered, site-keyed design matrices; the sweep's guaranteed row order; raw inputs for fold-local learned reductions | Splits, ridge fits, alpha selection, voxel scores -> himalaya / sklearn / Net2Brain `Encoding` |
| 8 | veRSA | Features, plus `rdm()` accepting externally predicted responses (documented composition) | Encoding fit and veRSA orchestration -> Net2Brain veRSA / rsatoolbox |
| 9 | Stacked encoding + variance partitioning | Multi-site banks with GUARANTEED identical row order and stable identities (the sweep's stimulus-order invariant) | Stacked fits, voxel R2, unique/shared variance, layer selection, SEM/tests -> Net2Brain / brainML / himalaya |

Two rows are genuinely stronger than the competition and are safe to lead
with: **CT-RSA** (row 3: pass-qualified per-timestep RDMs fall out of
TorchLens's recurrence grouping with zero per-model work) and **stacked
encoding** (row 9: its prerequisite -- a complete, honestly-labeled,
identically-ordered per-layer feature bank -- IS the sweep).

What the trade is, stated honestly (memo section 1): module-hook extractors
(DeepJuice) have no capture floor and no overhead multiple precisely
because module hooks do not see functional ops, fused attention, or
per-pass recurrence. TorchLens pays a capture cost and in exchange sees
every operation, prints a measured cost table before running, emits a
manifest a reviewer can audit, and never silently drops a site. Coverage
and provenance are the trade; this page names it rather than claiming
dominance on both axes.
