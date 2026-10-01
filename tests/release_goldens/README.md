# Release golden corpus (harvested bytes -- NEVER regenerated)

`genuine_release_artifacts.tar.gz` is the harvested release-artifact corpus:
seven genuine `.tlspec` artifacts written by six distinct torchlens writers
(five released wheels plus the unreleased schema-8 main tree), each produced
by running the era-appropriate generator script under that exact installed
release. The bytes are the authority; the scripts are provenance, not a
recipe for regeneration.

Pinned digest (enforced by `tests/test_tlspec_envelope_goldens.py`):

    sha256 3429a84fbd406d374afde5b50f6a8be0867ada77b728d6e88309713c6eadcdb3
    size   4804520 bytes

## Governance (plan of record: ecosystem MEMO section 3.7)

- Goldens are harvested BYTES with provenance rows. They are NEVER
  regenerated from tags: tags are not a time machine (v2.18.0's git-LFS
  object is a proven 404; v2.19/v2.20 refuse python 3.10; the v2.16 API has
  no `level=` kwarg). Parts of this corpus are IRREPLACEABLE.
- No golden is regenerated in place. Deleting one requires a ledger
  retirement row in the artifact-compatibility ledger.
- Hand-authored fixtures are negative/forgery tests only and never live in
  this directory.
- This directory is test data. It ships in the sdist/tests tree only, never
  in the wheel.

## Contents

| Member | Writer | tlspec | Notes |
|---|---|---|---|
| `art_v2.16.0_portable/` | torchlens 2.16.0 | io_format_version=2 | The only genuine v2.16.0 ModelLog bundle in existence (resnet18, 111 layers). Uses `io_format_version` / `n_activation_blobs` / `n_gradient_blobs`; the checked-in "v2.16" fixtures elsewhere use modern spellings and are negative tests only. |
| `art_v2.31.0_audit/` | torchlens 2.31.0 | 6 | Audit-level save. |
| `art_v2.31.0_portable/` | torchlens 2.31.0 | 6 | Producer-floor writer. |
| `art_v2.32.4_portable/` | torchlens 2.32.4 | 6 | |
| `art_v2.33.0_portable/` | torchlens 2.33.0 | 6 | Same stamp as v2.34.1, different persisted grammar -- the drift pair the writer contract digest exists to catch. |
| `art_v2.34.1_portable/` | torchlens 2.34.1 | 6 | Same stamp as v2.33.0, different persisted grammar. |
| `art_main_portable/` | unreleased main (self-reports 2.34.1) | 8 | Dev-build writer; support class internal-dev, never invented stable history. |

All artifacts: torchvision resnet18 IMAGENET1K_V1 at 1x3x32x32 random input,
written on py3.10.20 / torch 2.13.0+cu130. Per-artifact rows: `PROVENANCE`.

`generators/` holds the era generator scripts as run at harvest time (one
`# ruff: noqa` provenance header added at import; nothing else touched)
(`write_artifact.py` for the level-aware modern API, `write216.py` for the
v2.16 API, plus the `load216.py` / `symmetry.py` / `ceiling.py` reader
probes). `reference_manifests/` holds the seven manifest.json files extracted
at harvest for zero-cost inspection without unpacking the tarball.
