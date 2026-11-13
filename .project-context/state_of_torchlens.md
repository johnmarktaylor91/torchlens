# State of TorchLens 3.x

A short map of the package as it stands. The public API is defined by the glossary
(`docs/reference/glossary.md`); when this page and the glossary disagree, the glossary wins.

## What TorchLens is

TorchLens runs a model and records every tensor operation as a graph: operations, modules,
parameters, buffers, activations, gradients, and source metadata. PyTorch eager capture is the
stable path; JAX, TensorFlow, MLX, PaddlePaddle and tinygrad are previews (`docs/backends.md`).
Torch functions are wrapped lazily on the first `tl.trace()` call and stay wrapped; the wrappers
check a logging flag, so calls outside a capture take the original PyTorch path. The main output
is a `Trace`, whose layers are aggregate `Layer` objects over per-pass `Op` records.

## Public API

`torchlens.__all__` (in `torchlens/__init__.py`) is the root surface. In groups:

- Capture and extraction: `trace`, `record` / `Recording` (sparse, predicate-selected
  recording; the `recording` namespace), `pluck`, `extract`, `extract_dataset`, `summary`,
  `draw`, `release_model`, `clear_cache`.
- Records and containers: `Trace`, `Layer`, `Op`, `Bundle`, `Recipe`, `Edit`.
- Save and load: `trace.save(path)` and `tl.load(path)`; `to_disk` for streaming saves.
- Validation: `validate`.
- Selectors: `label`, `func`, `module`, `where`, `in_module`, `site`, `head`, `facet`,
  `followed_by`, `preceded_by`, `label_contains`, `label_matches`, `all_sites`, `grad` and its
  siblings, `at_step`, plus unit selectors such as `top_k`, `dead`, `changed` and `subspace`.
- Interventions: `trace.do(site, edit)` and `when(where, what)` for live edits; edit helpers
  such as `zero_ablate`, `mean_ablate`, `scale`, `steer`, `clamp`, `add_noise`,
  `project_out`, `replace_with`, `patch_from` and `splice_module`; `sweep` for
  dose-response runs.
- Namespaces: `options`, `stats`, `facets`, `export`, `fingerprint`, and the submodules
  reached as attributes (`tl.types`, `tl.io`, `tl.viz`, `tl.debug`, `tl.report`, `tl.neuro`,
  `tl.backends`, `tl.agent`, ...); `torchlens.mechinterp` is imported explicitly.

A removed top-level name raises an `AttributeError` that names its new home
(`docs/reference/deprecations.md`).

## Saved format

`trace.save()` writes a directory of typed JSON record files plus safetensors tensor blobs;
nothing is pickled (`docs/maintainers/saved_format_3.md`, code in `torchlens/_io/format3/`).
Artifacts saved by TorchLens 2.x are not read: `tl.load` raises `tl.errors.ArtifactTooOldError` (code
`artifact_version_below_floor`).

## Where to look

| Question | Start here |
|----------|------------|
| Logging flag and global state | `torchlens/_state.py` |
| Torch wrapping | `torchlens/backends/torch/wrappers.py`, `torchlens/backends/torch/model_prep.py` |
| Forward capture | `torchlens/_capture/trace.py` |
| Raw operation records | `torchlens/backends/torch/ops.py` |
| Backward capture | `torchlens/backends/torch/backward.py` |
| Labels, loops, branches | `torchlens/_postprocess/labeling.py`, `loop_detection.py`, `control_flow.py` |
| Layer and module records | `torchlens/_postprocess/finalization.py`, `torchlens/_data_classes/` |
| Field orders | `torchlens/_constants.py` |
| Save and load | `torchlens/_io/bundle.py`, `torchlens/_io/format3/` |
| Selectors | `torchlens/_selection/selectors.py` |
| Interventions | `torchlens/intervention/` |
| Bundles and comparisons | `torchlens/bundle/` |
| Sparse recording | `torchlens/recording/` |
| Graphviz rendering | `torchlens/_visualization/` |
| Validation | `torchlens/validation/` |
| Environment and model report (`tl.report.doctor`) | `torchlens/report/_doctor.py`, `torchlens/compat/_report.py` |
| Optional integrations | `torchlens/bridge/` |
| Release guards | `scripts/check_release_major.py`, `scripts/check_no_breaking_markers.py`, `.github/workflows/release.yml` |

## Release

A release is a maintainer-pushed `vX.Y.Z` tag on main whose `pyproject.toml` version and
`torchlens/__init__.py:__version__` already carry that version (`.github/workflows/release.yml`).
`scripts/check_release_major.py` stops an unapproved major version before every build. A major
version ships only on the maintainer's explicit command, through `.github/MAJOR_RELEASE_APPROVED`.

## Deeper docs

- `README.md` for the user tour; `docs/how-it-works.md` for the capture mechanism.
- `docs/reference/glossary.md` for every public name.
- `LIMITATIONS.md` and `ROADMAP.md` for limits and plans.
- Root `AGENTS.md` and the per-subpackage `AGENTS.md` files under `torchlens/` for
  implementation maps; `architecture.md` beside this page for the architecture narrative
  (verify against code).
