# conda-forge recipe

`recipe.yaml` is the canonical conda-forge recipe for torchlens. It uses the v1
recipe format (CEP 13, built by `rattler-build`), which is the format
conda-forge's staged-recipes repository documents for new packages.

The package is `noarch: python`: one build serves every platform and every
Python that conda-forge supports at or above its global `python_min`. It is built
from the PyPI sdist of the release named in `context.version`, installed with
pip, with these run requirements:

| pyproject.toml dependency | conda-forge package |
| --- | --- |
| `numpy>=1.26.4` | `numpy >=1.26.4` |
| `packaging>=23.2` | `packaging >=23.2` |
| `torch>=2.1` (with per-Python floors) | `pytorch >=2.1` |
| `safetensors>=0.4` | `safetensors >=0.4` |
| `tqdm>=4.66` | `tqdm >=4.66` |
| `graphviz>=0.20.1` | `python-graphviz >=0.20.1` |
| `typing_extensions>=4.0` | `typing_extensions >=4.0` |
| `pillow>=9` | `pillow >=9` |

The per-Python torch floors in `pyproject.toml` (2.2 on Python 3.12, 2.6 on
3.13 and later) have no conda counterpart: each conda-forge `pytorch` build is
tied to the interpreters it was built for, so the solver already pairs them, and
the recipe keeps the lowest floor. Optional extras (`[neuro]`, `[tabular]`,
`[viz]` and the rest) are not run requirements; users add those packages to their
environment themselves. The recipe's Python floor is conda-forge's `python_min`,
which may be higher than `requires-python` because conda-forge drops old
interpreters on its own schedule.

The recipe's tests import torchlens, run `pip check`, and trace a two-layer MLP
with `tl.trace`, on the oldest and the newest Python the channel offers.

## Keeping it in sync

`tests/test_conda_recipe_sync.py` fails when this recipe drifts from
`pyproject.toml`: the version, the runtime dependencies and their floors, the
setuptools build floor, the license, or an optional extra appearing in the run
requirements. When a release bumps the version in `pyproject.toml` and
`torchlens/__init__.py`, bump `context.version` here in the same commit. The
`source.sha256` can only be filled once that sdist is on PyPI:

```bash
pip download torchlens==X.Y.Z --no-deps --no-binary :all: -d /tmp/tl-sdist
sha256sum /tmp/tl-sdist/torchlens-X.Y.Z.tar.gz
```

After the conda-forge feedstock exists, conda-forge's update bot opens a pull
request on the feedstock for each new PyPI release (new version and sha256), so
this copy matters mainly for the first submission and for changes to the
dependency list, which the bot does not infer.

## Building it locally

With `rattler-build` installed from conda-forge, and a variant file that names
conda-forge's current `python_min` (for example `python_min: ["3.11"]` in
`variants.yaml`):

```bash
rattler-build build --recipe packaging/conda/recipe.yaml -m variants.yaml -c conda-forge
```

`conda-smithy recipe-lint --conda-forge packaging/conda` runs the same linter
that staged-recipes runs on a submission.

## Submitting to conda-forge

1. Fork `conda-forge/staged-recipes` and create `recipes/torchlens/recipe.yaml`
   as a copy of this file (check `context.version` and `source.sha256` name the
   latest PyPI release).
2. Open the pull request; every name under `extra.recipe-maintainers` must
   confirm in it that they agree to maintain the package.
3. Once it is merged, conda-forge creates the `torchlens-feedstock` repository
   and publishes the package; after that, recipe changes go to the feedstock.
