# TorchLens Menagerie

The menagerie is a public, reproducible toolkit for cataloging neural-network model
families and rendering their TorchLens computational graphs. It has two parts:

- `catalog.py`: builds, normalizes, queries, and exports the model catalog.
- `generate_menagerie.py`: resolves dependency clusters, renders models one at a time,
  purges per-model caches, and writes browsable gallery indexes.

Bulk graph output should go outside the repository, for example
`/tmp/torchlens_menagerie_gallery` or a large external volume.

## Catalog Schema

The source TSV lives at `menagerie/data/master_catalog.tsv` and uses nine columns:

```text
name    zoo    constructor_call    input_shape    input_dtype    family    domain    era    notes
```

`python -m menagerie.catalog build` normalizes that source into:

- `menagerie/data/catalog_canonical.tsv`
- `menagerie/data/catalog.db`

The canonical rows add:

- `model_id`: stable integer row id after sorting.
- `family_normalized`: canonical family label.
- `verified`: recipe signal inferred from notes and known zoos.

The catalog intentionally keeps distinct rows when the same model name appears in different
zoos or has different constructor/input recipes.

## Catalog Commands

```bash
python -m menagerie.catalog build
python -m menagerie.catalog stats
python -m menagerie.catalog query --family vit --zoo timm --verified --limit 10
python -m menagerie.catalog query --domain detection --limit 10
python -m menagerie.catalog recipe resnet18
```

The importable API lives in `menagerie.catalog`: `build_canonical_rows()`,
`write_catalog()`, `load_rows()`, `find_recipe()`, and `stats()`.

## Rendering

Render a small verified sample:

```bash
python -m menagerie.generate_menagerie \
  --verified-only \
  --name alexnet \
  --name resnet18 \
  --name efficientnet_b0 \
  --max-models 10 \
  --out-dir /tmp/torchlens_menagerie_gallery \
  --no-install-deps
```

Render output is organized as:

```text
OUT_DIR/
  INDEX.md
  FEATURED.md
  index.html
  featured/
  <domain>/<family>/<model>.svg
  <domain>/<family>/INDEX.md
  <domain>/INDEX.md
  manifest.tsv
```

The manifest is append-only and resumable. Its status values include:

- `rendered`
- `skipped:<reason>`
- `failed:<reason>`

Useful render controls:

```bash
# Full run, dependency installs enabled by default.
python -m menagerie.generate_menagerie --out-dir /big/menagerie

# Rebuild only indexes from the current catalog and manifest.
python -m menagerie.generate_menagerie --index-only --out-dir /big/menagerie

# Incremental render by catalog id.
python -m menagerie.generate_menagerie --since 10216 --only-new --out-dir /big/menagerie

# Retry failures without redoing rendered rows.
python -m menagerie.generate_menagerie --retry-failed --out-dir /big/menagerie
```

## Disk Safety

Before every model, the renderer checks free space on the output filesystem. If free
space is below `--min-free-gb` (default `15`), the run aborts.

For each model, it snapshots cache contents before construction and removes only cache
entries created during that model. The guarded cache roots include Hugging Face, Torch
Hub, Torch checkpoints, and timm caches. Each model also gets an isolated temporary
directory under `OUT_DIR/_tmp`, which is deleted after the worker exits.

Each render runs in a child process with `--timeout-sec`. A timeout or exception is
recorded in the manifest and the run continues.

## Dependency Resolution

The renderer infers dependency clusters from `zoo` and `constructor_call`, installs each
cluster once with pip, then processes all rows in that cluster. Use `--no-install-deps`
for dry local tests or environments where dependencies are already installed.

Rows are honestly skipped when the catalog recipe is not runnable PyTorch code, such as
JAX-native entries, web/config sketches, missing public code, or weights-gated models
without a random-init path.

## Validating Every Model

`validate_menagerie.py` runs TorchLens replay validation over menagerie recipes without
rendering graphs:

```bash
python -m menagerie.validate_menagerie \
  --verified-only \
  --out-dir /tmp/torchlens_menagerie_validation \
  --no-install-deps
```

Validation is independent from rendering. It uses its own default output directory and
append-only `validation_manifest.tsv`, then writes `validation_summary.json` and
`VALIDATION_REPORT.md`. The renderer's `manifest.tsv` is not read or updated.

The default `--scope forward` calls `torchlens.validate_forward_pass(...,
validate_metadata=True)` for the claim that saved activations replay the forward pass
and satisfy metadata invariants. `--scope forward+backward` additionally tries backward
validation with a scalar loss over floating tensor outputs.

Useful validation controls:

```bash
# Retry only non-validated rows in the validation manifest.
python -m menagerie.validate_menagerie --revalidate-failed

# Rebuild validation_summary.json and VALIDATION_REPORT.md from the manifest.
python -m menagerie.validate_menagerie --report-only

# Validate a tiny local sample.
python -m menagerie.validate_menagerie \
  --zoo classics-pytorch \
  --subset 3 \
  --no-install-deps \
  --out-dir /tmp/val_smoke
```

## Cross-environment rendering (rerun-safe)

Some zoos require mutually incompatible dependency stacks, so `run_across_envs.py`
orchestrates rendering or validation across separate conda environments named
`tlmenagerie_<env>`. The one-command dry-run for a repeatable render pass is:

```bash
python -m menagerie.run_across_envs --task render --dry-run --out-dir /big/menagerie
```

Only `--execute` creates environments, installs packages, and runs jobs:

```bash
python -m menagerie.run_across_envs --task render --execute --out-dir /big/menagerie
```

Environment recipes live in `menagerie/data/env_specs.json`, not in code. Each recipe
declares the conda Python version, ordered packages, optional extra pip index, zoo
patterns, an import-based `post_install_check`, status, and notes. Plain `pip_packages`
entries install with pip; `mim:<package>` entries install with `mim install` after
`openmim` is available.

The runner is idempotent. If `tlmenagerie_<env>` already exists and its
`post_install_check` succeeds, setup is skipped and the existing environment is reused.
If setup succeeds, `env_specs.json` is updated with `status: "working"` and captured
`pip freeze` output so future reruns keep the resolved recipe. If setup fails, the
recipe is marked `failed`, the per-env error is written under the output directory's
`env_logs/`, and the runner continues to the next environment.

Disk safety is checked before creating or installing an environment. The default
threshold is `--min-free-gb 20`; an environment is skipped with a clear log message if
free space is below the threshold. Use `--cleanup-env` to remove each
`tlmenagerie_<env>` after its render or validation task finishes, which keeps at most
one extra environment around at a time. The default keeps environments for faster
reruns.

To add a new dependency island, edit `menagerie/data/env_specs.json`: add a new key,
choose `zoo_patterns` that match the catalog `zoo` field, set a conservative
`post_install_check`, leave `status` as `untested`, then run:

```bash
python -m menagerie.run_across_envs --setup-only --envs <env> --execute
```

## Update Flow

See `DISCOVER_MODELS.md` for the recurring discovery prompt and `UPDATE_RECIPE.md` for
the incremental update procedure.
