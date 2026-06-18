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

## Update Flow

See `DISCOVER_MODELS.md` for the recurring discovery prompt and `UPDATE_RECIPE.md` for
the incremental update procedure.
