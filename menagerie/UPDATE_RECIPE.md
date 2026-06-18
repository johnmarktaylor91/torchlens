# Menagerie Update Recipe

Use this procedure for incremental catalog updates. Keep bulk rendered graphs outside the
repository.

## 1. Discover Candidates

Run the process in `DISCOVER_MODELS.md`.

Expected output:

- genuine new families
- grep evidence proving absence
- rejected already-present candidates
- rejected variant/config/backbone candidates
- ambiguous cases

Do not add scale variants, backbone swaps, dataset configs, or checkpoint names as new
families.

## 2. Verify and Deduplicate

Search the existing catalog before editing:

```bash
rg -i "candidate|alias|paper_acronym" menagerie/data/master_catalog.tsv
python -m menagerie.catalog query --family candidate
python -m menagerie.catalog recipe candidate
```

For each accepted row, fill the source schema:

```text
name    zoo    constructor_call    input_shape    input_dtype    family    domain    era    notes
```

Prefer tiny random-init constructors. If a row is only a public config sketch, say that in
`notes` rather than pretending it is renderable.

## 3. Append Source Rows

Append new rows to `menagerie/data/master_catalog.tsv`. Keep it TSV-clean: exactly nine
columns, no embedded tabs.

Rebuild canonical outputs:

```bash
python -m menagerie.catalog build
python -m menagerie.catalog stats
```

Review the new family/domain counts. If normalization created a duplicate spelling,
update `catalog.py` normalization rules and rebuild.

## 4. Render Only New Deltas

If the previous highest `model_id` was `10216`, render only new rows:

```bash
python -m menagerie.generate_menagerie \
  --since 10216 \
  --only-new \
  --out-dir /tmp/torchlens_menagerie_gallery
```

For local smoke tests without installing dependencies:

```bash
python -m menagerie.generate_menagerie \
  --since 10216 \
  --only-new \
  --no-install-deps \
  --timeout-sec 60 \
  --out-dir /tmp/torchlens_menagerie_gallery
```

The renderer appends `manifest.tsv`, skips rows already rendered, and rebuilds gallery
indexes unless `--skip-index` is set.

## 5. Rebuild Indexes

After rendering or copying manifests between machines:

```bash
python -m menagerie.generate_menagerie \
  --index-only \
  --out-dir /tmp/torchlens_menagerie_gallery
```

Review:

- `INDEX.md`
- `FEATURED.md`
- representative `domain/INDEX.md`
- representative `domain/family/INDEX.md`

## 6. Record Outcomes

Summarize:

- rows added
- families added
- constructors verified
- rows rendered
- rows skipped with reasons
- dependency clusters that failed
- any normalization rule changes

Do not commit bulk graph output. Only commit catalog/tooling/docs changes that are meant
to be public.
