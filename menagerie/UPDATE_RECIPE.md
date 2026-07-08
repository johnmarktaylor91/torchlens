# Menagerie Update Recipe

Use this procedure for public catalog updates. Keep rendered graphs and local
databases outside commits.

## 1. Choose the Submission Path

There are two source-of-truth paths:

- **Non-classics:** append one typed JSONL record to
  `menagerie/data/master_catalog.jsonl`. Deferred non-classics go in
  `menagerie/data/deferred.jsonl` with a `deferral.reason`.
- **Classics:** add or edit a Python module under `menagerie/classics/` and expose
  entries through `MENAGERIE_ENTRIES` or the existing classics registry pattern.
  Classics are not duplicated in JSONL; the registry is their sole source.

SQLite files are derived caches. `catalog.db` is rebuilt from JSONL plus the classics
registry. `verification.db` is the append-only audit/provenance database.

## 2. Decide Whether the Entry Is Distinct

Before adding a row, search for the name, aliases, paper acronym, and family:

```bash
rg -i "candidate|alias|paper_acronym" menagerie/data/master_catalog.jsonl menagerie/classics
python -m menagerie.catalog build
python -m menagerie.catalog query --family candidate
python -m menagerie.catalog recipe candidate
```

Use the `variant` field only when two rows are genuinely different designs under the
same `(name, zoo)` natural key. Do not use it for ordinary scale, checkpoint,
resolution, dataset, or backbone swaps. If the architecture is the same design, keep
one family entry and document aliases or caveats in `notes`.

## 3. Write the Recipe

Preferred non-classics recipe type:

- `import-callable`: a constructor expression with explicit imports. Use this whenever
  the model can be built from a normal import and call.

Legacy or exceptional recipe types:

- `expression`: an eval expression. This is reported as code execution but is not
  quarantined by itself.
- `statement`: statement code assigning `model`. Simple import-plus-constructor
  statements are allowed; statements with local classes/functions, lambdas, control
  flow, or dynamic `exec`/`eval` are quarantined.
- `exec-string`: arbitrary multi-line exec body. Discouraged and quarantined.

Every record has an always-callable `input` builder. The builder must return the real
runtime input object whenever possible. Use `NoInput` only for wrappers whose forward
method intentionally ignores the input, and set `input_is_real=false` honestly. Do not
reintroduce prose input parsing.

## 4. Validate the Source

Run the schema gate before rendering:

```bash
python -m menagerie.tools.validate_catalog
python -m menagerie.catalog build
python -m menagerie.catalog stats
python -m menagerie.status
```

The validation pre-commit gate rejects unknown fields, duplicate natural keys, classics
rows in JSONL, and deferred rows without a reason.

## 5. Render, Validate, Verify, Status

Render a delta or focused sample:

```bash
python -m menagerie.generate_menagerie \
  --only-new \
  --out-dir /tmp/torchlens_menagerie_gallery
```

Validate TorchLens replay separately:

```bash
python -m menagerie.validate_menagerie \
  --out-dir /tmp/torchlens_menagerie_validation \
  --no-install-deps
```

Use the gates before finishing:

```bash
python -m menagerie.tools.validate_catalog
python -m menagerie.tools.parity_check
python -m menagerie.status
python -m menagerie.status --provenance
```

`status` reports the honest funnel: total catalog models, expected models, rendered
coverage, current-version validation, deferred rows, narrowed quarantined arbitrary-exec
rows, and the separate count of all models built via code execution.

## 6. Record Provenance

When a discovery sweep adds entries, record the wave in `verification.db` through
`menagerie.provenance.record_sweep(...)` or the historical importer pattern in
`menagerie.tools.import_provenance`. Set `added_wave` on new JSONL rows to the
corresponding `sweep_id` so `python -m menagerie.status --provenance` can join models
back to the sweep.

## 7. Commit Scope

Commit only public source changes: JSONL records, classics modules, tests, and docs.
Do not commit generated `catalog.db`, `verification.db`, `.candidate` files, or rendered
gallery output.
