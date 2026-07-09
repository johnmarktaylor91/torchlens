# Audit Notebooks — Maintenance Recipe

Audience: agents and developers maintaining the `notebooks/audit/` tree.

## What this folder is

Coverage-optimized notebooks that exercise EVERY human-facing TorchLens surface.
Sliced by **user workflow** (not data-structure, not submodule). Locked decisions:

1. **Public/committed.** Notebook source, `_models.py`, README, CLAUDE.md, and
   `visual/generate_visual_pack.py` + `visual/coverage_matrix.md` are all tracked.
   Heavy regenerable artifacts (executed HTML, intermediate PDFs, the stapled PDF) are
   gitignored via `.gitignore` in this directory.
2. **Sliced by user workflow.** Each notebook = one user-facing workflow, not one class.
3. **Visual pack = single script** (`visual/generate_visual_pack.py`), render-then-staple.
4. **The suite executes green end-to-end.** Awkward/broken surfaces are shown via GAP
   cells that PRINT the problem (guarded with try/except); they never crash the run.

## Import pinning — the notebooks audit THEIR OWN checkout

torchlens is often pip-installed `-e` against a *different* checkout, and nbconvert
executes each notebook with `notebooks/audit/` as cwd — so a bare `import torchlens`
would silently audit the wrong code. Two defenses are in place; keep both:

1. Every setup cell inserts the repo root at `sys.path[0]` and **asserts** that
   `torchlens.__file__` resolves under this checkout. Copy that block into any new
   notebook (take it from `00_setup_and_first_capture.ipynb`).
2. The run recipe below also exports `PYTHONPATH` as belt-and-braces.

## How to run the full audit suite

```bash
# Activate the torchlens dev environment first, then:

cd /path/to/torchlens
export PYTHONPATH="$(pwd)"   # ensure the notebooks import THIS checkout

# 1. Smoke-test the model zoo (~1 min)
python notebooks/audit/_models.py

# 2. Execute all notebooks in place, sequentially (~10-14 min total;
#    most notebooks take 20-60 s; torch import dominates, so do NOT parallelize)
for nb in notebooks/audit/[0-9]*.ipynb; do
    echo "=== $nb ==="
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

# 3. Export to HTML for review (goes to _exports/; the canonical review surface)
mkdir -p notebooks/audit/_exports
for nb in notebooks/audit/[0-9]*.ipynb; do
    jupyter nbconvert --to html --output-dir notebooks/audit/_exports/ "$nb"
done

# 4. Regenerate the visual pack
python notebooks/audit/visual/generate_visual_pack.py
# => writes notebooks/audit/visual/visual_audit.pdf

# 5. BEFORE COMMITTING: strip outputs from the on-disk notebooks
nbstripout notebooks/audit/[0-9]*.ipynb
```

Step 5 matters: the `detect-secrets` pre-commit hook scans WORKING-TREE files, and
executed notebooks with inline images (the collapse slider embeds, matplotlib figures)
contain base64 blobs it flags as high-entropy secrets. The `nbstripout` git filter
only cleans the staged blob, not the file on disk — so commit attempts fail until you
strip. The HTML in `_exports/` keeps the executed outputs for review.

Execution-time expectations: 00-06 and 08-10 run in ~20-30 s each; 07 (intervention),
11 (visualization), 12 (debug), 13 (export sweep) run ~30-90 s; 14 (HF, guarded)
~60-90 s when the tiny models are cached, seconds when skipped. A failing notebook
stops `nbconvert`; add `--allow-errors` only when triaging, never for the committed run.

## Lockstep rule — MANDATORY

Whenever a **public surface is added, renamed, or removed** anywhere in `torchlens/`,
update the matching audit notebook AND `README.md`'s coverage matrix **in the same
commit**. This mirrors the glossary lockstep rule in the project root `CLAUDE.md`
(spec drives code; a rename is not done until docs + notebooks match).

Concretely:
- Added `tl.foo` -> add a cell in the workflow notebook where a user would meet it;
  add it to that notebook's "Surfaces covered" checklist; update the README row.
- Renamed `trace.bar` -> search each notebook's "Surfaces covered" checklist and
  `README.md` first (they are the rename targets), then the code cells; re-execute.
- Removed a surface -> keep a GAP callout showing the removal so it stays visible.
- New submodule family (a new `tl.xyz`) -> decide which USER WORKFLOW it belongs to;
  only create a new notebook if it is genuinely a new workflow (15/16 are the pattern).

## Coverage is derived, never asserted

Do not hand-maintain coverage counts. Re-derive them:

```python
# from the repo root, with PYTHONPATH set as above
import json, glob, re, torchlens as tl
text = "\n".join(
    "".join(c["source"]) for p in glob.glob("notebooks/audit/[0-9]*.ipynb")
    for c in json.load(open(p))["cells"]
)
missing = [n for n in sorted(tl.__all__) if not re.search(rf"\b{re.escape(n)}\b", text)]
print(len(tl.__all__), "names;", "missing:", missing)
```

Notebook 00 performs the runtime half of this check (every `__all__` entry must
`getattr` cleanly).

## The GAP callout rule — never fake output

Every notebook ends with a **"⚠️ GAPs / ergonomic smells"** markdown cell. If a
surface errors or feels awkward:
- Keep the failing code cell, wrapped in try/except that PRINTS the error.
- Write a short `⚠️ GAP: expected X, got Y` note in the final cell.
- **Never comment out a failure silently. Never fabricate output.**

This is a tripwire, the same spirit as `validation/`. A GAP callout = the audit
working correctly. Silencing it defeats the point.

### Re-verifying GAP markers (do this every overhaul pass)

GAPs go stale in both directions: bugs get fixed, and new regressions appear. On each
maintenance pass:
1. Re-run the minimal repro for every GAP bullet against the live code (a scratch
   script is fine; the notebook cells themselves are the repros).
2. A FIXED gap: delete the failing framing, keep a one-line
   `*(fixed since YYYY-MM)*` note in the GAP cell and README so reviewers know the
   history.
3. A still-live gap: keep it verbatim.
4. Anything NEW goes in the GAP cell of the notebook that surfaces it AND the
   README row.

## How to add a new notebook

1. Copy the structure from `00_setup_and_first_capture.ipynb`:
   - Cell 1 markdown: title + purpose + "Surfaces covered" checklist.
   - Setup cell WITH the repo-root pin + `tl.__file__` assert.
   - Sections: markdown header + code cell + shown human output.
   - Final cell: "⚠️ GAPs / ergonomic smells".
2. Add an entry to `README.md`'s coverage matrix.
3. Execute green: `jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`.
4. Commit with `docs(audit): ...` (NOT feat/fix/perf — those cut a release).

## How to add a new visual page

1. Add a row to `visual/coverage_matrix.md` with: model name, options dict, what to nit-check.
2. `generate_visual_pack.py` reads the matrix and rebuilds `visual_audit.pdf` idempotently.
3. Run: `python notebooks/audit/visual/generate_visual_pack.py`

## Model zoo

`_models.py` is the single source of truth for all tiny models used across notebooks.
- Do NOT import from `tests/` (tests are not a package).
- Every entry must pass `tl.trace(model, x)` before being added.
- Run `python _models.py` (from `notebooks/audit/`) to smoke-test all entries.
- ZOO is a `dict[str, callable]` mapping name -> zero-arg factory returning `(model, x)`.
- `deep_blocks` exists specifically for collapse-v2 (deep enough for a multi-step
  schedule); keep it 6 blocks unless notebook 11 changes with it.

## Commit discipline

- Type: `docs(audit):` or `chore(audit):` ONLY. NEVER `feat`, `fix`, `perf` — those
  trigger semantic-release and cut a package version bump.
- No AI attribution anywhere (commits, PR bodies, code comments).
- Cell ids are sequential strings ("0", "1", ...). After inserting cells, renumber —
  nbformat's random hex ids trip the `detect-secrets` pre-commit hook.
- Pre-commit hooks (ruff-format/eof/nbstripout) may auto-fix files and fail the first
  attempt: re-add the changed files and re-commit; confirm HEAD advanced.
- If `git commit` fails on `.git/index.lock`, wait 2s and retry once.
