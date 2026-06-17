# Audit Notebooks — Maintenance Recipe

Audience: agents and developers maintaining the `notebooks/audit/` tree.

## What this folder is

Coverage-optimized notebooks that exercise EVERY human-facing TorchLens surface.
Sliced by **user workflow** (not data-structure, not submodule).  Three locked decisions:

1. **Public/committed.** Notebook source, `_models.py`, README, CLAUDE.md, and
   `visual/generate_visual_pack.py` + `visual/coverage_matrix.md` are all tracked.
   Heavy regenerable artifacts (executed HTML, intermediate PDFs, the stapled PDF) are
   gitignored via `.gitignore` in this directory.
2. **Sliced by user workflow.** Each notebook = one user-facing workflow, not one class.
3. **Visual pack = single script** (`visual/generate_visual_pack.py`), render-then-staple.

## How to run the full audit suite

```bash
# Activate the torchlens dev environment first, then:

cd /path/to/torchlens

# 1. Smoke-test the model zoo
python notebooks/audit/_models.py

# 2. Execute all notebooks (saves outputs in-place)
for nb in notebooks/audit/[0-9]*.ipynb; do
    echo "=== $nb ==="
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

# 3. Export to HTML for review (goes to _exports/)
mkdir -p notebooks/audit/_exports
for nb in notebooks/audit/[0-9]*.ipynb; do
    jupyter nbconvert --to html --output-dir notebooks/audit/_exports/ "$nb"
done

# 4. Regenerate the visual pack
python notebooks/audit/visual/generate_visual_pack.py
# => writes notebooks/audit/visual/visual_audit.pdf
```

## Lockstep rule — MANDATORY

Whenever a **public surface is added, renamed, or removed** anywhere in `torchlens/`,
update the matching audit notebook AND the `visual/coverage_matrix.md` **in the same
commit**. This mirrors the glossary lockstep rule from `CLAUDE.md` at the project root.

Concretely:
- Added `tl.foo` -> add a cell in the appropriate notebook; update README coverage row.
- Renamed `trace.bar` -> find it in the "Surfaces covered" list at the top of the
  notebook (the checklist is the rename target), update the cell, re-execute.
- Removed a surface -> mark it as a GAP callout (see below) so the removal is visible.

## Centralized vocabulary convention

Each notebook starts with a markdown cell listing **"Surfaces covered"** as a checkbox
list.  This is the canonical rename checklist: when a name changes, search `README.md`
and that checklist first, then update the code cells.  Never hard-code a name only in
a code cell without listing it in the header — future-you won't find it.

## The GAP callout rule — never fake output

Every notebook ends with a **"⚠️ GAPs / ergonomic smells"** markdown cell.  If a
surface errors or doesn't exist:
- Write a short `⚠️ GAP: expected X, got Y` note in that cell.
- Keep the failing code cell but wrap it in a try/except that prints the error.
- **Never comment out a failure silently. Never fabricate output.**

This is a tripwire, the same spirit as `validation/`.  A GAP callout = the audit
working correctly.  Silencing it defeats the point.

## How to add a new notebook

1. Copy the structure from `00_setup_and_first_capture.ipynb`:
   - Cell 1 markdown: title + purpose + "Surfaces covered" checklist.
   - Sections: markdown header + code cell + shown human output.
   - Final cell: "⚠️ GAPs / ergonomic smells".
2. Add an entry to `README.md`'s coverage matrix.
3. Execute green: `jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb`.
4. Commit with `docs(audit): ...` (NOT feat/fix/perf -- those cut a release).

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

## Commit discipline

- Type: `docs(audit):` or `chore(audit):` ONLY. NEVER `feat`, `fix`, `perf` -- those
  trigger semantic-release and cut a package version bump.
- No AI attribution anywhere (commits, PR bodies, code comments).
- If `git commit` fails on `.git/index.lock`, wait 2s and retry once.
- Pre-commit hooks (ruff-format/eof) may auto-fix files and fail the first attempt:
  re-add the changed files and re-commit; confirm HEAD advanced.
