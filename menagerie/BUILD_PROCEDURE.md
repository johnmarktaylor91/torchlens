# Menagerie Bulk Build Procedure

This procedure is for adding large batches of model families to the Menagerie. It
complements `DISCOVER_MODELS.md` for search, `UPDATE_RECIPE.md` for one-off model
adds, `METHODOLOGY.md` for standards, and `SEARCH_PROVENANCE.md` for audit logs.

## Pipeline

1. Discover candidates.
   - Use `menagerie/DISCOVER_MODELS.md` as the durable search prompt.
   - Seed broad searches with `python -m menagerie.discover_crawler`.
   - Record each sweep, source class, date, and unresolved lead in
     `menagerie/SEARCH_PROVENANCE.md`.

2. Reconcile against what is already built.
   - Rebuild the dedup baseline before every new batch.
   - Compare candidates against the three registry stages: public catalog,
     `menagerie/classics/`, and active staging/quarantine work.
   - Dedup by model name and by computational graph distinctness. The durable
     distinctness rule is: a unique model is a unique computational graph,
     represented by `graph_shape_hash`. `resnet18`, `resnet34`, and `resnet50`
     are distinct; the same model at a different input resolution is not a new
     model. Preserve aliases and names so every model remains findable.

3. Triage candidates into build tracks.
   - Real importable library class.
   - Vendorable real repository code.
   - Faithful port from real code.
   - Faithful reimplementation from a sufficiently detailed description.
   - Skip only when there is genuinely no code anywhere and no sufficient
     description anywhere.

4. Build with the source ladder.
   - Rung 1: use a real library class.
   - Rung 2: vendor real repository code with minimal compatibility shims.
   - Rung 3: make a faithful port from available source.
   - Rung 4: make a faithful reimplementation from a detailed paper, appendix,
     config, or architecture description.
   - Rung 5: skip only if the search evidence proves that no usable code or
     sufficient description exists.

5. Integrate.
   - Place staged modules under `.research/menagerie-redesign/tierA/staging/`.
   - Each module must define `MENAGERIE_ENTRIES` as
     `(name, build, example, year, code)`. `build` and `example` may be either
     callables or string attribute names in the module.
   - Run the integrator with a long timeout:

     ```bash
     python3 .research/menagerie-redesign/tierA/integrate.py
     ```

   - Allow at least 5 minutes. Startup can take 1-2 minutes because staged
     modules may import heavy ML stacks.
   - The integrator preserves the trace gate: every entry must pass
     `torchlens.trace(build(), example())` before moving into
     `menagerie/classics/` or appending to the JSONL catalog.
   - Failed staged modules move to
     `.research/menagerie-redesign/tierA/staging/quarantine/`; reasons are
     written to
     `.research/menagerie-redesign/tierA/staging/quarantine_reasons.tsv`.

6. Build env-island models.
   - For models marked `needs_env`, use the environment-island flow instead of
     weakening the main trace gate.
   - Keep dependency-specific work isolated, then bring back only trace-verified
     registry outputs.

7. Validate.
   - Check that classics load cleanly:

     ```bash
     python3 -c "import warnings;warnings.filterwarnings('ignore');from menagerie.classics import CLASSICS,CLASSICS_LOAD_ERRORS;print(len(CLASSICS),len(CLASSICS_LOAD_ERRORS))"
     ```

   - Run the applicable Menagerie validation and project quality gates for the
     changed surface.

## Build-Tier Assignment

Use evidence to decide which agent does which work.

- Codex vendors real code for rungs 1-2 and performs cheap triage. This is the
  zero-cheat path: import the real class or vendor the real repository code.
- Sonnet reimplements from detailed descriptions for rungs 3-4.
- Codex is not trusted to reimplement from descriptions in this pipeline; it is
  too likely to fabricate plausible architecture details.
- Every model must pass the trace gate, and every batch needs audit sampling
  against the source evidence.

## Skip Maxim

Skip only when there is genuinely no code anywhere and no sufficient description
anywhere. Never fabricate missing architecture details. Document every skip's
search evidence in `SEARCH_PROVENANCE.md`; skips are audited so valid models are
not lost.

## Adding a Later Batch

1. Rebuild the dedup baseline against the catalog, classics, staging, and
   quarantine.
2. Run discovery and log the sweep in `SEARCH_PROVENANCE.md`.
3. Triage candidates into the source ladder and assign build tiers by evidence.
4. Stage real-code modules or recipe TSVs.
5. Run `python3 .research/menagerie-redesign/tierA/integrate.py` with at least a
   5 minute timeout.
6. Build `needs_env` models through environment islands.
7. Validate registry load count and run the applicable tests before committing.
