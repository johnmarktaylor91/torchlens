# Facet-Recipe Maintenance Sweep (durable prompt)

You are auditing TorchLens's semantic facet recipes (`torchlens/semantic/recipes/`)
against the current model landscape. Re-run this sweep periodically, after each
conference cycle, or whenever a more capable model becomes available (a smarter
auditor finds more gaps). Your output is **PROPOSALS FOR REVIEW ONLY** -- you must
NEVER register, merge, or auto-apply a facet recipe. A wrong facet label is a
confidently mislabelled part of someone's model, which is strictly worse than no
label; every proposal goes through human review.

## Mission

Hunt exhaustively for architecture families whose modules TorchLens traces but does
not semantically classify, and for classified modules whose recipes produce wrong or
needlessly absent facets. Be adversarial: assume the current recipe set is
incomplete and biased toward GPT-2/Llama-era decoder LMs.

## Inputs

1. Run the repeatable audit for coverage evidence:

   ```bash
   python tools/facet_maintenance/run_facet_audit.py --out /tmp/facet-proposals
   ```

   Extend the roster with `--spec your_models.py` (a module exposing
   `iter_models()` yielding `(name, model, input_args)`), e.g. drawn from the
   Model Menagerie catalog (modelmenagerie.ai) or fresh HF releases. Traces feed
   `torchlens.semantic.facet_coverage`, whose per-module rows split
   recipe-classified modules from structural-only ones and carry typed absence
   reasons.

2. The registered-recipe inventory: `tl.facets.list()` and
   `torchlens.semantic.recipes.BUILTIN_FACET_CAPABILITY_INVENTORY`.

## Hunt axes (cover ALL of them)

- **Attention variants**: MQA/GQA, sliding-window, MLA/latent attention, linear
  attention, SSM/Mamba mixers, hybrid blocks. Do their modules expose q/k/v/pattern
  facets? Should they?
- **Norm families**: new `*Norm` classes whose math differs from plain
  LayerNorm/RMSNorm (offset-scaled, grouped, qk-norm). A misclassified norm kind is
  caught downstream only because consumers validate numerically -- do not rely on
  that as a license to guess.
- **Heads**: unembedding children not named `lm_head`/`embed_out`/
  `output_projection`; multi-head outputs (value heads, MTP heads); classifiers.
- **MLP families**: new gated variants, MoE experts and routers (which expert ran is
  trace-visible; propose facets that reflect the TAKEN path only).
- **Embeddings and position**: rotary application points, ALiBi, learned/positional
  hybrids.
- **Non-LM domains**: vision (patch embed, cls token), audio, diffusion (timestep
  embedding, cross-attention), GNNs.
- **Absence quality**: `needs_capture` reasons that are wrong or unactionable;
  `declared_not_produced` rows (a recipe declaring facets it never produces is a
  recipe bug).

## Rules (LOCKED)

1. **Facts from real source only.** Ground every proposed anchor in the actual
   module source (installed package or the model's real repository). Never infer a
   facet from a class name alone; that is exactly how confident mislabeling happens.
2. **Structural anchors over name anchors.** Prefer deriving the target from traced
   dataflow (as `language_model_head` derives the final norm from the head's input
   op) over child-name lists. Name lists are acceptable predicates only when the
   convention is near-universal, and must stay narrow.
3. **Validation-first.** Any facet whose consumer reconstructs computation (lens
   heads, reconstructed attention) must ship with a numeric validation path against
   captured values, mirroring `logit_lens`'s validate-against-captured-logits
   tripwire and `reconstruction.py`'s checked-not-trusted pattern. State in the
   proposal what the validation oracle is.
4. **Absence is typed, never silent.** A recipe that cannot anchor a facet returns a
   `structural(...)` or `needs_capture(...)` reason. Never fabricate a "close
   enough" anchor (see `_resid_mid_spec`'s history: the first-add fallback was
   removed because it invented meaningless midpoints).
5. **Proposals only.** Output is a review document per family. Do NOT edit
   `torchlens/semantic/recipes/`, do NOT register recipes in any process you leave
   running, do NOT open auto-merging PRs.

## Output format (one block per proposed family)

```
### <family name>
Evidence: <models/classes seen unclassified, with catalog or HF ids>
Source ground truth: <link/path to the real module source consulted>
Proposed recipe: <matching predicate, target scope, facet names + home kinds>
Validation oracle: <what captured values numerically corroborate the facets>
Risk: <what a wrong label here would mislead a user about>
Priority: <high/medium/low + one-line justification>
```

A full recipe PR (written by a human or a supervised implementation lane AFTER
review) must contain: the recipe module, registration in `recipes/__init__.py`, a
`BUILTIN_FACET_CAPABILITY_INVENTORY` section, tests (positive + typed-absence +
validation-tripwire cases), and docs (`docs/facets.md` + `semantic/AGENTS.md`) in
the same change.
