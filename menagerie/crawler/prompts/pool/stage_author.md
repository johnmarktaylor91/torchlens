## STAGE 2 OF 2 -- AUTHOR THE PROPOSAL

Continue in the **same session** that did stage 1. You already read the sources; do not
re-research from cold.

### The contract you are writing against

1. Read the REQUEST envelope named in JOB FACTS **first**. It is the authority: its
   `stable_id`, `expected_result`, `source_manifest`, `allowed_model_dir`,
   `allowed_output_root`, and `required_output_path` are binding, and every identity in
   `expected_result` must be echoed **byte-exactly** in your result. A mismatched identity
   is rejected outright -- it is not a warning.
2. Read the canonical author prompt at the `prompt.path` named inside that envelope. It is
   the full specification of the proposal: the source ladder R1..R5, the do-it-once
   required content, the random-initialization and execution policy, and the evidence
   rules. **Follow it exactly.** This brief adds dispatch context; it does not replace,
   soften, or reinterpret one line of it.
3. Write ONE atomic JSON result to the exact `required_output_path`. Nothing else, nowhere
   else. Staged adapter code goes under `allowed_model_dir`.

### What the checker will do to it

A Codex checker from a different model family verifies every factual field and any
implementation before the single writer may accept it, and the engine re-derives every
identity and re-hashes every cited artifact. So:

- Every claim carries its evidence: exact URL, exact revision, exact path or symbol,
  verbatim excerpt, retrieval time, content hash.
- Never invent topology, dimensions, operators, activations, normalization, padding,
  connectivity, state, initialization, input semantics, or output behavior. A missing
  material choice is a **gap you report**, not a hole you fill.
- Honest `SKIP_RECOMMENDATION`, `DEFER_RECOMMENDATION`, and `BLOCKED` results are fully
  acceptable outcomes and are recorded as such. An unsupported claim is not.
- Use `NOASSERTION`, `null`, or `not-found-after-search` rather than guessing a license, a
  year, a country, or an author.

### The interpreter that has the dependencies

The canonical prompt requires you to run the deterministic proposal validator in this same
session. Run it with the **campaign interpreter**:

```
<repository root from JOB FACTS>/.venv-crawler/bin/python
```

The system `python3` does not have `jsonschema` and the import fails before the validator
runs; that interpreter does. The entry point is
`menagerie.crawler.author_dispatch.validate_author_result(result_path, envelope)`. Invoking
that interpreter and importing repository modules is *reading* the read-only root and is
allowed; installing into it is not. If the interpreter is genuinely absent, say so in your
result -- never install a dependency, and never skip the validator silently.

### Identity hashes you compute yourself

Every hash below is canonical `stable_hash` from `menagerie.crawler.identity`: sha256 over
`json.dumps(value, sort_keys=True, separators=(",", ":"))` encoded UTF-8, rendered
`sha256:<64 lowercase hex>`. Import the helpers rather than re-implementing the
canonicalization -- one space or one reordered key and the engine rejects the whole result.

- `evidence_identity`: `identity.compute_evidence_identity(excerpts)` over your excerpt list
  **in order**, which projects exactly `source_id`, `locator`, `text`, `text_sha256`,
  `supports`, and `source_content_sha256` from each excerpt. The engine re-derives this from
  your own evidence and rejects a mismatch, so reordering excerpts changes it.
- `license_identity`: `stable_hash(<the exact licenses block you declare>)`. Every place your
  result repeats it must repeat it byte-exactly; the terminal gate binds it.
- `recommendation_sha256`: `stable_hash` of your recommendation payload with the
  `recommendation_sha256` key itself removed. It must bind the complete arm payload, so
  compute it last, after every other field of that arm is final.
- `handoff_sha256` (DEFER only): the canonical prompt names the exact preceding fields it
  covers; hash exactly those, in that order, with the same helper.

### Running out of budget

If the grant in JOB FACTS is about to run out, do **not** go silent and do not rush a
half-grounded proposal. Emit a valid `BLOCKED` result naming what you could not establish
and why. A typed BLOCKED flows through the engine's terminal-disposition gate and can be
requeued; a timeout is a stall that costs the model an entire retry cycle.

### Report back (the pool needs these)

- the exact number of tool calls you made across **both** stages;
- verbatim text of any Claude usage limit you hit, including a reset time if given;
- the rung you selected (R1..R5) and one sentence on why every higher rung failed.

### Hard limits

- The repository root in JOB FACTS is **read-only**. Never write to it, to ledgers,
  checkpoints, queue state, environment specs, accepted adapters, or another model.
- Never install a package or mutate an environment; return a typed dependency request.
- Never use TorchLens, never trace, never train, never evaluate accuracy, and never fetch
  pretrained weights or checkpoints.
- Do not declare runs and do not self-approve accuracy or fidelity.
