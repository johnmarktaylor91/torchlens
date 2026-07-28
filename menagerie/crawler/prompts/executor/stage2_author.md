## STAGE 2 OF 2 -- AUTHOR THE PROPOSAL

Continue in the **same session** that did stage 1. You already read the sources; do not
re-research from cold. (If this brief says COLD START, the prior session was lost; its
recorded discovery output is inlined below and you may re-ground what you need.)

### The contract you are writing against

1. Read the REQUEST envelope named in JOB FACTS **first**. It is the authority: its
   `stable_id`, `expected_result`, `source_manifest`, and `allowed_model_dir` are
   binding, and every identity in `expected_result` must be echoed **byte-exactly** in
   your result. A mismatched identity is rejected outright -- it is not a warning.
2. Read the canonical author prompt at the `prompt.path` named inside that envelope. It is
   the full specification of the proposal: the source ladder R1..R5, the do-it-once
   required content, the random-initialization and execution policy, and the evidence
   rules. **Follow it exactly.** This brief adds dispatch context; it does not replace,
   soften, or reinterpret one line of it.
3. Write ONE atomic JSON result to the exact RESULT output path in JOB FACTS. **That path
   supersedes the envelope's `required_output_path`:** the executor verifies your result
   and publishes it to the envelope's path itself, so a superseded attempt can never
   overwrite a live one.
4. Stage adapter code under the STAGED MODEL dir in JOB FACTS (your attempt's own
   `model/` tree -- the only tree you can write). The executor mirrors it into the
   envelope's `allowed_model_dir` at publication, so your proposal's `code_manifest`
   paths must name the files **as they will appear under `allowed_model_dir`**.

### Exact strings are machine facts

The frozen `source_manifest` was produced by the coordinator's source broker: every
`revision` is a forge-resolved commit SHA and every digest was computed from fetched
bytes. Cite those values **verbatim from the manifest**; never reconstruct, shorten, or
"remember" an identifier. Broker-derived citation metadata, when present, rides in the
manifest's `broker` block -- your citation claims must match it, and your judgment is
whether it names the *introducing* work.

### One supplementary source round

If exactly one more file would materially change the proposal, write
`supplement-request.json` next to your RESULT output path **instead of** the result:

```json
{"supplement_version": "menagerie.crawler.author-supplement-request.v1",
 "sources": [<same descriptor shape as stage 1, no SHAs, no digests>],
 "why": "one sentence"}
```

The executor runs one broker pass and resumes this session once with the supplementary
manifest. You get exactly one such round; after it you must write the result.

### What the checker will do to it

A Codex checker from a different model family verifies every factual field and any
implementation before the single writer may accept it, and the engine re-derives every
identity and re-hashes every cited artifact. So:

- Every claim carries its evidence: exact URL, exact revision, exact path or symbol,
  verbatim excerpt, retrieval time, content hash -- all read from the frozen manifest.
- Never invent topology, dimensions, operators, activations, normalization, padding,
  connectivity, state, initialization, input semantics, or output behavior. A missing
  material choice is a **gap you report**, not a hole you fill.
- Honest `SKIP_RECOMMENDATION`, `DEFER_RECOMMENDATION`, and `BLOCKED` results are fully
  acceptable outcomes and are recorded as such. An unsupported claim is not.
- Use `NOASSERTION`, `null`, or `not-found-after-search` rather than guessing a license, a
  year, a country, or an author.

### Running out of budget

The wall deadline in JOB FACTS is enforced by an external kill shortly after it passes.
If it is approaching, do **not** go silent and do not rush a half-grounded proposal. Emit
a valid `BLOCKED` result naming what you could not establish and why. A typed BLOCKED
flows through the engine's terminal-disposition gate and can be requeued; a timeout costs
the model an entire retry cycle.

### Hard limits

- The repository root in JOB FACTS is **read-only**. Never write to it, to ledgers,
  checkpoints, environment specs, accepted adapters, or another model.
- Never install a package or mutate an environment; return a typed dependency request.
- Never use TorchLens, never trace, never train, never evaluate accuracy, and never fetch
  pretrained weights or checkpoints.
- Do not declare runs and do not self-approve accuracy or fidelity.
- Effort is counted by the machine from the harness record; you do not report tool-call
  numbers.
