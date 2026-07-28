## STAGE 2 OF 2 -- AUTHOR THE PROPOSAL

Continue in the **same session** that did stage 1. You already read the sources; do not
re-research from cold. (If this brief says COLD START, the prior session was lost; its
recorded discovery output is inlined below and you may re-ground what you need.)

### The contract you are writing against

1. Read the REQUEST envelope named in JOB FACTS **first**. It is the authority: its
   `source_manifest` and `allowed_model_dir` are binding. The executor owns
   `expected_result`, `stable_id`, `work_id`, timestamps, and result hashes; do not copy
   those outer-envelope facts into your result.
2. Read the canonical author prompt at the `prompt.path` named inside that envelope. It is
   the full specification of the proposal: the source ladder R1..R5, the do-it-once
   required content, the random-initialization and execution policy, and the evidence
   rules. **Follow it exactly.** This brief adds dispatch context; it does not replace,
   soften, or reinterpret one line of it.
3. Read the exact PROPOSAL schema in JOB FACTS and follow its local `$ref` files in the
   REFERENCED schema directory. The registered schema, not an inferred summary shape, is
   authoritative for every proposal key and nesting level.
4. Write ONE atomic JSON result to the exact RESULT output path in JOB FACTS. Write only
   `kind` and its authored `payload`, in the exact shape below. **That path supersedes the
   envelope's `required_output_path`:** the executor adds the registered result envelope,
   verifies it, and publishes it to the envelope's path itself, so a superseded attempt
   can never overwrite a live one.
5. Stage adapter code under the STAGED MODEL dir in JOB FACTS (your attempt's own
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

<!-- CONTRACT_FIXTURE: supplement-author-payload -->
```json
{
  "sources": [
    {
      "source_id": "supplement-doc",
      "kind": "raw-url",
      "url": "https://example.org/model-config.txt",
      "requested_role": "documentation",
      "basis": "Observed direct object containing the missing configuration detail."
    }
  ],
  "why": "This one object resolves the remaining material configuration gap."
}
```

These are the same closed source descriptors as stage 1: no `revision`, hashes, final
URLs, broker roles, authoritative media types, or invented `requested_role` values. The
executor adds `supplement_version`, validates the descriptors through the registered
source-discovery schema, runs one broker pass, and resumes this session once with the
supplementary manifest. You get exactly one such round; after it you must write the result.

### Exact result transport shape

The result file is an author-owned inner object, not `author-result.v4` itself. The
executor adds the full `expected_result` bindings, `schema_version`, `created_at`,
`result_id`, `result_sha256`, the redundant payload `arm`, and
`recommendation_sha256`.

For `PROPOSED`, `payload.proposal` is one complete registered
`menagerie.crawler.author-proposal.v3` object. Its catalog sections do **not** sit directly
under `proposal`: all fourteen must sit under `proposal.proposed_facts`, using exactly
these registered keys:

`identity`, `taxonomy`, `external_metadata`, `website`, `people_and_origin`, `dates`,
`citation`, `licenses`, `source_resolution`, `evidence`, `implementation`,
`input_contract`, `modes`, and `fidelity`.

Do not emit an informal parallel vocabulary such as `website_english`, `first_public`,
`rung`, `source_selection`, or `output_contract` beside those registered blocks. A
proposal with `identity` or `implementation` directly beneath `proposal` is invalid even
when every claim inside it is excellent.

<!-- CONTRACT_FIXTURE: stage2-proposed-author-payload -->
```json
{
  "kind": "PROPOSED",
  "payload": {
    "proposal": {
      "$complete_author_proposal_v3": "Replace this fixture sentinel with the complete object required by the PROPOSAL schema in JOB FACTS."
    }
  }
}
```

The `$complete_author_proposal_v3` sentinel documents the insertion point only. Never
write it literally. The inserted object must contain every top-level proposal field and
the complete `proposed_facts` object required by the registered schema.

For `DEFER_RECOMMENDATION`, use `platform` (exactly `cuda` or `x86`), not
`recommended_target`; include `source_ids` and `evidence_ids`; and put the same complete
registered proposal at `handoff_execution.proposal`. The executor derives the other four
handoff identity fields. A declarative proposal still carries
`proposal.proposed_facts.implementation.code_manifest: []`; staged code carries its
non-empty manifest there. Never put `implementation` directly under `proposal`.

<!-- CONTRACT_FIXTURE: stage2-defer-author-payload -->
```json
{
  "kind": "DEFER_RECOMMENDATION",
  "payload": {
    "platform": "cuda",
    "source_ids": ["impl-main"],
    "evidence_ids": ["ev-platform"],
    "evidence_identity": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "license_identity": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "handoff_execution": {
      "proposal": {
        "$complete_author_proposal_v3": "Replace this fixture sentinel with the complete registered proposal; code_manifest belongs under proposed_facts.implementation."
      }
    }
  }
}
```

For `SKIP_RECOMMENDATION`, the exact authored payload keys are `status_code`,
`source_ids`, `evidence_ids`, `evidence_identity`, `search_report_identity`, and
`license_identity`. For `BLOCKED`, they are `stage`, `reason_code`, `prerequisite_ids`,
`evidence_ids`, `evidence_identity`, and `license_identity`, plus `research_summary` only
for the higher-tier promotion case described by the canonical prompt.

<!-- CONTRACT_FIXTURE: stage2-author-payload -->
```json
{
  "kind": "BLOCKED",
  "payload": {
    "stage": "source",
    "reason_code": "missing-material-source",
    "prerequisite_ids": ["source-needed"],
    "evidence_ids": ["evidence-gap"],
    "evidence_identity": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "license_identity": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  }
}
```

Do not add any outer request identity, payload `arm`, or result/recommendation/handoff
hash. Those are facts the executor derives from the exact object you authored.

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
