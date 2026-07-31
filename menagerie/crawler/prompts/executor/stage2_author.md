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
`recommendation_sha256`. For every recommendation arm it also derives
`evidence_identity` from the exact terminal evidence pack and `license_identity` from the
exact license disposition. For `SKIP_RECOMMENDATION` it additionally derives
`search_report_identity`. Never send any of those identity fields: an authored value is
rejected rather than trusted or allowed to override the machine.

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
the complete `proposed_facts` object required by the registered schema, **except the
eleven the executor supplies** -- see immediately below.

### Eleven proposal fields the executor fills in

The registered proposal schema marks these required, but they are the machine's, not
yours. Omit them; the executor supplies each one and rejects a value that disagrees with
what it holds.

- `schema_version`, and the eight request bindings `campaign_id`, `stable_id`, `work_id`,
  `intake_snapshot_id`, `intake_snapshot_sha256`, `intake_item_sha256`,
  `source_manifest_identity`, and `dispatcher_identity`. All nine come straight off the
  REQUEST envelope the executor is holding, so transcribing them can only introduce a
  wrong copy.
- `proposal_sha256`, the digest of the finished proposal. The executor derives it last,
  after it has filled the fields above, so an authored value could not be correct anyway.
- `author`, the closed `{provider, model, version, prompt_sha256}` object naming the model
  that wrote the proposal. **Omit it.** It describes the machine's own dispatch of you,
  not anything you can observe: `version` is the campaign's author-version string, not the
  prompt's file name and not a model name. The executor stamps it and overwrites whatever
  is there, so writing one is wasted effort rather than an error.

`proposed_facts.modes.per_mode_run` is likewise the executor's, and the reason is not
bookkeeping: nothing has run when you write a proposal, so there is no per-mode outcome
to report. Omit it. Supplying a non-empty one claims attempts that do not exist and is
refused.

Everything else in the proposal is yours, **including `proposed_facts.evidence`
`excerpts[].text_sha256`** -- unlike a terminal `evidence_records` entry, that digest
feeds the evidence identity the engine re-derives, so it is still required here. Compute
it as `identity.hash_bytes(text.encode("utf-8"))` over the exact text you quoted.

### The five identities are yours, and there is a calculator for them

`source_identity`, `evidence_identity`, `recipe_revision`, `vet_identity`, and
`fidelity_identity` stay **yours**, and the engine recomputes every one of them from your
published `proposed_facts` and refuses a mismatch. That is not bookkeeping: each is a pure
function of the facts you declared, so an identity that does not follow from your own facts
is the engine catching a claim you did not actually make. It is not relaxed, and the
executor will not fill these in for you.

What you are not expected to do is the arithmetic. The derivation is canonical JSON over a
projected excerpt subset, SHA-256, nested six deep; reproducing it by hand is how ten
straight proposals arrived with three of the five wrong. Run the calculator instead, using
the exact interpreter and module named in the JOB FACTS `identity calculator` line:

```
<identity calculator> --request <REQUEST envelope path> --facts <your drafted facts file>
```

Write your drafted proposal (or just its `proposed_facts` object) to a JSON file under your
attempt directory and pass that as `--facts`. The tool prints a JSON object with the five
identities. Copy each into the matching top-level proposal field, and copy **two of them a
second time** into their embedded homes, which the engine checks separately:

- `recipe_revision` also into `proposed_facts.implementation.recipe_revision`
- `evidence_identity` also into `proposed_facts.evidence.evidence_identity`

Both embedded copies are required, and a proposal carrying the top-level five without them
is refused as `embedded recipe/evidence identities are stale`. The tool already accounts
for them when it computes `vet_identity`, so copy its numbers verbatim -- do not compute a
draft without the copies and then add them, because the two are authored leaves and adding
them afterwards moves `vet_identity`.

Two things it will not do. It never fetches, infers, or invents a fact -- it computes only
from the facts you hand it, and refuses with a message naming what is missing if they are
incomplete. And it is not a way around the check: it faithfully returns the identity of
whatever facts you give it, so an identity computed from a fact you have not actually
grounded still fails against the real artifacts. Draft the facts honestly first, then
compute. Re-run it if you change any fact afterwards -- the identities move when the facts
move.

### Every terminal arm carries its excerpts in `evidence_records`

`DEFER_RECOMMENDATION`, `SKIP_RECOMMENDATION`, and `BLOCKED` each cite `evidence_ids`.
An ID on its own is not inspectable, so every terminal payload also carries
`evidence_records`: one record per cited ID, holding the literal text you read.

Each record has exactly these keys, and every one of them is something you read and can
quote:

- `evidence_id` -- your own correlation label; it must match an entry of `evidence_ids`.
- `source_id` -- the frozen manifest source you read it out of.
- `locator` -- where in that source it sits. A `bytes:start-end` locator gets checked as
  an exact byte range; anything else is checked as substring presence.
- `text` -- the excerpt, **verbatim**. The engine reads the frozen source bytes back out
  of content-addressed storage and requires your text to appear in them exactly. Text
  that is paraphrased, reflowed, or reconstructed from memory will not match and the
  claim will be reported as ungrounded.
- `supports` -- the claims this excerpt is offered in support of.

`family_level` (boolean) and `disposition` (short string) are optional.

Never put `text_sha256`, `content_sha256`, or `evidence_identity` in a record. You have no
hashing primitive, you are not asked for a digest, and the engine derives every digest
itself from the bytes it re-read.

`license_record` is optional and takes `source_id`, `locator`, `text`, and
`declared_license` -- the license text you actually read at a frozen source. It is quoted
and grounded the same way; the engine still derives `license_identity` itself.

Omitting `evidence_records` is allowed and is recorded as a **named gap** on the terminal
envelope: the checker is told plainly that the cited IDs have no inspectable excerpt. It
is not treated as grounding, and it will not be silently forgiven. Quote what you read.

For `DEFER_RECOMMENDATION`, use `platform` (exactly `cuda` or `x86`), not
`recommended_target`; include `source_ids`, `evidence_ids`, and `evidence_records`; and put the same complete
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
    "evidence_records": [
      {
        "evidence_id": "ev-platform",
        "source_id": "impl-main",
        "locator": "setup.py lines 41-43",
        "text": "CUDAExtension(name='deform_conv_cuda'",
        "supports": ["needs-cuda"]
      }
    ],
    "handoff_execution": {
      "proposal": {
        "$complete_author_proposal_v3": "Replace this fixture sentinel with the complete registered proposal; code_manifest belongs under proposed_facts.implementation."
      }
    }
  }
}
```

For `SKIP_RECOMMENDATION`, the exact authored payload keys are `status_code`,
`source_ids`, `evidence_ids`, and `evidence_records`.

<!-- CONTRACT_FIXTURE: stage2-skip-author-payload -->
```json
{
  "kind": "SKIP_RECOMMENDATION",
  "payload": {
    "status_code": "skipped:insufficient-description",
    "source_ids": ["impl-main"],
    "evidence_ids": ["evidence-gap"],
    "evidence_records": [
      {
        "evidence_id": "evidence-gap",
        "source_id": "impl-main",
        "locator": "README.md lines 12-13",
        "text": "architecture details are omitted here",
        "supports": ["insufficient-description"]
      }
    ]
  }
}
```

For `BLOCKED`, the authored keys are `stage`, `reason_code`, `prerequisite_ids`,
`evidence_ids`, and `evidence_records`, plus `research_summary` only for the higher-tier promotion case described
by the canonical prompt. `prerequisite_ids` names missing external facts or capabilities
with stable semantic labels such as `runtime-dependency` or `faithful-source`; it never
lists a schema path, an output field, or a digest the executor owns.

<!-- CONTRACT_FIXTURE: stage2-author-payload -->
```json
{
  "kind": "BLOCKED",
  "payload": {
    "stage": "source",
    "reason_code": "missing-material-source",
    "prerequisite_ids": ["source-needed"],
    "evidence_ids": ["evidence-gap"],
    "evidence_records": [
      {
        "evidence_id": "evidence-gap",
        "source_id": "impl-main",
        "locator": "README.md lines 4-5",
        "text": "Code release is pending.",
        "supports": ["blocked-prerequisite"]
      }
    ]
  }
}
```

Do not add any outer request identity, payload `arm`, terminal identity, or
result/recommendation/handoff hash. Those are facts the executor derives from the exact
objects it holds.

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
