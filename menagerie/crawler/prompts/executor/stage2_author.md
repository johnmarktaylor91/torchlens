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

### `proposed_facts.fidelity`: write all eight leaves, judge only two

`fidelity` is a **gate-state** block, and it is the one such block you must write. Nothing
in it is filled in for you: all eight keys are required, including the six that describe a
fidelity gate which has not run yet. Do **not** omit a leaf because it has no value --
"not yet judged" has a spelling, and this is it:

```json
"fidelity": {
  "required": true,
  "reason": "<your judgment, in prose>",
  "verdict": null,
  "fidelity_identity": null,
  "gate_id": null,
  "current": false,
  "permanent_scar": false,
  "deviations": []
}
```

`required` and `reason` are yours (R3 and R4 always require a gate; below that it is your
call). The other six are the gate's, and at authoring time they take exactly the values
above. Writing them is not self-approving fidelity -- `verdict: null` and `current: false`
are precisely the claim that *no* verdict exists -- and the checker overwrites all six when
it runs.

`permanent_scar` is the leaf that has actually been missed, and it cost a model its whole
authoring campaign: it is a gate-set indelible mark that a record was once judged `slop`,
never an author's self-assessment, so an authored proposal always writes `false`. The
embedded `fidelity.fidelity_identity` is likewise `null` here; it is a different field from
the proposal's own top-level `fidelity_identity`, which you do compute with the calculator.

Everything else in the proposal is yours, **including `proposed_facts.evidence`
`excerpts[].text_sha256`** -- unlike a terminal `evidence_records` entry, that digest
feeds the evidence identity the engine re-derives, so it is still required here. Compute
it with the calculator's hash mode over the exact text you quoted (see below).

**Hash the STRING YOU PASTED, never the region your `locator` names.** The engine
recomputes the digest from the `text` field alone and refuses any disagreement, and both
observed failures came from hashing the source instead:

- **A line range wider than the quote.** `lines 371-382` counted the trailing blank line;
  the pasted text stopped at line 381. The excerpt was genuinely verbatim in the source,
  so nothing else caught it -- the digest was the only check standing between a partial
  quote and the record.
- **A character your paste did not reproduce.** The paragraph really held U+00A0 before
  `<cite`; the pasted text held an ASCII space. The digest was right about the bytes and
  the quote was wrong, which is the worse way round: it would have published a paraphrase
  under a correct-looking hash.

So: paste the excerpt into your facts file first, then digest **that exact string**. If a
source region resists byte-exact quoting -- no-break spaces, zero-width joiners, unusual
line endings -- quote a shorter span you can reproduce exactly rather than widening the
digest to cover it. A short exact excerpt grounds a claim; a long approximate one grounds
nothing and costs the whole proposal.

Always compute the digest with the granted calculator -- the ONLY command you can run.
`sha256sum` is **not granted** and auto-denies, like every other shell tool; do not try
it. Write the exact pasted string to a file under your attempt directory, then run:

```
<identity calculator> --hash-file <that file>
```

(`--hash-string '<text>'` also works for short single-line strings.) The tool prints the
`sha256:<hex>` digest of the exact bytes. If your file-writing tool appended a final
newline the quote does not contain, use the reported
`sha256_without_trailing_newline` value instead -- that trap has produced real
mismatches. A digest written from memory, copied from a different excerpt, or eyeballed
off the source matches nothing: a real proposal died carrying eight sequential
hand-declared placeholder digests that matched no bytes anywhere, while its quoted text
was byte-perfect.

**A citation value is checked against the BOUND EXCERPTS, never against the page.** The
engine concatenates the text of exactly the excerpts named in
`citation.source_evidence_ids` and requires each non-null citation leaf -- title, venue,
year, every author, and every declared `arxiv_id`/`doi`/`openreview_id` -- to occur in
that concatenation. Nothing else in the frozen source is read. This is the failure that
killed a real MetaFormer proposal: the author bound the `citation_title`/`citation_author`
meta-tag run and the `CVPR 2022 (Oral)` comments row, both honest, neither containing
`2111.11418` -- which sat twenty-eight times elsewhere on the same fetched page. Having
the bytes is not having quoted them. So after you write the citation, read each non-null
leaf back against your own bound excerpt text, and bind one more excerpt whenever a leaf
is not literally inside it. The one tolerance is arXiv revisions: a bound excerpt reading
`arXiv:2111.11418v3` grounds a declared `2111.11418`, because the version names a revision
of that same work. It does not run the other way -- a bare mention does not license
declaring `v3` -- so declare the identifier as the excerpt shows it or plainer.

Sources the ONE supplementary round fetched for you are citable exactly like frozen ones:
quote them by their `source_id` from the supplementary manifest. They are our fetch and
our digest, and the engine grounds evidence against them alongside the frozen manifest.

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

### Exact-value traps that have each killed a real proposal

Each of these is enforced to the byte, and a near-synonym is a dead proposal, not a
warning. Check every one before you write the result:

- `licenses.weights.status` is always **exactly `"not-used"`** -- checkpoint access is
  forbidden for every proposal, so no other value is ever true. `"not-applicable"`
  killed a real proposal.
- The `proposed_facts.citation` block and `proposed_facts.external_metadata.citation`
  must be **exactly equal**, leaf for leaf. Divergence anywhere refuses the proposal.
- `source_resolution.sources` must list **every** source in the frozen manifest
  (supplementary rows included) -- one row per `source_id`, no subset, no extras -- and
  each row's machine fields (`url`, `revision`, `content_sha256`, `byte_count`,
  `media_type`) must echo the manifest verbatim. A real proposal listed only the 9
  sources it had used out of 18 and was refused at staging.
- `source_resolution.mandatory_link_status` must be `"ok"`, and `primary_source_id` must
  name one declared source whose `url` starts with `http`. Never add `cas_path` to a
  `source_resolution.sources` row: CAS locations are machine-owned, and an authored one
  refuses the proposal.
- `source_resolution.sources[].retrieved_at` is machine-derived: copy the manifest row's
  `retrieved_at` verbatim. The broker stamps it from its own resolver receipt; an
  invented or estimated timestamp is a false provenance claim.
- Every `source_to_code_map` row's `code_path` is a **non-empty string** naming a staged
  file. Only the top-level `implementation.code_path` may be `null` -- and for a
  declarative R1 it must be, with `source_to_code_map: []`.
- `initialization.policy` is the const `"random"` and `pretrained_disabled` the const
  `true`. The library's real initialization behavior belongs in
  `source_specified_choices`, never as prose in `policy`.

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

`family_level` (boolean) and `disposition` (short string) are optional. `disposition`, and
`license_record`'s `declared_license`, are **non-empty** strings: omit the key when you have
nothing to put in it rather than sending `""`, which is refused.

Never put `text_sha256`, `content_sha256`, or `evidence_identity` in a record. You have no
hashing primitive, you are not asked for a digest, and the engine derives every digest
itself from the bytes it re-read.

`license_record` is optional and takes `source_id`, `locator`, `text`, and
`declared_license` -- the license text you actually read at a frozen source. It is quoted
and grounded the same way; the engine still derives `license_identity` itself.

Omitting `evidence_records` is allowed and is recorded as a **named gap** on the terminal
envelope: the checker is told plainly that the cited IDs have no inspectable excerpt. It
is not treated as grounding, and it will not be silently forgiven. Quote what you read.

**The pack resolves per record, so cite everything you can ground -- and nothing else.**
Every ID in `evidence_ids` is re-derived on its own. An ID whose record's `text` matches
the frozen source named by its `source_id` byte-for-byte is shown to the checker; an ID
that has no record, or whose `source_id` is not in the REQUEST envelope's
`source_manifest`, or whose text does not match, is reported beside them as a
**named unresolved gap** in `unresolved_evidence_ids` and shown to nobody. The two sets partition
`evidence_ids` exactly, so a bad row can never launder itself through good neighbours --
and good rows are no longer destroyed by a bad one. A pack where some IDs ground and some
do not resolves `partially-grounded`, not `grounded`; only a pack where every ID grounds
resolves `grounded`.

Two floors are absolute. If nothing verifies, or if no verified record's `supports` names
your terminal arm's own predicate (`blocked-prerequisite`, the platform predicate, or the
R5 status), the whole pack is unresolved and the arm terminalizes as
disposition-unverifiable: grounding only decoration around a dead predicate is not
partial grounding. So ground the excerpt your recommendation actually rests on FIRST.

Two consequences worth internalizing. First, an ID you can ground is never worse than
silence, so cite the whole basis for your arm -- above all the excerpt carrying the
`blocked-prerequisite` claim a `BLOCKED` rests on. An arm whose own predicate has no
grounded excerpt is what the checker is entitled to reject. Second, a fact you read
somewhere the broker never froze -- a web page, a metadata API response, an abstract you
fetched outside the manifest -- has no groundable `source_id`, will be named as a gap the
checker reads, and belongs in `findings` rather than `evidence_ids`. Drop the ID, or
request that object through the one supplementary source round so it is in the manifest
before you cite it. Padding buys nothing: an unresolved ID stays visible as a defect in
your pack, never a verified row.

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
lists a schema path, an output field, or a digest the executor owns. A `BLOCKED` whose
`reason_code` names a real prerequisite is adjudicated by the checker against frozen
source bytes; running out of wall budget is not such a claim and has its own exact
spelling -- see "Running out of budget" below.

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

You cannot feel time passing, so never GUESS that the deadline "is approaching" --
OBSERVE it. The JOB FACTS `clock` line is your granted observation: run it and it prints
the current instant and the exact `remaining_seconds` before your deadline. The full
grant in JOB FACTS is yours from the attempt-start instant; a prior attempt's failure
did not consume one second of it.

**Do not bail early.** In one real rung, eight of twenty sessions published
`wall-exceeded` with 63-88% of their grant remaining -- most before minute eleven of a
thirty-minute grant, every one a model lost for nothing. The executor VERIFIES every
wall claim against its own clock: a `wall-exceeded` result published with less than
half the grant observably consumed and no external timeout is **refused as an
unverified self-report**, never recorded as a budget terminal. So claim exhaustion only
after a clock observation shows the remaining seconds genuinely cannot fit the
remaining work, and cite that observation (the printed `now` and `remaining_seconds`)
in your result.

When the budget truly is exhausted, do **not** go silent and do not rush a
half-grounded proposal. The wall deadline in JOB FACTS is enforced by an external kill
shortly after it passes. Publish a `BLOCKED` result whose `reason_code` is **exactly
`wall-exceeded`**, with `stage` naming the stage in flight when time ran out (for this
brief, `author`) and `prerequisite_ids` naming the budget itself, such as
`["authoring-wall-budget"]`. That exact spelling is recognized as a budget outcome: the
engine records `failed:<stage>` with a stage-valid effort reason -- a terminal an
operator can requeue with a larger grant -- and asks no checker to adjudicate it, so it
needs no `blocked-prerequisite` excerpt. Cite whatever evidence you already grounded,
or none.

Do not spell exhaustion any other way, and never dress it as a prerequisite. A
`blocked-prerequisite` claim is adjudicated against frozen source bytes, and no frozen
source can witness your wall clock: in one real rung, eight sessions invented their own
spellings (`authoring-budget-exhausted`, `author-wall-deadline-reached`, and variants
like them) and every one terminalized as a rejected or unverifiable disposition instead
of a requeueable budget record. Running out of time is a budget fact about this
session, never a missing prerequisite of the model.

### Hard limits

- The repository root in JOB FACTS is **read-only**. Never write to it, to ledgers,
  checkpoints, environment specs, accepted adapters, or another model.
- Never install a package or mutate an environment; return a typed dependency request.
- Never use TorchLens, never trace, never train, never evaluate accuracy, and never fetch
  pretrained weights or checkpoints.
- Do not declare runs and do not self-approve accuracy or fidelity.
- Effort is counted by the machine from the harness record; you do not report tool-call
  numbers.
