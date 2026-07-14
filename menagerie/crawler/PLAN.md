# Menagerie get-every-model-to-RUN crawler: canonical implementation plan

Status: authoritative implementation specification

Decision date: 2026-07-14

Implementation branch: `menagerie/crawler-pipeline`

Implementation root: `menagerie/crawler/`

This document consolidates the converged Fable R2 and Sol R2 designs and JMT's final rulings. It
is a build specification, not a new design exercise. Where an R2 document disagrees with a ruling,
the ruling and the reconciliation decisions encoded here are final.

## 1. Goal, success bar, and non-goals

The crawler takes the complete discovered-model intake, currently assembled from
`menagerie/data/master_catalog.jsonl`, `menagerie/data/deferred.jsonl`, the stable-ID map,
`menagerie/classics/`, and discovery-only records, and gives every stable ID one honest current
disposition. The positive bar is deliberately narrow:

1. construct the sourced architecture with random initialization;
2. construct a semantically valid dummy input with no checkpoint;
3. execute an explicit `forward(*args, **kwargs)` in a fresh subprocess; and
4. retain enough provenance, literal evidence, judgment, metadata, and failure evidence that the
   source never has to be read again for this campaign.

`runs` does not claim checkpoint accuracy, training viability, backward support, TorchLens
traceability, graph validity, or numerical equivalence to a pretrained model. This crawler must not
import TorchLens, trace, render, or validate a trace. Static and runtime tripwires enforce that
boundary.

The primary machine is JMT's Apple-Silicon Mac mini, CPU-only for the crawl, with at least 100 GiB
available to the campaign. The only platform deferrals are CUDA and x86. They are swept later on the
existing Linux/NVIDIA machine.

Native TensorFlow/Keras, JAX/Flax, and Paddle models run in their native frameworks behind a
transparent adapter in a phase-2 tail after the PyTorch sweep. Every record stores both
`original_framework` and `run_framework`. Whether a later tracing campaign should reimplement these
models in PyTorch or use a TorchLens native backend is explicitly deferred and outside this crawl.

## 2. Locked invariants

The implementation must preserve all of the following.

1. The deterministic Python execution driver is the only authority that awards `runs`.
2. All inherited source, recipe, rung, run, fidelity, and verification claims are untrusted hints.
   Every row re-earns its current claims and is re-executed by this driver.
3. The run bar is random initialization plus a source-valid dummy input; checkpoints and hidden
   weight downloads are forbidden.
4. Every model record, including failures and skips, has at least one exact public model-identity,
   implementation, or primary-description URL. A model whose retained discovery/identity URL cannot be
   resolved to an authoritative implementation or description fails at `source`; it never earns `runs`.
5. Real source wins: library, exact upstream, faithful port, detailed-description reimplementation,
   then epistemic skip.
6. The expensive source-reading pass is done once. Every source-read or agent-judgment field and its
   literal grounding are captured now. Only cheaply derivable fields may be postponed.
7. Every agent-authored field is blocked at canonical write until an independent Codex gate checks
   it. This covers descriptions, citations, year, country, license, taxonomy, source mapping, input
   semantics, and port/reimplementation fidelity.
8. Metadata vetting gates crawl completion, not the mechanical run award. Codex throughput or quota
   must never stop eligible mechanical R1/R2 forward execution.
9. Exact platform target locks, artifact hashes, resolved exports, and probe results define an
   environment generation. A fuzzy freeze is not a lock.
10. Source, evidence, recipe, adapter, input, environment, runner, author-prompt, or checker-prompt
    changes mechanically stale exactly the dependent facts.
11. The status taxonomy is closed. There are no `other`, generic `unsupported`, convenience defer,
    or silent drop buckets.
12. Intake and current terminal records satisfy an exact, pairwise-disjoint partition before the
    crawl can be declared complete.
13. Target a small set of thick conda environments, created, used, and torn down sequentially. The
    listed intent roster (~8 PyTorch + ~3 native-tail) is a starting point, not a ceiling. Cluster models
    by real dependency compatibility; if honest clustering needs a few more thick environments, that is
    fine. Never force-merge incompatible dependencies into a fragile mega-environment just to hit a
    number, and do not proliferate toward one environment per model. There is no hard cap; keep it
    reasonable.
14. The pipeline, prompts, schemas, procedures, records, manifests, locks, and our source code live
    on the dedicated `menagerie/crawler-pipeline` branch and are pullable to the mini. Runtime caches,
    full logs, installed envs, and full third-party artifacts follow the boundary in section 8.
15. The unattended process is crash-safe and usage-limit-safe. Every pause is visible, reset-time
    wakeups are scheduled, and an OS lock plus idempotent work identities prevent duplicate crawls.
16. The driver never invokes a shell for model work. It passes an argv vector to a credential-
    scrubbed subprocess with offline flags, empty caches, and a socket-attempt tripwire.
17. This is deliberately not a distributed system. There are no leases, heartbeats, spool queues,
    ULID event segments, or multi-worker arbitration. One single-writer driver and one execution
    worker at a time are the supported mini topology.

## 3. Architecture and authority boundaries

### 3.1 Components

The system has three planes and one bounded escalation path.

1. **Deterministic plane.** Token-free Python owns intake migration, derived queues, environment
   lifecycle, exact-lock validation, controlled fetches, proposal validation, subprocess execution,
   receipts, current-view reduction, status, partition checks, and append-only JSONL writes.
2. **Author plane.** Claude Sonnet is invoked only when judgment is needed. It has WebSearch plus Exa
   (`web_search_exa` and `web_fetch_exa`) for research and grounding only. One rich session handles
   one model's source triage, recipe/adapter authorship, input contract, literal evidence, and full
   one-time metadata together. It is not split into source/recipe/input/metadata sessions. The author
   uses web research to find sources and captures each exact URL plus verbatim excerpt; the controlled
   fetcher then pins those sources into the local CAS. Opus or Sol is a bounded escalation after ordinary
   attempts fail.
3. **Checker plane.** A fresh Codex context independently checks frozen proposals. Description and
   lightweight metadata are checked in batch; required fidelity remains per-model. Primary verification is
   against the author's stored grounding excerpts, with optional web corroboration of citations or facts.
   Codex never edits.
4. **Human/Sol review.** Accuracy-gate cap exhaustion becomes terminal
   `failed:accuracy-gate` with `human_review.required=true`. A later reviewed result appends a
   superseding revision; history is never erased.

Mechanical family templates and source manifests should keep the majority of R1/R2 rows out of the
author plane. When an author session is necessary, all source-reading work is combined into that one
session. Bounded repair sessions receive the same frozen evidence pack plus exact checker findings; they
do not restart research.

### 3.2 Authority matrix

| Component | May | Must never |
| --- | --- | --- |
| Python driver/reducer | derive work, manage one env, validate hashes, invoke worker, append ledgers, award terminal status, materialize views | invent a worker receipt; synthesize a per-model result from a batch exit; edit third-party source silently |
| Execution worker | build one model, make one dummy call, invoke `forward` in every meaningful train/eval mode, write one atomic receipt per mode under scratch | access network/checkpoints; touch ledgers; write outside scratch; import TorchLens |
| Claude author | use WebSearch + Exa to research and ground sources; choose a rung; stage our adapter/port/patch; author evidence and metadata; recommend defer/skip/failure | write canonical records; declare `runs`; install packages; edit outside its staging root; self-approve accuracy/fidelity |
| Codex checker | read the exact frozen pack, optionally corroborate with web research, verify claims, write one gate envelope | edit code, proposal, evidence, queues, envs, or records; repair its own finding |
| Supervisor | launch bounded author/check sessions, checkpoint, commit at ruled cadence, escalate hard cases | poll live subprocesses with tokens; hand-edit ledgers; push |

A missing or invalid worker receipt is explicit `failed:runner` evidence after retry policy. A parent may
record only facts it observed—exit code, signal, peak RSS, and timestamps—and must not fabricate worker-
internal fields.

### 3.3 Data flow

```text
trusted discovery snapshot + stable IDs
                 |
                 v
deterministic migration: every row UNTRIAGED, legacy claims retained only as hints
                 |
        +--------+-----------------------------+
        |                                      |
        v                                      v
mechanical R1/R2 source/template path     one rich Claude author session
        |                                      |
        |                               atomic proposal result.json
        |                                      |
        +----------> deterministic proposal/evidence/hash validation
                                               |
                                               v
                                      fresh Codex gate result.json
                                               |
                          inaccurate/cannot-verify -> bounded same-pack repair
                                               |
                                               v
                                 single writer accepts gated authored facts
                                               |
                                               v
                                 exact locked thick-env assignment
                                               |
                                               v
                                  isolated driver-owned forward(s)
                                               |
                                               v
                       append attempt + model revision; reduce current status/view
```

Mechanical execution may continue while metadata waits for Codex. Agent-authored bytes cannot enter
accepted model facts until their gate passes. R3/R4 cannot earn `runs` until their required fidelity
verdict is current; a successful forward meanwhile is reported as `forward_observed_but_blocked`.

## 4. Canonical storage and append-only rules

### 4.1 Canonical facts

Canonical truth consists of three append-only JSONL ledgers using the exact schemas in section 5:

- `records/models/<shard>.jsonl`: accepted `model.v2` full revisions and terminal revisions;
- `records/attempts/<machine>-<env>.jsonl`: immutable `attempt.v2` facts; and
- `records/gates/<shard>.jsonl`: immutable `gate.v2` checker decisions.

Single-writer operation means ordinary append-only JSONL is sufficient. Each append is one complete
UTF-8 line, written under the process lock, flushed, and `fsync`ed. The line contains a payload hash and
the ledger keeps a monotonic local sequence. Startup scans every line. A torn final line is copied with
its byte offset and hash into a recovery report before the driver truncates only that incomplete tail;
valid preceding facts are never rewritten. A malformed complete line is a hard `failed:runner` campaign
error, not silently skipped.

Current truth is deterministic: highest accepted `record_seq` for a stable ID wins only if revision
parentage is valid. Two different payloads claiming the same stable ID and revision are a hard conflict.
Byte-identical replay is idempotent. SQLite, queue snapshots, cursors, dashboards, and release JSON are
derived and disposable.

Agent and checker outputs are not ledgers. Each writes
`<allowed_output_root>/result.json.tmp`, flushes and `fsync`s it, then atomically renames it to
`result.json`. The dispatcher ignores chat prose, rejects writes outside the staging root, validates the
required schema and hashes, and allows one contract-only correction before counting the session failed.

### 4.2 Identity and staleness

- `stable_id`: preserve the existing ID. New IDs are assigned once as
  `m_` plus the first 20 base32 characters of SHA-256 over a namespace and immutable natural key; the
  full digest is stored. Aliases and duplicate merges never delete an ID.
- `source_identity`: SHA-256 over normalized source URLs, exact revisions, locators, content hashes, and
  the source-search report.
- `evidence_identity`: SHA-256 over ordered literal excerpts, locators, supports lists, and source
  content hashes.
- `recipe_revision`: SHA-256 over declarative constructor data or adapter bytes, dummy-call adapter,
  declared choices, initialization/mode/device policy, and `source_identity`.
- `env_generation`: SHA-256 over environment intent, exact target lock including artifact hashes,
  actual resolved export, compiler/SDK facts, and probe results.
- `fidelity_identity`: SHA-256 over `source_identity`, `evidence_identity`, implementation/code hash,
  source-to-code map, fidelity prompt hash, and exact checker model/version.
- `vet_identity`: SHA-256 over every authored metadata field, its evidence references, source/evidence
  identities, vet prompt hash, and exact checker model/version.
- `execution_identity`: SHA-256 over stable ID, recipe revision, env generation, worker/runner version,
  target/machine class, seed policy, framework adapter, and device.

Changing any byte causes the reducer to stop treating dependent attempts or verdicts as current. Old
facts remain history. An environment-generation change requires re-execution of all accepted models in
that env; the previous success is not current.

## 5. Versioned schemas

All three JSON Schemas set `additionalProperties: false`, define every enum, and require RFC 3339 UTC
timestamps and `sha256:<64 lowercase hex>` hashes. The field lists below are normative; the JSON Schema
files are the executable expression of them.

### 5.1 `model.v2`

Every accepted current view has the following complete field tree.

To keep Codex throughput out of the mechanical hot loop without violating block-at-write, `model.v2`
supports an interim revision with `authored_metadata_state="pending"` (or `"failed"` after the accuracy
gate terminalizes). In that revision the source-read blocks `taxonomy`, `website`, `people_and_origin`,
`dates`, `citation`, and agent-authored license annotations are null as whole typed blocks—never populated
from unvetted legacy/Claude text—and completeness names them as pending. A driver-awarded mechanical
`runs` status may coexist with those nulls. A current accurate gate appends a superseding revision with
`authored_metadata_state="accepted"`, every block and child below present, and no pending issue. Keys may
not be partially populated to bypass the gate.

#### Record and intake

- `schema_version: "menagerie.crawler.model.v2"`.
- `stable_id: string`.
- `record_seq: integer >= 1`; `record_revision: sha256`; `parent_revision: sha256|null`;
  `created_at: timestamp`; `revised_by: {actor, model, version}|{actor:"driver"}`.
- `authored_metadata_state: "pending"|"accepted"|"failed"`.
- `intake.snapshot_id`; `intake.snapshot_sha256`; `intake.legacy_row_sha256|null`;
  `intake.legacy_recipe_sha256|null`; `intake.legacy_module_sha256|null`;
  `intake.legacy_claims_untrusted: true`; `intake.preserved_legacy_flags: string[]`;
  `intake.discovery_sources: string[]`.

#### Identity and taxonomy

- `identity.canonical_name`; `identity.aliases[]`; `identity.acronym|null`;
  `identity.variant`; `identity.variant_scope`; `identity.family_representative_id`;
  `identity.duplicate_of|null`; `identity.alias_of|null`.
- `taxonomy: object|null`; when accepted: `taxonomy.family`; `taxonomy.domains[]`;
  `taxonomy.tasks[]`; `taxonomy.modalities[]`;
  `taxonomy.era`; `taxonomy.architecture_tags[]`; `taxonomy.novel_ops[]`.

#### Website text and family templating

- `website: object|null`; when accepted: `website.kind:
  "family-representative"|"size-variant-template"`.
- `website.tagline`; `website.description` (two to four sentences);
  `website.key_contribution`; `website.voice_version`.
- `website.family_grounding_id`; `website.template_source_model_id|null`;
  `website.variant_parameter_input_line|null`; `website.template_hash|null`.

Claude authors and Codex vets the family representative. Same-family size variants instantiate a
near-verbatim deterministic template from that accepted text; the only prose difference is the measured
parameter/input line. The variant record points to the family evidence and accepted representative gate.
No model-specific architectural claim may be added through the template path.

#### People, origin, dates, and citation

- `people_and_origin: object|null`; when accepted: `people_and_origin.authors[]`;
  `people_and_origin.labs[]`;
  `people_and_origin.institutions[]`; `people_and_origin.origin_countries[]` using ISO 3166-1 alpha-2;
  `people_and_origin.country_basis`; `people_and_origin.country_confidence:
  "high"|"medium"|"low"|"cannot-determine"`; `people_and_origin.country_note`.
- `dates: object|null`; when accepted: `dates.year: integer|null`; `dates.year_basis`;
  `dates.first_public_date: date|null`;
  `dates.first_public_date_basis`.
- `citation: object|null`; when accepted: `citation.status:
  "present"|"not-found-after-search"|"not-applicable"`;
  `citation.title|null`; `citation.authors[]`; `citation.year: integer|null`;
  `citation.venue|null`; `citation.arxiv_id|null`; `citation.doi|null`;
  `citation.openreview_id|null`; `citation.url|null`; `citation.bibtex|null`;
  `citation.source_evidence_ids[]`.

Citation is mandatory for R3/R4 and whenever a publication defines the architecture. A bounded,
evidence-backed `not-found-after-search` is a complete value for an R1/R2 model with no introducing paper.
Country describes institutional origin of the introducing work, never author nationality. It is never
guessed from names.

#### Licenses

- `licenses: object|null`; when accepted: `licenses.code:
  {spdx, status, source_id, locator, evidence_ids[]}` where `status` is
  `declared|not-found|custom|not-applicable` and unknown SPDX is `NOASSERTION`.
- `licenses.paper_text: {status, source_id|null}` with
  `linked-not-redistributed|short-excerpt-committed|not-applicable`.
- `licenses.weights: {status:"not-used"}`.
- `licenses.data: {spdx|null, status, source_id|null, evidence_ids[]}`.
- `licenses.redistribution_class:
  "public-compatible"|"restricted-private"|"manifest-only"|"not-applicable"`.

#### Source resolution and mandatory links

- `source_resolution.rung: "R1_LIBRARY"|"R2_VENDOR"|"R3_PORT"|
  "R4_REIMPLEMENT"|"R5_SKIP"`.
- `source_resolution.decision`; `source_resolution.rung_evidence`;
  `source_resolution.searched_at`; `source_resolution.attempted_rungs[]`, each with
  `{rung, result, reason_code, evidence_ids[]}`.
- `source_resolution.search_report: {queries[], places_checked[], links_checked[], languages_checked[],
  archives_checked[], started_at, finished_at, conclusion}`.
- `source_resolution.mandatory_link_status: "ok"|"failed"` and
  `source_resolution.primary_source_id`.
- `source_resolution.sources[]`, each with
  `{source_id, role, kind, url, revision_kind, revision, locator, content_sha256, byte_count,
  media_type, retrieved_at, fetch_recipe, mirror_class, mirror_digest|null}`.

`role` is one of `implementation`, `introducing-paper`, `supplement`, `project-page`,
`documentation`, `license`, `affiliation`, or `archive`. `kind` is one of `repository`, `package`,
`paper`, `web-page`, or `archive`. Every model, including `failed:source`, retains at least one exact public
identity/source-lead URL. Every selected rung has an authoritative implementation or primary-description
URL; R5 stores the primary evidence URL plus the complete negative search report.

#### Literal evidence and grounding

- `evidence.excerpts[]`, each with
  `{evidence_id, source_id, locator, text, text_sha256, supports[], family_level, license_disposition}`.
- `evidence.coverage: {all_agent_fields_have_support, missing_support[], family_grounding_complete}`.
- `evidence.evidence_identity`; `evidence.family_grounding_path|null`.

Excerpts are literal, minimal, and sufficient. They support every source-read positive claim, including
description, citation, year, country, license, taxonomy, input semantics, rung choice, and fidelity.
Family-level grounding is stored once and referenced by all size variants.

#### Implementation and framework

- `implementation.original_framework`; `implementation.run_framework`;
  `implementation.native_object_type`; `implementation.native_call_method`;
  `implementation.transparent_forward_adapter: boolean`.
- `implementation.recipe_type: "declarative-library"|"typed-adapter"|"port"|
  "reimplementation"|"none"`; `implementation.code_path|null`; `implementation.code_sha256|null`;
  `implementation.builder_symbol: "build_model"|null`; `implementation.dummy_call_symbol:
  "make_dummy_call"|null`.
- `implementation.library_recipe|null` with
  `{distribution, version, artifact_sha256, module, symbol, kwargs, pretrained_disable_fields[]}`.
- `implementation.upstream_files[]` with `{source_id, path, sha256, use}`.
- `implementation.patches[]` with `{path, sha256, classification, semantic:boolean, rationale,
  evidence_ids[]}`.
- `implementation.source_to_code_map[]` with
  `{material_item, source_id, source_locator, evidence_ids[], code_path, code_locator, disposition}`.
- `implementation.declared_choices[]` with
  `{field, value, source_status, material:boolean, rationale, evidence_ids[]}`.
- `implementation.initialization: {policy:"random", pretrained_disabled:true,
  source_specified_choices[]}`; `implementation.mode`; `implementation.device_policy`;
  `implementation.required_construct_asset: null`.
- `implementation.recipe_revision`; `implementation.torchlens_import_static_check: "passed"`.

Arbitrary expression, statement, `eval`, and `exec` recipes are forbidden. R1 may use the declarative
recipe. R2-R4 use a typed module. Native objects expose a transparent Python `forward` that delegates to
the recorded native method without changing tensor math.

#### Input and output contract

- `input_contract.code_path|null`; `input_contract.builder_symbol`; `input_contract.seed`;
  `input_contract.semantic_description`; `input_contract.source_basis[]`;
  `input_contract.smallest_valid_probe_rationale`.
- `input_contract.args[]` and `input_contract.kwargs[]`, each leaf with
  `{path, kind, semantic_role, shape, dtype, device_policy, distribution, constraints[],
  source_evidence_ids[]}`.
- `input_contract.non_tensor_values[]` with
  `{path, type, value, semantic_role, constraints[], source_evidence_ids[]}`.
- `input_contract.masks_state_and_control[]`; `input_contract.expected_output_semantics`.

The shape list contains integers or declared symbolic dimensions. Distribution is typed, such as
`normal`, `uniform`, `integer-range`, `zeros`, `ones`, `categorical`, or `constructor`. Batch one and the
smallest source-valid variable dimensions are preferred; fixed architectural dimensions are never shrunk.

#### Mechanical observations

- `observed.parameter_count_total`; `observed.parameter_count_trainable`;
  `observed.output_signature: {tree, leaves[]}` where each leaf has
  `{path, kind, shape|null, dtype|null, device|null, python_type}`.
- `observed.constructor_seconds`; `observed.forward_seconds`; `observed.peak_rss_bytes`;
  `observed.measurement_attempt_ids[]`; `observed.snippet` generated from the accepted recipe;
  `observed.snippet_sha256`.

#### Runtime modes

`identity.variant` remains the family/size variant identity; it does not describe runtime mode. The
separate `modes` object records the train/eval execution axis:

- `modes.meaningful_modes: string[]`, a subset of `"train"|"eval"`; default both. A model with no
  BatchNorm, Dropout, or mode-dependent branch may declare only its one meaningful mode.
- `modes.per_mode_run: {train?: run receipt reference/status, eval?: run receipt reference/status}` for
  every meaningful mode's forward outcome.
- `modes.train_eval_divergence: "none"|"statistical"|"structural"`; `statistical` means identical
  output shapes with differing values (for example BatchNorm or Dropout), while `structural` means a
  different output shape or op graph (for example a YOLO Detect head or RPN/detection post-processing).
  Structural divergence is explicit because later tracing must capture both modes.
- `modes.divergence_evidence`: a brief note, such as `output-shape-train vs output-shape-eval`.

#### Fidelity and metadata gate

- `fidelity.required`; `fidelity.reason`;
  `fidelity.verdict: "match"|"minor-drift"|"major-drift"|"slop"|"cannot-verify"|null`;
  `fidelity.fidelity_identity|null`; `fidelity.gate_id|null`; `fidelity.current`;
  `fidelity.permanent_scar: boolean`; `fidelity.deviations[]`.
- `accuracy_gate.required: true`; `accuracy_gate.vet_identity|null`;
  `accuracy_gate.gate_id|null`; `accuracy_gate.verdict:
  "accurate"|"inaccurate"|"cannot-verify"|null`; `accuracy_gate.current`;
  `accuracy_gate.checker_model`; `accuracy_gate.checker_version`;
  `accuracy_gate.prompt_sha256`.

`match` means no architectural deviation. `minor-drift` is allowed only for explicitly declared,
nonmaterial, source-unspecified probe or initialization choices; it can never hide changed tensor math,
topology, dimensions, connectivity, state, or output. Any material divergence is `major-drift`.
`slop` names a generic stand-in or knowingly nonfaithful construction and sets a permanent historical
scar even after a later repair. `cannot-verify` means the frozen evidence cannot decide.

Every R3/R4 record requires a current verdict. Every classic, every inherited faithful-claimer, every
known/presumed slop row, and all ruled audit samples receive a stored five-way verdict before crawl
completion.

#### Execution, status, provenance, budgets, and completeness

- `execution.execution_identity`; `execution.environment_id`; `execution.env_generation`;
  `execution.accepted_attempt_ids[]`; `execution.confirmation_policy:
  "two-cold-r3-r4"|"single-mechanical"|"mechanical-canary"`;
  `execution.network_attempted:false`; `execution.checkpoint_accessed:false`;
  `execution.last_verified_at`; `execution.current`.
- `status.kind: "runs"|"deferred"|"skipped"|"failed"`; `status.code` from section 6;
  `status.stage|null`; `status.reason_code|null`; `status.detail|null`;
  `status.traceback|null`; `status.no_traceback_reason|null`; `status.attempted_rungs[]`;
  `status.retries: {source, fetch, evidence, author, gate, environment, import, constructor,
  input, forward, fidelity}`; `status.environment|null`; `status.timestamp`;
  `status.attempt_ids[]`; `status.root_cause_fingerprint|null`;
  `status.supersedes_revision|null`; `status.human_review:
  {required, reason|null, queue|null, requested_at|null}`.
- `provenance.author_model`; `provenance.author_version`; `provenance.author_prompt_sha256`;
  `provenance.checker_model`; `provenance.checker_version`;
  `provenance.producer_run_id`; `provenance.machine_id`.
- `budget.author_sessions_used`; `budget.author_sessions_max`; `budget.gate_rounds_used`;
  `budget.run_revisions_used`; `budget.explicit_grants[]`.
- `flags[]`; `notes`; `scar_history[]`.
- `completeness.schema_valid`; `completeness.mandatory_source_present`;
  `completeness.source_read_fields_complete`; `completeness.evidence_coverage_complete`;
  `completeness.accuracy_gate_current`; `completeness.required_fidelity_current`;
  `completeness.execution_current`; `completeness.family_template_valid`;
  `completeness.release_eligible`; `completeness.issues[]`.

### 5.2 `attempt.v2`

Every controlled fetch, environment build/probe, candidate build/input/forward, confirmation, repair, and
policy failure produces an immutable attempt. Required fields are:

- `schema_version: "menagerie.crawler.attempt.v2"`; `attempt_id`; `ledger_seq`; `payload_sha256`;
  `work_id`; `stable_id|null`; `attempt_no`; `parent_attempt_id|null`; `actor`;
  `stage`; `mode: "train"|"eval"|null`; `started_at`; `finished_at`;
  `result: "succeeded"|"failed"|"observed"`. `mode` is required for a meaningful-mode forward
  receipt and null for non-forward attempts.
- `attempted_rungs[]`; `retries: {stage_attempt, root_cause_repeat, author_round, gate_round}`.
- `identities: {source, evidence, recipe, environment, execution, runner, author_prompt,
  checker_prompt}` with inapplicable values null.
- `environment: {family, target, env_id, lock_sha256, resolved_export_sha256, python,
  packages_manifest_sha256, compiler_identity, sdk_identity}`.
- `host: {machine_id, os, os_build, architecture, cpu, ram_bytes, accelerator,
  accelerator_runtime}`.
- `invocation: {argv[], cwd, safe_env, seed, device, mode, network_policy,
  timeout_seconds, rss_limit_bytes, scratch_limit_bytes}`. Secrets are never recorded; `safe_env`
  contains only allowlisted keys.
- `worker_receipt: {present, receipt_sha256, constructor_started, constructor_completed,
  input_completed, forward_started, forward_completed, mode, input_signature, output_signature,
  parameter_count_total, parameter_count_trainable, native_framework, delegated_method}`.
- `supervisor_observation: {exit_code, signal, wall_seconds, cpu_seconds, peak_rss_bytes,
  stdout_sha256, stdout_bytes, stdout_tail, stderr_sha256, stderr_bytes, stderr_tail,
  full_log_local_path, full_log_retention}`.
- `policy_observation: {network_attempted, socket_targets[], checkpoint_or_weight_read_attempted,
  checkpoint_paths[], write_outside_scratch_attempted, write_paths[], credentials_present,
  torchlens_import_attempted, cache_read_attempted}`.
- `error: null|{stage, reason_code, exception_type, message, traceback,
  no_traceback_reason, native_crash, root_cause_fingerprint, details}`.
- `defer_evidence: null|{target_status, source_ids[], probe_attempt_ids[], explanation}`.

`traceback` stores the full Python traceback without truncation when Python produced one. Full stdout and
stderr remain local, but their hashes, sizes, paths, retention, and bounded 1,500-character tails are
committed. A dead worker has `worker_receipt.present=false`; only the parent observation is populated.

### 5.3 `gate.v2`

One immutable gate record is either a `metadata_batch` envelope for 10--20 independent model items or a
per-model `fidelity` envelope. A metadata batch carries only description and lightweight metadata
(citation, year, country, and license) checks; every item carries its own stored source excerpt and receives
its own verdict. Required envelope fields are:

- `schema_version: "menagerie.crawler.gate.v2"`; `gate_id`; `ledger_seq`; `payload_sha256`;
  `gate_kind: "metadata_batch"|"fidelity"`; `batch_size`; `gate_round`; `gate_identity`.
- `checker: {provider:"openai", model, version, prompt_sha256, started_at, finished_at}`.
- `items[]`: `metadata_batch` contains 10--20 items and `fidelity` contains exactly one item. Each item has
  `work_id`, `stable_id`, `family_representative_id`, `fidelity_identity|null`, `vet_identity`, and its
  item-specific `verified_hashes`, `integrity`, `verdict`, `field_checks`, `fidelity`, `rung_check`,
  `unsupported_claims`, `required_repairs`, and `confidence` fields below.
- `verified_hashes: {proposal, source_manifest, evidence, code, source_to_code_map,
  family_template|null}`.
- `integrity: {verdict, hash_mismatches[], excerpt_discrepancies[], locator_failures[]}` where the
  verdict is `accurate|inaccurate|cannot-verify`.
- `verdict: "accurate"|"inaccurate"|"cannot-verify"`.
- `field_checks[]`, each
  `{field, verdict, evidence_ids[], checked_source_ids[], reason, required_repair|null}`.
- `fidelity: {required, verdict, material_checks[], unsupported_choices[], contradictions[],
  omissions[], permanent_scar}`. Fidelity verdict is one of the five canonical values or
  `not-applicable`; each material check is
  `{category, verdict, source_id, source_locator, evidence_ids[], code_path, code_locator, reason}`.
- `rung_check: {selected_rung, highest_applicable, verdict, findings[]}`.
- `unsupported_claims[]`; `required_repairs[]`; `confidence: "high"|"medium"|"low"`;
  `result_envelope_sha256`.

The fields after `items[]` are item fields, not batch-wide judgments; an envelope-level summary may not
replace an item's verdict. `metadata_batch` items have `fidelity.required=false` and
`fidelity.verdict="not-applicable"`. Top-level `accurate` for an item requires all of its scoped checks to
be accurate. `inaccurate` wins over `cannot-verify` when any source contradiction exists. A deliberately
null best-effort field can be accurate only when the stored bounded search supports the null. For R3/R4,
the per-model fidelity envelope also requires fidelity `match` or properly delimited `minor-drift`; material
drift is never acceptable.

## 6. Status taxonomy, error taxonomy, and partition invariant

### 6.1 Public terminal statuses

The only public current terminal status codes are:

```text
runs
deferred:needs-cuda
deferred:needs-x86
skipped:no-usable-description
skipped:not-a-real-NN
failed:<stage>
```

Closed failure stages are:

```text
intake source fetch evidence accuracy-gate environment import constructor input
forward fidelity resource policy runner
```

The canonical human-gate terminal is exactly `failed:accuracy-gate`; it has
`human_review.required=true`, is excluded from the runnable website set, and may later be superseded by
an append-only reviewed result.

Workflow labels such as `UNTRIAGED`, `queued`, `authoring`, `awaiting-gate`, `fidelity-pending`,
`environment-pending`, `forward-observed-but-blocked`, and `paused:usage-limit` are scheduler/operational
states, never public model terminal statuses. A model may already have driver-awarded `runs` while its
agent-authored metadata is `awaiting-gate`; it is not crawl-complete or release-eligible until the gate
is current.

### 6.2 Closed reason codes

Each `failed:<stage>` carries exactly one versioned reason code from the following initial vocabulary.
Free-form detail supplements but never replaces it.

| Stage | Allowed reasons |
| --- | --- |
| `intake` | `schema-invalid`, `stable-id-conflict`, `duplicate-revision-conflict`, `migration-invariant` |
| `source` | `identity-unresolved`, `missing-mandatory-link`, `source-model-mismatch`, `higher-rung-unresolved`, `effort-cap-exhausted` |
| `fetch` | `unreachable`, `revision-missing`, `hash-mismatch`, `access-denied`, `artifact-missing`, `effort-cap-exhausted` |
| `evidence` | `locator-missing`, `excerpt-mismatch`, `insufficient-detail`, `coverage-incomplete`, `search-incomplete`, `effort-cap-exhausted` |
| `accuracy-gate` | `inaccurate-cap-exhausted`, `cannot-verify-cap-exhausted`, `identity-mismatch`, `checker-contract-invalid`, `effort-cap-exhausted` |
| `environment` | `solve-failed`, `lock-missing`, `artifact-hash-mismatch`, `build-failed`, `probe-failed`, `resolved-export-mismatch`, `island-cap`, `below-minimum-island-size`, `effort-cap-exhausted` |
| `import` | `module-missing`, `symbol-missing`, `abi-load-failed`, `import-exception`, `effort-cap-exhausted` |
| `constructor` | `exception`, `requires-checkpoint`, `requires-weight-asset`, `invalid-model-object`, `effort-cap-exhausted` |
| `input` | `contract-invalid`, `source-invalid-shape`, `generation-exception`, `semantic-constraint`, `effort-cap-exhausted` |
| `forward` | `exception`, `mode-run`, `incomplete-receipt`, `invalid-output-signature`, `confirmation-mismatch`, `effort-cap-exhausted` |
| `fidelity` | `major-drift-cap-exhausted`, `slop-cap-exhausted`, `cannot-verify-cap-exhausted`, `identity-mismatch`, `effort-cap-exhausted` |
| `resource` | `timeout`, `oom`, `disk-floor`, `scratch-cap`, `rss-cap`, `effort-cap-exhausted` |
| `policy` | `network-attempt`, `checkpoint-read`, `write-outside-scratch`, `credentials-exposed`, `torchlens-import`, `opaque-code`, `effort-cap-exhausted` |
| `runner` | `native-crash`, `signal`, `missing-receipt`, `protocol-violation`, `ledger-corruption`, `internal-error`, `effort-cap-exhausted` |

The full failure object always contains `kind`, `code`, `stage`, `reason_code`, full traceback or an
explicit `no_traceback_reason`, attempted rungs, retries by stage, environment, timestamp, attempt IDs,
root-cause fingerprint, and `human_review`.

A failure in one meaningful train/eval forward is never silently collapsed into another mode's success. It
is visibly reported as `failed:mode-run{mode,...}` in the mode outcome and materialized as
`failed:forward` with `reason_code="mode-run"` in the closed terminal taxonomy.

Deferrals require positive evidence. `needs-cuda` means source or a focused probe proves unavoidable
NVIDIA/CUDA use and no source-provided CPU/reference path. `needs-x86` means package/source-build evidence
proves the arm64 path unavailable. Memory, checkpoint need, checker trouble, and ordinary environment
failure are failures, not deferrals. The designated detectron2/mmcv heavy-build policy in section 11 is
the ruled bounded path into `deferred:needs-x86` after two recorded arm64 CPU source-build failures.

Skips are epistemic only. Operational failure never becomes a skip. Both skip records contain the
primary source URL, literal evidence, and search report.

### 6.3 Partition and completion

For intake stable-ID set `I`, let the current status sets be `R`, `D_cuda`, `D_x86`, `S_desc`, `S_nn`,
and `F_stage`. The reducer must assert:

```text
I = R ∪ D_cuda ∪ D_x86 ∪ S_desc ∪ S_nn ∪ (∪ F_stage)
all sets are pairwise disjoint
no current stable ID appears more than once
no current terminal record points to a stale parent or identity
```

The crawl-complete gate additionally requires:

```text
zero scheduler/workflow rows except superseded history
zero missing mandatory source links
zero accepted agent fields without literal evidence references
zero missing or stale metadata gates
zero missing or stale required fidelity verdicts
all classics, faithful-claimers, and known/presumed slop pools have stored Codex verdicts
zero stale runs under the current recipe, runner, and env generation
all family variants reference a current accepted representative template
license sweep passed before any public merge
```

An operational `paused:usage-limit` is visible but prevents a complete report while pending. A later
reviewed result supersedes the old terminal revision; the current partition still contains exactly one
status while the old failure remains auditable.

## 7. Source-rung ladder and anti-slop rules

The author or deterministic source template evaluates rungs in order. Selecting a lower rung stores why
every higher rung was unavailable.

### R1 — `R1_LIBRARY`

Use the exact, materially unmodified architecture in a maintained public library. Required evidence is
the official registry/docs URL, upstream repo at the package's exact tag/commit, distribution/version and
artifact hash, exact symbol/kwargs, explicit disabling of pretrained weights, and a supported mapping
from the Menagerie name to that symbol. A library that changed topology under the same name is not R1.

### R2 — `R2_VENDOR`

Use the real upstream repository implementation at an exact revision. It may be installed from a locked
artifact, imported from the controlled source CAS, or reached by a small committed adapter. Allowed
patches are nonsemantic only: import-path/API shims, renamed equivalent kwargs, removal of unreachable
training/data entry points, and transparent framework/forward adapters. Every patch is hashed and
classified. A change to tensor math, topology, parameters, initialization semantics, state/control, or
return structure moves the work to R3.

A custom CUDA performance kernel may use a CPU/reference implementation only when exact evidence and the
Codex fidelity check establish mathematical equivalence. Shape similarity is not proof. Otherwise the
real source defers to the capable machine.

### R3 — `R3_PORT`

Faithfully translate real implementation code only when the exact upstream cannot reasonably execute on
either planned target as written—for example an extinct framework/runtime, unsupported source language,
or unavailable proprietary operator with a complete reference. Store repo/revision/path/hash, literal
original code and descriptive excerpts, a material source-to-code map, committed port path/hash, all
deviations and source-unspecified choices, and proof R1/R2 were unavailable.

Apple inconvenience, CUDA, or x86 incompatibility is not permission to port. Fidelity must be current and
`match` or a strictly nonmaterial `minor-drift`; any material difference blocks `runs`.

### R4 — `R4_REIMPLEMENT`

Use only when a bounded documented search finds no usable implementation in any framework and primary
material completely specifies the intended forward architecture. The search checks author/lab repos,
paper and supplement, project page, packages/hubs, Papers With Code, exact title/acronym/author repository
search, archived and non-English surfaces, and cited implementations.

Sufficiency is semantic, not a word count. Evidence must determine:

- full layer/block inventory and order;
- dimensions or dimension rules;
- connectivity, branches, skips, recurrence, attention, and aggregation;
- operators/equations and tensor axes;
- activations, normalization, padding, stride, dilation, groups, and heads;
- state, masks, loops, stochastic/control behavior;
- material initialization when specified;
- input semantics and legal dimensions; and
- output contract.

If the implementation must invent one material choice, R4 is forbidden. Abstracts, gists, blog summaries,
names, and generic diagrams are search leads only. Literal source prose and equations-as-text with exact
locators are mandatory. Fidelity has the same acceptance rule as R3.

### R5 — `R5_SKIP`

- `skipped:no-usable-description`: no usable code exists and the best primary evidence cannot specify the
  forward without material invention.
- `skipped:not-a-real-NN`: the item is not a trainable neural architecture—for example a dataset, loss,
  optimizer, preprocessing algorithm, or conceptual label.

Both require source/search evidence and an accurate Codex-vetted skip justification. A source identity
that cannot be established is `failed:source`, not a source-free skip.

### Anti-slop gates

Before an authored proposal can be accepted, the deterministic validator enforces:

1. rung-specific source evidence and mandatory link are present;
2. R2/R3 references exact original source bytes in the mirror/CAS manifest;
3. a lexical/structural generic-Sequential-versus-exotic-family tripwire flags the measured slop pattern;
4. `compact`, `stand-in`, `simplified`, `representative`, or equivalent approximation language blocks
   acceptance unless it describes a source-authored official variant; and
5. offline construction proves no checkpoint/weight dependency. A structured required construction asset
   is recorded as `failed:constructor`, never silently fetched or converted to a success.

The Codex checker is the semantic backstop, not a substitute for these cheap gates.

## 8. Repository, public mirror, and private mirror boundary

### 8.1 Committed to `menagerie/crawler-pipeline`

```text
menagerie/crawler/
  Python package, schemas, frozen prompts, tests, and procedures
  envs/<intent>/environment.yml
  envs/<intent>/locks/{osx-arm64,linux-x86_64-cuda}.lock
  envs/<intent>/{probes,resolved-exports,capabilities}/...
  adapters/<prefix>/<stable_id>.py
  ports/<prefix>/<stable_id>/...
  patches/<prefix>/<stable_id>/*.patch
  evidence/families/<family_id>.json
  evidence/models/<prefix>/<stable_id>.json
  source_manifests/<prefix>/<stable_id>.json
  mirrors/public-manifest.jsonl
  mirrors/private-manifest.jsonl
  records/intake/*.jsonl + manifests
  records/models/*.jsonl
  records/attempts/*.jsonl
  records/gates/*.jsonl
  records/operational/*.jsonl
  views/current-models/*.jsonl, status-summary.json, deferred-linux.jsonl
```

Committed content includes records, short literal grounding excerpts, hashes, exact fetch recipes, our
adapters/ports/patches, complete Python tracebacks, bounded log tails, target locks, and mirror manifests.
Records and our code remain public at merge.

### 8.2 Durable public content-addressed release store

Legally redistributable full source archives and exact package artifacts are copied to a durable release
store keyed by SHA-256. The committed public manifest records digest, size, media type, upstream URL,
revision, license decision, release-store object key, and verification timestamp. Fetching verifies the
digest before use. JMT owns mirror credentials and backup policy; credentials never enter the repo or
worker environment.

### 8.3 Private restricted mirror

GPL/AGPL and no-license upstream bytes are treated as restricted under JMT's ruling: they remain in a
private content-addressed mirror. The public repository commits only their URL, revision, content hash,
byte count, license finding, and deterministic fetch recipe. No restricted full bytes are committed or
published in the public release store. Our independent port/adapter may be committed only after the
license sweep confirms its treatment.

### 8.4 Local and gitignored

```text
.crawl-local/
  state.sqlite                 # rebuildable queue/current-state cache
  logs/                        # full stdout/stderr and environment logs
  source-cas/                  # fetched full repositories/archives
  artifact-cas/                # wheels, sdists, conda packages, compiler outputs
  envs/                        # installed conda environments
  caches/                      # disposable per-env and per-attempt caches
  scratch/                     # agent/worker staging roots
  locks/driver.lock            # OS advisory lock and owner metadata
  wakeups/                     # generated launchd/systemd state
```

The repo `.gitignore` and checkpoint allowlist enforce this boundary. Garbage collection is mark-and-sweep
from committed manifests and starts in report-only mode.

Before any merge to public main, the license-sweep command must classify every committed code/excerpt and
every mirror entry, fail on unknown disposition, prove restricted full bytes are absent, and emit a
committed report. The pipeline never bypasses this gate.

### 8.5 Commit cadence and branch guard

The checkpoint command refuses to operate unless the current branch is exactly
`menagerie/crawler-pipeline`. It validates ledgers, schemas, identities, partition progress, mirror
manifests, generated views, and the allowlisted change set. The supervisor makes one conventional commit
per completed environment and one daily checkpoint commit when an environment lasts more than a day.
The pipeline does not push. No crawler records are mixed into ordinary TorchLens development branches.

## 9. Do-it-once source-reading checklist

Before a proposal leaves its one rich author session, deterministic validation asks whether any missing
value would force a future agent to reopen a source. The proposal must contain:

- canonical name, aliases/acronym, family relationship, representative/variant scope, and duplicates;
- family, domain, tasks, modalities, era, architecture tags, and novel operators;
- neutral English tagline, two-to-four-sentence description, and key contribution;
- exact implementation, paper, project, documentation, affiliation, and license links;
- repo commits/tags, package artifacts, paths/symbols, paper versions/sections/pages, and hashes;
- the complete structured citation object, including BibTeX;
- year, first-public-date basis, authors, labs, institutions, origin countries, confidence, and note;
- code, paper-text, weights, and data license findings with locators, never inferred;
- rung, why every higher rung failed, every query/place/URL/archive/language checked, and conclusion;
- `original_framework`, `run_framework`, native object/call, exact builder/config, init/mode/device,
  patches, and source-to-code mapping;
- semantic input and output contract, fixed/variable dimensions, dtypes, value ranges, masks, state,
  non-tensor arguments, and smallest-valid-probe rationale;
- literal minimal excerpts supporting every authored positive field;
- for R3/R4, the full material architecture evidence, candidate code path/hash, declared choices and
  deviations, and fidelity map; and
- uncertainties and explicit bounded `not-found-after-search` results.

The following are intentionally mechanical and may be derived later from accepted source/code/receipts:
parameter counts, output pytrees/shapes/dtypes, timings, peak RSS, exact package exports, deterministic
website recipe snippets, graph statistics, FLOPs, render assets, and embeddings. Benchmarks, training
recipes, checkpoint inventories, and accuracy claims are outside the crawl rather than deferred source
fields.

Website voice is versioned and neutral: no hype, marketing adjectives, benchmark claims, or "state of the
art"; name the distinctive mechanism; distinguish introducing work from a later library implementation;
and say task/domain only when grounded.

## 10. Block-at-write Codex gate loop

1. Claude atomically writes one complete proposal and frozen source/evidence/code manifest.
2. Deterministic validation checks schema, hashes, URL syntax, exact locators, evidence coverage,
   import syntax, typed adapter contract, no-pretrained policy, no opaque expressions, staging-root
   containment, and source-rung mechanical gates.
3. The checker dispatcher packs 10--20 eligible description/lightweight-metadata proposals into one fresh
   Codex `metadata_batch` envelope; each item includes its own frozen pack and stored source excerpt.
   Required R3/R4 fidelity review uses a separate fresh per-model `fidelity` envelope.
4. Codex atomically writes `gate.v2` with a per-item verdict. The dispatcher independently recomputes all
   identities and rejects any batch summary that lacks an item result.
5. `accurate`: the single writer appends that item's authored facts and gate only after its batch result.
6. `inaccurate`: exact findings go to Claude for a bounded repair against the same frozen pack; the repaired
   item joins the next eligible metadata batch (or its next per-model fidelity check).
7. `cannot-verify`: one evidence repair may use already fetched sources. One controlled additional fetch
   is allowed only if the stored bounded search was incomplete; it remains part of the same campaign.
8. The initial check plus at most two repair/check rounds are allowed. An identical root finding twice
   stops early.
9. Cap exhaustion writes `failed:accuracy-gate`, preserves evidence/findings/hashes, sets
   `human_review.required=true`, and excludes the model from the runnable release set. Later human/Sol
   review appends a superseding revision.

All changed authored bytes create a new vet identity. Changed source/evidence/code bytes also create a new
fidelity identity where applicable. There is no purge-after workflow because ungated authored claims never
enter accepted facts.

For family variants, Codex gates the representative text and family grounding once. The deterministic
variant templater may then add only the measured parameter/input line. A template-hash or nonallowed prose
difference fails validation and requires a new gate.

Fidelity routing uses the canonical five outcomes:

- `match`: fidelity passes.
- `minor-drift`: passes only under the strict nonmaterial definition in section 5.1.
- `major-drift`: bounded repair, then `failed:fidelity`.
- `slop`: permanent scar, full re-triage from source, then `failed:fidelity` if not repaired.
- `cannot-verify`: one evidence-repair round; unresolved becomes `failed:fidelity`.

## 11. Thick environments, exact locks, and arm64 routing

### 11.1 Intent roster

The planner seeds from the empirical successes and failures in `menagerie/data/env_specs.json`,
`validation_envs.yml`, and the routing manifest, but none of those old recipes is current truth. Target a
small set of thick environments. The listed intent roster (~8 PyTorch + ~3 native-tail) is a starting
point, not a ceiling: cluster models by real dependency compatibility; if honest clustering needs a few
more thick environments, that is fine. Never force-merge incompatible dependencies into a fragile
mega-environment just to hit a number, and do not proliferate toward one environment per model. No hard
cap; keep it reasonable. The starting roster is:

**Phase 1 — PyTorch sweep (eight intents):**

1. `core`: PyTorch, torchvision, timm, Transformers, Diffusers, common pure-Python libraries, and most
   R3/R4 code.
2. `graph`: PyG/DGL, molecular, equivariant, geometric, and scientific packages.
3. `audio`: torchaudio, speech, codec, vocoder, and music stacks.
4. `mmlab`: compatible OpenMMLab detection, segmentation, pose, OCR, video, and restoration stacks.
5. `tabular-ts`: tabular, recommendation, time-series, survival, and forecasting packages.
6. `detect-misc`: detectron2, compiled CV ops, specialized detectors, and compatible misc vision stacks.
7. `legacy-torch`: old but source-runnable Python/PyTorch/NumPy combinations.
8. `oddballs`: exact research repositories that do not fit another compatible thick env.

**Phase 2 tail — native frameworks (three intents):**

9. `tf-keras-arm64`.
10. `jax-flax-arm64`.
11. `paddle-arm64`.

A controlled split is appropriate when empirical solving proves one listed intent incompatible. The planner
may merge only genuinely compatible intents, and may add a few thick environments when needed for honest
dependency clustering. An outlier with fewer than 25 models must fit an existing env or end as structured
`failed:environment`; the driver does not create a convenience island or defer status.

R2/R3/R4 code targets `core` unless the exact original source proves another dependency set necessary.
Phase 2 begins only after every PyTorch model has a current Phase-1 terminal status.

### 11.2 Exact-lock discipline

Each target intent commits:

- readable direct-dependency `environment.yml`;
- exact conda-lock target files for `osx-arm64` and, when applicable, `linux-64`/CUDA, including package
  URLs and hashes;
- hashes and build recipes for pip wheels/sdists and source-built artifacts;
- compiler, SDK, framework, Python, and accelerator constraints;
- actual resolved export after creation and its hash;
- import probes and at least three fixed model canaries; and
- a capability manifest consumed by the author and router.

The driver creates a new env from the exact target lock and never updates a mutable old env. Forbidden:
editable SSH/private dependencies, unpinned moving revisions, opportunistic resolver drift, unofficial
indexes without a reviewed manifest, `--no-deps` hacks, or `sitecustomize` version spoofing. Any lock,
artifact, resolved-export, compiler, or probe change creates a new `env_generation` and requires
re-verification.

### 11.3 Sequential lifecycle

For each environment, in order:

1. require at least 30 GiB free and at least twice its last measured installed size;
2. materialize/fetch hashed artifacts, create from the exact lock, and capture build attempts;
3. verify the resolved export equals the declared identity;
4. run imports and fixed canaries;
5. process assigned models with exactly one worker at a time;
6. run the deterministic 2% mechanical canary sample and current env canaries;
7. capture metrics, resolved export, event cursor, and status report;
8. validate and commit the completed env (or daily if it is long-running);
9. remove only that env and its dedicated cache/scratch; and
10. verify disk recovery before advancing.

An environment medic gets at most two source-root-cause repair attempts. It proposes exact intent/lock
changes, which receive a new env identity; it never mutates an active env per model.

### 11.4 CUDA/x86 and the ruled heavy-build pool

Static hints prioritize probes but cannot alone award a deferral. Source or focused probe evidence is
stored in the attempt/defer object.

Detectron2/mmcv and the associated roughly 500-model arm64 heavy-build pool receive at most two CPU
source-build attempts on the mini. If both fail for the designated arm64 build path, the pool records both
full build attempts and becomes `deferred:needs-x86` for the Linux sweep. If source proves unavoidable
CUDA sooner, it becomes `deferred:needs-cuda`. No third mini build is attempted.

Runtime CUDA/x86 signals add learned routing evidence for matching exact package/source identities; they
do not create an unreviewed global name denylist.

## 12. Execution worker and run-award rule

### 12.1 Typed recipe contract

R1 may use a closed declarative `{distribution, version, module, symbol, kwargs}` recipe. All other runnable
recipes expose:

```python
def build_model() -> object:
    """Build the random-initialized model without external weight assets."""

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    """Build a deterministic and semantically valid forward call."""
```

The model or transparent native adapter must expose `forward`. The worker invokes that attribute
explicitly, never `model(...)` as an ambiguous substitute. No `eval`, `exec`, statement string, or arbitrary
callable string is accepted.

### 12.2 Isolation and receipt

There are three deliberately separate network postures: author research is web-enabled through WebSearch
and Exa solely to locate and ground sources; controlled fetch is pinned-network-only through `fetcher.py`
for exact URLs/revisions into the local CAS; and model execution is offline. The offline policy applies to
the execution subprocess, not the author's research phase. Web-search capability never enters
`worker.py` or `worker_supervisor.py`, so a forward cannot depend on a network, checkpoint, or credential.

Every meaningful-mode forward uses a fresh process with seeded framework/Python/NumPy RNGs, the explicit
`model.train()` or `model.eval()` setting for that mode, no-grad/inference mode where valid, capped OMP
threads, fresh framework caches, offline flags, and a credential-scrubbed environment. The worker detects
meaningful modes, invokes `forward()` in both declared meaningful modes, records one per-mode receipt, and
classifies train/eval divergence as none, statistical, or structural from the captured per-mode outputs.
Source/env files are read-only; only scratch/result roots are writable. The cheap security boundary is
argv-not-shell, no credentials or SSH agent, offline framework flags, write auditing, and a socket
tripwire. A socket attempt fails `policy` even if the connection would have failed. VMs and fail-closed
firewall engineering are not required.

Default timeout is 300 seconds, with a declared per-model override no greater than 1,800 seconds. Default
RSS cap is 12 GiB. The parent kills the process group on timeout/RSS violation and records only observable
facts. The receipt records the input and output pytrees without tensor payloads, parameter counts, timing,
and policy observations.

### 12.3 Cold-forward policy

- Agent-authored R3/R4 rows require two driver-owned cold forwards in separate fresh processes with
  separate empty caches, spanning all meaningful train/eval modes. The author dev-run is diagnostic and
  never counts. Receipts for the same mode must match on output tree, shapes, and dtypes; exact random
  values need not match.
- Mechanical R1/R2 rows require one driver-owned cold forward for every meaningful mode.
- Every mechanical row is re-verified whenever its `env_generation` changes.
- A deterministic 2% sample per environment/batch receives a second cold confirmation. Membership is
  `sha256(stable_id + "mechanical-canary-v1") mod 100 < 2` and therefore reproducible.

### 12.4 Driver-only award rule

The reducer awards `runs` only when all applicable predicates are true:

```text
mandatory exact source link and current source identity exist
the rung was earned in this pipeline, never inherited
recipe and input identities match the accepted/current source
driver-owned receipt(s) show constructor, input, and explicit forward completion
every meaningful mode has a successful per-mode receipt
receipt execution identity matches current recipe, runner, target, and env_generation
network/checkpoint/write/TorchLens policy observations are clean
output tree, shape, and dtype signature is complete
anti-slop gates pass
the cold-forward policy is satisfied
for R3/R4: current fidelity is match or strictly nonmaterial minor-drift
```

The metadata accuracy gate is not in this award predicate. Thus mechanical R1/R2 execution continues and
may earn `runs` during Codex backlog. Such a row remains `completeness.accuracy_gate_current=false` and is
excluded from crawl-complete/release views. For R3/R4, fidelity is part of the architecture execution bar,
so a checker outage leaves a visible `forward-observed-but-blocked` workflow state rather than a false run
award.

## 13. Resumable single-writer driver and effort caps

### 13.1 Scheduler phases

```text
INTAKE -> SOURCE_TEMPLATE or AUTHOR_COMPLETE_MODEL -> VALIDATE_PROPOSAL
       -> CODEX_GATE -> ACCEPT_AUTHORED_FACTS -> ENV_PLAN -> RUN
       -> optional RUN_REPAIR + RE-GATE -> REDUCE -> TERMINAL
```

The queue is a rebuildable SQLite view over intake and append-only ledgers. `cursor.json` may accelerate
restart but is never authoritative. SIGTERM finishes or kills the one in-flight worker, appends its honest
observation, checkpoints, and exits. SIGKILL loses at most the uncommitted in-flight result; on restart its
work identity is still unsatisfied. No batch return code creates individual model facts.

An OS `flock` on `.crawl-local/locks/driver.lock` is acquired before any mutable operation. The lock file
records PID, process start time, boot ID, run ID, target, and command. A wakeup that finds a live owner exits
successfully after appending/printing an idempotent `wake-noop-already-running` event. Stale PID metadata is
not enough to break a live kernel lock.

### 13.2 Default effort caps

- mechanical execution: two recipe/input attempts per env generation;
- complete Claude author campaign: one rich initial session plus at most two narrow repair sessions,
  30 tool calls, 20 controlled fetches, and 30 minutes per session;
- proposal contract correction: one short correction;
- Codex: initial check plus two author/check repair rounds;
- source search: the fixed checklist plus one recorded justified extension;
- run repair: two authored recipe/input revisions, each re-gated when bytes change;
- fidelity repair: two rounds; metadata regeneration: two rounds;
- normal forward: 300 seconds; declared override: at most 1,800 seconds;
- environment medic: two attempts; designated arm64 heavy-build: two attempts; and
- identical root-cause fingerprint: stop on the second occurrence.

Cap exhaustion is `failed:<actual-stage>` with `reason_code=effort-cap-exhausted`, not a skip or convenience
defer. Only `tools/requeue --reason ... --grant ...` creates an explicit new work generation; it records the
grant and preserves history.

## 14. Unattended operation and usage-limit self-wake

The driver checkpoints continuously through fsynced facts; relaunch is therefore cheap.

### Claude usage window

When the Claude dispatcher reports a usage limit, the driver:

1. validates and checkpoints all completed envelopes;
2. appends an operational event with status `paused:usage-limit`, provider, observed limit response,
   exact `reset_at`, queued work counts, and current env;
3. leaves mechanical execution running if it does not need Claude; otherwise exits cleanly;
4. installs or updates a one-shot macOS `launchd` wakeup at the actual reset timestamp, with jitter no
   greater than 60 seconds—not a blind five-hour timer; and
5. on wake, reacquires the driver lock, appends `resumed:usage-limit`, rebuilds state, and continues the
   first unsatisfied identity.

The Linux implementation uses the same wake contract through a one-shot systemd timer, with cron as a
documented fallback. Wake installation and removal are idempotent.

### Codex rate/quota limit

Transient rate limits use exponential backoff with bounded jitter and the server reset time when supplied.
Quota exhaustion appends a visible checker pause and schedules a reset-time wake. It never blocks the
mechanical run lane: eligible R1/R2 rows can be `runs` with `accuracy-gate`/`fidelity-pending` completeness
work. R3/R4 successful forwards remain `forward-observed-but-blocked` until fidelity is current. The next
vet sweep drains pending work without reopening sources.

Daily status prints provider pause/reset times, pending gate counts, models with runs but incomplete
metadata, `forward_observed_but_blocked`, session budgets, and wakeup health. Repeated scheduler-launch
failure becomes a visible campaign `failed:runner` and stops rather than looping silently.

## 15. Retro-audit campaign

The current catalog is not grandfathered. Measured debt includes roughly 419 concrete generic-Sequential
slop rows, roughly 1,400 presumptive compact/local/unregistered pools, 2,709 classics, 2,222 unsubstantiated
faithful claimers, and source links on only 44 of 8,533 JSONL rows at the measured snapshot.

`migrate_legacy.py` deterministically snapshots the complete roster and imports every stable ID as
`UNTRIAGED`. It preserves exact legacy row/recipe/module/notes hashes and flags as hints only. Nothing
inherits `runs`, rung, source link status, verification, or fidelity.

Campaign waves are:

1. a balanced 100-model calibration set covering known slop, genuine classics, official libraries,
   non-neural items, CUDA/x86 source, ports, and description-only candidates;
2. all 419 known slop rows, permanently flagged `slop-detected-r1-audit`;
3. all roughly 1,400 compact/local/unregistered presumptive-slop rows, treating the old recipe as absent;
4. all roughly 2,222 faithful/reimplementation claimers;
5. all 2,709 classics, each receiving a stored Codex five-way fidelity verdict;
6. high-volume official library families using deterministic source manifests, representative-level
   description/gate, and per-variant measured templates;
7. remaining repository and research-tail rows; and
8. deferred and discovery-only names, whose deferrals are re-derived rather than inherited.

Every old row is re-executed. Every classic, faithful-claimer, and known/presumed slop entry has a stored
Codex verdict before completion. Stand-ins are never re-blessed because they happen to run. The retro-audit
is complete only when the ordinary partition and completeness gates pass.

## 16. One-time metadata harvest

Agentic models harvest metadata in their one rich model session. Mechanical families use one Sonnet
session for the family representative and deterministic size-variant templating; families that cannot
share grounding get separate representative sessions. Codex vets every representative and every
model-specific authored claim.

The harvest fills all `identity`, `taxonomy`, `website`, `people_and_origin`, `dates`, `citation`,
`licenses`, `source_resolution`, `evidence`, framework, and semantic input fields from section 5.1. It
captures literal grounding while sources are open. The driver fills measured observations and generates
the tested-recipe snippet.

Description contract: tagline no more than 12 words; paragraph two to four factual sentences; contribution
one line; no benchmarks or marketing; every architectural clause grounded. Citation comes from the exact
paper/arXiv/DOI/OpenReview record. Country uses affiliation/project evidence and an explicit confidence;
`cannot-determine` beats guessing. License uses the pinned revision; `NOASSERTION` beats inference.

## 17. Setup, run, resume, teardown, and transfer procedures

### 17.1 Initial setup on the mini

```bash
git switch menagerie/crawler-pipeline
python -m menagerie.crawler doctor --target osx-arm64 --strict
python -m menagerie.crawler intake --all-existing --snapshot-date 2026-07-14
python -m menagerie.crawler plan --target osx-arm64 --phase pytorch
python -m menagerie.crawler run --target osx-arm64 --phase pytorch --sequential-envs --resume
```

`doctor` verifies branch, host/architecture, at least 100 GiB campaign capacity and current disk floor,
conda/lock tooling, mirror configuration, author-agent WebSearch plus Exa (`web_search_exa` and
`web_fetch_exa`) availability in Claude Code on the mini, no worker-visible credentials, schema/prompt/
ledger hashes, offline/socket tripwire, exact-lock availability, static TorchLens import ban, OS lock,
wakeup tooling, and repo/checkpoint policy. It does not require a VM or fail-closed host firewall.

The intake command snapshots master/deferred JSONL, stable IDs, classics registry/module hashes, and
discovery-only records. Re-running it with identical input is a no-op; changed intake creates a new
snapshot while preserving prior history.

### 17.2 Resume and status

```bash
python -m menagerie.crawler status --full --verify-partition
python -m menagerie.crawler run --target osx-arm64 --resume
```

Resume acquires the OS lock, validates JSONL tails and result envelopes, rebuilds SQLite, verifies/recreates
the current env from its exact lock, checks scheduled wakeups, and resumes the first unsatisfied identity.
It never edits a complete accepted fact.

### 17.3 Phase-2 native tail

```bash
python -m menagerie.crawler plan --target osx-arm64 --phase native-tail
python -m menagerie.crawler run --target osx-arm64 --phase native-tail --sequential-envs --resume
```

The planner refuses this phase until the PyTorch phase has no workflow rows. Transparent adapters record
original/run frameworks and delegated native calls. Tracing policy remains deferred.

### 17.4 Clean teardown

```bash
python -m menagerie.crawler checkpoint --verify-ledgers --verify-views
python -m menagerie.crawler teardown --target osx-arm64 --active-env --verify-disk
python -m menagerie.crawler status --full --verify-partition
```

Teardown stops dispatch, handles the one in-flight subprocess, closes facts, removes only the active env
and dedicated caches/scratch, verifies disk recovery, rebuilds views in a clean process, compares digests,
and removes obsolete wakeups. Artifact GC is report-only unless explicitly authorized.

### 17.5 One-command Linux/NVIDIA deferred sweep

After the branch and private/public mirror credentials are available on the Linux/NVIDIA box, the entire
handoff is:

```bash
python -m menagerie.crawler handoff-linux --resume --only-status 'deferred:*'
```

That command is self-contained and performs, in order: branch guard and pull-safety check; strict
`linux-x86_64-cuda` doctor; fetch-by-hash of required public/private mirror objects; generation of the
current deferred view; selection only of `deferred:needs-cuda` and `deferred:needs-x86`; exact Linux/CUDA
lock validation; sequential env creation; the same driver/worker/gate logic; superseding Linux terminal
revisions; per-env/daily checkpoints; teardown; and a final partition/completeness report. It never copies
installed Mac envs or mutable SQLite. Mac deferrals remain history. If git pull must be done manually by
policy, run it immediately before this one command.

### 17.6 Machine transfer and commit behavior

Only git-committed facts/code/prompts/locks/manifests move between machines. Full artifacts are fetched by
content hash. Attempt ledgers use machine-specific filenames, but the same single driver owns each active
machine campaign; concurrent Mac/Linux crawling is not a supported mode. The checkpoint reducer merges
completed machine facts deterministically and detects conflicting current revisions.

## 18. Frozen verbatim agent prompts

The following two templates are canonical. Implementation copies their text byte-for-byte into the prompt
files; only bracketed placeholders are substituted. Their exact file hashes participate in `vet_identity`
and `fidelity_identity`.

### 18.1 Claude crawler-author prompt

```text
SYSTEM ROLE
You are the Claude Sonnet/Opus author for ONE Menagerie model. This is the model's single expensive
source-reading campaign. Produce a complete, cold-auditable proposal now so no future agent must reopen
the sources. A fresh Codex checker from a different model family will verify every factual field and any
implementation before the single writer may accept it. Honest uncertainty, skips, deferrals, and failures
are acceptable. Unsupported claims are not.

SCOPE
The only execution goal is to construct a random-initialized architecture, build a semantically valid dummy
call, and explicitly call model.forward(*args, **kwargs). Never use TorchLens. Never trace, render, train,
evaluate accuracy, or fetch pretrained weights/checkpoints. Native TensorFlow/Keras, JAX/Flax, and Paddle
models remain native behind a transparent forward adapter during the phase-2 tail. Record both
original_framework and run_framework. Do not decide any later TorchLens tracing strategy.

AUTHORITY
1. Read WORK_ENVELOPE_PATH first. Its stable_id, budgets, frozen source manifests, environment capability
   sheets, and allowed_output_root are binding.
2. Write only below allowed_output_root. Do not edit canonical records, ledgers, queue state, environment
   specs, accepted adapters, other models, or committed files.
3. Do not install packages or mutate an environment. Return a typed dependency request when needed.
4. You are granted WebSearch plus Exa (`web_search_exa` and `web_fetch_exa`) for the research/authoring
   phase. Use them to find real source repositories, papers, grounded descriptions, citations, and
   provenance; capture each exact URL and verbatim excerpt. Web results discover and ground sources, but
   only the controlled fetcher may retrieve public source bytes into the campaign: record every URL, exact
   revision/version, locator, retrieval time, content hash, and fetch recipe. Never invent a URL or cite a
   search-results page. This web capability never authorizes network access by the execution subprocess.
5. Do not declare runs and do not self-approve accuracy or fidelity. Only the deterministic driver can
   award runs; only Codex can issue the independent gate.
6. Treat every inherited recipe, note, source claim, run label, and fidelity label as an untrusted hint.
   Re-earn every claim.
7. Work on this model completely in this one rich session. Source triage, recipe or adapter, dummy input,
   literal evidence, description, citation, year, country, license, and taxonomy are not separate phases.

NON-NEGOTIABLE SOURCE LADDER
Evaluate in order and select the first applicable rung:
R1_LIBRARY: the exact materially unmodified architecture in a maintained public library.
R2_VENDOR: the exact runnable upstream repository implementation at a pinned revision.
R3_PORT: a faithful translation of real code only when R1/R2 cannot reasonably run on either planned target.
R4_REIMPLEMENT: only when a documented search finds no usable code in any framework and detailed primary
material specifies every material forward choice.
R5_SKIP: no usable description, or not a real trainable neural architecture.

Apple CUDA/x86 incompatibility is a deferral, not permission to port or reimplement. A gist, abstract,
blog summary, architecture name, or generic diagram is never sufficient. A CUDA performance kernel may be
replaced only when exact evidence establishes mathematical equivalence; declare it for Codex review.
Never invent topology, dimensions, operators, activations, normalization, padding, connectivity, state,
initialization, input semantics, or output behavior. If a material choice is missing, do not fill it in.

RANDOM-INITIALIZATION AND EXECUTION POLICY
- Explicitly disable every pretrained, weights, and checkpoint flag.
- Do not download or read a checkpoint. If construction requires one, report BLOCKED at constructor with
  reason requires-checkpoint.
- Default to eval mode and CPU. Cite any source-required different mode or device.
- Use batch 1 and the smallest source-valid variable dimensions. Never shrink a fixed architectural
  dimension or alter architecture for convenience.
- R1 may use the closed declarative library recipe. Other runnable work must stage a typed adapter defining
  build_model() and make_dummy_call(seed, device). The returned object must expose forward.
- A transparent native adapter must record its native object type and delegated method.
- Record every positional and keyword leaf's semantic role, shape, dtype, legal value range or distribution,
  device policy, constraints, masks/state, and all non-tensor values.
- Your candidate dev-run is diagnostic only. It does not count toward runs. The deterministic driver will
  execute every accepted recipe independently using the canonical cold-forward policy.

DO-IT-ONCE REQUIRED CONTENT
Before finishing, fill every source-reading field in author-proposal.v2:
1. canonical name, aliases/acronym, family relationship, representative/variant scope, duplicate links,
   domains, tasks, modalities, era, architecture tags, and novel operators;
2. website English: a neutral tagline of at most 12 words, a grounded two-to-four-sentence paragraph, and
   one key-contribution line;
3. implementation, paper, project, documentation, affiliation, and license links; exact revisions,
   paths/symbols, paper versions/locators, content hashes, and fetch recipes;
4. structured citation: title, authors[], year, venue, arxiv_id, doi, openreview_id, url, and bibtex;
5. first-public year/date basis; authors, labs, institutions; institutional origin countries as ISO alpha-2,
   confidence, basis, and note. Never infer nationality or country from a name;
6. code, paper-text, weight, and data license facts. Use NOASSERTION, null, or not-found-after-search rather
   than guessing;
7. rung decision and why every higher rung failed, including queries, places, URLs, dates, languages,
   archives, and conclusions;
8. original_framework, run_framework, native object/call, exact build symbol/config, initialization, mode,
   device policy, patches, code path, and source-to-code map;
9. semantic input and output contract, legal dimensions, dtypes, values, masks, state, and control;
10. LITERAL VERBATIM excerpts copied exactly with source_id and precise locator. Every authored factual
    leaf must name supporting evidence_id values. Keep excerpts minimal but sufficient. Size variants may
    reference the family representative's stored grounding and may differ only in the parameter/input line;
11. for R3/R4: literal architecture code/prose/equations-as-text, staged candidate code path, every declared
    deviation or source-unspecified choice, and the complete material-choice checklist; and
12. explicit uncertainties and bounded not-found-after-search results.

WEBSITE VOICE
Use clear, factual, non-promotional English. Say what the architecture is, what mechanism distinguishes it,
and its sourced task/domain. Do not use marketing language, benchmark claims, "state of the art," vague
praise, or facts from memory. Distinguish the introducing work from the library that implements it.

R4 SUFFICIENCY TEST
R4 is allowed only if literal primary evidence determines layer inventory and order; dimensions or rules;
connectivity, branches, skips, recurrence, attention, and aggregation; operators, equations, and axes;
activations, normalization, padding, stride, dilation, groups, and heads; state, masks, control, and
stochastic behavior; material initialization; input semantics; and output contract. There is no word-count
shortcut. One unspecified material choice means skipped:no-usable-description.

PROCESS
1. Inspect intake hints and prior failures, but trust only newly verified exact sources.
2. Complete the ladder and freeze exact sources through the controlled fetcher.
3. Capture literal evidence while each source is open and attach evidence_ids to all authored fields.
4. Author complete metadata and, when applicable, staged adapter, patch, port, or reimplementation code.
5. Run the deterministic proposal validator and fix schema/evidence errors within this same session.
6. If candidate execution is authorized, use only the standard candidate-run command. Preserve its receipt;
   do not paraphrase success.
7. Compute hashes for every staged artifact and atomically write result.json.

ACCURACY WARNING
Codex will independently check EVERY field you author, including description, taxonomy, citation, year,
country, license, source mapping, input claims, and implementation fidelity, against the exact sources and
literal evidence you cite. An invented fact, ungrounded clause, altered excerpt, wrong year, inferred
license, or unsupported country is a recorded inaccuracy. null, NOASSERTION, or cannot-determine beats
wrong, always. Material drift cannot be relabeled minor.

SELF-CHECK
- I used the highest available rung and documented every higher-rung failure.
- At least one exact public primary source link supports every proposed terminal outcome.
- Every authored factual leaf points to literal evidence or an explicit bounded not-found result.
- Citation, year, country, authors/labs/institutions, license, taxonomy, description, contribution, build,
  framework, and input claims are complete.
- R3/R4 includes exact source, literal text/code evidence, code path, source map, and declared choices.
- No pretrained asset, opaque eval/exec string, environment mutation, or invented architecture detail exists.
- All files are under allowed_output_root and result.json validates against required_result_schema.

OUTPUT
Write exactly one UTF-8 JSON object to <allowed_output_root>/result.json using a temporary file, flush,
fsync, and atomic rename. Use required_result_schema exactly. Important information must be in typed fields,
never only prose. Allowed outcomes are PROPOSED, DEFER_RECOMMENDATION, SKIP_RECOMMENDATION, or BLOCKED.
Do not output runs, accurate, or a fidelity verdict. End after result.json is written.
```

### 18.2 Codex accuracy and fidelity checker prompt

```text
SYSTEM ROLE
You are the independent Codex accuracy gate. Claude authored the proposals. For a `metadata_batch` envelope,
check 10--20 independent items' descriptions and lightweight metadata (citation, year, country, and license)
and return a separate accurate, inaccurate, or cannot-verify verdict for every item. For a `fidelity` envelope,
check exactly ONE model's complete proposal and required implementation fidelity. Catch
unsupported, misattributed, invented, stale, or architecturally unfaithful claims before they enter
canonical records. The inherited dataset contained generic Conv/Linear/Sequential stand-ins labeled as
faithful exotic architectures. Executability, plausible prose, names, and comments are never proof.

AUTHORITY
1. Read WORK_ENVELOPE_PATH and verify every item proposal, source-manifest, evidence, code,
   family-template, author-prompt, and checker-prompt hash before judging content.
2. Work in a fresh context. Trust exact source bytes and exact public sources, not Claude summaries,
   comments, class names, legacy notes, or the claimed rung.
3. Primary verification is against each item's frozen proposal/source/evidence/code pack and its stored
   grounding excerpt. You may optionally use web research to corroborate citations or facts, but it cannot
   replace missing captured evidence. A moving branch, search-results page, missing locator, altered excerpt,
   or hash mismatch cannot support a claim.
4. Do not edit code, evidence, records, queues, environments, or templates. Never use TorchLens. Identify
   exact repairs; do not perform them.
5. For `metadata_batch`, inspect only each item's scoped description/lightweight-metadata fields and never
   let one item's evidence or verdict stand in for another. For `fidelity`, one focused read covers every
   agent-authored field and the required implementation fidelity.

TOP-LEVEL FIELD VERDICTS
accurate: every non-null agent-authored claim is supported by cited exact evidence; bounded null and
not-found results are honest; website prose is grounded; and any required fidelity verdict is acceptable.
inaccurate: at least one claim contradicts a source, cites the wrong model/work, fabricates or materially
alters evidence, overstates support, or the implementation materially drifts from source.
cannot-verify: evidence, hash, locator, source specificity, or source detail is insufficient to determine
one or more material claims.

Use inaccurate for contradiction. Use cannot-verify for missing or ambiguous proof. "Looks right," common
knowledge, typical implementation, and likely equivalence are not accurate.

FIDELITY VERDICTS
match: candidate code implements every sourced material architectural item with no architectural deviation.
minor-drift: only an explicitly declared nonmaterial source-unspecified probe or initialization choice
differs; tensor math, topology, dimensions, connectivity, state/control, and output are unchanged.
major-drift: any material sourced item is contradicted, omitted, replaced, or invented.
slop: the candidate is a generic stand-in, superficial sketch, knowingly simplified substitute, or bears
no defensible material correspondence to the named architecture. This is a permanent historical scar.
cannot-verify: frozen evidence cannot decide one or more material implementation choices.

Never call a material difference minor. A source-unspecified choice may be minor only when it is declared
and demonstrably nonmaterial to the architecture.

CHECK IN THIS ORDER
1. INTEGRITY: verify hashes, exact URLs/revisions, source identity, locators, and that every literal excerpt
   exists verbatim at its locator. List altered or fabricated excerpts.
2. IDENTITY AND RUNG: verify this source is this architecture and variant and that the selected rung is the
   highest applicable. For R4, inspect the stored search protocol and spot-check the highest-risk searches;
   finding usable code invalidates R4.
3. CITATION AND DATE: verify title, full authors, year, venue, identifiers, URL, and BibTeX all name the same
   introducing work. Check the first-public-year basis, not a later library release.
4. PEOPLE AND COUNTRY: verify authors, labs, institutions, and institutional origin countries against
   affiliations/project evidence. Country is not nationality. Check confidence and note honesty.
5. TAXONOMY AND LICENSE: verify family, domains, tasks, modalities, era, tags, novel operators, and every
   license claim at the pinned revision. Missing license is never permission to infer one.
6. WEBSITE TEXT: verify tagline, paragraph, and contribution clause-by-clause. They must be grammatical,
   neutral, mutually consistent, source-grounded, and free of benchmark/marketing overclaim. Each factual
   clause needs evidence. For size variants, verify the prose is the accepted family template and only the
   measured parameter/input line differs.
7. BUILD, FRAMEWORK, AND INPUT: verify exact symbol/kwargs, pretrained disabling, original_framework,
   run_framework, transparent native delegation, mode/device, patches, dummy-input semantics, dimensions,
   dtypes, values, masks/state/control, and output contract.
8. EVIDENCE COVERAGE: enumerate every agent-authored leaf. Each positive factual claim needs adequate
   literal evidence. A null/not-found value passes only when it makes no positive claim and the bounded
   search is documented.
9. FIDELITY, WHEN REQUIRED:
   a. Before reading candidate code, derive topology, dimensions, operators, connectivity, state/control,
      material initialization, input, and output from the exact frozen source.
   b. Read candidate code and map every material source item to exact code locators.
   c. Check branches, skips, recurrence, attention/aggregation, axes, heads/groups, padding/stride/dilation,
      activations, normalization, masks, state updates, stochastic behavior, and return structure.
   d. Check every declared deviation and source-unspecified choice. A performance-kernel replacement matches
      only when exact evidence establishes mathematical equivalence.
   e. Apply exactly one of match, minor-drift, major-drift, slop, or cannot-verify. Name a generic stand-in
      explicitly as slop.

FIELD RESULTS
For every scoped authored field in every batch item, output accurate, inaccurate, or cannot-verify with
evidence_ids, checked source_ids, and a concise reason. Group only truly identical fields within the same
item; never hide a failed leaf or emit a batch-wide substitute. Output the five-way fidelity verdict and
material checks only for the per-model `fidelity` envelope whenever fidelity.required is true.

DECISION RULE
Top-level accurate requires every field result accurate and required fidelity match or strictly
nonmaterial minor-drift. Any inaccurate field, major-drift, or slop makes top-level inaccurate. Otherwise
any cannot-verify makes top-level cannot-verify. Confidence never overrides these rules. Metadata accuracy
gates canonical authored-field write and crawl completion; it does not award runs. Only the deterministic
driver awards runs.

OUTPUT
Write exactly one UTF-8 JSON object to <allowed_output_root>/result.json using a temporary file, flush,
fsync, and atomic rename. It must validate against menagerie.crawler.gate.v2. A `metadata_batch` result must
contain 10--20 independently identified per-item results, each with exact work/model/gate identities,
verified hashes, integrity result, verdict, exhaustive scoped field_checks, excerpt discrepancies,
unsupported claims, required repairs, and confidence. A `fidelity` result must contain the same per-model
identity and integrity fields plus rung_check and fidelity block. Do not place findings only in prose. End
after result.json is written.
```

## 19. Extensibility

New discoveries enter the same intake schema and receive a durable ID. `run --new-only` compares stable
IDs and revision hashes; unchanged current terminal models cost nothing. New rows use the same one-pass
authorship, ladder, gate, typed adapter, planner, worker, partition, mirrors, and release policy.

Framework-specific behavior belongs in a versioned framework adapter. Model-specific construction/input
belongs in a declarative recipe or typed model adapter. Dependency changes belong in intent/locks. There
must be no central model-name input branch, zoo-regex executable recipe, per-model runner exception, or
opaque code string.

Schema evolution is additive under a new explicit version. A proposed field is assessed with the do-it-
once test: if later production requires source reading or judgment, capture it now; if it is derivable from
accepted evidence/code/receipts, add it later without recrawling. New checker or author prompts invalidate
only identities that include their hash.

## 20. Reports and operational observability

`status --full` reports intake and current terminal totals; every status/stage/reason; untriaged/workflow
counts; source-link/revision/evidence coverage; citation/year/country/license completeness; gate and fidelity
verdicts/repairs; slop scars; rung/framework/domain/era/country distributions; env/lock/probe health;
timing/resource clusters; `forward_observed_but_blocked`; runs with metadata/fidelity pending; usage-limit
pause/reset/wakeup state; and Linux deferral conversion.

No report calls a partial wave complete. The authoritative completion report includes the exact partition
equation, all completeness counts at zero, ledger/view rebuild digests, license-sweep report hash, and
machine/env generations.

## 21. Acceptance tests

The implementation is not ready for the real crawl until tests prove all of the following.

1. Kill during author result temp-write, rename, JSONL append, env creation, constructor, first forward,
   second R3/R4 forward, and checkpoint; resume loses no complete fact and duplicates no work.
2. A byte-identical duplicate envelope is idempotent; a conflicting work/revision identity hard-fails.
3. A malformed complete JSONL line stops recovery; a torn final line is evidenced and safely repaired.
4. Mutating source, excerpt, authored metadata, code, source map, input, lock, resolved export, runner,
   author prompt, or checker prompt stales exactly the dependent identities.
5. A legacy generic Sequential that executes cannot inherit `runs`, rung, or fidelity and trips anti-slop
   validation/re-triage.
6. A plausible but wrong citation, year, country, description, license, family, or input claim is rejected
   before canonical authored-field write.
7. An altered or fabricated literal excerpt is `inaccurate` with an integrity finding.
8. An underspecified R4 source becomes `cannot-verify` and then skip/failure under the bounded policy; no
   plausible stand-in is generated.
9. A source-contradicting port that forwards successfully is `major-drift`; a generic substitute is `slop`;
   neither earns `runs`. A real nonmaterial source-unspecified choice is the only `minor-drift` fixture.
10. Hidden socket, checkpoint, cache-weight, outside-scratch write, credential exposure, or TorchLens import
    can never earn `runs` and emits the correct structured policy reason.
11. Worker death records only supervisor-observed fields and never fabricates a worker receipt/traceback.
12. R3/R4 requires two cold driver receipts spanning all meaningful train/eval modes; R1/R2 requires one
    per meaningful mode; deterministic 2% membership and env-generation re-verification are correct.
13. Metadata Codex outage does not block eligible mechanical R1/R2 execution; completion remains false and
    pending work is visible. Required R3/R4 fidelity outage shows `forward-observed-but-blocked`.
14. Claude usage exhaustion records exact reset time, schedules a one-shot wake, resumes idempotently, and a
    simultaneous live process/wakeup cannot obtain two locks.
15. Codex rate limit backs off with jitter and quota reset schedules a wake without silent loss.
16. Every intake ID has exactly one current terminal status; duplicate, missing, and workflow rows prevent
    completion.
17. `failed:accuracy-gate` is terminal, human-review-required, excluded from runnable views, and a later
    reviewed append supersedes it without deleting history.
18. Family representative prose is authored/vetted once; size variants are byte-equivalent except the
    allowed measured parameter/input line and reference family grounding.
19. Every classic, faithful-claimer, and known/presumed slop fixture lacks completion until a stored current
    five-way Codex verdict exists.
20. The env planner targets a small compatible set rather than a hard cap, creates no one-model env,
    verifies exact locks/exports, tears down sequentially, and stales runs on env-generation change.
20a. A model's meaningful train/eval modes each produce a receipt; a single-mode failure is visibly
    `failed:mode-run{mode,...}`, and statistical versus structural divergence is classified correctly.
21. Detectron2/mmcv receives no more than two arm64 CPU source-build attempts before its ruled platform
    deferral.
22. Phase-2 native models record different original/run framework fields only when a transparent adapter
    actually delegates; the PyTorch phase must finish first.
23. The Linux one-command handoff selects only both deferred statuses, recreates exact target envs, and
    appends superseding results while preserving Mac history.
24. Public/private mirror tests verify hashes, reject restricted bytes from committed/public-store roots,
    and fail the pre-merge license sweep on unknown disposition.
25. Current views and SQLite rebuild byte-identically from intake plus the three JSONL ledgers.
26. A clean mini can git-pull the branch, run strict doctor, fetch declared artifacts by hash, and resume
    using only committed code/prompts/procedures/schemas/records/locks/manifests.
27. Static and dynamic tests prove no crawler module or executed third-party worker imports TorchLens.
28. Checkpoint refuses a wrong branch, disallowed path, failed ledger/schema/partition check, or public
    merge without the license report; it never pushes.

## 22. Implementation and release sequence

1. Build schemas, identities, JSONL I/O, reducer, status, partition, and recovery spine.
2. Build typed recipes, worker isolation, policy tripwires, and the driver mechanical loop; validate on 20
   representative R1/R2 models with no agents.
3. Build deterministic legacy migration and run the 100-model retro-audit calibration.
4. Build env intents, exact target locks, probes, sequential lifecycle, and arm64 routing.
5. Build author/check dispatch, atomic envelopes, frozen prompts, evidence/source/mirror manifests, and
   the block-at-write loop.
6. Enable family representative metadata, deterministic size variants, and all ruled retro-audit pools.
7. Complete the Apple PyTorch sweep, then native-framework tail, checkpointing per env/daily.
8. Run the one-command Linux/NVIDIA deferred sweep.
9. Rebuild views, pass all acceptance and project gates, run the license sweep, and emit the final complete
   partition report. Do not merge or push automatically.
