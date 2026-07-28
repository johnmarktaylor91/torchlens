# Menagerie get-every-model-to-RUN crawler: canonical implementation plan

Status: authoritative implementation specification

Decision date: 2026-07-14

Implementation branch: `menagerie/crawler-pipeline`

Implementation root: `menagerie/crawler/`

This document consolidates the converged Fable R2 and Sol R2 designs and JMT's final rulings. It
is a build specification, not a new design exercise. Where an R2 document disagrees with a ruling,
the ruling and the reconciliation decisions encoded here are final.

## 1. Scope and authoritative sources

This committed contract retains the crawler clauses that define live invariants, schemas, status,
storage boundaries, environments, execution, procedures, prompt identities, and acceptance. Runtime
behavior is authoritative in `menagerie/crawler/`, executable schemas in `schemas/`, and tests in
`tests/`. Historical rollout narrative and copied generated representations are intentionally omitted.

`runs` does not claim checkpoint accuracy, training viability, backward support, TorchLens
traceability, graph validity, or numerical equivalence to a pretrained model. This crawler must not
import TorchLens, trace, render, or validate a trace. Static and runtime tripwires enforce that
boundary.

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
6. The expensive source-reading pass is done once. The crawl exhaustively captures now every metadata
   field that requires web search, external sources, or human judgment, together with literal grounding:
   that external set is the only re-crawl-forcing set. For every field, ask: “does this require reading
   papers/web?” If yes, capture it now, generously; if TorchLens can derive it from the loaded model,
   capture it cheaply only when already available or defer it to TorchLens, and never re-crawl for it.
7. Every external-metadata and other agent-authored field is blocked at canonical write until an
   independent Codex gate checks it. This covers descriptions, citations, year, country, license,
   taxonomy, source mapping, input semantics, and port/reimplementation fidelity.
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
17. This is deliberately not a distributed system. There are no heartbeats, spool queues, ULID event
    segments, or multi-worker arbitration. One single-writer driver and one execution worker at a time
    are the supported mini topology. A child-held kernel `worker.lock` is the no-double-worker authority;
    its durable `WorkerLease` metadata exists only for bounded recovery and visibility.

## 5. Versioned schemas

All canonical JSON Schemas set `additionalProperties: false`, define every enum, and require RFC 3339 UTC
timestamps and `sha256:<64 lowercase hex>` hashes. The executable JSON files under `schemas/` are the
sole schema definitions; section 5.1 provides the generated schema reference procedure.

### 5.0 Round-14 interface freeze and schema ownership

The current contract discriminators are `attempt.v3`, `model.v3`, `author-proposal.v3`,
`author-result.v4`, `gate.v3`, `artifact-event.v1`, and `operational-event.v1`. Historical v2 rows remain
readable immutable history, but they are not current authority. The executable JSON files under
`schemas/` are the sole schema definitions, with the generated reference procedure in section 5.1. The
executable v3 files add the closed raw receipt/parent attestation, mandatory dependency and artifact/
family authority, discriminated author recommendation, and terminal-disposition gate shapes.

Every normalized collection-item leaf in author, attempt, model, and gate v3 has exactly one schema-owned
classification: `author-gated`, `worker-observed`, `parent-observed`, `reducer-derived`,
`trusted-intake`, or `untrusted-history`. The schema-derived registry is the only source for authored-leaf
policy, vet identity, canonical assembly, variant inheritance, and exact checker field coverage. CI
expands schema references and fails on every missing, extraneous, unknown, or multiply owned leaf.

### 5.0.1 Round-15 shared contract freeze

The execution capability v2 discriminator is
`menagerie.crawler.execution-read-manifest.v2`. `ExecutionReadManifestV2` contains
`{manifest_version, manifest_id, stable_id, work_id, execution_identity,
code_manifest_identity, environment_generation, installed_package_inventory_sha256,
code_members[], runtime_members[], standard_input_asset|null, lookup_directories[]}`. Every code/runtime
member is a `RuntimeMember {path, sha256, kind, provenance}` naming one exact regular unaliased file.
Every `RuntimeLookupDirectory {path, provenance}` is traversal or mount scaffolding only and grants no
descendant read or execute authority. The manifest identity is recomputed from every listed field,
including all member digests and both environment identities. No repository root, environment root, or
other semantic root grant is a v2 member. The driver collects this closed digest-bound inventory before
deriving execution identity, binds the closure identity into execution currency, and then binds that same
closure into the final manifest. The supervisor, worker policy, and OS renderers consume only exact v2
members; lookup directories authorize traversal, never descendant bytes.

Current `author-result.v4` deferrals contain a separately authenticated `handoff_execution` with one
complete `author-proposal.v3`, its proposal and code-manifest identities, the terminal source-manifest
identity, and a digest over all handoff fields. Artifact reconstruction v2 retains that handoff and its
gate bindings so a clean Linux deferred-handoff run rehydrates and executes before consulting the author
lane. A historical code-less deferral is visibly `handoff-authority-unavailable`; it is never converted to
`failed:source`. Attempt v3 also persists authenticated `worker_receipt.output_value_sha256` and the
parent-owned nullable `capability_observation`, making none/statistical mode comparison and positive
platform probes representable.

Shutdown control flow uses `DriverShutdown(BaseException)`, never an ordinary exception. One interrupted
invocation produces only `worker-shutdown-interrupted` / `interrupted:shutdown` operational evidence and
no attempt or model row. Its closed details are `{invocation_id, admission_boundary, stable_id|null,
work_id|null, execution_identity|null, request_identity|null, lease_id|null, child_pid|null,
child_start_token|null, child_pgid|null, signal|null, parent_observation|null, partial_receipt|null}`.
`parent_observation`, when present, is the complete parent supervisor observation; `partial_receipt` is
non-awarding diagnostics. `DriverResult.shutdown_interruption` carries the same typed fact and the durable
driver state name is exactly `interrupted:shutdown`. Runtime propagation and admission guards land later.

Every driver invocation has one closed origin: `ordinary-run`, `manual-resume`, or `wake-callback`.
Later wake integration must use this enum, and pause or `DriverShutdown` exits must never resolve an active
wake episode.

Artifact validation will return one read-only `ArtifactCheckpointProjection` whose transaction index is
keyed by exact `(stable_id, work_id, transaction_id)`. Each `ArtifactTransactionProjection` retains the
verified final/authorization identities, immutable reconstruction path/digest/inputs, normalized
`MirrorObject` rows, and independent `ArtifactClaim` rows. The projection-level object tuple has one row
per intrinsic physical identity; the claim tuple preserves every claim even when multiple models share
one object digest. Reconstruction and canonical checkpoint must consume this same projection; neither may
derive a second path-keyed or latest-event view.

Current v3 `input_contract` has no `code_path` property. Presence with either null or a string rejects at
proposal, embedded author-result, model, supervisor, and worker boundaries. Executable authorship remains
only in the distinct `implementation.code_path` and `implementation.source_to_code_map[].code_path`
fields. V2 history retains its untrusted historical leaf. Phase 0 intentionally does not perform the
bundled identity/staleness bump; the later final migration recomputes identities from the frozen inputs
above together with every other changed dependency closure.

### 5.1 Generated schema reference

The executable JSON files under `schemas/` are the sole schema definitions. Do not copy field tables
into this contract. Generate the current version/root/file table and resolved leaf inventory directly:

```bash
python - <<'PY'
from menagerie.crawler.schema import SCHEMA_FILES, load_schema, schema_leaf_paths

for version, filename in SCHEMA_FILES.items():
    schema = load_schema(version)
    print(version, schema["$id"], filename)
    for leaf in sorted(schema_leaf_paths(schema)):
        print(" ", leaf)
PY
```

## 6. Status taxonomy, error taxonomy, and partition invariant

### 6.1 Public terminal statuses

The only public current terminal status codes are:

```text
runs
deferred:needs-cuda
deferred:needs-x86
skipped:insufficient-description
skipped:no-description
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
| `source` | `identity-unresolved`, `missing-mandatory-link`, `source-model-mismatch`, `source-target-invalid`, `higher-rung-unresolved`, `effort-cap-exhausted` |
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

Skips are epistemic only. Operational failure never becomes a skip. All three skip records contain the
primary source URL and search report. `skipped:insufficient-description` additionally retains the vague
source text verbatim with the insufficient excerpt disposition and a non-empty sufficiency gap;
`skipped:no-description` retains the negative search report only; `skipped:not-a-real-NN` retains evidence
that the item is outside the trainable-neural-network scope.

### 6.3 Partition and completion

For intake stable-ID set `I`, let the current status sets be `R`, `D_cuda`, `D_x86`, `S_insufficient`,
`S_none`, `S_nn`, and `F_stage`. The reducer must assert:

```text
I = R ∪ D_cuda ∪ D_x86 ∪ S_insufficient ∪ S_none ∪ S_nn ∪ (∪ F_stage)
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
the execution subprocess, not the author's research phase. Because the author never fetches source into the
campaign, it cannot compute a content digest for a target it has not read: `expected_sha256` is therefore
optional on a source target. When the author supplies one -- from a release manifest, lockfile, or package
index -- it is enforced byte-exactly and a mismatch fails the model. When it is genuinely absent the
controlled fetch is what learns the digest, and the frozen manifest pins exactly the bytes retrieved. Either
way `content_sha256` in the manifest is the digest of the verified CAS bytes, and every downstream consumer
re-verifies against it. A target the fetch contract cannot accept at all is `failed:source` /
`source-target-invalid`; an unretrievable one is `failed:fetch` / `unreachable`; bytes that contradict a
supplied digest are `failed:fetch` / `hash-mismatch`. Web-search capability never enters
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

### 12.3 Standard inputs

The worker prefers a canonical, license-clean standard input keyed by `external_metadata.modality` when it
can be shaped to the model's input specification: for example, a resized standard image for vision or a
tokenized fixed prompt for language. Coverage is deliberately practical rather than exhaustive; unknown
modalities or any standard-input shape/dtype mismatch use a random tensor fallback. Every per-mode receipt
records `input_kind`, `input_asset` (the standard asset identifier/hash or null), and a brief `input_note`.
The same chosen input is used across a model's meaningful modes and all cold runs, so this rule composes with
the two-cold-forwards policy without changing its consistency requirement.

### 12.4 Cold-forward policy

- Agent-authored R3/R4 rows require two driver-owned cold forwards in separate fresh processes with
  separate empty caches, spanning all meaningful train/eval modes. The author dev-run is diagnostic and
  never counts. Receipts for the same mode must match on output tree, shapes, and dtypes; exact random
  values need not match.
- Mechanical R1/R2 rows require one driver-owned cold forward for every meaningful mode.
- Every mechanical row is re-verified whenever its `env_generation` changes.
- A deterministic 2% sample per environment/batch receives a second cold confirmation. Membership is
  `sha256(stable_id + "mechanical-canary-v1") mod 100 < 2` and therefore reproducible.

### 12.5 Driver-only award rule

Locked derivation rule: no run-award or terminal-deciding field is admissible unless the reducer derives
it from retained parent-attested raw evidence or resolves it to an append-only fact and re-checks the
specific status-proving predicate. Existence, nonemptiness, or subset membership never suffices. The
frozen `TerminalProof` and `DependencyVector` are the only driver/reducer boundary values for these
decisions; locally shaped mappings are forbidden.

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

### 13.1 Scheduler phases (`LP-13.1`)

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

Before worker spawn the driver also acquires `.crawl-local/locks/worker.lock` in the fixed order
`driver.lock -> canonical ledgers -> worker.lock`, appends `worker-lease-opened`, and fsyncs a
`WorkerLease`. The trusted bootstrap fills child PID/start-token/PGID and transfers the open lock
description to the child before model import. The child-held kernel lock, not PID metadata, excludes
replacement execution. Startup reconciles held/free leases against boot ID, PID start token, process
group, raw receipt, and the bounded deadline; it never guesses a PID or promotes an unattested receipt.

### 13.2 Default effort caps (`LP-13.2`)

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

### 13.3 Human review checkpoint and progress notifications (`LP-13.3`)

The single-writer driver owns two terminal-count notification policies. `review_checkpoint_at` defaults
to `1000`; `0` or `null` disables it. When the terminal partition count first reaches that value, the
driver writes a runtime check-in report containing the funnel, fidelity-verdict distribution, accepted
sample, and concerning patterns; appends one `checkpoint-review` operational event; notifies JMT; sets
its disposable state to `paused:review-checkpoint`; tears down the active environment; and stops. This is
a blocking, one-shot checkpoint. `crawler resume --after-review`, or an already recorded
`review-signoff` event, appends/consumes the sign-off and allows the campaign to continue without
blocking again at the same checkpoint.

`progress_milestones` defaults to `[2000, 3000, 5000, 10000, 15000, 20000]` and may be empty. Crossing a
configured value appends exactly one `progress-notification` event with the completed count and funnel,
notifies JMT, and continues immediately. Persisted operational events make review and milestone attempts
idempotent across resume.

Both policies use `notify_command`. Its default resolves `send-to-jmt.sh` from `PATH`,
`~/scripts`, or `~/bin`, and otherwise uses log-only delivery. Notification text is a single plain-ASCII
summary line. A missing or failing notifier is recorded in the driver log and never crashes, pauses, or
blocks campaign progress.

External notification delivery is explicitly **at-least-once**. Each notification carries a durable
idempotency key that recipients may use for deduplication; no process claims exactly-once delivery across
a crash between external delivery and the canonical delivery event.

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

## 18. Frozen agent prompt identities

The prompt files are authoritative runtime inputs. The dispatch-brief fragments under `prompts/pool/`
are equally authoritative: `author_pool.render_dispatch_brief` renders them into every author session,
so their bytes steer author behavior exactly as the two top-level prompts do and they are pinned the
same way. These independently pinned literal digests are the committed drift oracle. The whole surface
is checked by `python -m menagerie.crawler.tools.verify_pool_prompts`, which also proves this pinned
inventory and the shipped inventory are the same set, so an unpinned new fragment and a deleted pinned
fragment both fail; `python -m menagerie.crawler.tools.verify_prompts` remains the two-prompt oracle it
reuses. Editing any of these files stales its digest by construction: re-pin the literal row here in the
same change.

- `claude_crawler_author_v2.txt`: `sha256:bc609db91f34a2fb41ae3b14f925c4660db4405bf4b8d97d2b25c71e0f18bd5d`
- `codex_accuracy_checker_v2.txt`: `sha256:93d82284c3f9f250b55d6eb700f3f63d6e1abf586f259a6350920c5912b9f2d8`
- `campaign_c1-mech.md`: `sha256:c4f38a682416ed82b995a7d06941d04d6559d4ec10f8a5bb0ee6d5436b00b1e8`
- `campaign_c2-disco.md`: `sha256:3ffbfaf9ab56c713b301f28071ff019e1173aa44a7d3e0ed890a980f59daafd1`
- `campaign_c3-classics.md`: `sha256:e6a277891824b72e378c52b1e472357d968d30a91a87b1b8d562ab923b5a11a4`
- `campaign_c4-native.md`: `sha256:b9ff9531f0f330dfabe09275c61aaa72d0a699a29b2c5cd463cae6f439c36683`
- `stage_author.md`: `sha256:07e47db576a72fce43d3f09d2b94dad5477b096ef98d5ca8d1b5da2d02d255f4`
- `stage_capability_probe.md`: `sha256:6eebdcd49603b44335ccc873aa61945e4415664be5275af2ddee212fcc224b0b`
- `stage_source_request.md`: `sha256:bebdd38c7779efb01e1d0d897ab7940a1cd6235bdd1dfb17ae7966c479394c81`

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
14. Claude usage exhaustion records exact reset time, schedules a recurring wake episode until a durable
    resolution fact, resumes idempotently, and a simultaneous live process/wakeup cannot obtain two locks.
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
