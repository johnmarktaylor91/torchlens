# Menagerie crawler implementation manifest

This is the ordered build inventory for `PLAN.md`. Each row names one concrete file, its single
responsibility, and the files/slices it may depend on. Rows sharing a slice may be dispatched in parallel
once their listed dependencies are complete. Generated crawl records, source evidence, adapters, ports,
resolved exports, and mirror objects are campaign outputs and are not counted as implementation files.

Slice order is `A -> B`, then `C`, `D`, and `E` may proceed in parallel; `F` integrates them; `G` supplies
operations; `H` supplies acceptance tests and gates. Environment intent/lock rows in slice C may be split by
environment after the registry contract is fixed.

| # | Slice | Path | Purpose | Depends on |
| ---: | :---: | --- | --- | --- |
| 1 | A | `menagerie/crawler/__init__.py` | Define the crawler package version and intentionally small public surface. | none |
| 2 | A | `menagerie/crawler/__main__.py` | Route `python -m menagerie.crawler` to the typed CLI. | 1, 41 |
| 3 | A | `menagerie/crawler/constants.py` | Hold schema versions, enums, caps, prompt names, targets, and closed status/reason vocabularies. | 1 |
| 4 | A | `menagerie/crawler/models.py` | Define typed internal dataclasses/enums shared by driver, reducer, worker, and gates. | 3 |
| 5 | A | `menagerie/crawler/schemas/model-v2.schema.json` | Executable full `model.v2` contract, including the exhaustive Codex-gated `external_metadata` group (with TorchLens-derivable structural fields optional), runtime modes, and unknown fields forbidden. | PLAN section 5.1 |
| 6 | A | `menagerie/crawler/schemas/attempt-v2.schema.json` | Executable full per-mode `attempt.v2` contract and closed errors. | PLAN section 5.2 |
| 7 | A | `menagerie/crawler/schemas/gate-v2.schema.json` | Executable `gate.v2` metadata-batch (10--20 per-item verdicts) and per-model fidelity contract. | PLAN section 5.3 |
| 8 | A | `menagerie/crawler/schemas/author-proposal-v2.schema.json` | Validate staged Claude proposals before any gate or canonical write. | 5 |
| 9 | A | `menagerie/crawler/schemas/operational-event-v1.schema.json` | Validate usage pauses, wakeups, checkpoints, and campaign-health events. | 3 |
| 10 | A | `menagerie/crawler/schema.py` | Load, cache, and strictly validate all JSON Schema payloads. | 4-9 |
| 11 | A | `menagerie/crawler/identity.py` | Compute stable/source/evidence/recipe/env/fidelity/vet/execution hashes and staleness. | 3, 4 |
| 12 | A | `menagerie/crawler/recordio.py` | Implement single-writer fsynced JSONL append, tail recovery evidence, scan, and idempotency. | 4, 10, 11 |
| 13 | A | `menagerie/crawler/state.py` | Rebuild disposable SQLite queue/current state from intake and JSONL ledgers. | 10-12 |
| 14 | A | `menagerie/crawler/reducer.py` | Sole canonical writer; validate parentage, gates, attempts, per-mode receipts/statuses, and materialized current revisions. | 5-13, 21a |
| 15 | A | `menagerie/crawler/status.py` | Enforce terminal partition/completeness, including meaningful-mode outcomes, and expose queryable funnel reports. | 13, 14, 21a |
| 16 | B | `menagerie/crawler/intake.py` | Snapshot all trusted discovery inputs and assign/preserve durable IDs idempotently. | 10-14 |
| 17 | B | `menagerie/crawler/migrate_legacy.py` | Import every old row/classic/deferred/discovery-only item as UNTRIAGED hints with risk flags. | 16; existing catalog/classics readers |
| 18 | B | `menagerie/crawler/recipe.py` | Implement closed declarative R1 recipes and typed adapter loading with no eval/exec. | 4, 10, 11 |
| 19 | B | `menagerie/crawler/frameworks.py` | Define transparent PyTorch/TF/JAX/Paddle forward-adapter protocol and native call metadata. | 4, 18 |
| 20 | B | `menagerie/crawler/policy.py` | Build credential-scrubbed offline envs and network/checkpoint/write/TorchLens tripwires. | 3, 4 |
| 20a | B | `menagerie/crawler/assets/standard/` | Canonical license-clean standard inputs (image, text, [audio/video/...]) for representative forwards. | none |
| 20b | B | `menagerie/crawler/standard_inputs.py` | Given a model's modality (from `external_metadata.modality`) + its input spec (shape/dtype), materialize the appropriate canonical standard input (load+resize the standard image / tokenize the standard text / etc.) shaped to the model's expected input; fall back to a random tensor when modality is unknown or the standard input cannot be shaped; report which was used. | modality + input-spec; 20a |
| 21 | B | `menagerie/crawler/worker.py` | Build one model/input via `standard_inputs` (canonical-by-modality where feasible, random fallback), explicitly invoke `forward` in BOTH `train()` and `eval()` meaningful modes, and atomically emit per-mode honest receipts recording the input kind. | 6, 18-20, 20b, 21a |
| 21a | B | `menagerie/crawler/modes.py` | Detect meaningful modes and classify train/eval divergence (`none`/`statistical`/`structural`) from captured per-mode outputs. | 4 |
| 22 | B | `menagerie/crawler/worker_supervisor.py` | Launch argv-only fresh subprocesses, enforce timeout/RSS, and record parent-only observations. | 6, 12, 20, 21 |
| 23 | D | `menagerie/crawler/fetcher.py` | Fetch controlled exact URLs/revisions into local CAS and emit hash-bound source manifests. | 4, 11, 20 |
| 24 | D | `menagerie/crawler/evidence.py` | Validate literal excerpts, locators, supports coverage, and family-level grounding. | 5, 8, 10, 11, 23 |
| 25 | D | `menagerie/crawler/proposal.py` | Validate staged author output, path isolation, typed code, source ladder, and anti-slop gates. | 8, 10, 18, 24 |
| 26 | D | `menagerie/crawler/author_dispatch.py` | Create one-model rich-session envelopes and validate atomic Claude result files. | 8, 11, 12, 25, 43 |
| 27 | D | `menagerie/crawler/checker_dispatch.py` | Create fresh batch-N metadata envelopes and per-model fidelity envelopes, validate per-item atomic gate results, and handle rate/quota responses. | 7, 11, 12, 24, 44 |
| 28 | D | `menagerie/crawler/gates.py` | Apply bounded block-at-write per batch item, requeue failed metadata items into the next batch, and route five-way per-model fidelity without editing proposals. | 7, 11, 14, 25-27 |
| 29 | D | `menagerie/crawler/family_templates.py` | Instantiate vetted family text for size variants with only the measured param/input line changed. | 5, 11, 24, 28 |
| 30 | D | `menagerie/crawler/metadata.py` | Enforce the external-metadata checklist (exhaustive web/source-derived fields: modality, architecture_class, field/domain/task, paradigm, lineage, tags, venue, year, country, authors, institution, citation, license, key_contribution, description) under the demarcation rule; TorchLens-derivable structural fields are optional. | 5, 24, 28, 29 |
| 31 | E | `menagerie/crawler/retro_audit.py` | Define calibration and ruled slop/claimer/classics/library waves and their completion gates. | 13-17, 28, 30 |
| 32 | E | `menagerie/crawler/effort.py` | Track per-stage caps, root-cause fingerprints, grants, and actual-stage cap failures. | 3, 4, 12 |
| 33 | C | `menagerie/crawler/envs.py` | Load the intent registry (small-set target, no hard cap), exact locks, exports, probes, and compute env generations. | 3, 4, 10, 11, 55-92 |
| 34 | C | `menagerie/crawler/env_lifecycle.py` | Create/probe/use/teardown one exact locked env sequentially and verify disk recovery. | 12, 22, 32, 33 |
| 35 | C | `menagerie/crawler/routing.py` | Assign models to intents, phase native tail, collect platform evidence, and enforce arm64 attempt caps. | 4, 13, 32-34 |
| 36 | E | `menagerie/crawler/mirrors.py` | Address public/private/local artifacts by hash and verify fetch/retention manifests. | 11, 12, 23 |
| 37 | E | `menagerie/crawler/licenses.py` | Classify redistribution, keep restricted bytes private, and implement the pre-public-merge sweep. | 5, 24, 36 |
| 38 | E | `menagerie/crawler/wakeup.py` | Record usage pauses and manage idempotent reset-time launchd/systemd/cron one-shot wakeups. | 9, 12, 13, 32 |
| 39 | E | `menagerie/crawler/checkpoint.py` | Validate branch/ledgers/views/mirrors/licenses, create allowlisted checkpoint sets, and never push. | 12-15, 36-38 |
| 40 | F | `menagerie/crawler/driver.py` | Run the single-writer scheduler, lock guard, env lifecycle, author/check lanes, forwards, and reduction. | 14-15, 22, 26-35, 38-39 |
| 41 | F | `menagerie/crawler/cli.py` | Expose doctor/intake/plan/run/status/checkpoint/teardown/requeue/handoff commands with typed args. | 15-17, 31-40 |
| 42 | F | `menagerie/crawler/doctor.py` | Verify target, branch, disk, locks, mirrors, Claude Code author WebSearch + Exa availability, secrets, policy tripwires, wakeups, and TorchLens ban. | 20, 33, 36-39 |
| 43 | G | `menagerie/crawler/prompts/claude_crawler_author_v2.txt` | Byte-for-byte frozen canonical author prompt from PLAN section 18.1. | PLAN section 18.1 |
| 44 | G | `menagerie/crawler/prompts/codex_accuracy_checker_v2.txt` | Byte-for-byte frozen canonical checker prompt from PLAN section 18.2. | PLAN section 18.2 |
| 45 | G | `menagerie/crawler/tools/requeue.py` | Append explicit budget grants/new work generations without mutating history. | 12-15, 32 |
| 46 | G | `menagerie/crawler/tools/license_sweep.py` | CLI wrapper for public/private byte-boundary and license-report enforcement. | 37, 39 |
| 47 | G | `menagerie/crawler/tools/rebuild_views.py` | Rebuild and digest current/release/deferred views from canonical JSONL only. | 12-15 |
| 48 | G | `menagerie/crawler/tools/verify_prompts.py` | Assert prompt files exactly equal PLAN's frozen blocks and report hashes. | 11, 43, 44 |
| 49 | G | `menagerie/crawler/procedures/QUICKSTART.md` | Give the pull-to-mini, strict-doctor including Claude Code WebSearch + Exa availability, intake, and press-go commands. | 40-44 |
| 50 | G | `menagerie/crawler/procedures/SETUP.md` | Document prerequisites including Claude Code WebSearch + Exa, mirror config, exact-lock tooling, three network postures, security posture, and branch policy. | 33-44 |
| 51 | G | `menagerie/crawler/procedures/RUN.md` | Document phase-1 PyTorch and phase-2 native sequential execution. | 31-44 |
| 52 | G | `menagerie/crawler/procedures/RESUME.md` | Document crash recovery, JSONL-tail evidence, locks, pending gates, and usage-reset wakeups. | 12-15, 38-44 |
| 53 | G | `menagerie/crawler/procedures/TEARDOWN.md` | Document safe env/cache cleanup, checkpointing, disk verification, and report-only CAS GC. | 34, 36, 39-42 |
| 54 | G | `menagerie/crawler/procedures/LINUX_SWEEP.md` | Document the one-command deferred-bucket Linux/NVIDIA handoff and supersession semantics. | 35-42 |
| 55 | C | `menagerie/crawler/envs/registry.yml` | Declare the intent registry (small-set target, no hard cap): eight PyTorch intents, three native-tail intents, split guidance, and phase order. | 3; PLAN section 11 |
| 56 | C | `menagerie/crawler/envs/core/environment.yml` | Declare direct dependencies for the core PyTorch intent. | 55; empirical existing env specs |
| 57 | C | `menagerie/crawler/envs/graph/environment.yml` | Declare direct dependencies for graph/science packages. | 55; empirical existing env specs |
| 58 | C | `menagerie/crawler/envs/audio/environment.yml` | Declare direct dependencies for audio/speech packages. | 55; empirical existing env specs |
| 59 | C | `menagerie/crawler/envs/mmlab/environment.yml` | Declare direct dependencies for compatible OpenMMLab stacks. | 55; empirical existing env specs |
| 60 | C | `menagerie/crawler/envs/tabular-ts/environment.yml` | Declare direct dependencies for tabular/recsys/time-series stacks. | 55; empirical existing env specs |
| 61 | C | `menagerie/crawler/envs/detect-misc/environment.yml` | Declare detectron2/mmcv and compiled miscellaneous CV build intent. | 55; empirical existing env specs |
| 62 | C | `menagerie/crawler/envs/legacy-torch/environment.yml` | Declare the bounded legacy Python/PyTorch compatibility intent. | 55; empirical existing env specs |
| 63 | C | `menagerie/crawler/envs/oddballs/environment.yml` | Declare the exact-repository research-tail intent. | 55; empirical existing env specs |
| 64 | C | `menagerie/crawler/envs/tf-keras-arm64/environment.yml` | Declare the phase-2 native TensorFlow/Keras intent. | 19, 55 |
| 65 | C | `menagerie/crawler/envs/jax-flax-arm64/environment.yml` | Declare the phase-2 native JAX/Flax intent. | 19, 55 |
| 66 | C | `menagerie/crawler/envs/paddle-arm64/environment.yml` | Declare the phase-2 native Paddle intent. | 19, 55 |
| 67 | C | `menagerie/crawler/envs/core/locks/osx-arm64.lock` | Pin core Apple packages/artifacts with exact URLs and hashes. | 56; target solve/probe |
| 68 | C | `menagerie/crawler/envs/graph/locks/osx-arm64.lock` | Pin graph Apple packages/artifacts exactly. | 57; target solve/probe |
| 69 | C | `menagerie/crawler/envs/audio/locks/osx-arm64.lock` | Pin audio Apple packages/artifacts exactly. | 58; target solve/probe |
| 70 | C | `menagerie/crawler/envs/mmlab/locks/osx-arm64.lock` | Pin mmlab Apple packages/artifacts exactly. | 59; target solve/probe |
| 71 | C | `menagerie/crawler/envs/tabular-ts/locks/osx-arm64.lock` | Pin tabular/time-series Apple packages/artifacts exactly. | 60; target solve/probe |
| 72 | C | `menagerie/crawler/envs/detect-misc/locks/osx-arm64.lock` | Pin/source-build detect-misc Apple artifacts exactly. | 61; two-attempt build probe |
| 73 | C | `menagerie/crawler/envs/legacy-torch/locks/osx-arm64.lock` | Pin legacy Apple packages/artifacts exactly. | 62; target solve/probe |
| 74 | C | `menagerie/crawler/envs/oddballs/locks/osx-arm64.lock` | Pin oddballs Apple packages/artifacts exactly. | 63; target solve/probe |
| 75 | C | `menagerie/crawler/envs/tf-keras-arm64/locks/osx-arm64.lock` | Pin native TensorFlow/Keras Apple artifacts exactly. | 64; target solve/probe |
| 76 | C | `menagerie/crawler/envs/jax-flax-arm64/locks/osx-arm64.lock` | Pin native JAX/Flax Apple artifacts exactly. | 65; target solve/probe |
| 77 | C | `menagerie/crawler/envs/paddle-arm64/locks/osx-arm64.lock` | Pin native Paddle Apple artifacts exactly. | 66; target solve/probe |
| 78 | C | `menagerie/crawler/envs/core/locks/linux-x86_64-cuda.lock` | Pin core Linux/CUDA packages/artifacts exactly. | 56; Linux target solve/probe |
| 79 | C | `menagerie/crawler/envs/graph/locks/linux-x86_64-cuda.lock` | Pin graph Linux/CUDA packages/artifacts exactly. | 57; Linux target solve/probe |
| 80 | C | `menagerie/crawler/envs/audio/locks/linux-x86_64-cuda.lock` | Pin audio Linux/CUDA packages/artifacts exactly. | 58; Linux target solve/probe |
| 81 | C | `menagerie/crawler/envs/mmlab/locks/linux-x86_64-cuda.lock` | Pin mmlab Linux/CUDA packages/artifacts exactly. | 59; Linux target solve/probe |
| 82 | C | `menagerie/crawler/envs/tabular-ts/locks/linux-x86_64-cuda.lock` | Pin tabular/time-series Linux packages/artifacts exactly. | 60; Linux target solve/probe |
| 83 | C | `menagerie/crawler/envs/detect-misc/locks/linux-x86_64-cuda.lock` | Pin deferred detectron2/mmcv Linux/CUDA artifacts exactly. | 61; Linux target solve/probe |
| 84 | C | `menagerie/crawler/envs/legacy-torch/locks/linux-x86_64-cuda.lock` | Pin deferred legacy x86 packages/artifacts exactly. | 62; Linux target solve/probe |
| 85 | C | `menagerie/crawler/envs/oddballs/locks/linux-x86_64-cuda.lock` | Pin deferred research-tail Linux/CUDA artifacts exactly. | 63; Linux target solve/probe |
| 86 | C | `menagerie/crawler/envs/probes.yml` | Define imports, three fixed canaries per intent, resolved-export checks, and source-build probes. | 55-85 |
| 87 | G | `menagerie/crawler/records/README.md` | Specify committed ledger sharding, append/recovery rules, derived views, and no-secret policy. | 12-15, 36-39 |
| 88 | G | `menagerie/crawler/mirrors/README.md` | Specify public release-store/private-mirror manifests, retention, and fetch-by-hash contract. | 36, 37 |
| 89 | G | `.gitignore` | Ignore `.crawl-local/` runtime state without hiding committed crawler records/manifests. | PLAN section 8 |
| 90 | H | `tests/crawler/test_schemas.py` | Prove all three canonical schemas, author/operational schemas, enums, and unknown-field rejection. | 5-10 |
| 91 | H | `tests/crawler/test_identity_staleness.py` | Prove byte-level dependency hashing and exact stale-result propagation. | 11 |
| 92 | H | `tests/crawler/test_recordio_recovery.py` | Prove fsync append, torn-tail evidence/recovery, corruption stop, conflicts, and idempotency. | 12 |
| 93 | H | `tests/crawler/test_reducer_partition.py` | Prove single current terminal partition, supersession, completeness, and view determinism. | 13-15 |
| 94 | H | `tests/crawler/test_legacy_migration.py` | Prove all inherited claims are hints and ruled retro-audit pools are flagged/covered. | 16, 17, 31 |
| 95 | H | `tests/crawler/test_recipe_and_worker.py` | Prove typed recipes, explicit per-mode forward, honest receipts, native adapters, and no eval/exec. | 18-22, 21a |
| 96 | H | `tests/crawler/test_policy_tripwires.py` | Prove network/checkpoint/write/credential/TorchLens violations cannot earn runs. | 20-22 |
| 97 | H | `tests/crawler/test_gate_loop.py` | Prove batch-N per-item metadata block-at-write, per-model fidelity, all-field checks, bounded repair/rebatching, and human failure. | 23-30 |
| 98 | H | `tests/crawler/test_family_templates.py` | Prove representative-only authorship/vet and the exact allowed variant-line difference. | 24, 28-30 |
| 99 | H | `tests/crawler/test_environment_lifecycle.py` | Prove small-set exact-lock intent environments (no hard cap), sequential teardown, export identity, canaries, and re-verification. | 33-35, 55-86 |
| 100 | H | `tests/crawler/test_arm64_routing.py` | Prove the detectron2/mmcv two-attempt cap and only-evidenced CUDA/x86 deferrals. | 32-35, 61, 72, 83 |
| 101 | H | `tests/crawler/test_driver_resume.py` | Kill each critical boundary and prove lock-safe, duplicate-free single-writer resume. | 38-42 |
| 102 | H | `tests/crawler/test_usage_limits.py` | Prove visible Claude/Codex pauses, backoff, exact-reset wakeups, and mechanical nonblocking. | 27, 38, 40 |
| 103 | H | `tests/crawler/test_run_award.py` | Prove driver-only award, meaningful-mode completion, two cold R3/R4 spanning modes, one mechanical per mode, 2% canary, and gate/run split. | 14, 22, 28, 35, 40, 21a |
| 104 | H | `tests/crawler/test_mirror_and_license.py` | Prove public/private artifact separation, hashes, restricted-byte rejection, and merge gate. | 36, 37, 39, 46 |
| 105 | H | `tests/crawler/test_linux_handoff.py` | Prove one-command deferred selection, exact Linux locks, superseding results, and Mac history. | 35, 39-42, 54, 78-85 |
| 106 | H | `tests/crawler/test_static_boundaries.py` | Prove crawler/worker import ban, branch/checkpoint allowlist, and no runtime artifacts in git. | all implementation files |

## Dispatch notes

- Slice A is the spine and should land first.
- Slice B can begin after schemas/identity/record I/O stabilize.
- Slice C can be divided by environment; exact target locks are accepted only after their target probes and
  resolved-export hashes exist.
- Slice D can be divided into source/evidence, author dispatch, checker dispatch, and family templating, but
  the main agent must freeze the two prompt files before identity/gate fixtures are finalized.
- Slice E can be divided into retro-audit, mirror/license, and wake/checkpoint work.
- Slice F is the integration critical path and begins after B-E contracts are stable.
- Slice G documentation is executable: every command is exercised by slice H.
- Slice H tests should be built alongside their owning slice, then run as one acceptance suite.

No row authorizes a git commit or push by an implementation worker. The supervising pipeline applies JMT's
per-environment/daily commit cadence after validation.
