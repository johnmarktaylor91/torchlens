# Split `derive_merge` into named stages (behaviour-preserving)

## Summary

`torchlens/_merged/_engine.py:derive_merge` was one 373-line function with
cyclomatic complexity 67 (radon grade F). It is now a 76-line orchestrator
(complexity 7, grade B) that calls one function per stage, in the same
normative order. Stages that need no join record moved to the new module
`torchlens/_merged/_derive_steps.py`. Join assembly, meaning anything that
builds a `PerRankRef` or `JoinRecord`, stays in `_engine.py`. `_engine.py`
went from 951 to 790 lines, under the 800-line style cap. No size ratchet or
allowlist covers either file: `scripts/checks/file_size_ratchet.py` has no
`_merged` entry, and `git grep -n "_engine.py" tests/` shows only
global-state rows. The signature of `derive_merge`, every public name, error
class, message and code are unchanged.

`_derive_steps.py` never imports `_engine.py`, so the import graph stays
acyclic. It defines no `__all__` and no mutable module globals, so the
global-state inventory needs no new row. `_STRUCTURAL_KINDS` is now defined
there and imported by `_engine.py`, so `_artifact.py`'s existing
`from ._engine import _STRUCTURAL_KINDS` still resolves.

## Complexity (radon 6.0.1 `cc -s`, complexipy 8.3.0 cognitive)

Before:

| Function | radon CC | grade | cognitive |
|---|---|---|---|
| `derive_merge` | 67 | F | 118 |
| `_guard_scope` | 9 | B | 17 |

After. These are all the functions the split produced or moved:

| Function | Module | Stage | Lines | radon CC | grade | cognitive |
|---|---|---|---|---|---|---|
| `derive_merge` | `_engine.py` | entry point / orchestrator | 715-790 | 7 | B | 6 |
| `_join_channel` | `_engine.py` | per-channel joining in seq-delta order | 673-712 | 3 | A | 3 |
| `_join_key` | `_engine.py` | one key: refs, relation table, cross-checks, gap, witness verdict, record | 586-670 | 10 | B | 8 |
| `_per_rank_refs` | `_engine.py` | per-rank references at one key | 539-583 | 3 | A | 2 |
| `_guard_scope` (moved) | `_derive_steps.py` | scope guard (shard-local, p2p, DTensor) | 43-86 | 9 | B | 17 |
| `check_membership_authority` | `_derive_steps.py` | membership-authority belt | 89-150 | 7 | B | 15 |
| `audit_membership_lineages` | `_derive_steps.py` | pre-join lineage audit | 153-194 | 6 | B | 5 |
| `build_group_table` | `_derive_steps.py` | group table, member-order and backend disputes | 197-274 | 9 | B | 14 |
| `collect_channel_records` | `_derive_steps.py` | identity matching: boundaries by correlation channel | 277-314 | 4 | A | 6 |
| `align_seq_deltas` | `_derive_steps.py` | seq-delta alignment | 317-340 | 3 | A | 3 |
| `c10d_delta_mismatch` | `_derive_steps.py` | c10d group-seq cross-check | 343-396 | 9 | B | 9 |
| `absolute_seq_mismatch` | `_derive_steps.py` | absolute-seq cross-check (armed ranks) | 399-457 | 4 | A | 3 |
| `presence_gap` | `_derive_steps.py` | presence gap at a key | 460-488 | 2 | A | 1 |
| `value_divergence` | `_derive_steps.py` | value-divergence finding | 491-521 | 1 | A | 0 |
| `declared_rank_gap` | `_derive_steps.py` | declared expected ranks (widen-only) | 524-554 | 3 | A | 2 |
| `stored_alignment` | `_derive_steps.py` | alignment verdict | 557-578 | 7 | B | 6 |
| `stored_value_status` | `_derive_steps.py` | value-status verdict | 581-605 | 9 | B | 8 |

Every function above is grade B or better, so none needs a "stays high"
comment. The deleted single-boundary helper `_per_rank_ref` (complexity 1) is
replaced by `_per_rank_refs`.

Out of scope and unchanged: `_witness_consistency` (E 37),
`_relation_findings` (E 36) and `_check_order` (C 15). These helpers sat
beside `derive_merge` before the split; this task did not touch them. They
also stayed in `_engine.py` for a second reason. `_check_order` and
`_relation_findings` carry section-number citations in their docstrings, so
moving either would retire a row of the numeral baseline. The only way to
regenerate that baseline is a full rewrite
(`tests/oracles/_regen.py`), which would also absorb 45 unrelated new
numeral rows from elsewhere in the repository.

## What stayed identical, and how

- Order of operations: the scope guard, then the `expected_ranks`
  normalization, the membership-authority belt, the lineage audit, the
  group table, channel collection, joining (channels sorted, deltas sorted,
  ranks sorted), the declared-ranks gap, the order check and the verdicts.
- Findings order, across stages and within a key: relation findings, the
  c10d mismatch, the absolute-seq mismatch, the presence gap, then the value
  divergence. Insertion order of `per_rank`, `joined_nodes` and the audit
  dict is also unchanged.
- Raise order within a key: role-digest belt (per rank, sorted), then the
  presence-subset check, then the group-rank belt inside
  `_witness_consistency`.
- Two equivalent rewrites, each covered by the characterization cases:
  - `_join_key` reads `entries[presence[0]]["kind"]` once; the old code read
    it twice from the same unmutated dict.
  - `stored_value_status` tests `MISMATCHED in applicable`; the old code
    built a list and tested its truthiness.
- The presence-outside-membership raise in `_join_key` was kept verbatim.
  `derive_merge` cannot reach it, because `check_membership_authority`
  refuses the same input first (case `refuse_presence_outside_membership`).

## Characterization coverage

`tests/test_merged_derive_steps.py` was written and committed before the
refactor. Two cases were added afterwards; their expected values were
recorded by running the saved pre-split `_engine.py`. The final file passes
against both the original and the split engine. Each case records one of:

- Success: a SHA-256 over the full `repr` of the returned `MergeDerivation`
  plus its canonical payload. Alongside it, the alignment, value status,
  finding kinds in order, and join count.
- Refusal: the exception class, message, and every payload field (code
  included).

The 46 cases cover:

- **Scope guard:** shard-local, p2p kind, non-`coll` channel, DTensor
  geometry. Ordering: shard-local before p2p on one rank; the lower rank's
  p2p before a higher rank's shard-local; any scope refusal before an
  authority refusal.
- **Membership authority:** rank outside membership, digest mismatch,
  duplicate correlation key. Ordering: authority before the audit.
- **Audit:** an asymmetric-arming conflict (with declared ranks) and a
  complete-witness disagreement.
- **Seed discharge:** mixed armed and seeded ranks.
- **Group table:** member-order mismatch; a three-way backend dispute, which
  produces a single finding and a demoted backend.
- **Witness backends:** unknown backend and `None` backend, both not
  applicable.
- **Relation table:** kind disagreement, reduce-op and shape disagreement,
  tensorless barrier.
- **c10d cross-check:** mismatch, agreement, and a base taken only at
  delta 0.
- **Absolute-seq cross-check:** armed mismatch, armed equality, and a seeded
  rank's base offset, which is never compared.
- **Witness verdicts:** divergent, partial, attested gather slice; the
  string-digest belt; the out-of-range group rank; and the digest belt firing
  before a later channel's group-rank belt.
- **Presence and declared ranks:** missing key, member without a core, no
  boundaries, empty evidence, declared ranks without cores, declared ranks
  that widen and declared ranks that never narrow.
- **Order contradiction:** with a declared gap, to pin the
  declared-gap-before-order sequence.
- **Two composite cases:**
  - `every_finding_stage_at_once` has a finding from every stage in one
    derivation.
  - `every_key_check_on_one_key` puts relation, c10d, absolute-seq, gap and
    divergence findings on the same key.

Evidence that the test detects changes:

- **Whole-object comparison:** a script loaded the saved pre-split
  `_engine.py` next to the new one and compared every case. On success it
  compared the recursive dataclass projection, `repr`, payload and the
  finding properties; on failure, the exception class, message and fields.
  Result: 46 of 46 identical.
- **Mutation checks:** moving the presence-gap check ahead of the
  cross-checks fails one case. Mis-gating the value-divergence finding fails
  seven.

## Census rows moved and why

- `tests/producer_parity/ledger/static_scan.json`, field `kind`:
  `torchlens/_merged/_engine.py` sites 7 -> 5, plus a new row for
  `torchlens/_merged/_derive_steps.py` with 2 sites. The two
  `Finding.kind` attribute reads in the alignment verdict (structural kinds
  and presence gaps) moved with `stored_alignment`; the total stays 7.
  Regenerated with the repository's generator:
  `TORCHLENS_REFRESH_PRODUCER_LEDGER=1 TORCHLENS_GOLDEN_REASON=... pytest
  tests/producer_parity/test_ledger.py::test_generate_and_close_ledger`.
  The generator also appends a record to
  `tests/producer_parity/ledger/PROVENANCE`. That path is outside this
  task's file list, so the append was reverted. The reason is recorded here
  and in the commit message, and no check reads PROVENANCE.
- `tests/composition_expectations/data/refusal_site_classification.tsv`: no
  change. `_engine.py` had 9 coded `MergeInputError` sites before and 9
  coded sites across the two modules after; none has a row in the ledger.
  Moved keys:
  - `torchlens._merged._engine:_guard_scope:MergeInputError::1..3` became
    `torchlens._merged._derive_steps:_guard_scope:MergeInputError::1..3`.
  - `torchlens._merged._engine:derive_merge:MergeInputError::1..3` became
    `torchlens._merged._derive_steps:check_membership_authority:MergeInputError::1..3`.
  - `torchlens._merged._engine:derive_merge:MergeInputError::4` became
    `torchlens._merged._engine:_join_key:MergeInputError::1`.

  The unclassified and uncoded ratchet counts are unchanged.
- `tests/oracles/data/numeral_baseline.tsv`: no change. The census reports no
  stale rows and no new keys for the touched files. New docstrings avoid
  numerals; for example, the "c10d" narrative stays in comments, as it was in
  the original code.

## Test results

Environment: `.venv-ci-smoke/` with python 3.10.20, built by the
repository's `ensure_venv`. Its torch install failed because
`download.pytorch.org` is blocked by the network policy (HTTP 403), so the
same venv was completed with `uv pip install -e ".[dev,tabular,viz]"
torch==2.13.0` from PyPI. That is the CUDA build, running on CPU. torchvision
and transformers are not installed.

- `ruff format torchlens tests`: 2971 files unchanged.
- `ruff check torchlens tests`: all checks passed.
- `mypy torchlens/`: no issues in 1125 source files.
- Targeted pytest, `-n 4`, 18 files: **492 passed, 0 failed, 0 skipped.**

  | File | Tests |
  |---|---|
  | `tests/test_merged_derive_steps.py` | 48 |
  | `tests/test_merged_engine.py` | 103 |
  | `tests/test_merged_engine_hardening.py` | 45 |
  | `tests/test_merged_gloo.py` | 40 |
  | `tests/test_merged_slice_witness.py` | 6 |
  | `tests/test_r58_merged_distributed_rce.py` | 40 |
  | `tests/test_w051_capt3_merged_ledger_shape.py` | 34 |
  | `tests/test_distributed_census_topologies.py` | 46 |
  | `tests/test_distributed_funcol_boundaries.py` | 20 |
  | `tests/test_distributed_shard_local_marker.py` | 12 |
  | `tests/test_error_code_coverage.py` | 6 |
  | `tests/test_error_code_provocations.py` | 3 |
  | `tests/test_generative_properties_r73.py` | 7 |
  | `tests/test_union_merge_integrity.py` | 29 |
  | `tests/composition_expectations/test_diagnostic_site_contract.py` | 5 |
  | `tests/oracles/test_oracle_w0_lints.py` | 14 |
  | `tests/producer_parity/test_ledger.py` | 12 |
  | `tests/test_global_state_inventory.py` | 22 |

- The file-size ratchet (`scripts/checks/file_size_ratchet.py`) passed.
  `scripts/checks/run_all.py` matches the pre-change baseline. Its two
  failures predate this change and are unrelated:
  - `known_failures_ledger`: stale node ids.
  - `schema_lockstep` collapse images: needs torchvision.

  A PLR0913 (too-many-arguments) increase that `ruff_ratchet` caught during
  the work was fixed by passing the group-table row as one argument.
- Known flake, not caused by this change:
  `tests/test_distributed_census_topologies.py::TestZeroInterferenceSuiteAndPerf::test_zi3_armed_overhead_inside_p2_gate`
  is a wall-clock overhead gate (10% ceiling). It failed once during a
  loaded `-n 4` run, at 10.6%. Serial reruns failed 1 of 6 on the original
  tree (10.3%) and 1 of 6 on the split tree (12.4%), and it passed in the
  final targeted run. It measures capture overhead, which this change does
  not touch.
