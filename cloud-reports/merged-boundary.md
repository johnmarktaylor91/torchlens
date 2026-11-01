# Split `_validate_boundary` into rule groups (behaviour-preserving)

## Summary

`torchlens/_merged/_evidence.py:_validate_boundary` (one function, lines 138-477, radon F (102))
is now a 45-line orchestrator (A (1)) that calls named rule groups in the original order. The
rule groups, the closed boundary vocabularies and the shared `_refuse` constructor moved to a new
leaf module, `torchlens/_merged/_boundary_rules.py`. Every function in the new module is grade A
or B (highest cyclomatic complexity 10), so no function needs a justification comment.

Behaviour is byte-identical. All 50 raise sites keep their class, code and message text and are
checked in the same order. The same entries are accepted. Evidence: a 134-case characterization
test recorded on the unmodified code, plus a 58,537-case differential fuzz whose output files are
byte-identical before and after.

No public name changed. `_evidence` still exports `BOUNDARY_SCHEMA`, `KNOWN_KINDS`, `P2P_KINDS`,
`REDUCE_OP_KINDS`, `TENSORLESS_KINDS` and `_validate_boundary`, and the reader vocabularies
(`_COMPLETION_BINDINGS`, `_WITNESS_POLICIES`, `_DISCLOSURE_TOKENS`, `_NOT_PRESENT_REASONS`,
`_INSTALL_EPOCHS`) are re-exported as redundant aliases. The writer-lockstep tests read them from
there. No glossary, docs or notebook update applies.

## Environment

The pinned CI venv (`.venv-ci-smoke/`, python 3.10) could not be built as scripted: the egress
proxy refuses `download.pytorch.org` (HTTP 403 on CONNECT). The same venv was populated from PyPI
instead: `torch==2.13.0` (PyPI build `2.13.0+cu130`, run on CPU), `torchvision==0.28.0`,
`pip install -e ".[dev,test,tabular,viz]"`, plus `radon` and `complexipy`. `torch.distributed`
and gloo are available, so the gloo tests ran (no gloo skips).

## Complexity before and after (`radon cc -s`; cognitive from `complexipy`)

Before (`torchlens/_merged/_evidence.py` at the base commit):

| function | lines | radon CC | cognitive |
|---|---|---|---|
| `_validate_boundary` | 138-477 | F (102) | 102 |
| `extract_rank_evidence` | 480-683 | D (24) | 33 |
| `resolve_rank_inputs` | 725-783 | B (6) | 10 |
| `_require_mergeable_outcome` | 686-722 | A (3) | 2 |
| `_refuse` | 113-135 | A (1) | 0 |

After:

| file | function | lines | radon CC | cognitive |
|---|---|---|---|---|
| `_evidence.py` | `_validate_boundary` | 100-144 | A (1) | 0 |
| `_evidence.py` | `extract_rank_evidence` | 147-350 | D (24) | 33 |
| `_evidence.py` | `resolve_rank_inputs` | 392-450 | B (6) | 10 |
| `_evidence.py` | `_require_mergeable_outcome` | 353-389 | A (3) | 2 |
| `_boundary_rules.py` | `_refuse` | 87-109 | A (1) | 0 |
| `_boundary_rules.py` | `_is_strict_int` | 112-115 | A (2) | 1 |
| `_boundary_rules.py` | `check_boundary_identity` | 118-137 | A (3) | 2 |
| `_boundary_rules.py` | `check_correlation_key` | 140-171 | B (9) | 8 |
| `_boundary_rules.py` | `check_group_membership` | 174-205 | B (8) | 7 |
| `_boundary_rules.py` | `check_group_coherence` | 208-268 | B (9) | 8 |
| `_boundary_rules.py` | `_check_membership_digest_binding` | 271-299 | A (3) | 1 |
| `_boundary_rules.py` | `check_roles` | 302-332 | A (5) | 4 |
| `_boundary_rules.py` | `_check_role_entry` | 335-396 | B (10) | 7 |
| `_boundary_rules.py` | `check_scalar_records` | 399-435 | B (7) | 7 |
| `_boundary_rules.py` | `check_completion_events` | 438-467 | A (5) | 4 |
| `_boundary_rules.py` | `check_disclosures` | 470-520 | B (9) | 6 |
| `_boundary_rules.py` | `check_kind_class_records` | 523-549 | B (7) | 6 |
| `_boundary_rules.py` | `check_witness_policy` | 552-601 | B (9) | 7 |
| `_boundary_rules.py` | `check_witness_digests` | 604-644 | A (3) | 3 |
| `_boundary_rules.py` | `_check_digest_provenance` | 647-681 | A (4) | 3 |
| `_boundary_rules.py` | `_check_digest_list` | 684-711 | B (7) | 6 |
| `_boundary_rules.py` | `check_op_label_backrefs` | 714-758 | B (8) | 8 |
| `_boundary_rules.py` | `check_lifetime_evidence` | 761-777 | A (4) | 2 |

`extract_rank_evidence` (D (24)) is outside this task's scope and is unchanged. File sizes:
`_evidence.py` went from 783 to 450 lines, and `_boundary_rules.py` is 777 lines (under the
800-line cap).

## New function map (call order in `_validate_boundary`)

| # | rule group | what it checks | raise sites |
|---|---|---|---|
| 1 | `check_boundary_identity` | schema tag; kind in `KNOWN_KINDS` (returns kind) | 2 |
| 2 | `check_correlation_key` | exact key set; string digest and channel; non-negative strict-int ordinal and seq | 5 |
| 3 | `check_group_membership` | group record and rank list present; `my_global_rank` int; members strict ints, distinct, contain self (returns ranks) | 5 |
| 4 | `check_group_coherence` | `my_group_rank` type and position; `size` equals member count; backend type; then `_check_membership_digest_binding` (digest equals `membership_digest_for_ranks(ranks)`) | 4 + 1 |
| 5 | `check_roles` | roles is a list; non-empty for tensor-carrying kinds; then `_check_role_entry` per role (mapping, role name, shape, index, per-role unique index) | 2 + 5 |
| 6 | `check_scalar_records` | `reduce_op` present for reduce kinds; `c10d_group_seq` key present and int-or-null (returns it) | 3 |
| 7 | `check_completion_events` | events record and completion binding; `async_op` boolean and coherent (returns binding) | 3 |
| 8 | `check_disclosures` | list of strings; closed token vocabulary; in-flight-read disclosure matches binding; failed-seq disclosure only without a seq value | 4 |
| 9 | `check_kind_class_records` | `op_node` matches the tensorless class; peer record present exactly for p2p kinds | 3 |
| 10 | `check_witness_policy` | policy in vocabulary; not-present reason in vocabulary and equal to the one policy and binding imply (returns witness) | 3 |
| 11 | `check_witness_digests` | per field, contribution first: `_check_digest_provenance` (none under policy `none`; no destination digests when unobserved), then `_check_digest_list` (list, non-empty for tensor kinds, lowercase hex digests) | 2 + 2 |
| 12 | `check_op_label_backrefs` | list of strings; none on tensorless kinds; at least one on op-bearing kinds; no duplicates | 5 |
| 13 | `check_lifetime_evidence` | `install_epoch` in the closed vocabulary | 1 |

Groups follow the order of the original checks, so refusal precedence is unchanged. The
original comments moved with the checks they explain. `_is_strict_int` (a `TypeGuard[int]`)
replaces the eight spelled-out "int but not bool" tests with the same short-circuit order, so no
comparison runs on a value it did not run on before.

## Proving equivalence

1. **Characterization test** (`tests/test_merged_boundary_rules.py`, written and recorded on
   the unmodified code, unchanged after the refactor; 616 lines):
   - 17 recorder-coherent entries that must be accepted and not mutated: tensorless, digest
     empty lists, p2p, async digest, null and absent group rank, read-failed seq, role
     position hole, unsorted membership, and others.
   - 113 refusal rows, one or more mutations per raise site. 96 of them each target one check.
   - 17 `order_*` rows carry two defects in adjacent groups and pin which refusal wins.
   - Each refusal asserts the exact message, the class (`MergeInputError`), the code
     (`merged_schema_invalid`) and the full payload (`code`, `findings`, `remedy`, `source`).
   - 3 rows go through `extract_rank_evidence` end to end.
   - A trace of the 113 refusal rows executes all 50 raise statements: 50 of 50 in the old
     function, and 50 of 50 in the new module.
   - Result: 134 passed before the refactor and 134 passed after.
2. **Differential fuzz** (scratch only, not committed): 58,537 seeded random mutations (1-3 edits
   per entry from a 39-value pool that includes unhashable values) over all 17 accepted bases. The
   recorded `(class, message, payload)` outputs are byte-identical before and after: 52,568
   `MergeInputError`, 3,326 accepted, 2,643 `TypeError`.

### Finding (pre-existing, preserved, not fixed here)

The 2,643 `TypeError` outcomes are a pre-existing escape. An unhashable value (a list or dict)
where a closed-vocabulary string is expected raises a bare `TypeError` from the `in frozenset`
membership tests instead of the typed `merged_schema_invalid` refusal. Affected fields include
`kind`, `role`, `completion_binding`, `policy_resolved`, `not_present_reason`, `install_epoch`
and the ranks in `my_global_rank not in global_ranks`. For example, `kind=[]` gives
`TypeError: unhashable type: 'list'`. The refactor preserves this exactly, because the brief
requires identical error classes. A fix would be a deliberate behaviour change, with new rows in
the characterization test.

## Census rows moved and why

- **Refusal-site census (`refusal_site_classification.tsv`)**: no change needed.
  - The 50 sites changed scope from `torchlens._merged._evidence:_validate_boundary` to
    `torchlens._merged._boundary_rules:<rule>`; the counts per rule are in the map above.
  - None of them had a classification row, and all were and remain uncoded (`_refuse` default
    code, no explicit `code=`), so `s17_total_sites`, `s17_uncoded_sites` and
    `s17_unclassified_sites` are unchanged.
  - `test_diagnostic_site_contract.py` passes unmodified.
- **Producer-parity ledger (`static_scan.json`)**: no change needed; `test_ledger.py` passes.
- **Numeral baseline (`numeral_baseline.tsv`)**: 4 rows removed, 5 added, all moved numerals.
  - `error-strings _evidence.py 256` became `error-strings _boundary_rules.py 256`: "SHA-256"
    in the moved witness-digest message.
  - New `error-strings _boundary_rules.py 10`: "c10d" in the moved `c10d_group_seq` messages.
    `error-strings _evidence.py 10` stays, because `extract_rank_evidence` messages also say
    "c10d".
  - `package-docstrings _evidence.py 12`, `17` and `65` became the same rows for
    `_boundary_rules.py`: "R65-12" and "S-17" in the moved `_refuse` docstring.
  - Regeneration: `python tests/oracles/_regen.py` also rewrites many rows unrelated to this
    change (pre-existing drift in other files, plus a re-sort). Only the rows above were applied,
    each at the position the generator sorts it to; the census's live rows for both files match
    the baseline exactly.
  - New docstrings were worded so they introduce no new numerals. No row was added for a new
    numeral, and no count changed.

## Test results

Venv as above, `OMP_NUM_THREADS=1`.

- The brief's full list plus the three census tests and `tests/test_global_state_inventory.py`
  (the new module's globals), run together with `-n 4`: 578 passed, 0 skipped.
  - `test_merged_boundary_rules.py`, `test_merged_engine.py`, `test_merged_engine_hardening.py`
  - `test_merged_gloo.py`, `test_merged_slice_witness.py`, `test_r58_merged_distributed_rce.py`
  - `test_w051_capt3_merged_ledger_shape.py`, `test_distributed_census_topologies.py`
  - `test_distributed_funcol_boundaries.py`, `test_distributed_shard_local_marker.py`
  - `test_error_code_coverage.py`, `test_error_code_provocations.py`
  - `test_generative_properties_r73.py`, `test_union_merge_integrity.py`
  - `composition_expectations/test_diagnostic_site_contract.py`
  - `oracles/test_oracle_w0_lints.py`, `producer_parity/test_ledger.py`
- Other tests that import `torchlens._merged`, plus import hygiene: 412 passed, 2 skipped (jax
  not installed).
  - `test_api_surface.py`, `test_coverage_floor_governance.py`, `test_exception_taxonomy.py`
  - `test_merged_load_identity.py`, `test_merged_save_durability.py`
  - `test_no_string_private_getattr.py`, `test_punchlist_surface.py`
  - `test_io_determinism_fixes.py`, `test_import_hygiene.py`
- Gates:
  - `ruff format torchlens tests`: clean, no files changed.
  - `ruff check torchlens tests`: all checks passed.
  - `mypy torchlens/`: one error, in `torchlens/callbacks/lightning.py:43` (incompatible
    `Callback` import, because both `lightning` and `pytorch_lightning` are installed in this
    venv). The same error appears on the unmodified base commit in the same venv, in a file this
    change does not touch. There are no errors in `torchlens/_merged/`.
