# Duplicate-test cleanup: exact copies removed, literal-only pairs parametrized

Base: `wip/mp-train` at `79eafa509`. Only test files from the group table plus this report and
`cloud-reports/dup-tests-tests.txt` changed. No assertion, fixture or helper behaviour changed.

## Environment

- `.venv-ci-smoke/` (python 3.10.20, torch 2.13.0, ruff, mypy). `scripts/smoke_ci_parity.py`'s
  `ensure_venv` could not reach `download.pytorch.org` (proxy policy 403), so the same venv was
  filled from PyPI: `torch==2.13.0` (the PyPI `+cu130` build, run on CPU),
  `torchvision==0.28.0`, and the same extras and pins (`-e .[dev,test,tabular,viz]`,
  `pydot==4.0.1`, `graphviz==0.21`, `transformers==5.18.0`).
- The container had no Graphviz binary, so the rendering tests in `tests/test_buffer_visibility.py`,
  `tests/test_metadata.py` and `tests/test_conditional_branches.py` failed with `'dot'` missing
  until `apt-get install graphviz` (this happened before and after the change). All results below
  are with Graphviz installed.
- `OMP_NUM_THREADS=1` for every pytest run.

## Repository guards that limit where copies can be deleted

- `scripts/checks/file_size_ratchet.py` ledgers `tests/test_conditional_branches.py` at 2150 lines
  (2135 at base) and fails any ledgered row more than `_MAX_LEDGER_SLACK = 100` lines under its
  ceiling. Deleting the five conditional copies from that file leaves 2049 lines (slack 101), which
  fails the ratchet, and the ratchet is not a file this change may edit. The copies were therefore
  deleted from the unledgered `tests/test_conditional_integration.py`. Both files test the same
  feature (conditional branch attribution); the kept copies also run under
  `test_conditional_branches.py`'s autouse AST/linecache isolation fixture. The five model classes
  used only by the deleted copies (`DecoratedForwardModel`, `NestedTernaryModel`,
  `SaveSourceContextOffAssertModel`, `TernaryInsideIfModel`, `TernaryWithBoolCastModel`) were
  removed from `test_conditional_integration.py` with them. Each was used nowhere else in that
  file and was identical to its `test_conditional_branches.py` twin, which stays.
- `tests/oracles/test_oracle_w0_lints.py::test_numeral_census_baseline_mechanics` fails on stale
  rows of `tests/oracles/data/numeral_baseline.tsv`, which keys numerals in test names and
  docstrings per file. Merged docstrings keep every numeral of the old docstrings, and the per-file
  census over all member files is the same before and after (72 keys each, no row lost or added).
  The oracle file passes (14 passed).
- `ruff format` / `ruff check` dropped imports the deleted copies alone used:
  `Callable` and `PathFaithfulness` in `tests/test_tlspec_runnable_r77_fact_components.py`.

## Case ids

Every parametrized case uses its old test name minus the `test_` prefix as its id, so
`test_legacy_bool_refuses_typed[legacy_true_refuses_typed]` replaces
`test_legacy_true_refuses_typed`, and a grep for an old name still finds its case.

## Exact-duplicate groups

| group | action | collected before -> after | marks |
| --- | --- | --- | --- |
| X27 | kept `tests/test_conditional_branches.py` copy, deleted `tests/test_conditional_integration.py` copy (+ its model class) | 2 -> 1 | unmarked, unchanged |
| X28 | skipped: not an exact duplicate any more. The bodies match, but the two files' `IfBoolCastModel` differ (`bool(x.sum() > 0)` vs `bool(x.mean() > 0)`), so each copy traces a different graph | 2 -> 2 | unchanged |
| X29 | kept `test_conditional_branches.py` copy, deleted integration copy (+ model class) | 2 -> 1 | `requires_code_positions` skipif kept on the survivor |
| X30 | kept `test_conditional_branches.py` copy, deleted integration copy (+ model class) | 2 -> 1 | unmarked, unchanged |
| X31 | kept `test_conditional_branches.py` copy, deleted integration copy (+ model class) | 2 -> 1 | `requires_code_positions` kept |
| X32 | skipped: the name `test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution` is cited in the module docstring of `torchlens/_postprocess/ast_branches.py` (line 24), a file outside this change (the comments in both test files cite it too) | 2 -> 2 | unchanged |
| X33 | kept `test_conditional_branches.py` copy, deleted integration copy (+ model class) | 2 -> 1 | `requires_code_positions` kept |
| X34 | kept `tests/test_r4_metadata_unpickle_rce.py` copy, deleted `tests/test_r7_torch_type_io_rce.py` copy. Both files' `_save_all_levels`, `_ControlFlow` and `_StatefulLinear` are AST-identical (docstrings and comments aside). r4 owns the restricted-load path and is the file `torchlens/_io/_safe_unpickle.py` cites. `_save_all_levels` stays in r7 for `test_all_save_levels_still_load` | 2 -> 1 | `filterwarnings` kept on the survivor |
| X35 | kept `tests/test_tlspec_runnable_r75_envelope_path.py` copy (the envelope-path lane r77 extends), deleted `tests/test_tlspec_runnable_r77_fact_components.py` copy. `_save`, `_MetaBranch` and `_LiteralCarrier` are AST-identical in both files, and r77's helpers stay in use by its other tests | 4 -> 2 (2 ids per copy) | unmarked |

X29, X31 and X33 carry `requires_code_positions`, so under the pinned python 3.10 the kept copies
skip exactly as both copies did before; they run on python 3.11+.

## Literal-only groups

| group | file | action | new test (cases) | collected before -> after | marks |
| --- | --- | --- | --- | --- | --- |
| X36 | `tests/test_r47_crossthread_op_surface.py`, `tests/test_r49_private_c_op_surface.py` | skipped: members live in two files, each owning a different observer surface (r47 `_torch_ops_call_classes`, r49 `_private_c_module_callables`). One parametrized test would move one surface's install-failure pin into the other surface's file | - | 2 -> 2 | unchanged |
| S36 | `tests/test_r45_torch_symbol_decode.py` | skipped: the "literals" are whole fresh-interpreter build and attack programs (runnable registry-key tamper, portable manifest dtype tamper, runnable slot-dtype tamper) with different hazard sets (2, 3, 3 values), different outcome markers and skip texts, and docstrings carrying distinct regression provenance. A merged test would be a table of program text, not of constant values | - | 8 -> 8 | `heavy` unchanged |
| S84 | `tests/test_two_pass_inplace_fix.py` | parametrized | `test_two_pass_succeeds_with_relu_module` (`inplace` True/False) | 2 -> 2 | unmarked |
| S87 | `tests/test_observers_backward.py` | parametrized | `test_tap_backward_tensor_mount` (`label` `output_1` / `relu_1_2`) | 2 -> 2 | unmarked |
| S90 | `tests/test_buffer_visibility.py` | parametrized | `test_legacy_bool_refuses_typed` (`legacy_value` True/False) | 2 -> 2 | unmarked |
| S100 | `tests/test_themes_families_budget.py` | parametrized; the smoke member's case carries `marks=pytest.mark.smoke` | `test_zero_coverage_refuses_typed` (`summed_member`, `expected_code`) | 2 -> 2 | case 1 unmarked, case 2 `smoke` (per `pytest.param`) |
| S108 | `tests/test_metadata.py` (`TestConditionalBranchDetection`) | parametrized method | `test_if_label_in_visualization` (`stem`, `view`, assertion `message`) | 2 -> 2 | unmarked |
| S110 | `tests/test_tlspec_rehydrate_no_planted_call.py` | parametrized; both docstrings and the in-body comment folded into one docstring | `test_planted_key_refused_on_load` (`planted_key` `_internal_set` / `save`) | 2 -> 2 | unmarked |
| S111 | `tests/test_runnable_r36_regressions.py` | skipped: the two methods belong to different finding classes (`TestHon21MultiElementEscape`, `TestHon22ValueCollision`), each already parametrized over its own probe table with a different input shape. A merged test would leave one of the classes | - | 7 -> 7 | unchanged |
| S119 | `tests/validation_goldens/test_arm_campaign_killers_mutants3.py` | parametrized | `test_loop_detection_invariants_fires_on_shortcut_mismatch` (`first_is_input`, `second_func_name`, `second_num_passes`, `match`) | 2 -> 2 | unmarked |
| S121 | `tests/test_w051_bind_fixwave.py` | parametrized | `test_forged_injection_provenance_refuses_at_load` (`provenance_key`, `forged_value`, `reason`) | 2 -> 2 | unmarked |
| S124 | `tests/test_fastlog_halt.py` | parametrized | `test_halt_from_inside_module_predicate` (`halt_kind`, `reason`, `halted_label`) | 2 -> 2 | unmarked |
| S128 | `tests/test_episode_capture.py` | parametrized | `test_tampered_episode_header_refuses_typed` (`header_key`, `header_value`, `code`) | 2 -> 2 | unmarked |
| S133 | mutants3 | parametrized | `test_module_containment_logic_fires` (`nested_module`, `module`, `match`) | 2 -> 2 | unmarked |
| S137 | mutants3 | parametrized | `test_pass_count_consistency_fires_on_op_mismatch` (`op_pass_index`, `op_num_passes`, `match`) | 2 -> 2 | unmarked |
| S142 | mutants3 | parametrized | `test_layer_pass_layer_log_xrefs_fires` (`op_pass_index`, `op_layer_label`, `match`) | 2 -> 2 | unmarked |
| S143 | `tests/test_doctor.py` | parametrized; both docstrings kept | `test_capability_row_optional_absences_stay_pass` (`flags`) | 2 -> 2 | unmarked |
| S148 | mutants3 | parametrized | `test_graph_ordering_fires` (`first_raw_index`, `second_raw_index`, `match`) | 2 -> 2 | unmarked |
| S150 | `tests/test_io_security_r27.py` | parametrized | `test_load_intervention_spec_rejects_symlinked_child_json` (`child`, `outside_name`) | 2 -> 2 | unmarked |
| S152 | `tests/test_perf_method.py` | parametrized | `test_gate_handles_baseline_row_disappearance` (`operation`, `expected_passed`, `report_key`) | 2 -> 2 | unmarked |
| S153 | `tests/test_ledger_exp_persistence.py` | parametrized; the two tamper closures become one that walks the old subscript chain (`("operations", 0, "kind")`, `("member_construction", "changed", "origin")`) and writes the old value | `test_tampered_lineage_refuses_typed` (`bundle_name`, `field_path`, `forged_value`) | 2 -> 2 | unmarked |
| S183 | `tests/test_prebump_roundtrip_acceptance.py` | parametrized | `test_tamper_structure_only_refuses_typed` (`forged_field`, `forged_value`, `tamper_name`, `reason`) | 2 -> 2 | unmarked |
| S187 | `tests/test_subfolder_imports.py` | parametrized; each case keeps its two `importorskip` calls | `test_attribute_access_when_deps_present` (`first_dep`, `second_dep`, `module_name`) | 2 -> 2 | unmarked |

The mutants3 per-arm killers stay one case per raise arm. `tests/support/mutation_driver.py`
selects that file by path, not by test name, so it still selects every killer.

No member name appears in a smoke selection list, ledger, durations file, conftest list or docs
page, apart from the X32 reference above. Checked with `git grep` over the tracked tree.

## Totals

- Groups changed: 7 exact (X27, X29, X30, X31, X33, X34, X35), 20 literal-only.
  Skipped: X28, X32 (exact); X36, S36, S111 (literal-only).
- Test functions removed: 27 (7 duplicate copies + 20 functions folded into the surviving
  parametrized test of their group). Also removed: 5 model classes used only by the deleted
  copies.
- Cases: 40 literal-group cases preserved (20 groups x 2 -> 20 tests x 2 cases); 8 duplicate cases
  removed (X27, X29, X30, X31, X33: 1 each; X34: 1; X35: 2 parametrize ids).

## Collected cases per member file (`--collect-only -q`, before -> after)

| file | before | after | delta |
| --- | --- | --- | --- |
| `tests/test_buffer_visibility.py` | 9 | 9 | 0 |
| `tests/test_conditional_branches.py` | 41 | 41 | 0 |
| `tests/test_conditional_integration.py` | 34 | 29 | -5 (X27, X29, X30, X31, X33) |
| `tests/test_doctor.py` | 16 | 16 | 0 |
| `tests/test_episode_capture.py` | 23 | 23 | 0 |
| `tests/test_fastlog_halt.py` | 20 | 20 | 0 |
| `tests/test_io_security_r27.py` | 10 | 10 | 0 |
| `tests/test_ledger_exp_persistence.py` | 5 | 5 | 0 |
| `tests/test_metadata.py` | 145 | 145 | 0 |
| `tests/test_observers_backward.py` | 13 | 13 | 0 |
| `tests/test_perf_method.py` | 14 | 14 | 0 |
| `tests/test_prebump_roundtrip_acceptance.py` | 30 | 30 | 0 |
| `tests/test_r45_torch_symbol_decode.py` | 24 | 24 | 0 (skipped group) |
| `tests/test_r47_crossthread_op_surface.py` | 21 | 21 | 0 (skipped group) |
| `tests/test_r49_private_c_op_surface.py` | 16 | 16 | 0 (skipped group) |
| `tests/test_r4_metadata_unpickle_rce.py` | 19 | 19 | 0 |
| `tests/test_r7_torch_type_io_rce.py` | 11 | 10 | -1 (X34) |
| `tests/test_runnable_r36_regressions.py` | 99 | 99 | 0 (skipped group) |
| `tests/test_subfolder_imports.py` | 6 | 6 | 0 |
| `tests/test_themes_families_budget.py` | 15 | 15 | 0 |
| `tests/test_tlspec_rehydrate_no_planted_call.py` | 11 | 11 | 0 |
| `tests/test_tlspec_runnable_r75_envelope_path.py` | 6 | 6 | 0 |
| `tests/test_tlspec_runnable_r77_fact_components.py` | 9 | 7 | -2 (X35) |
| `tests/test_two_pass_inplace_fix.py` | 9 | 9 | 0 |
| `tests/test_w051_bind_fixwave.py` | 21 | 21 | 0 |
| `tests/validation_goldens/test_arm_campaign_killers_mutants3.py` | 85 | 85 | 0 |
| total | 712 | 704 | -8 |

## Smoke-tier selection (`tests/ -m smoke --collect-only -q`)

Before: `1484/7776 tests collected (6292 deselected)`. After: `1484/7776 tests collected
(6292 deselected)`. All deleted copies were unmarked, so no smoke item was removed. The sorted
node-id diff is the S100 smoke member becoming a parametrize cell, with the same assertions and
the same `smoke` mark:

```diff
< tests/test_themes_families_budget.py::test_one_view_zero_coverage_names_the_zero_coverage_code
> tests/test_themes_families_budget.py::test_zero_coverage_refuses_typed[one_view_zero_coverage_names_the_zero_coverage_code]
```

`pytest tests/test_themes_families_budget.py -m smoke`: 5 passed, 10 deselected in 0.50s (the
cell is far under the 5 s smoke budget).

## Gates

| gate | result |
| --- | --- |
| `.venv-ci-smoke/bin/ruff format torchlens tests` | 2969 files left unchanged |
| `.venv-ci-smoke/bin/ruff check torchlens tests` | All checks passed |
| `ruff format --check` over the CI lint paths | 3176 files already formatted |
| `.venv-ci-smoke/bin/mypy torchlens/` | 1 error, `torchlens/callbacks/lightning.py:43` (`Callback` import type mismatch). Same error at the base commit with this change stashed: this venv has both `lightning` and `pytorch_lightning` installed. No file under `torchlens/` changed |
| the 22 touched files plus `tests/test_conditional_branches.py`, `-n 4` | 530 passed, 14 skipped (python 3.10 code-position skips, optional deps `IPython`/`rsatoolbox` absent) |
| `tests/test_marker_lint.py`, `-n 4` | 59 passed |
| `tests/oracles/test_oracle_w0_lints.py` (incl. heavy numeral census) | 14 passed |
| `python scripts/checks/file_size_ratchet.py` | 5/5 checks passed |
| each file in `cloud-reports/dup-tests-tests.txt`, serial | all pass, 5-15 s wall each |

Not run, per the task scope: the full suite, the `not slow` tiers, and the full smoke run.
