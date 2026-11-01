# `tl.validate` returns False on a Linear + ReLU chain

## Verdict

**Bug in a metadata invariant check, fixed.** The capture, forward replay and
perturbation checks are correct. The `equivalence_symmetry` invariant
(`torchlens/validation/_invariants_equivalence.py`, `_check_equivalence_symmetry`)
looked up the op's module-suffixed `equivalence_class` as a key of
`Trace.op_equivalence_classes`, whose keys are the pre-suffix classes. On a
`Sequential` with a ReLU at module `"11"` the suffixed class of the ReLU at
module `"1"` spells exactly that other ReLU's key. The check then compared an
honest op against a foreign group and raised. The check was wrong for this
input within its own contract (per-op `equivalent_ops` must equal the op's own
trace-level group). The fix changes how the expected group is resolved. No
tolerance or exemption changed.

The failure has nothing to do with depth, op count, recursion or float drift.
It needs a ReLU at module address `"11"`, i.e. at least 6 `Linear` + `ReLU`
pairs.

## Reproduction matrix (pre-fix, CPU)

`tl.validate(model, x, scope="forward", seed=s, validate_metadata=m)` with
`model = nn.Sequential(*[Linear(w, w), ReLU()] * depth)` (distinct modules),
`x = randn(2, w)`.

| width | depth | dtype | seed | validate_metadata | result |
|---|---|---|---|---|---|
| 8 | 1-5 | float32 | 0 | True | True |
| 8 | 6, 7, 8 | float32 | 0 | True | **False** |
| 32 | 1-5 | float32 | 0 | True | True |
| 32 | 6, 7, 8, 10, 50 | float32 | 0 | True | **False** |
| 128 | 1-5 | float32 | 0 | True | True |
| 128 | 6, 7, 8 | float32 | 0 | True | **False** |
| 32 | 5 | float64 | 0 | True | True |
| 32 | 6 | float64 | 0 | True | **False** |
| 32 | 6 | float32 | 7, 123 | True | **False** |
| 32 | 6 | float32 | 0 | False | True |
| 32 | 600 | float32 | 0 | False | True |
| 8, 32, 128 | 600 | float32 | 0 | True | **False** |
| 32 | 600 | float64 | 0 | True | **False** |

The smallest failing depth is 6 at every width. The result does not depend on
width, dtype or seed. It depends only on depth >= 6 and on
`validate_metadata=True`: with metadata invariants off, replay and perturbation
pass even at depth 600. Every failure carries the same message:

```
metadata_invariant mismatch -- MetadataInvariantError: [equivalence_symmetry]
relu_1_2:1.equivalent_ops=['relu_1_2:1'] != expected ['relu_6_12:1']
```

Post-fix, every row above returns True (depth 600 at widths 8, 32, 128;
float64; seeds 0 and 7; depth 6, 12, 60).

## Failing check and op

- Check: `equivalence_symmetry` (metadata invariant), reached from
  `validate()` in `torchlens/validation/consolidated.py` through
  `validate_forward_pass`. The forward verdict is False because a
  `MetadataInvariantError` settles False (`consolidated.py`). The structured
  record comes from `torchlens.validation.last_validation_failure()`.
- Op: `relu_1_2:1`, the ReLU in module `"1"`. Its `equivalent_ops` is
  `{'relu_1_2:1'}`, which is correct: it is the only ReLU in that module. The
  check expected `{'relu_6_12:1'}`, the group of the ReLU in module `"11"`.

## Root cause

The torch capture writes the equivalence data in two namespaces:

- `torchlens/backends/torch/_ops_activations.py:260-262` registers each op under
  `base_equivalence_class` (pre-suffix) in `Trace.op_equivalence_classes`, and
  `equivalent_ops` is that group.
- The op itself is stamped with `base + "_".join(module addresses)`
  (`_append_module_suffix_to_equivalence_class`,
  `torchlens/backends/torch/tensor_tracking.py:1399`). For a non-parameterized
  op the base already ends in `_module<innermost address>`
  (`tensor_tracking.py:1439-1441`).

So the ReLU at module `"1"` gets registry key `relu_<hash>_module1` and stamped
class `relu_<hash>_module1` + `"1"` = `relu_<hash>_module11`. That stamped class
is byte-identical to the registry key of the ReLU at module `"11"`. Captured
values for a depth-6 chain:

```
registry key 'relu_bb836ef1f1396e4e_module1'  -> ['relu_1_2:1']
registry key 'relu_bb836ef1f1396e4e_module11' -> ['relu_6_12:1']
relu_1_2:1.equivalence_class  = 'relu_bb836ef1f1396e4e_module11'
relu_6_12:1.equivalence_class = 'relu_bb836ef1f1396e4e_module1111'
```

`_check_equivalence_symmetry` resolved the expected group with
`ml.op_equivalence_classes.get(op.equivalence_class)` first and fell back to
membership only on a miss. Its own comment already said the per-op class
carries the suffix and the keys are pre-suffix, so this first lookup was a
cross-namespace lookup. Usually it just misses, but here it hit the foreign
group. Below depth 6 no stamped class spells another key, which is why
resnet18 and the other models passed. A probe over a 12-pair chain, a nested
shared block, resnet18, a 2-layer LSTM and a 2-layer TransformerEncoder found
exactly one foreign hit, in the chain.

The same lookup also made the check **blind** to a real corruption: rebinding
the module-`"1"` ReLU's `equivalent_ops` onto the foreign group's labels passed
pre-fix, because that is the group it was wrongly compared against
(`test_equivalence_symmetry_still_fires_on_one_sided_rebind` fails pre-fix).

## Fix

`torchlens/validation/_invariants_equivalence.py`, `_check_equivalence_symmetry`
(only the expected-group resolution changed):

An op's own registry key always prefixes its stamped class. Torch appends the
module path to it, and the preview backends (`backends/_finalize.py`) stamp
the key itself. The check now takes the group that holds the op under a key
that prefixes the op's stamped class. If no group does, it uses the old order:
the stamped-class key lookup, then membership.

Effect on detection, compared with the old resolution:

- Honest traces: every op resolves to its own group. In the probe, every op of
  the chain, LSTM and transformer, and all 71 non-buffer ops of resnet18, take
  the new path. The 80 resnet18 buffer ops (registry key `buffer:<address>`,
  stamped `buffer_<address><path>`) keep the old path unchanged.
- A label scrubbed out of its group, or a registry group replaced by a subset,
  takes the old path, so `test_corruption_arm_equivalence_registry_group_mismatch`
  and the other existing arming tests still fire.
- A one-sided rebind of an op whose stamped class collides with a foreign key
  is now caught (it was missed before).
- Residual, unchanged in kind: a label moved into another group whose key also
  prefixes the op's stamped class, with `equivalent_ops` re-pointed to match,
  is not caught by this check. For module-suffixed ops the old code resolved
  by membership alone and missed every such move.

Alternative not taken: rebuilding the exact suffix from `Op.modules` and
stripping it. It is also correct, but it adds a `modules` read site to
`_invariants_equivalence.py`, so the committed producer-parity ledger
(`tests/producer_parity/ledger/static_scan.json`) needs a reviewed refresh,
outside the files this change may touch. The chosen resolution reads no new
field, and the ledger test passes unchanged.

Not changed, recommended follow-ups (outside this change's files):

- The capture joins the module path to the class without a separator
  (`tensor_tracking.py:1399`), so a stamped class can collide with another
  class's key. A separator would make the stamped class unambiguous. That
  changes every stamped `equivalence_class` string and would need a golden and
  artifact review, so it is not the smallest fix.
- `docs/reference/glossary.md` describes `Op.equivalence_class` as the "Key of
  the equivalence class". For torch ops inside a submodule it is the key plus
  the module path, not a key of `Trace.op_equivalence_classes`. The entry
  should say so.

No public name changed, and no validation tolerance, exemption, skip or
ceiling changed.

## Tests

New `tests/test_validate_deep_chain.py` (depth 6, width 8; the smallest
reproducing chain). It holds 4 tests, each under 1 s measured (0.6 s for the
`tl.validate` test). They are left unmarked, which is the <5 s budget tier:
`tests/AGENTS.md` keeps new tests out of the capped smoke tier unless they
cover a critical path smoke does not already reach, and this invariant's
arming tests are already in smoke.

- `test_validate_passes_on_linear_relu_chain_at_collision_depth`: fails pre-fix.
- `test_equivalence_symmetry_resolves_registry_key_without_suffix_collision`:
  pins the collision as a precondition, then runs the invariants. Fails pre-fix.
- `test_equivalence_symmetry_still_fires_on_scrubbed_registry_group`: arming.
- `test_equivalence_symmetry_still_fires_on_one_sided_rebind`: arming. Fails
  pre-fix (the blind spot above).

Pre-fix, 3 of the 4 new tests fail; post-fix all 4 pass.

Gates, run in `.venv-ci-smoke/`:

- `ruff format torchlens tests`: 2970 files left unchanged.
- `ruff check torchlens tests`: all checks passed.
- `mypy torchlens/`: no issues in 1124 source files.
- Covering tests: the new file, `tests/test_validate_consolidated.py`,
  `tests/test_metadata_invariant_dispatch.py`, every test file that imports
  `torchlens.validation.invariants` or `_invariants_equivalence` (48 files,
  including `tests/test_validation.py`, the `validation_goldens` mutant killers
  and `tests/test_split_rebind_patchability.py`), plus the census gates that
  read this module (`tests/producer_parity/test_ledger.py`,
  `tests/oracles/test_oracle_w0_lints.py`,
  `tests/composition_expectations/test_diagnostic_site_contract.py`).
  54 files, `-n 4`: 1522 passed, 14 skipped, 12 failed. All 12 failures were
  `GraphvizUnavailableError` (no `dot` binary) in
  `tests/test_orphan_rendering.py` and `tests/test_loop_recurrence_hardening_2.py`.
  After installing Graphviz, those two files pass: 30 passed.
- `cloud-reports/deep-chain-tests.txt` (the checker's list): 128 passed in
  about 10 s.

Environment note: the pinned venv builder could not reach the PyTorch CPU
wheel index (proxy denial), so `.venv-ci-smoke/` was built by the same
`uv pip install` with the PyPI wheel of the same version: python 3.10.20,
torch 2.13.0 (`+cu130` build, run on CPU only), torchvision 0.28.0, and the
same other pins.
