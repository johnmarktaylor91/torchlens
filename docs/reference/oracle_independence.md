# Oracle Independence Table

Doc of record for the R75 oracle-independence census (three hunt rounds,
2026-08). Every verdict-steering oracle in the validation pipeline is listed
with its observation root, its independence class from the capture machinery
it judges, and the planted-corruption test that proves the classification.
The table is the reviewed authority: a new oracle, or a change to an
observation root, edits this file in the same change.

## Independence classes

- **INDEPENDENT** — the oracle observes through a root the wrapper/capture
  machinery cannot influence (fresh process, pristine namespace, raw bytes).
- **PARTIAL** — independent of the primary defect surface it exists to catch,
  but sharing at least one disclosed root with the subject.
- **SHARED (quarantined)** — same root as the subject, and the pipeline
  already refuses to let it bless alone (typed refusal, `not_applicable`
  attestation, fail-closed gap).
- **SHARED (disclosed residual)** — same root as the subject for a named
  class of defects; the class, its boundary, and its compensating external
  oracle are documented and pinned by a test.

## The table

| Oracle | Observation root | Class | Arming / boundary test |
|---|---|---|---|
| Phase-0 pristine ground truth | Fresh forward on UNWRAPPED torch (`validation/_pristine.py`; wrappers restored from the `_decorated_to_orig` ledger, reinstalled after) | INDEPENDENT of the wrapper layer; the ledger itself is integrity-scanned (wrapper-marked "orig" ⇒ typed refusal `pristine_ledger_poisoned`) and a poisoned ledger fails validation closed | `tests/test_validation_pristine_ground_truth.py` (planted 0.1% tanh wrapper distortion fails; poisoned-ledger refusal; honest-ledger control) |
| Per-op forward replay | The RECORDED capture callable (`layer.func`) re-executed on saved parent payloads | SHARED (disclosed residual) for distortions masked from the final output; PARTIAL otherwise — an output-reaching wrapper distortion diverges from the pristine phase-0 truth | Masked-interior boundary pinned by `test_masked_interior_distortion_boundary_is_pinned`; edge-drop/dup plants in `tests/test_oracle_independence.py` |
| Replay comparator (`tensor_nanequal`) | Single judge for per-op replay AND capture-side alias/mutation bookkeeping — a dominant hub | PARTIAL (hub risk, not a formal SPOF); entry SELF-TEST proves it on sentinel pairs (incl. NaN doctrine and signed-zero flips) before any verdict is trusted | `tests/test_validation_comparator_selftest.py`; mutation driver margin |
| Metadata invariants (registry) | Finished-Trace fields; a subset re-derives from independent roots (raw journal seq domains, capture-sealed edge-survival witness, ancestry/distance closure recompute, live-payload truth, CSR edge-occurrence multiplicity) | PARTIAL overall; a named TAUTOLOGICAL subclass (producer-arithmetic re-execution: `num_passes == len(recurrent_ops)`, lookup bidirectionality, param aggregates) catches post-hoc mutation only, never a systematically wrong producer | Whole-registry mutation scoring (`tests/support/mutation_driver.py`, roster derived from `METADATA_INVARIANT_CONTRACTS`); sub-check deletion operator (BLOCK mutants) |
| Backward / layer-grad oracle | Stock autograd on the untraced model under the pristine context, plain hooks | INDEPENDENT/PARTIAL (properly differential) | `tests/test_backward.py`; pristine refusal path shared with phase-0 |
| Numeric attestation (loaded sparse) | Fresh namespace resolution at load + `_decorated_to_orig` force-unwrap with typed WRAPPER_SHADOWED refusal + three-way SHA-256, zero float tolerance | INDEPENDENT — the model oracle of the pipeline | `tests/test_tlspec_runnable_r35_attestation_lattice.py` (ceiling lattice) plus the runnable contract suites; the byte tripwire itself tests its residual roots |
| Live `run()` provider | The capture machinery itself (fork `_save_new_outs`) | SHARED (quarantined): attestation reports `not_applicable`, never `attested` | `tests/test_tlspec_runnable_execution.py` (LIVE-provider pins); `tests/test_tlspec_runnable_r35_attestation_lattice.py` (not_applicable ceilings) |
| Loaded-sparse / `fast=True` run guards | Registry callables unwrapped through the SAME ledger, executed under `pause_logging` through installed wrapper shells | SHARED (disclosed residual) — verdicts are wrap-state-correlated on this surface, and the two shared-root defect directions are now PINNED FAIL-CLOSED: a hostile shell shadowing the live namespace at load refuses typed (`ReattachError`), and a poisoned `_decorated_to_orig` row for an installed shell refuses typed the same way; post-load namespace swaps never enter the resolved path (bound at load) | `tests/test_oracle_independence.py` (`test_loaded_sparse_refuses_wrapper_shadowed_callable`, `test_loaded_sparse_refuses_poisoned_unwrap_ledger_entry`, plus the verified/attested positive control) |
| RF `verify` | Autograd through the CAPTURED retained graph | PARTIAL — catches TL indexing/sampling bugs; structurally blind to capture-time forward corruption (scope documented) | `tests/test_r19b_rf_gradient_verify.py`; `tests/test_rf_crossval.py` |
| Non-PyTorch replay oracles (tf/mlx/jax/paddle/tinygrad) | NumPy/`jnp` comparison of replayed vs saved payloads under the ONE shared eps-derived band (`backends/_validation_shared.float_replay_tolerances`) | PARTIAL (capture-side payloads; fail-closed sidecars keyed to raw labels) | `tests/test_preview_replay_tolerance_tripwires.py` (corruption plants per dtype family, incl. fp64/complex/NaN/signed bands); per-backend stale-sidecar tamper tests |
| JAX derived-grad second oracle | Central finite difference with a dtype-derived step; an unmoved probe fails CLOSED | PARTIAL (same process, but numerically independent of the VJP it checks) | `tests/test_preview_grad_tolerance_tripwires.py::TestJaxFiniteDifferenceStep` |
| Mode-vs-wrapper differential harness | `TorchFunctionMode` on PRISTINE torch in a FRESH subprocess vs the wrapper stream; exception authority is the harness's own `PINNED_NOT_LOGGED` (subject-table drift is a loud mismatch) | INDEPENDENT for op-stream structure; the independent root for the masked-interior class. Composite interiors are invisible to the mode side (protocol pops before the body) — a documented limitation, not a laundering channel | `tests/test_differential_oracle_authority.py` (subject-added exemption plant) |
| Golden suites (capture/surface/selector/export/viz) | Prior torchlens output regenerated by torchlens | SHARED by construction — drift detectors, not correctness oracles | Compensating hand-derived first-principles pins (`tests/test_golden_independent_fact_pins.py` + per-suite pin files); coverage of the 63 golden-consuming files is partial and tracked |

## Standing rules

1. **No oracle may receive its exception authority from the subject under
   test.** Exemption tables, tolerance bands, and not-logged inventories are
   pinned on the oracle side; drift between the pin and the subject's live
   table is a loud failure requiring a reviewed harness edit
   (`NOT_LOGGED_AUTHORITY_DRIFT` is the template).
2. **Every SHARED classification must be either quarantined or disclosed with
   a pinned boundary test.** A shared root with neither is a finding.
3. **Tolerance policy lives in exactly one reviewed module per ecosystem**:
   `utils/tensor_utils` (torch, dtype-ULP derivation) and
   `backends/_validation_shared.py` (the non-PyTorch backends). A backend re-growing
   a private copy fails `test_no_stray_local_copies_remain`.
4. **Fail closed on resolution gaps.** An oracle that cannot prove a record
   is outside its contract keeps the tripwire armed (dead-branch backward
   carve-out, fdiff unmoved-probe refusal, ledger-poisoning refusal are the
   templates).
