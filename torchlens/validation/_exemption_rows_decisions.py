"""Reason-code rows of the validation exemption table.

Each row declares a guarded decision the pipeline emits under the row's code
(early exits, recorders, posthoc value proofs), a numeric tolerance, or a
metadata-invariant carve-out. The assembled table and its views live in
:mod:`torchlens.validation._exemption_table`.
"""

from __future__ import annotations

from ._exemption_schema import ExemptionKind, ExemptionRow

_HARDENING = "tests/validation_goldens/test_validation_exemption_hardening.py::"
_TIGHTENING = "tests/test_validation_exemption_tightening.py::"
_TABLE_TESTS = "tests/test_validation_exemption_table.py::"
_VALIDATION = "tests/test_validation.py::"
_VHARD = "tests/test_validation_hardening.py::"
_GAP = "tests/test_replay_gap_value_proofs.py::"
_RUNG = "tests/test_validation_rung1_class3.py::"
_EX = "torchlens.validation.exemptions:"
_CORE = "torchlens.validation.core:"
_C1 = "C1 replay determinism: the output is not a deterministic function of the saved inputs"
_C2 = "C2 perturbation sensitivity: the parent value provably cannot reach the output"
_C3 = "C3 replay-surface completeness: the op has no traceable function by design"
_C4 = "C4 numeric resolution: the difference is below what the dtype can represent"


def _decision(
    code: str,
    kind: ExemptionKind,
    clause: str,
    guard: str,
    admits: str,
    removal_test: str,
    negative_test: str | None,
) -> ExemptionRow:
    """Declare a guarded decision row keyed by its emitted reason code."""

    return ExemptionRow((code,), kind, admits, clause, guard, removal_test, negative_test)


_GOLDEN_CATEGORIES = (
    "tests/validation_goldens/test_validation_decision_golden.py::"
    "test_validation_decision_snapshot_covers_required_categories"
)
_DISCRETE = _EX + "_posthoc_discrete_output_decision"
_STRUCTURAL_OUT = _EX + "_posthoc_structural_output_decision"
_VALUE_PROOF = _EX + "_posthoc_value_proof_decision"
_SETITEM_TOTAL = _EX + "_setitem_destination_coverage_is_total"
_OPERAND = "torchlens.validation._operand_value_proofs:"
_SDPA = "torchlens.validation._sdpa_value_proofs:"
_SINGLE_KEY = _GAP + "test_sdpa_single_key_edge_is_exempted"
_NON_FINITE = _VHARD + "test_annihilator_and_singleton_proofs_refuse_a_non_finite_saved_output"

_DECISION_ROWS: tuple[ExemptionRow, ...] = (
    # -- early exits: the op leaves the pipeline before replay or perturbation
    _decision(
        "functionless_source_or_boundary",
        "early_exit",
        _C3,
        _CORE + "_is_provable_functionless_source_or_boundary",
        "a func-less op that is a corroborated intervention replacement, a buffer boundary "
        "with buffer_source and buffer_write_kind set, a parentless source, or a declared "
        "transform boundary; a func-less computational op in plain capture fails",
        _GOLDEN_CATEGORIES,
        _HARDENING + "test_functionless_computational_op_fails_loudly",
    ),
    _decision(
        "uninitialized_by_design",
        "early_exit",
        _C1,
        _EX + "uninitialized_by_design_applies",
        "an op in the skip_validation rows whose saved output is uninitialized memory; a "
        "value-bearing overload of the same spelling is not exempt",
        _HARDENING + "test_empty_like_is_justified_exempted_not_unverified",
        _TIGHTENING + "test_uninitialized_by_design_proof_is_fail_closed",
    ),
    _decision(
        "not_saved_by_user",
        "early_exit",
        _C3,
        _CORE + "_classify_user_excluded_replay_surface",
        "an op whose payload the user's own save predicate excluded, decided by the exact "
        "negative predicate result; a checkable mismatch still fails",
        _HARDENING + "test_selective_save_interior_gap_is_unverified",
        _HARDENING + "test_selective_save_checkable_mismatch_still_fails",
    ),
    _decision(
        "pre_perturbation_exemption",
        "early_exit",
        _C2,
        _CORE + "_check_perturbation_exemptions",
        "a parent edge admitted by a structural_position or custom_check row before the "
        "perturbation runs; the exemption is counted, never a vacuous pass",
        _HARDENING + "test_instance_norm_running_stats_are_training_update_targets",
        "tests/test_validation_immunizer_liveness.py::"
        "test_neutralization_assertion_cannot_pass_vacuously",
    ),
    _decision(
        "torchscript_metadata_only_argument",
        "early_exit",
        _C2,
        "torchlens.validation._script_metadata_proof:script_metadata_only_proof",
        "a scripted-function argument whose graph reads only size, dim, dtype or device; any "
        "element read keeps the perturbation check",
        "tests/test_torchscript_metadata_only_perturbation.py::"
        "test_every_enumerated_metadata_read_passes",
        "tests/test_torchscript_metadata_only_perturbation.py::"
        "test_one_element_read_still_fails_perturbation",
    ),
    _decision(
        "ulp_swamped_perturbation",
        "early_exit",
        _C4,
        _CORE + "_perturbation_delta_below_output_spacing",
        "a perturbation whose propagated delta is below the saved output's ULP spacing at "
        "every element; a similar-magnitude influential add is not exempt",
        _HARDENING + "test_swamped_fp32_add_uses_ulp_predicate",
        _HARDENING + "test_similar_magnitude_influential_add_is_not_ulp_exempted",
    ),
    _decision(
        "parent_inplace_rng_bernoulli",
        "early_exit",
        _C1,
        _CORE + "_check_whether_func_on_saved_parents_yields_saved_tensor",
        "a replay mismatch whose parent is an in-place bernoulli_ draw, re-run from the "
        "child's own saved arg snapshots; the snapshot proof is required",
        _HARDENING + "test_bernoulli_models_pass_forward_validation",
        _HARDENING + "test_bernoulli_parent_replay_exemption_requires_snapshot_proof",
    ),
    # -- recorders: the decision is recorded for a whole op
    _decision(
        "intentional_intervention_replacement",
        "recorder",
        _C3,
        _CORE + "_is_intentional_intervention_replacement",
        "an edited, non-internal op whose replacement the trace-level replacement ledger "
        "corroborates; a self-claimed placeholder in plain capture fails",
        _VALIDATION + "test_validation_genuine_replacement_alone_validates",
        _VHARD + "test_w32_forged_placeholder_in_plain_capture_fails",
    ),
    _decision(
        "skip_perturbation_entirely",
        "recorder",
        _C2,
        _EX + "SKIP_PERTURBATION_ENTIRELY",
        "every parent edge of an op in the skip_perturbation rows; replay still runs",
        _GOLDEN_CATEGORIES,
        _TIGHTENING + "test_meshgrid_perturbation_decisions_are_real_not_whole_op_skipped",
    ),
    # -- posthoc: a failed perturbation is re-examined against a saved-value proof
    _decision(
        "discrete_bool_output",
        "posthoc",
        _C2,
        _DISCRETE,
        "a bool output whose comparison the straddle probe proves far from threshold, or a "
        "bool predicate whose nan or finiteness probe flips; a spurious edge fails",
        "tests/test_validation_bool_exemption.py::"
        "test_far_threshold_comparison_keeps_evidence_backed_exemption",
        "tests/test_validation_bool_exemption.py::test_bool_comparison_spurious_edge_now_fails",
    ),
    _decision(
        "discrete_index_output",
        "posthoc",
        _C2,
        _DISCRETE,
        "an integer index output of the topk, sort, max, min, multinomial family after the "
        "magnitude ladder moved no index, and the one-arg where index form",
        _VHARD + "test_geometric_ladder_scoped_to_value_discretizing_children",
        _VALIDATION + "test_float_output_where_is_not_one_arg_index_exempt",
    ),
    _decision(
        "discrete_value_output",
        "posthoc",
        _C2,
        _VALUE_PROOF,
        "a non-floating max output; the result is a discrete value, not a continuous one",
        _VHARD + "test_negative_decimals_round_dead_zone_validates_true",
        _VALIDATION + "test_posthoc_integer_cast_quantization_is_exempt_only_for_float_to_int",
    ),
    _decision(
        "type_template_output",
        "posthoc",
        _C2,
        _STRUCTURAL_OUT,
        "the template tensor (arg 1 or other=) of a type-casting op; only its dtype and "
        "device flow into the output, a perturbed data source is not exempt",
        "tests/test_validation_registry_ledger.py::test_every_structural_position_has_a_proof",
        _HARDENING + "test_to_tensor_template_exemption_refuses_perturbed_data_source",
    ),
    _decision(
        "integer_cast_quantization",
        "posthoc",
        _C4,
        _STRUCTURAL_OUT,
        "a float-to-integer cast whose perturbation stays inside one quantization bucket",
        _VHARD + "test_zero_const_integer_cast_chain_validates_true",
        _VALIDATION + "test_posthoc_integer_cast_quantization_is_exempt_only_for_float_to_int",
    ),
    _decision(
        "structural_output_template",
        "posthoc",
        _C2,
        _STRUCTURAL_OUT,
        "the template parent of a factory op (full_like, zeros_like, ...) whose shape, dtype "
        "and device alone reach the output; a runtime fill value is never exempt",
        _TIGHTENING + "test_full_like_template_parent_stays_structurally_exempt",
        "tests/test_validation_immunizer_liveness.py::"
        "test_structural_template_decision_must_exclude_runtime_fill_values",
    ),
    _decision(
        "rng_probability_template",
        "posthoc",
        _C1,
        _STRUCTURAL_OUT,
        "the destination of an in-place RNG fill that overwrites every element; a "
        "non-destination parent is not exempt",
        _RUNG + "test_rng_fill_destination_validates",
        _RUNG + "test_rng_fill_template_refuses_a_non_destination_parent",
    ),
    _decision(
        "uninitialized_destination_overwrite",
        "posthoc",
        _C2,
        _EX + "_perturbed_parent_is_uninitialized_setitem_dest",
        "a __setitem__ destination born uninitialized whose every element is overwritten; a "
        "partial overwrite of a blank destination is not exempt",
        _VALIDATION + "test_validation_with_setitem_slice_full_overwrite",
        _HARDENING + "test_setitem_blank_destination_partial_overwrite_is_not_exempt",
    ),
    _decision(
        "full_destination_overwrite",
        "posthoc",
        _C2,
        _SETITEM_TOTAL,
        "a __setitem__ or index_put destination fully covered by a tensor value, proved on "
        "the saved index without duplicate or negative-index aliasing",
        _VALIDATION + "test_validation_with_index_put_destination_full_overwrite",
        _HARDENING + "test_posthoc_setitem_duplicate_indices_are_not_full_overwrite",
    ),
    _decision(
        "scalar_destination_overwrite",
        "posthoc",
        _C2,
        _SETITEM_TOTAL,
        "a __setitem__ destination fully covered by a scalar value; a partial scalar write "
        "is not exempt",
        _TABLE_TESTS + "test_scalar_setitem_full_overwrite_is_exempted",
        _HARDENING + "test_posthoc_scalar_partial_setitem_is_not_full_overwrite",
    ),
    _decision(
        "index_domain_value_irrelevant",
        "posthoc",
        _C2,
        _EX + "_index_domain_value_irrelevance_decision",
        "an index parent whose every in-domain alternative yields the saved output "
        "(uniform rows); a live index edge fails",
        _TIGHTENING + "test_uniform_embedding_rows_are_proved_irrelevant_not_failed",
        _TIGHTENING + "test_dead_embedding_index_edge_now_fails",
    ),
    _decision(
        "multiplicative_zero_annihilator",
        "posthoc",
        _C2,
        _VALUE_PROOF,
        "a mul or multiply operand whose saved partner is exactly zero at every element, "
        "keyed by argument position with a finite saved output",
        _HARDENING + "test_multiplicative_zero_annihilator_uses_structural_proof",
        _NON_FINITE,
    ),
    _decision(
        "addcmul_zero_annihilator",
        "posthoc",
        _C2,
        _EX + "_addcmul_zero_annihilator_decision",
        "an addcmul tensor1 or tensor2 operand whose partner or value is exactly zero",
        _VHARD + "test_addcmul_zero_gain_validates",
        _VHARD + "test_addcmul_zero_annihilator_refuses_unproved_cases",
    ),
    _decision(
        "binary_extrema_dominated",
        "posthoc",
        _C2,
        _EX + "_binary_extrema_nonperturbed_arg_dominates",
        "a maximum or minimum operand strictly dominated by the saved other operand at "
        "every element",
        _HARDENING + "test_binary_extrema_requires_actual_nonperturbed_dominance",
        _HARDENING + "test_binary_extrema_requires_actual_nonperturbed_dominance",
    ),
    _decision(
        "mod_divisor_irrelevant",
        "posthoc",
        _C2,
        _VALUE_PROOF,
        "a remainder divisor (arg 1) when the saved output equals the saved dividend",
        _VALIDATION + "test_validation_with_remainder_dividend_below_divisor",
        _VALIDATION + "test_remainder_divisor_not_exempt_when_output_differs_from_dividend",
    ),
    _decision(
        "integer_mod_unit_divisor",
        "posthoc",
        _C2,
        _EX + "_integer_mod_unit_divisor_decision",
        "an integer remainder dividend when the saved divisor is exactly one everywhere",
        _RUNG + "test_integer_mod_unit_divisor_dividend_is_exempt",
        _RUNG + "test_integer_mod_unit_divisor_refuses_a_divisor_parent",
    ),
    _decision(
        "batch_norm_input_at_running_mean",
        "posthoc",
        _C2,
        "torchlens.validation._norm_running_var_proof:batch_norm_input_at_running_mean",
        "the running_var edge of an eval-mode batch_norm whose input equals running_mean "
        "exactly, so the normalized term is zero",
        "tests/test_replay1_bn_running_var_proof.py::"
        "test_running_var_edge_is_exempted_by_the_value_proof",
        "tests/test_replay1_bn_running_var_proof.py::"
        "test_dead_running_var_edge_off_the_mean_still_fails",
    ),
    _decision(
        "sdpa_zero_query_uniform_attention",
        "posthoc",
        _C2,
        _EX + "_sdpa_zero_query_decision",
        "the key edge of scaled_dot_product_attention with an exactly zero query, a finite "
        "output and no mask",
        _VHARD + "test_sdpa_zero_query_key_insensitivity_validates",
        _VHARD + "test_sdpa_zero_query_decision_scoped_correctly",
    ),
    _decision(
        "softmax_singleton_reduction_dim",
        "posthoc",
        _C2,
        _EX + "_softmax_singleton_dim_decision",
        "a softmax input whose reduction dim has size one, so the output is all ones",
        _VHARD + "test_softmax_singleton_reduction_dim_validates",
        _VHARD + "test_softmax_singleton_dim_decision_scoped_correctly",
    ),
    _decision(
        "layer_norm_singleton_normalized_shape",
        "posthoc",
        _C2,
        _EX + "_layer_norm_singleton_shape_decision",
        "a layer_norm input with a singleton normalized shape, so the output is bias",
        _VHARD + "test_layer_norm_singleton_normalized_shape_validates",
        _VHARD + "test_layer_norm_singleton_shape_decision_scoped_correctly",
    ),
    _decision(
        "empty_output_no_elements",
        "posthoc",
        _C2,
        _OPERAND + "output_has_no_elements",
        "any parent of an op whose saved output has zero elements",
        _GAP + "test_empty_output_scalar_edge_is_exempted",
        _GAP + "test_dead_scalar_edge_with_one_output_element_still_fails",
    ),
    _decision(
        "sdpa_zero_value_output",
        "posthoc",
        _C2,
        _SDPA + "sdpa_zero_value_output",
        "query and key edges of scaled_dot_product_attention whose value operand is exactly "
        "zero and whose rows are not fully masked",
        _GAP + "test_sdpa_zero_value_edge_is_exempted",
        _GAP + "test_dead_sdpa_query_edge_with_near_zero_value_still_fails",
    ),
    _decision(
        "sdpa_single_key_identity",
        "posthoc",
        _C2,
        _SDPA + "sdpa_single_key_identity",
        "the query edge of scaled_dot_product_attention over exactly one finite key",
        _SINGLE_KEY,
        _GAP + "test_dead_sdpa_edge_over_two_keys_still_fails",
    ),
    _decision(
        "sdpa_single_key_key_edge",
        "posthoc",
        _C2,
        _SDPA + "sdpa_single_key_key_edge",
        "the key edge of scaled_dot_product_attention over exactly one finite key",
        _SINGLE_KEY,
        _GAP + "test_dead_sdpa_key_edge_with_a_non_finite_key_still_fails",
    ),
    _decision(
        "einsum_zero_operand",
        "posthoc",
        _C2,
        _OPERAND + "einsum_zero_operand",
        "an einsum operand contracted against a partner that is exactly zero everywhere",
        _GAP + "test_einsum_nested_operand_edge_is_exempted",
        _GAP + "test_dead_einsum_edge_with_near_zero_state_still_fails",
    ),
    _decision(
        "scatter_reduce_destination_overwritten",
        "posthoc",
        _C2,
        _OPERAND + "scatter_reduce_destination_overwritten",
        "a scatter_reduce destination with include_self False whose saved index covers "
        "every element",
        _GAP + "test_scatter_reduce_destination_edge_is_exempted",
        _GAP + "test_dead_scatter_reduce_edge_still_fails",
    ),
    _decision(
        "grid_sampler_zero_input",
        "posthoc",
        _C2,
        _OPERAND + "grid_sampler_zero_input",
        "the grid edge of grid_sampler over an exactly zero image",
        _GAP + "test_grid_sampler_grid_edge_on_a_zero_image_is_exempted",
        _GAP + "test_dead_grid_edge_on_a_near_zero_image_still_fails",
    ),
    _decision(
        "conv_zero_input_output_is_bias",
        "posthoc",
        _C2,
        _OPERAND + "conv_zero_input_output_is_bias",
        "the weight edge of a convolution over an exactly zero input, so the output is bias",
        _GAP + "test_conv_weight_edge_on_a_zero_input_is_exempted",
        _GAP + "test_dead_conv_weight_edge_on_a_near_zero_input_still_fails",
    ),
    _decision(
        "or_absorbing_true_operand",
        "posthoc",
        _C2,
        _OPERAND + "or_absorbing_true_operand",
        "a logical or operand whose saved partner is true at every element",
        _GAP + "test_or_edge_against_an_all_true_operand_is_exempted",
        _GAP + "test_dead_or_edge_against_an_operand_with_one_false_still_fails",
    ),
    _decision(
        "pow_unit_base",
        "posthoc",
        _C2,
        _OPERAND + "pow_unit_base",
        "the exponent edge of pow whose saved base is exactly one everywhere",
        _GAP + "test_pow_exponent_edge_with_unit_base_is_exempted",
        _GAP + "test_dead_pow_exponent_edge_with_a_base_just_above_one_still_fails",
    ),
    _decision(
        "locally_constant_by_construction",
        "posthoc",
        _C2,
        _VALUE_PROOF,
        "a piecewise-constant op (floor, round, sign, comparisons) whose perturbation stays "
        "inside the saved plateau; a boundary crossing validates and is not exempt",
        _VHARD + "test_nan_multiplication_proof_still_covers_plain_mul",
        _HARDENING + "test_boundary_crossing_validates_piecewise_constant_ops",
    ),
)

_TOLERANCE_ROWS: tuple[ExemptionRow, ...] = (
    _decision(
        "deep_numeric_replay_band",
        "tolerance",
        _C4,
        _CORE + "_deep_numeric_replay_matches_saved",
        "a replay mismatch on a reduction of derived depth >= 64 compared under the "
        "depth-scaled DEEP_NUMERIC_REPLAY_* band instead of dtype ULP equality; a gross "
        "error still fails",
        _VALIDATION + "test_deep_numeric_replay_recovers_deep_conv_cancellation_drift",
        _VALIDATION + "test_deep_numeric_replay_gross_error_on_deep_conv_still_fails",
    ),
    _decision(
        "ground_truth_ulp_headroom",
        "tolerance",
        _C4,
        _CORE + "_GROUND_TRUTH_ULP_HEADROOM",
        "the ground-truth output comparison uses per-dtype ULP headroom (4 for fp16 and "
        "bf16, 8 otherwise); fp8 stays strict",
        _VALIDATION + "test_ground_truth_output_check_is_dtype_aware",
        _VALIDATION + "test_ground_truth_fp8_doctrine_stays_strict",
    ),
    _decision(
        "replay_dtype_tolerances",
        "tolerance",
        _C4,
        "torchlens._utils.tensor_utils:derive_float_tolerances",
        "tensor comparisons with allow_tolerance use the dtype-derived (eps floor, rtol "
        "ceiling) pair; the rows are pinned literals",
        _VALIDATION + "test_tensor_nanequal_tolerances_are_dtype_derived_boundary",
        "tests/test_replay_tolerance_dtype_tripwire.py::"
        "test_every_row_respects_the_eps_floor_and_the_rtol_ceiling",
    ),
    _decision(
        "backward_grad_tolerances",
        "tolerance",
        _C4,
        "torchlens._utils.tensor_utils:param_grad_tolerances_for_dtype",
        "backward replay compares parameter and layer gradients under the dtype-derived "
        "PARAM_GRAD_VALIDATION and LAYER_GRAD_VALIDATION pairs when no override is given",
        "tests/test_grad_tolerance_dtype_tripwire.py::test_fp32_rows_are_the_pinned_defaults",
        "tests/test_grad_tolerance_dtype_tripwire.py::test_every_row_respects_the_eps_floor",
    ),
    _decision(
        "replay_warnings_ignored",
        "tolerance",
        _C1,
        "torchlens.validation._replay_warnings:replay_warnings_ignored",
        "warnings raised inside a validation re-execution are suppressed; no decision "
        "changes and the filter is scoped to the context manager",
        _TABLE_TESTS + "test_replay_warnings_filter_is_scoped_to_the_context",
        _TABLE_TESTS + "test_replay_warnings_filter_is_scoped_to_the_context",
    ),
)

_INVARIANT_ROWS: tuple[ExemptionRow, ...] = (
    _decision(
        "func_call_id_synthetic_or_replacement",
        "invariant",
        _C3,
        "torchlens.validation._invariants_backward_flow:_is_func_call_id_exempt",
        "input, output and buffer nodes, and a corroborated intervention replacement, carry "
        "no func_call_id; a plain-capture op missing one fails invariant S",
        _VALIDATION + "test_func_call_id_exemption_is_scoped_to_genuine_replacement",
        _VALIDATION + "test_func_call_id_invariant_rejects_missing_id_in_plain_capture",
    ),
    _decision(
        "module_call_boundary_external_inputs",
        "invariant",
        _C3,
        "torchlens.validation._invariants_modules_params:_check_module_call_boundary_and_tree",
        "a nested module's inputs may come from outside its ops; its outputs may not",
        _VALIDATION + "test_module_call_boundary_allows_external_inputs_on_nested_modules",
        _VALIDATION + "test_module_call_boundary_rejects_output_outside_ops",
    ),
    _decision(
        "payload_metadata_selective_save",
        "invariant",
        _C3,
        "torchlens.validation._invariants_payloads:_check_payload_metadata_invariants",
        "payload invariants skip ops whose payload the user's save predicate excluded",
        _VALIDATION + "test_payload_metadata_invariant_allows_selective_save_validation",
        _HARDENING + "test_selective_save_checkable_mismatch_still_fails",
    ),
)
