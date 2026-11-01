"""Op-keyed rows of the validation exemption table.

Each row here is derived into a run-time registry in
:mod:`torchlens.validation.exemptions` (``SKIP_VALIDATION_ENTIRELY``,
``SKIP_PERTURBATION_ENTIRELY``, ``STRUCTURAL_ARG_POSITIONS``,
``STRUCTURAL_ARG_KWARG_ALIASES``, ``CUSTOM_EXEMPTION_CHECKS``). The assembled
table and its views live in :mod:`torchlens.validation._exemption_table`.
"""

from __future__ import annotations

from ._exemption_schema import ExemptionKind, ExemptionRow

_HARDENING = "tests/validation_goldens/test_validation_exemption_hardening.py::"
_TIGHTENING = "tests/test_validation_exemption_tightening.py::"
_TABLE_TESTS = "tests/test_validation_exemption_table.py::"
_EX = "torchlens.validation.exemptions:"
_CORE = "torchlens.validation.core:"


def _row(
    names: str | tuple[str, ...],
    kind: ExemptionKind,
    admits: str,
    reason: str,
    guard: str,
    removal_test: str,
    negative_test: str | None,
    positions: frozenset[int] = frozenset(),
    kwarg_aliases: tuple[tuple[int, frozenset[str]], ...] = (),
) -> ExemptionRow:
    """Build a row from compact arguments (a single name becomes a 1-tuple)."""

    spelled = (names,) if isinstance(names, str) else names
    return ExemptionRow(
        spelled, kind, admits, reason, guard, removal_test, negative_test, positions, kwarg_aliases
    )


def _member_test(kind: str, name: str) -> str:
    """Return the table test that fails when a registry member disappears."""

    return f"{_TABLE_TESTS}test_registry_member_is_declared[{kind}:{name}]"


def _skip_validation(name: str, reason: str, negative_test: str | None) -> ExemptionRow:
    admits = (
        "forward replay and perturbation are both skipped for this op; the saved "
        "bytes are uninitialized memory and no deterministic replay exists"
    )
    return _row(
        name,
        "skip_validation",
        admits,
        reason,
        _EX + "uninitialized_by_design_applies",
        _member_test("skip_validation", name),
        negative_test,
    )


def _skip_perturbation(name: str, reason: str, negative_test: str | None) -> ExemptionRow:
    admits = (
        "perturbation is skipped for every parent of this op; forward replay still "
        "runs and must reproduce the saved output"
    )
    return _row(
        name,
        "skip_perturbation",
        admits,
        reason,
        _CORE + "validate_parents_of_saved_layer",
        _member_test("skip_perturbation", name),
        negative_test,
    )


def _structural(
    names: str | tuple[str, ...],
    positions: frozenset[int],
    admits: str,
    reason: str,
    negative_test: str | None,
    kwarg_aliases: tuple[tuple[int, frozenset[str]], ...] = (),
) -> ExemptionRow:
    spelled = (names,) if isinstance(names, str) else names
    return _row(
        spelled,
        "structural_position",
        admits,
        reason,
        _EX + "perturbed_layer_at_structural_position",
        _member_test("structural_position", spelled[0]),
        negative_test,
        positions,
        kwarg_aliases,
    )


def _custom(
    names: str | tuple[str, ...], check: str, admits: str, reason: str, negative_test: str | None
) -> ExemptionRow:
    spelled = (names,) if isinstance(names, str) else names
    return _row(
        spelled,
        "custom_check",
        admits,
        reason,
        _EX + check,
        _member_test("custom_check", spelled[0]),
        negative_test,
    )


_UNINIT_NEGATIVE = _HARDENING + "test_empty_like_is_justified_exempted_not_unverified"
_POSITION_NEGATIVE = (
    _HARDENING + "test_structural_arg_exemption_uses_parent_position_not_equal_value"
)
_FACTORY = "factory template: the self tensor supplies dtype, device and layout only"
_SHAPE = "shape template: the tensor is read for its size only, never its elements"
_COORDS = (
    "SAFETY skip, not a value-irrelevance proof: perturbed coordinates or offsets can index "
    "out of bounds inside the native kernel and segfault past Python; replay still guards "
    "the value edge"
)

_REGISTRY_ROWS: tuple[ExemptionRow, ...] = (
    # ------------------------------------------------------------ skip_validation
    _skip_validation(
        "empty_like",
        "returns uninitialized memory by construction; saved bytes are not replayable",
        _UNINIT_NEGATIVE,
    ),
    _skip_validation(
        "new",
        "torch.Tensor.new() returns uninitialized memory by construction for the "
        "argless/size-only overloads ONLY, proved per call by "
        "uninitialized_by_design_applies",
        _TIGHTENING + "test_value_bearing_tensor_new_corruption_fails_not_exempted",
    ),
    _skip_validation(
        "new_empty",
        "torch.Tensor.new_empty() returns uninitialized memory by construction",
        _UNINIT_NEGATIVE,
    ),
    _skip_validation(
        "new_empty_strided",
        "torch.Tensor.new_empty_strided() returns uninitialized memory by construction",
        _UNINIT_NEGATIVE,
    ),
    _skip_validation(
        "newempty",
        "canonical torch.Tensor.new_empty() spelling returns uninitialized memory",
        _UNINIT_NEGATIVE,
    ),
    _skip_validation(
        "newemptystrided",
        "canonical torch.Tensor.new_empty_strided() spelling returns uninitialized memory",
        _UNINIT_NEGATIVE,
    ),
    # ---------------------------------------------------------- skip_perturbation
    _skip_perturbation(
        "new_zeros", "output is all zeros by construction; no parent value reaches it", None
    ),
    _skip_perturbation(
        "new_ones", "output is all ones by construction; no parent value reaches it", None
    ),
    _skip_perturbation(
        "zero_", "in-place zero fill; the destination's prior values are discarded", None
    ),
    _skip_perturbation(
        "zeros_like",
        "shape/dtype/device template only; the output value is constant zero",
        None,
    ),
    _skip_perturbation(
        "ones_like", "shape/dtype/device template only; the output value is constant one", None
    ),
    _skip_perturbation(
        "rand_like", "values are RNG-drawn; the parent supplies shape/dtype/device only", None
    ),
    _skip_perturbation(
        "randn_like", "values are RNG-drawn; the parent supplies shape/dtype/device only", None
    ),
    _skip_perturbation(
        "exponential_",
        "in-place RNG draw; the output is determined by RNG state, not inputs",
        None,
    ),
    # -------------------------------------------------------- structural_position
    _structural(
        "new",
        frozenset({0}),
        "value-bearing new(tensor)/new(data): the self tensor at arg 0 is the template",
        _FACTORY + "; the data source at arg 1 stays perturbation-tested",
        _TIGHTENING + "test_value_bearing_tensor_new_corruption_fails_not_exempted",
    ),
    _structural(
        "copy_",
        frozenset({0}),
        "the destination at arg 0 is fully overwritten by the source",
        "destination values are overwritten; source values determine the output",
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("_foreach_copy_", "foreachcopy"),
        frozenset({0}),
        "each destination member (arg 0, zipped slot) is totally overwritten by its source",
        "destination members are value-irrelevant by construction; sources stay tested",
        _POSITION_NEGATIVE,
    ),
    _structural(
        "fill_",
        frozenset({0}),
        "the destination at arg 0 is overwritten by the fill value",
        "destination values are overwritten; the fill VALUE at arg 1 stays tested",
        _POSITION_NEGATIVE,
    ),
    _structural(
        "set_",
        frozenset({0}),
        "the receiver at arg 0 is rebound to the source storage",
        "the receiver's prior values never reach the output; a source at arg 1 stays tested",
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("expand_as", "expandas"),
        frozenset({1}),
        "the template at arg 1 supplies the target shape",
        _SHAPE + "; arg 0 values flow into the output and stay tested",
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("reshape_as", "reshapeas"),
        frozenset({1}),
        "reshape_as(other) is self.reshape(other.sizes())",
        _SHAPE,
        _POSITION_NEGATIVE,
        ((1, frozenset({"other"})),),
    ),
    _structural(
        ("view_as", "viewas"),
        frozenset({1}),
        "view_as(other) is self.view(other.size())",
        _SHAPE,
        _POSITION_NEGATIVE,
        ((1, frozenset({"other"})),),
    ),
    _structural(
        "resize_as_",
        frozenset({1}),
        "resize_as_(the_template) is self.resize_(the_template.size())",
        _SHAPE + "; the receiver keeps the element prefix it carries and stays tested",
        _POSITION_NEGATIVE,
        ((1, frozenset({"the_template"})),),
    ),
    _structural(
        "_pack_padded_sequence",
        frozenset({1}),
        "the lengths tensor at arg 1 (keyword lengths)",
        "lengths are packing metadata, not output values",
        _POSITION_NEGATIVE,
        ((1, frozenset({"lengths"})),),
    ),
    _structural(
        "_pad_packed_sequence",
        frozenset({1}),
        "the lengths tensor at arg 1 (keyword lengths)",
        "lengths are packing metadata, not output values",
        _POSITION_NEGATIVE,
        ((1, frozenset({"lengths"})),),
    ),
    _structural(
        "type_as",
        frozenset({1}),
        "the type template at arg 1 (keywords tensor, other)",
        "type template tensor; its values are irrelevant",
        _POSITION_NEGATIVE,
        ((1, frozenset({"tensor", "other"})),),
    ),
    _structural(
        ("new_tensor", "newtensor"),
        frozenset({0}),
        "the self tensor at arg 0 of Tensor.new_tensor",
        _FACTORY,
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("new_full", "newfull"),
        frozenset({0}),
        "the self tensor at arg 0 of Tensor.new_full; size and fill args determine the output",
        _FACTORY,
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("new_zeros", "newzeros"),
        frozenset({0}),
        "the self tensor at arg 0 of Tensor.new_zeros",
        _FACTORY,
        _POSITION_NEGATIVE,
    ),
    _structural(
        ("new_ones", "newones"),
        frozenset({0}),
        "the self tensor at arg 0 of Tensor.new_ones",
        _FACTORY,
        _POSITION_NEGATIVE,
    ),
    _structural(
        "nms",
        frozenset({0}),
        "boxes at arg 0 (keyword boxes); scores at arg 1 stay tested",
        _COORDS,
        None,
        ((0, frozenset({"boxes"})),),
    ),
    _structural(
        "deform_conv2d",
        frozenset({1}),
        "sampling offsets at arg 1 (keyword offset); input, weight, bias, mask stay tested",
        _COORDS,
        None,
        ((1, frozenset({"offset"})),),
    ),
    _structural(
        "roi_align",
        frozenset({1}),
        "boxes at arg 1 (keywords boxes, rois); the feature map at arg 0 stays tested",
        _COORDS,
        None,
        ((1, frozenset({"boxes", "rois"})),),
    ),
    _structural(
        "roi_pool",
        frozenset({1}),
        "boxes at arg 1 (keywords boxes, rois)",
        _COORDS,
        None,
        ((1, frozenset({"boxes", "rois"})),),
    ),
    _structural(
        "ps_roi_align",
        frozenset({1}),
        "boxes at arg 1 (keywords boxes, rois)",
        _COORDS,
        None,
        ((1, frozenset({"boxes", "rois"})),),
    ),
    _structural(
        "ps_roi_pool",
        frozenset({1}),
        "boxes at arg 1 (keywords boxes, rois)",
        _COORDS,
        None,
        ((1, frozenset({"boxes", "rois"})),),
    ),
    # --------------------------------------------------------------- custom_check
    _custom(
        ("meshgrid", "broadcast_tensors", "broadcasttensors"),
        "_check_zipped_sibling_exempt",
        "a perturbed parent that feeds only OTHER outputs of a zipped multi-output op",
        "per-output parent projection: each output's own value edge stays tested",
        _TIGHTENING + "test_meshgrid_dead_own_edge_now_fails_perturbation",
    ),
    _custom(
        "__getitem__",
        "_check_getitem_exempt",
        "a perturbed parent occupying only index positions (never arg 0, the data)",
        "index tensors select positions; the indexed data at arg 0 stays tested",
        _HARDENING + "test_getitem_exemption_uses_parent_position_not_equal_value",
    ),
    _custom(
        "__setitem__",
        "_check_setitem_exempt",
        "a perturbed destination whose written slice is provably fully overwritten, "
        "or an uninitialized destination",
        "a totally overwritten destination's prior values cannot reach the output",
        _HARDENING + "test_setitem_blank_destination_partial_overwrite_is_not_exempt",
    ),
    _custom(
        ("index_put", "index_put_"),
        "_check_index_put_exempt",
        "a perturbed destination (arg 0) whose indexed positions are fully overwritten "
        "with accumulate=False",
        "the destination's prior value at overwritten positions is provably irrelevant",
        _HARDENING + "test_index_put_overwrite_proof_rejects_negative_index_aliasing",
    ),
    _custom(
        "lstm",
        "_check_lstm_exempt",
        "a perturbed parent at the hidden or cell state argument position",
        "keyed on argument position, never on tensor equality with h0",
        _POSITION_NEGATIVE,
    ),
    _custom(
        "interpolate",
        "_check_interpolate_exempt",
        "a perturbed parent at the scale_factor argument position",
        "keyed on argument position, never on tensor equality with the saved scale",
        _POSITION_NEGATIVE,
    ),
    _custom(
        ("scatter", "scatter_"),
        "_check_scatter_or_index_domain_exempt",
        "a perturbed destination whose scattered dimension is fully overwritten, or an "
        "index domain with no in-domain alternative",
        "overwritten destination values and degenerate index domains cannot matter",
        _HARDENING + "test_scatter_exemption_uses_destination_position_not_equal_value",
    ),
    _custom(
        (
            "scatter_add",
            "scatter_add_",
            "scatteradd",
            "embedding",
            "gather",
            "index_select",
            "cross_entropy",
        ),
        "_check_index_domain_degenerate",
        "an index parent whose domain has n <= 1 valid values or zero in-domain entries",
        "value-irrelevance is forced by the domain constraint, not assumed; any perturbable "
        "domain runs the strict in-domain rotation check",
        _TIGHTENING + "test_dead_embedding_index_edge_now_fails",
    ),
    _custom(
        "where",
        "_check_where_exempt",
        "a condition parent when the saved branches are equal after broadcasting; a branch "
        "parent when the saved condition selects the other branch everywhere; the one-arg "
        "index form",
        "saved value semantics prove the parent cannot reach the output",
        "tests/test_validation.py::test_where_mixed_condition_true_branch_is_not_exempt",
    ),
    _custom(
        ("maskedfill", "masked_fill", "masked_fill_"),
        "_check_masked_fill_exempt",
        "input parent when the saved mask is all true; fill parent when all false; mask "
        "parent when the input already equals the fill value everywhere",
        "masked_fill(input, mask, value) is where(mask, value, input); saved values prove it",
        _HARDENING + "test_scalar_masked_fill_all_selected_input_parent_is_structural",
    ),
    _custom(
        ("batch_norm", "instance_norm"),
        "_check_norm_running_stat_exempt",
        "training-mode running-stat parents (update targets, not inputs) and parents "
        "annihilated by an all-zero weight",
        "running stats are written, not read, in training; a zero weight annihilates",
        _HARDENING + "test_instance_norm_running_stats_are_training_update_targets",
    ),
)
