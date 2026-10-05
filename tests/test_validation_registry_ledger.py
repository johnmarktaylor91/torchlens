"""Per-entry audit ledger for the two declared exemption registries (finding B1-06).

``STRUCTURAL_ARG_POSITIONS`` and ``CUSTOM_EXEMPTION_CHECKS`` both settle under the
single ``pre_perturbation_exemption`` code of the exemption ledger
(``test_validation_exemption_ledger.py``), which can only see that the tables are
non-empty. The sub-ledgers here pin every ENTRY by key and value, so a new entry,
a widened position set or a rebound check is red until its audit record is
written. Split from the exemption ledger along this seam (R43 file-size
ratchet); the records and checks are unchanged.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

import pytest
from _exemption_contracts import VALIDATION_CONTRACTS

from torchlens.validation import exemptions as ex

# ---------------------------------------------------------------------------
# FINDING B1-06: per-entry closure for the two registries the ledger covered
# only by NON-EMPTINESS.
#
# ``test_registry_tables_are_covered_by_a_ledger_entry`` asserted that
# ``STRUCTURAL_ARG_POSITIONS`` and ``CUSTOM_EXEMPTION_CHECKS`` are non-empty and
# that the code they settle under is ledgered. Both facts survive ANY membership
# edit: planting ``STRUCTURAL_ARG_POSITIONS["take_along_dim"] = {1}`` (a genuine
# index-VALUE dependency), ``["where"] = {0}`` (the condition mask), or an
# unconditional ``return True`` custom check left the whole suite green -- the
# protection was a name list that re-catches yesterday's mistake and misses
# tomorrow's.
#
# The two sub-ledgers below close them per ENTRY, on the pattern
# ``SKIP_PERTURBATION_JUSTIFICATIONS`` already uses: name -> justification +
# contract clause + the case the entry must still let FAIL, pinned against the
# live registry by KEY and by VALUE (exact position sets / exact bound check
# function). A new entry, a widened position set, or a rebound check is RED
# until its audit record is written.
#
# Each row also carries a ``proof_kind``, which is the honest ground the entry
# stands on:
#
# ``value_irrelevance_proved``
#     The parent's VALUE provably cannot reach the output (destination fully
#     overwritten, dtype/device/shape template) -- squarely outside C2.
# ``domain_forced``
#     A value dependency with NO admissible alternate value (degenerate index
#     domain): irrelevance is forced by the domain, not assumed.
# ``vehicle_limited``
#     The parent IS value-sensitive and the exemption is a PERTURBATION-VEHICLE
#     limitation, not a proof: no in-domain alternative can be synthesized
#     without crashing the kernel or changing the output shape. These rows are
#     the weak ground in the table; enumerating them is the point -- the class
#     is now bounded and auditable, and a new member cannot join it silently.
#     Each such row must name the check that still guards the value edge.
# ---------------------------------------------------------------------------

_PROOF_KINDS = frozenset({"value_irrelevance_proved", "domain_forced", "vehicle_limited"})


@dataclass(frozen=True)
class StructuralPositionExemption:
    """Audit record for one ``STRUCTURAL_ARG_POSITIONS`` row.

    Parameters
    ----------
    func_name:
        Captured function name, exactly as the registry keys it.
    positions:
        Argument positions the registry declares structural. Pinned EXACTLY
        against the live set, so widening a row is red.
    contract:
        Contract clause from the module docstring the exemption sits outside.
    proof_kind:
        Ground the entry stands on (see the section comment).
    justification:
        Why the declared positions cannot carry output values -- or, for
        ``vehicle_limited``, why no in-domain perturbation can be built.
    refuses:
        The case this row must still let FAIL.
    """

    func_name: str
    positions: frozenset[int]
    contract: str
    proof_kind: str
    justification: str
    refuses: str


@dataclass(frozen=True)
class CustomCheckExemption:
    """Audit record for one ``CUSTOM_EXEMPTION_CHECKS`` row.

    Parameters
    ----------
    func_name:
        Captured function name, exactly as the registry keys it.
    check:
        Attribute name in ``torchlens.validation.exemptions`` of the check the
        registry must be bound to. Pinned by IDENTITY, so rebinding an op to a
        laxer predicate is red.
    contract:
        Contract clause the exemption sits outside.
    proof_kind:
        Ground the entry stands on (see the section comment).
    justification:
        What the check proves from the saved call.
    refuses:
        The case the check must still let FAIL.
    """

    func_name: str
    check: str
    contract: str
    proof_kind: str
    justification: str
    refuses: str


_C2 = "C2 perturbation sensitivity"

_OVERWRITTEN_DESTINATION_REFUSES = (
    "the SOURCE argument, which carries every output value, and any parent "
    "recorded at any other position on the call"
)
_FACTORY_TEMPLATE_REFUSES = (
    "the size/fill/data arguments that actually determine the output values, "
    "and the same tensor appearing at any non-factory position"
)

#: Per-entry audit records for ``STRUCTURAL_ARG_POSITIONS``.
STRUCTURAL_POSITION_LEDGER: tuple[StructuralPositionExemption, ...] = (
    StructuralPositionExemption(
        func_name="new",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "value-bearing Tensor.new(tensor)/new(data): the arg-0 SELF tensor supplies "
            "dtype and device only, per torch's legacy constructor contract -- its "
            "values never reach the output; the size-only overloads are exempted "
            "earlier by uninitialized_by_design_applies and never reach perturbation "
            "(R08-1 narrowing companion)"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="copy_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "torch's copy_ contract overwrites the arg-0 destination in full from the "
            "arg-1 source, so the destination's prior values cannot reach the output"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="_foreach_copy_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "zipped spelling of copy_: each destination member (matched per zipped slot "
            "(0, j)) is totally overwritten by its source member, so no member's prior "
            "values reach the output"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="foreachcopy",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of _foreach_copy_; identical "
            "totally-overwritten zipped destination argument"
        ),
        refuses=_OVERWRITTEN_DESTINATION_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="fill_",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "the arg-0 destination is overwritten by the scalar/tensor fill VALUE at "
            "arg 1; r31 narrowing moved fill_ out of the whole-op skip precisely so "
            "that fill value stays perturbation-tested"
        ),
        refuses="the fill VALUE at arg 1, which determines every output element",
    ),
    StructuralPositionExemption(
        func_name="expand_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "arg 1 is consumed for its SHAPE only; r31 narrowing moved expand_as out of "
            "the whole-op skip because arg 0's values are the output values"
        ),
        refuses="arg 0, whose values are broadcast into the output unchanged",
    ),
    StructuralPositionExemption(
        func_name="expandas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of expand_as; the arg-1 tensor is a shape "
            "template whose values are never read"
        ),
        refuses="arg 0, whose values are broadcast into the output unchanged",
    ),
    StructuralPositionExemption(
        func_name="reshape_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.reshape_as(other) is self.reshape(other.sizes()): arg 1 is consumed for its SHAPE only, never its elements"
        ),
        refuses=(
            "arg 0, whose values are the output values in a new shape, including "
            "when the same parent also fills arg 1 (x.view_as(x))"
        ),
    ),
    StructuralPositionExemption(
        func_name="reshapeas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of reshape_as; the arg-1 tensor is a shape template whose values are never read"
        ),
        refuses=(
            "arg 0, whose values are the output values in a new shape, including "
            "when the same parent also fills arg 1 (x.view_as(x))"
        ),
    ),
    StructuralPositionExemption(
        func_name="view_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.view_as(other) is self.view(other.size()): arg 1 is consumed for its SHAPE only, never its elements"
        ),
        refuses=(
            "arg 0, whose values are the output values in a new shape, including "
            "when the same parent also fills arg 1 (x.view_as(x))"
        ),
    ),
    StructuralPositionExemption(
        func_name="viewas",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized TorchLens spelling of view_as; the arg-1 tensor is a shape template whose values are never read"
        ),
        refuses=(
            "arg 0, whose values are the output values in a new shape, including "
            "when the same parent also fills arg 1 (x.view_as(x))"
        ),
    ),
    StructuralPositionExemption(
        func_name="type_as",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "arg 1 is consumed for dtype/device only -- Tensor.type_as reads its "
            "template's type, never its elements"
        ),
        refuses="arg 0, the tensor actually being cast, whose values flow to the output",
    ),
    StructuralPositionExemption(
        func_name="new_tensor",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "self at arg 0 is a dtype/device/layout factory template for "
            "Tensor.new_tensor; the output values come from the data argument"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newtensor",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_tensor spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_full",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_full reads self only for dtype/device/layout; the output is "
            "determined entirely by the size and fill_value arguments"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newfull",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_full spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_zeros",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_zeros reads self only for dtype/device/layout; the output is "
            "constant zero at the requested size"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newzeros",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_zeros spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="new_ones",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "Tensor.new_ones reads self only for dtype/device/layout; the output is "
            "constant one at the requested size"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="newones",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized Tensor.new_ones spelling; the self tensor is the same "
            "dtype/device/layout factory template"
        ),
        refuses=_FACTORY_TEMPLATE_REFUSES,
    ),
    StructuralPositionExemption(
        func_name="_pack_padded_sequence",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 lengths tensor IS value-sensitive (its entries decide which "
            "timesteps enter the packed data and the batch_sizes output), but the "
            "perturbation vehicle can only step its integer entries, which leaves the "
            "[1, T] / enforce_sorted precondition and aborts inside the native packing "
            "kernel instead of producing a comparable output; forward replay still "
            "reconstructs and re-runs the call from the saved lengths, so a dropped "
            "lengths edge remains visible as a replay mismatch. NARROWING OWED: the "
            "F2-style fix is an in-domain lengths perturbation (a monotone-preserving "
            "shrink), not a position blanket -- tracked as a B1-06 residual finding"
        ),
        refuses=(
            "arg 0, the padded input whose values are packed into the output, and any "
            "lengths tensor reaching the call at another position or keyword"
        ),
    ),
    StructuralPositionExemption(
        func_name="_pad_packed_sequence",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "inverse of _pack_padded_sequence with the same arg-1 lengths descriptor: "
            "stepped lengths leave the admissible domain and abort the native kernel "
            "rather than yielding a comparable output, so no in-domain alternative can "
            "be synthesized by the vehicle; forward replay still re-runs the real call "
            "from the saved lengths. NARROWING OWED with its sibling entry"
        ),
        refuses=(
            "arg 0, the packed data whose values are padded into the output, and any "
            "lengths tensor reaching the call at another position or keyword"
        ),
    ),
    # -- torchvision PyCapsule ops (b1p2 D2 adjudicated NARROWING) ----------
    # Formerly whole-op rows in SKIP_PERTURBATION_ENTIRELY: the skip was wider
    # than its segfault justification. Only the coordinate/offset arg is
    # vehicle-limited (perturbed coordinates index out of bounds inside the
    # native kernel, past Python exception handling); feature and score args
    # returned to strict perturbation.
    StructuralPositionExemption(
        func_name="nms",
        positions=frozenset({0}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-0 boxes ARE value-sensitive (coordinates decide suppression), but "
            "perturbed coordinates can index out of bounds inside the torchvision "
            "PyCapsule kernel and segfault past Python exception handling, so no safe "
            "in-domain alternative can be synthesized by the vehicle; forward replay "
            "still re-runs the real call from the saved boxes. Scores (arg 1) stay "
            "strictly perturbation-tested (the b1p2 D2 narrowing)"
        ),
        refuses="the arg-1 scores, whose values order and gate every kept box",
    ),
    StructuralPositionExemption(
        func_name="deform_conv2d",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 sampling offsets ARE value-sensitive, but out-of-domain offsets "
            "index outside the feature map inside the native kernel and can segfault, "
            "so the vehicle cannot perturb them safely; forward replay still re-runs "
            "the real call from the saved offsets. Input, weight, bias, and modulation "
            "mask stay strictly perturbation-tested"
        ),
        refuses=(
            "the arg-0 input and the weight/bias/mask arguments, whose values flow "
            "arithmetically into every output element"
        ),
    ),
    StructuralPositionExemption(
        func_name="roi_align",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "the arg-1 boxes ARE value-sensitive (they place the pooling windows), but "
            "perturbed box coordinates can address out-of-bounds feature-map memory in "
            "the PyCapsule kernel; forward replay still re-runs the real call from the "
            "saved boxes. The arg-0 feature map stays strictly perturbation-tested"
        ),
        refuses="the arg-0 feature map, whose values are averaged into every output bin",
    ),
    StructuralPositionExemption(
        func_name="roi_pool",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "same coordinate-safety limitation as roi_align for the arg-1 boxes; the "
            "arg-0 feature map stays strictly perturbation-tested and replay re-runs "
            "the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are pooled into every output bin",
    ),
    StructuralPositionExemption(
        func_name="ps_roi_align",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "position-sensitive variant of roi_align with the same arg-1 coordinate "
            "safety limitation; the arg-0 feature map stays strictly tested and replay "
            "re-runs the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are averaged into every output bin",
    ),
    StructuralPositionExemption(
        func_name="ps_roi_pool",
        positions=frozenset({1}),
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "position-sensitive variant of roi_pool with the same arg-1 coordinate "
            "safety limitation; the arg-0 feature map stays strictly tested and replay "
            "re-runs the real call from the saved boxes"
        ),
        refuses="the arg-0 feature map, whose values are pooled into every output bin",
    ),
)

#: Per-entry audit records for ``CUSTOM_EXEMPTION_CHECKS``.
_ZIPPED_SIBLING_JUSTIFICATION = (
    "meshgrid/broadcast_tensors zip N inputs to N outputs: output j carries exactly "
    "input j's values, so a perturbed parent is exempt ONLY when every recorded "
    "position places it at a zipped index other than this output's own "
    "multi_output_index (fail-closed on missing index, unknown position shape, or an "
    "unmapped parent). Replaces the former whole-op skip that also exempted each "
    "output's OWN value edge (R08-2)"
)

_ZIPPED_SIBLING_REFUSES = (
    "the output's own zipped input (index == multi_output_index), any parent whose "
    "recorded position cannot be proven cross-member, and every call on a trace that "
    "did not record multi_output_index"
)

CUSTOM_CHECK_LEDGER: tuple[CustomCheckExemption, ...] = (
    CustomCheckExemption(
        func_name="meshgrid",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=_ZIPPED_SIBLING_JUSTIFICATION,
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="broadcast_tensors",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=_ZIPPED_SIBLING_JUSTIFICATION,
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="broadcasttensors",
        check="_check_zipped_sibling_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            _ZIPPED_SIBLING_JUSTIFICATION
            + "; canonicalized capture spelling -- the old snake_case-keyed skip never "
            "matched it (a silently dead registry row)"
        ),
        refuses=_ZIPPED_SIBLING_REFUSES,
    ),
    CustomCheckExemption(
        func_name="__getitem__",
        check="_check_getitem_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY a parent that occupies no arg-0 slot, i.e. an index/slice "
            "argument of the subscript; keyed on recorded arg POSITION, never on tensor "
            "equality with the index. The indexed data parent (arg 0) stays strictly "
            "perturbed. RESIDUAL: index VALUES do select which data flows out, so this "
            "row is the same class the F2 tightening replaced elsewhere with in-domain "
            "rotation; the value edge stays guarded by forward replay -- tracked as a "
            "B1-06 residual finding"
        ),
        refuses=(
            "the arg-0 data parent, and an index tensor that merely EQUALS the "
            "subscript while being recorded at position 0"
        ),
    ),
    CustomCheckExemption(
        func_name="__setitem__",
        check="_check_setitem_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "exempts a __setitem__ destination only for a structural mask/index slot or "
            "a PROVEN total overwrite of the destination, and the uninitialized-origin "
            "walk deliberately does not chain through prior in-place writes"
        ),
        refuses=(
            "a partial overwrite, a destination holding real written data from an "
            "earlier in-place write (the TwoIndexCopyDim2 hole), and duplicate advanced "
            "indices that would fake total coverage by numel equality"
        ),
    ),
    CustomCheckExemption(
        func_name="index_put",
        check="_check_index_put_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "exempts the index_put destination only when the saved indices provably "
            "cover every destination element, so its prior values cannot survive"
        ),
        refuses=(
            "a partial write, accumulate semantics, and the values argument that "
            "supplies the written data"
        ),
    ),
    CustomCheckExemption(
        func_name="index_put_",
        check="_check_index_put_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place spelling of index_put bound to the same proof: total index "
            "coverage of the destination is required before the destination is excused"
        ),
        refuses=(
            "a partial write, accumulate semantics, and the values argument that "
            "supplies the written data"
        ),
    ),
    CustomCheckExemption(
        func_name="lstm",
        check="_check_lstm_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY the arg-1 (h_0, c_0) initial-state slot, keyed on recorded "
            "arg position and never on equality with a zero-initialized state. "
            "RESIDUAL: the initial state genuinely feeds the recurrence, so this is a "
            "vehicle/position blanket rather than a value-irrelevance proof; the state "
            "edge stays guarded by forward replay, which rebuilds the call from the "
            "saved state -- tracked as a B1-06 residual finding"
        ),
        refuses=(
            "the arg-0 input sequence and the weight arguments, and a data tensor that "
            "merely equals a zero-initialized h0 while sitting at another position"
        ),
    ),
    CustomCheckExemption(
        func_name="interpolate",
        check="_check_interpolate_exempt",
        contract=_C2,
        proof_kind="vehicle_limited",
        justification=(
            "exempts ONLY a tensor occupying the scale_factor slot (arg 2 or the "
            "scale_factor keyword), keyed on recorded position. A stepped scale factor "
            "changes the OUTPUT SHAPE rather than producing a comparable output, so the "
            "vehicle cannot express an in-domain alternative; forward replay still "
            "re-runs the real call from the saved scale factor"
        ),
        refuses=(
            "the arg-0 input being resampled, and a scale-factor-valued tensor recorded "
            "at any other position"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter",
        check="_check_scatter_or_index_domain_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "destination is excused only when the saved index provably covers every "
            "slot along the scatter dim (identity-checked against the saved "
            "destination), else the degenerate-index-domain proof must hold"
        ),
        refuses=(
            "a partially-covering index, reduce semantics, and the src argument that "
            "supplies the scattered values"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_",
        check="_check_scatter_or_index_domain_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place scatter bound to the same total-coverage / degenerate-domain proof as scatter"
        ),
        refuses=(
            "a partially-covering index, reduce semantics, and the src argument that "
            "supplies the scattered values"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_add",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "scatter_add accumulates, so the destination is never excused; only an "
            "index parent with NO admissible alternate value (domain n<=1, or zero "
            "in-range entries) is exempt"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="scatter_add_",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "in-place scatter_add bound to the same degenerate-index-domain proof; "
            "accumulation means the destination stays strictly tested"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="scatteradd",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "canonicalized scatter_add spelling bound to the same degenerate-domain "
            "proof, so the canonical and torch spellings cannot drift apart"
        ),
        refuses=(
            "any perturbable index domain, which is rotated in-domain instead, and "
            "every non-index parent of the call"
        ),
    ),
    CustomCheckExemption(
        func_name="embedding",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "the F2 tightening removed embedding's blanket index exemption; the index "
            "parent is now excused only when the vocabulary domain admits no alternate "
            "index at all"
        ),
        refuses=(
            "a perturbable index domain (indices are rotated in-domain by "
            "index_domain_rotation_values) and the weight parent"
        ),
    ),
    CustomCheckExemption(
        func_name="gather",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "gather's index blanket was likewise removed by F2; only a domain with no "
            "admissible alternate index is exempt"
        ),
        refuses=(
            "a perturbable index domain (rotated in-domain) and the arg-0 source "
            "whose values are gathered"
        ),
    ),
    CustomCheckExemption(
        func_name="index_select",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "index_select's index blanket was removed by F2; only a domain with no "
            "admissible alternate index is exempt"
        ),
        refuses=(
            "a perturbable index domain (rotated in-domain) and the arg-0 source "
            "whose values are selected"
        ),
    ),
    CustomCheckExemption(
        func_name="cross_entropy",
        check="_check_index_domain_degenerate",
        contract=_C2,
        proof_kind="domain_forced",
        justification=(
            "the target blanket was removed by F2; an all-ignore_index target (zero "
            "in-range entries) or a single-class domain admits no alternate target, "
            "which is what this row excuses"
        ),
        refuses=(
            "a perturbable target domain (rotated in-domain with out-of-range "
            "sentinels preserved) and the logits parent"
        ),
    ),
    CustomCheckExemption(
        func_name="where",
        check="_check_where_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "where parents are excused only when the SAVED condition proves the branch "
            "is never taken at any output element (or the one-arg index form applies); "
            "the condition itself is not blanket-exempt"
        ),
        refuses=(
            "a branch the saved condition selects anywhere, and a condition mask whose "
            "perturbation can flip a selected element"
        ),
    ),
    CustomCheckExemption(
        func_name="maskedfill",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "canonicalized masked_fill spelling bound to the saved-mask proof: input "
            "excused only when the mask is true everywhere, fill value only when it is "
            "false everywhere, mask only when input already equals the fill value"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="masked_fill",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "the F2 tightening removed masked_fill's structural mask blanket; the mask "
            "is now excused only when the saved input already equals the fill value "
            "everywhere, so every mask yields the same output"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="masked_fill_",
        check="_check_masked_fill_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "in-place masked_fill bound to the same saved-value proof as the out-of-place spellings"
        ),
        refuses=(
            "any mixed saved mask, where each parent still carries output values, and "
            "the mask itself unless input == value at every broadcast position"
        ),
    ),
    CustomCheckExemption(
        func_name="batch_norm",
        check="_check_norm_running_stat_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "EITHER (a) in TRAINING mode (proved from saved arg 5 being exactly True) "
            "running_mean/running_var at args 3-4 are update TARGETS: the normalized "
            "output is computed from batch statistics, not from the running buffers; "
            "OR (b) the perturbed parent(s) are a subset of {input, running_mean, "
            "running_var} (args 0/3/4) and the saved weight operand (arg 1, the real "
            "ATen position) is proved all-zero: weight=0 annihilates the whole "
            "normalized term (output = ((input-running_mean)/sqrt(running_var+eps))*"
            "weight+bias), in ANY mode -- timm's zero_init_last convention zero-inits "
            "exactly this operand for the last BatchNorm in many ResNet-family blocks"
        ),
        refuses=(
            "eval-mode running_mean/running_var perturbation when weight is nonzero, "
            "where the running stats DO determine the output and stay strictly tested; "
            "perturbing weight or bias themselves, even when weight is zero (weight "
            "moving off zero, or bias directly, both genuinely change the output); "
            "and every parent outside args 0/3/4"
        ),
    ),
    CustomCheckExemption(
        func_name="instance_norm",
        check="_check_norm_running_stat_exempt",
        contract=_C2,
        proof_kind="value_irrelevance_proved",
        justification=(
            "same running-stat update-target and zero-weight annihilator proofs as "
            "batch_norm, bound to the same predicate so the two cannot drift apart"
        ),
        refuses=(
            "eval-mode running_mean/running_var perturbation when weight is nonzero, "
            "where the running stats DO determine the output and stay strictly tested; "
            "perturbing weight or bias themselves, even when weight is zero; and every "
            "parent outside args 0/3/4"
        ),
    ),
)

#: Kwarg spellings that may match a structural POSITION. Every alias widens the
#: reach of a ``STRUCTURAL_ARG_POSITIONS`` row, so it is audited against that
#: row: an alias for an undeclared func or an undeclared position is red.
STRUCTURAL_KWARG_ALIAS_LEDGER: dict[str, dict[int, frozenset[str]]] = {
    "_pack_padded_sequence": {1: frozenset({"lengths"})},
    "_pad_packed_sequence": {1: frozenset({"lengths"})},
    "type_as": {1: frozenset({"tensor", "other"})},
    "reshape_as": {1: frozenset({"other"})},
    "reshapeas": {1: frozenset({"other"})},
    "view_as": {1: frozenset({"other"})},
    "viewas": {1: frozenset({"other"})},
    # torchvision coordinate args (b1p2 D2 narrowing): keyword spellings of
    # the SAME audited positions above, nothing wider.
    "nms": {0: frozenset({"boxes"})},
    "deform_conv2d": {1: frozenset({"offset"})},
    "roi_align": {1: frozenset({"boxes", "rois"})},
    "roi_pool": {1: frozenset({"boxes", "rois"})},
    "ps_roi_align": {1: frozenset({"boxes", "rois"})},
    "ps_roi_pool": {1: frozenset({"boxes", "rois"})},
}


def structural_position_violations(
    live: dict[str, set[int]],
    ledger: tuple[StructuralPositionExemption, ...],
) -> list[str]:
    """Return audit violations between a structural-position registry and its ledger.

    Parameters
    ----------
    live:
        Registry mapping func name -> structural arg positions.
    ledger:
        Audit records to hold it to.

    Returns
    -------
    list[str]
        One message per unledgered entry, phantom record, or position-set
        mismatch. Empty means the registry is fully audited.
    """

    ledgered = {entry.func_name: entry for entry in ledger}
    violations = [
        f"{name}: structural exemption with no audit record (positions {sorted(positions)})"
        for name, positions in sorted(live.items())
        if name not in ledgered
    ]
    violations.extend(
        f"{name}: audit record for an exemption the registry no longer declares"
        for name in sorted(set(ledgered) - set(live))
    )
    violations.extend(
        f"{name}: registry declares positions {sorted(live[name])} but the audit record "
        f"justifies {sorted(ledgered[name].positions)}"
        for name in sorted(set(live) & set(ledgered))
        if frozenset(live[name]) != ledgered[name].positions
    )
    return violations


def custom_check_violations(
    live: dict[str, object],
    ledger: tuple[CustomCheckExemption, ...],
    module: object,
) -> list[str]:
    """Return audit violations between the custom-check registry and its ledger.

    Parameters
    ----------
    live:
        Registry mapping func name -> bound check callable.
    ledger:
        Audit records to hold it to.
    module:
        Module the cited check symbols must resolve in.

    Returns
    -------
    list[str]
        One message per unledgered entry, phantom record, unresolvable citation,
        or check rebound to a different predicate.
    """

    ledgered = {entry.func_name: entry for entry in ledger}
    violations = [
        f"{name}: custom exemption check with no audit record"
        for name in sorted(set(live) - set(ledgered))
    ]
    violations.extend(
        f"{name}: audit record for a custom check the registry no longer declares"
        for name in sorted(set(ledgered) - set(live))
    )
    for name in sorted(set(live) & set(ledgered)):
        cited = getattr(module, ledgered[name].check, None)
        if cited is None:
            violations.append(f"{name}: audit record cites missing check {ledgered[name].check!r}")
        elif cited is not live[name]:
            violations.append(
                f"{name}: registry is bound to {getattr(live[name], '__name__', live[name])!r} "
                f"but the audit record justifies {ledgered[name].check!r}"
            )
    return violations


def returns_true_unconditionally(source: str) -> bool:
    """Return whether a function's source returns True with no condition at all.

    The planted third defect of finding B1-06 was a custom check whose body is
    ``return True``: a blanket wearing a predicate's signature. A top-level
    (non-branched) ``return True`` is exactly that shape, so it is refused
    structurally rather than by reviewer attention.

    Parameters
    ----------
    source:
        Source text of a single function definition.

    Returns
    -------
    bool
        True when a ``return True`` sits at the function body's top level.
    """

    tree = ast.parse(textwrap.dedent(source))
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return False
    return any(
        isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is True
        for statement in definition.body
    )


def test_structural_arg_positions_have_a_per_entry_audit_record() -> None:
    """Every structural-position row is justified, by name AND by position set."""

    violations = structural_position_violations(
        dict(ex.STRUCTURAL_ARG_POSITIONS), STRUCTURAL_POSITION_LEDGER
    )
    assert not violations, (
        "STRUCTURAL_ARG_POSITIONS is not fully audited -- each row needs the contract "
        "clause it sits outside, the ground it stands on, and the case it must still "
        "let fail:\n" + "\n".join(violations)
    )


def test_custom_exemption_checks_have_a_per_entry_audit_record() -> None:
    """Every custom-check row is justified and bound to the predicate it cites."""

    violations = custom_check_violations(dict(ex.CUSTOM_EXEMPTION_CHECKS), CUSTOM_CHECK_LEDGER, ex)
    assert not violations, (
        "CUSTOM_EXEMPTION_CHECKS is not fully audited -- a new op or a rebound check "
        "needs its own audit record:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "entry",
    STRUCTURAL_POSITION_LEDGER + CUSTOM_CHECK_LEDGER,
    ids=lambda entry: entry.func_name,
)
def test_registry_audit_record_is_well_formed(
    entry: StructuralPositionExemption | CustomCheckExemption,
) -> None:
    """Each per-entry record cites a known clause, a ground, and a refusal."""

    assert entry.contract in VALIDATION_CONTRACTS, f"{entry.func_name}: uncited contract"
    assert entry.proof_kind in _PROOF_KINDS, f"{entry.func_name}: unknown proof kind"
    assert len(entry.justification.strip()) >= 60, (
        f"{entry.func_name}: the justification must prove value-irrelevance from the "
        "call (or name the vehicle limit), not restate the op's name"
    )
    assert len(entry.refuses.strip()) >= 30, (
        f"{entry.func_name}: 'refuses' must name the case this entry still lets fail; "
        "an entry with nothing it refuses is a blanket, not a carve-out"
    )
    if entry.proof_kind == "vehicle_limited":
        assert "replay" in entry.justification.lower(), (
            f"{entry.func_name}: a vehicle-limited exemption is NOT a proof, so it must "
            "name the check that still guards the value edge"
        )


def test_structural_kwarg_aliases_are_declared_by_an_audited_position() -> None:
    """A kwarg alias can only widen a position its own audit record justifies."""

    ledgered = {entry.func_name: entry for entry in STRUCTURAL_POSITION_LEDGER}
    live_aliases = {
        name: {position: frozenset(aliases) for position, aliases in by_position.items()}
        for name, by_position in ex.STRUCTURAL_ARG_KWARG_ALIASES.items()
    }
    assert live_aliases == STRUCTURAL_KWARG_ALIAS_LEDGER, (
        "STRUCTURAL_ARG_KWARG_ALIASES changed without an audit record; a kwarg alias "
        "extends a structural exemption to a keyword spelling"
    )
    for name, by_position in live_aliases.items():
        assert name in ledgered, f"{name}: kwarg alias for an unaudited structural func"
        for position in by_position:
            assert position in ledgered[name].positions, (
                f"{name}: kwarg alias widens position {position}, which the audit "
                "record does not justify as structural"
            )


def test_no_custom_exemption_check_is_an_unconditional_blanket() -> None:
    """No registered check can excuse a parent with no condition at all."""

    import inspect

    for func_name, check in ex.CUSTOM_EXEMPTION_CHECKS.items():
        assert getattr(check, "__name__", "<lambda>") != "<lambda>", (
            f"{func_name}: custom checks must be named module-level predicates so the "
            "audit record can cite and pin them"
        )
        assert not returns_true_unconditionally(inspect.getsource(check)), (
            f"{func_name}: custom check returns True with no condition -- that is a "
            "blanket exemption wearing a predicate's signature"
        )


def test_inline_pre_perturbation_predicates_still_exist() -> None:
    """The two non-registry sources of ``pre_perturbation_exemption`` resolve.

    ``_check_perturbation_exemptions`` also excuses empty-tensor parents and
    pure ``out=`` destinations inline. Neither is a registry, so neither can
    grow an unaudited ENTRY, but the citations must not rot.
    """

    from torchlens.validation import core as validation_core

    assert hasattr(validation_core, "_perturbed_parents_only_occupy_out_kwarg")
    assert hasattr(validation_core, "_check_perturbation_exemptions")


class TestRegistryLedgerMechanismIsRedCapable:
    """Replay finding B1-06's three plants and prove each one is now reported."""

    def test_planting_a_value_dependent_structural_position_is_reported(self) -> None:
        """``take_along_dim`` arg 1 is the index tensor -- an unaudited entry."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["take_along_dim"] = {1}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("take_along_dim" in violation for violation in violations)

    def test_planting_the_where_condition_mask_is_reported(self) -> None:
        """``where`` arg 0 is the condition mask -- an unaudited entry."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["where"] = {0}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("where" in violation for violation in violations)

    def test_widening_an_audited_position_set_is_reported(self) -> None:
        """Adding the SOURCE argument to ``copy_`` is a widening, not a rename."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted["copy_"] = {0, 1}
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("copy_" in violation for violation in violations)

    def test_removing_a_registry_entry_is_reported_as_a_phantom(self) -> None:
        """A narrowing must also update the audit record, in the same change."""

        planted = dict(ex.STRUCTURAL_ARG_POSITIONS)
        planted.pop("type_as")
        violations = structural_position_violations(planted, STRUCTURAL_POSITION_LEDGER)
        assert any("type_as" in violation for violation in violations)

    def test_planting_an_unledgered_custom_check_is_reported(self) -> None:
        """A new op in the custom registry needs its own audit record."""

        planted = dict(ex.CUSTOM_EXEMPTION_CHECKS)
        planted["take_along_dim"] = ex._check_index_domain_degenerate
        violations = custom_check_violations(planted, CUSTOM_CHECK_LEDGER, ex)
        assert any("take_along_dim" in violation for violation in violations)

    def test_rebinding_an_audited_check_to_another_predicate_is_reported(self) -> None:
        """Swapping ``masked_fill``'s proof for a laxer one is reported."""

        planted = dict(ex.CUSTOM_EXEMPTION_CHECKS)
        planted["masked_fill"] = ex._check_getitem_exempt
        violations = custom_check_violations(planted, CUSTOM_CHECK_LEDGER, ex)
        assert any("masked_fill" in violation for violation in violations)

    def test_unconditional_return_true_check_is_detected(self) -> None:
        """The planted blanket shape is caught structurally."""

        assert returns_true_unconditionally(
            "def blanket(trace, op, layers_to_perturb):\n    return True\n"
        )

    def test_a_real_branching_check_is_not_flagged_as_a_blanket(self) -> None:
        """A predicate that can return False is not a blanket."""

        assert not returns_true_unconditionally(
            "def real(trace, op, layers):\n"
            "    if not layers:\n"
            "        return False\n"
            "    if layers == ['x']:\n"
            "        return True\n"
            "    return False\n"
        )
