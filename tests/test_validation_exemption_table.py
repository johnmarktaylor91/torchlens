"""Governance tests for the declared validation exemption table.

``torchlens/validation/_exemption_table.py`` is the single declaration of every
exemption the validation tripwire admits. These tests pin its contents: the
run-time registries are derived from it and nothing else, every registry member
the table declared at consolidation time is still present with the same value,
every guard symbol resolves, every paired test exists, and the code-keyed rows
correspond one to one with the audit ledger in
``test_validation_exemption_ledger.py`` and with the reason codes the pipeline
can emit. Removing a row fails a test here; the table never widens a row to
make a test pass (``AGENTS.md``, Validation Integrity).
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _source_corpus import package_files, package_source
from test_validation_exemption_ledger import (
    EXEMPTION_LEDGER,
    core_exempting_reasons,
    posthoc_exempting_reasons,
)

from torchlens.validation import _exemptions as ex
from torchlens.validation._exemption_schema import DECISION_KINDS, REGISTRY_KINDS
from torchlens.validation._exemption_table import (
    EXEMPTION_TABLE,
    ExemptionRow,
    declared_custom_check_symbols,
    declared_structural_kwarg_aliases,
    declared_structural_positions,
    exemption_row,
    skip_perturbation_justifications,
    skip_validation_justifications,
)
from torchlens.validation._exemptions import (
    _check_index_domain_degenerate,
    _check_interpolate_exempt,
    _check_norm_zero_weight_annihilates,
    _posthoc_overwrite_decision,
)
from torchlens.validation._replay_warnings import replay_warnings_ignored

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THIS_FILE = "tests/test_validation_exemption_table.py::"

# ---------------------------------------------------------------------------
# The registry contents at consolidation (base b00f4010f). Each member is one
# parametrized case: the live registry must still hold it with this value and
# the table must declare it. Removing a table row removes the live member.
# ---------------------------------------------------------------------------
_SKIP_VALIDATION_MEMBERS: tuple[str, ...] = (
    "empty_like",
    "new",
    "new_empty",
    "new_empty_strided",
    "newempty",
    "newemptystrided",
)
_SKIP_PERTURBATION_MEMBERS: tuple[str, ...] = (
    "new_zeros",
    "new_ones",
    "zero_",
    "zeros_like",
    "ones_like",
    "rand_like",
    "randn_like",
    "exponential_",
)
_STRUCTURAL_MEMBERS: dict[str, frozenset[int]] = {
    "new": frozenset({0}),
    "copy_": frozenset({0}),
    "_foreach_copy_": frozenset({0}),
    "foreachcopy": frozenset({0}),
    "fill_": frozenset({0}),
    "set_": frozenset({0}),
    "expand_as": frozenset({1}),
    "expandas": frozenset({1}),
    "reshape_as": frozenset({1}),
    "reshapeas": frozenset({1}),
    "view_as": frozenset({1}),
    "viewas": frozenset({1}),
    "resize_as_": frozenset({1}),
    "_pack_padded_sequence": frozenset({1}),
    "_pad_packed_sequence": frozenset({1}),
    "type_as": frozenset({1}),
    "new_tensor": frozenset({0}),
    "newtensor": frozenset({0}),
    "new_full": frozenset({0}),
    "newfull": frozenset({0}),
    "new_zeros": frozenset({0}),
    "newzeros": frozenset({0}),
    "new_ones": frozenset({0}),
    "newones": frozenset({0}),
    "nms": frozenset({0}),
    "deform_conv2d": frozenset({1}),
    "roi_align": frozenset({1}),
    "roi_pool": frozenset({1}),
    "ps_roi_align": frozenset({1}),
    "ps_roi_pool": frozenset({1}),
}
_KWARG_ALIAS_MEMBERS: dict[str, dict[int, frozenset[str]]] = {
    "_pack_padded_sequence": {1: frozenset({"lengths"})},
    "_pad_packed_sequence": {1: frozenset({"lengths"})},
    "type_as": {1: frozenset({"other", "tensor"})},
    "reshape_as": {1: frozenset({"other"})},
    "reshapeas": {1: frozenset({"other"})},
    "view_as": {1: frozenset({"other"})},
    "viewas": {1: frozenset({"other"})},
    "resize_as_": {1: frozenset({"the_template"})},
    "nms": {0: frozenset({"boxes"})},
    "deform_conv2d": {1: frozenset({"offset"})},
    "roi_align": {1: frozenset({"boxes", "rois"})},
    "roi_pool": {1: frozenset({"boxes", "rois"})},
    "ps_roi_align": {1: frozenset({"boxes", "rois"})},
    "ps_roi_pool": {1: frozenset({"boxes", "rois"})},
}
_CUSTOM_MEMBERS: dict[str, str] = {
    "meshgrid": "_check_zipped_sibling_exempt",
    "broadcast_tensors": "_check_zipped_sibling_exempt",
    "broadcasttensors": "_check_zipped_sibling_exempt",
    "__getitem__": "_check_getitem_exempt",
    "__setitem__": "_check_setitem_exempt",
    "index_put": "_check_index_put_exempt",
    "index_put_": "_check_index_put_exempt",
    "lstm": "_check_lstm_exempt",
    "gru": "check_packed_rnn_batch_sizes_exempt",
    "rnn_tanh": "check_packed_rnn_batch_sizes_exempt",
    "rnn_relu": "check_packed_rnn_batch_sizes_exempt",
    "expand": "check_expand_pinned_size_exempt",
    "interpolate": "_check_interpolate_exempt",
    "scatter": "_check_scatter_or_index_domain_exempt",
    "scatter_": "_check_scatter_or_index_domain_exempt",
    "scatter_add": "_check_index_domain_degenerate",
    "scatter_add_": "_check_index_domain_degenerate",
    "scatteradd": "_check_index_domain_degenerate",
    "embedding": "_check_index_domain_degenerate",
    "gather": "_check_index_domain_degenerate",
    "index_select": "_check_index_domain_degenerate",
    "cross_entropy": "_check_index_domain_degenerate",
    "where": "_check_where_exempt",
    "maskedfill": "_check_masked_fill_exempt",
    "masked_fill": "_check_masked_fill_exempt",
    "masked_fill_": "_check_masked_fill_exempt",
    "batch_norm": "_check_norm_running_stat_exempt",
    "instance_norm": "_check_norm_running_stat_exempt",
}

_MEMBER_CASES: tuple[str, ...] = (
    *(f"skip_validation:{name}" for name in _SKIP_VALIDATION_MEMBERS),
    *(f"skip_perturbation:{name}" for name in _SKIP_PERTURBATION_MEMBERS),
    *(f"structural_position:{name}" for name in _STRUCTURAL_MEMBERS),
    *(f"custom_check:{name}" for name in _CUSTOM_MEMBERS),
)

# Rows whose negative test is a declared gap. Pinned exactly: a new gap cannot
# appear without being declared here, and closing one must remove it here.
_DECLARED_NEGATIVE_GAPS: frozenset[str] = frozenset(
    {
        *(f"skip_perturbation:{name}" for name in _SKIP_PERTURBATION_MEMBERS),
        "structural_position:nms",
        "structural_position:deform_conv2d",
        "structural_position:roi_align",
        "structural_position:roi_pool",
        "structural_position:ps_roi_align",
        "structural_position:ps_roi_pool",
    }
)


def _row_ids(row: ExemptionRow) -> str:
    return f"{row.kind}:{row.name}"


@pytest.mark.parametrize("case", _MEMBER_CASES)
def test_registry_member_is_declared(case: str) -> None:
    """Each registry member pinned at consolidation is live and table-declared."""

    kind, _, name = case.partition(":")
    row = exemption_row(name, kind)
    assert row.kind == kind
    if kind == "skip_validation":
        assert ex.SKIP_VALIDATION_ENTIRELY[name] == row.reason
    elif kind == "skip_perturbation":
        assert ex.SKIP_PERTURBATION_ENTIRELY[name] == row.reason
    elif kind == "structural_position":
        assert ex.STRUCTURAL_ARG_POSITIONS[name] == set(_STRUCTURAL_MEMBERS[name])
        assert row.positions == _STRUCTURAL_MEMBERS[name]
        expected_aliases = _KWARG_ALIAS_MEMBERS.get(name)
        if expected_aliases is None:
            assert name not in ex.STRUCTURAL_ARG_KWARG_ALIASES
            assert row.kwarg_aliases == ()
        else:
            live = ex.STRUCTURAL_ARG_KWARG_ALIASES[name]
            assert {pos: frozenset(names) for pos, names in live.items()} == expected_aliases
    else:
        assert ex.CUSTOM_EXEMPTION_CHECKS[name] is getattr(ex, _CUSTOM_MEMBERS[name])
        assert row.guard_symbol == _CUSTOM_MEMBERS[name]


def test_registries_have_no_undeclared_members() -> None:
    """The live registries hold exactly the pinned members: nothing was added."""

    assert set(ex.SKIP_VALIDATION_ENTIRELY) == set(_SKIP_VALIDATION_MEMBERS)
    assert set(ex.SKIP_PERTURBATION_ENTIRELY) == set(_SKIP_PERTURBATION_MEMBERS)
    assert set(ex.STRUCTURAL_ARG_POSITIONS) == set(_STRUCTURAL_MEMBERS)
    assert set(ex.STRUCTURAL_ARG_KWARG_ALIASES) == set(_KWARG_ALIAS_MEMBERS)
    assert set(ex.CUSTOM_EXEMPTION_CHECKS) == set(_CUSTOM_MEMBERS)


def test_registries_are_derived_from_the_table_only() -> None:
    """Every run-time registry equals its table view, object for object."""

    assert skip_validation_justifications() == ex.SKIP_VALIDATION_ENTIRELY
    assert skip_perturbation_justifications() == ex.SKIP_PERTURBATION_ENTIRELY
    assert declared_structural_positions() == ex.STRUCTURAL_ARG_POSITIONS
    assert declared_structural_kwarg_aliases() == ex.STRUCTURAL_ARG_KWARG_ALIASES
    symbols = declared_custom_check_symbols()
    assert set(ex.CUSTOM_EXEMPTION_CHECKS) == set(symbols)
    for spelling, symbol in symbols.items():
        assert ex.CUSTOM_EXEMPTION_CHECKS[spelling] is getattr(ex, symbol)


def test_table_rows_are_unique_and_well_formed() -> None:
    """Names are unique within a kind and every row carries its proof fields.

    A spelling may appear under two kinds (``new`` is both uninitialized-by-design
    and a structural template; ``new_zeros`` skips perturbation and is structural):
    the registries are disjoint by kind, so uniqueness is per kind.
    """

    seen: set[tuple[str, str]] = set()
    for row in EXEMPTION_TABLE:
        assert row.kind in REGISTRY_KINDS | DECISION_KINDS
        assert row.admits.strip() and row.reason.strip()
        assert ":" in row.guard
        assert "::" in row.removal_test
        for name in row.names:
            assert (row.kind, name) not in seen, (row.kind, name)
            seen.add((row.kind, name))
        if row.kind == "structural_position":
            assert row.positions
        else:
            assert not row.positions and not row.kwarg_aliases
    gaps = {_row_ids(row) for row in EXEMPTION_TABLE if row.negative_test is None}
    assert gaps == _DECLARED_NEGATIVE_GAPS


@pytest.mark.parametrize("row", EXEMPTION_TABLE, ids=_row_ids)
def test_guard_symbol_resolves(row: ExemptionRow) -> None:
    """The guard named by each row exists where the row says it does."""

    module = importlib.import_module(row.guard_module)
    assert hasattr(module, row.guard_symbol), row.guard


def _test_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assert_node_id_exists(node_id: str) -> None:
    file_part, _, test_part = node_id.partition("::")
    test_name = test_part.partition("[")[0]
    path = _REPO_ROOT / file_part
    assert path.is_file(), node_id
    assert test_name in _test_function_names(path), node_id


@pytest.mark.parametrize("row", EXEMPTION_TABLE, ids=_row_ids)
def test_paired_tests_exist(row: ExemptionRow) -> None:
    """Both paired tests of each row name a real test function."""

    _assert_node_id_exists(row.removal_test)
    if row.negative_test is not None:
        _assert_node_id_exists(row.negative_test)


def test_registry_rows_point_removal_at_the_member_test() -> None:
    """Registry rows are removal-tested by the member case for their first spelling."""

    for row in EXEMPTION_TABLE:
        if row.kind in REGISTRY_KINDS:
            expected = f"{_THIS_FILE}test_registry_member_is_declared[{row.kind}:{row.name}]"
            assert row.removal_test == expected


def test_decision_rows_match_the_exemption_ledger() -> None:
    """Code-keyed rows and the audit ledger agree on codes, tiers and guards."""

    ledger = {entry.code: entry for entry in EXEMPTION_LEDGER}
    rows = {
        row.name: row
        for row in EXEMPTION_TABLE
        if row.kind in {"early_exit", "posthoc", "recorder"}
    }
    assert set(rows) == set(ledger)
    for code, row in rows.items():
        assert row.kind == ledger[code].tier, code
        assert row.guard == ledger[code].proof, code
        clause = ledger[code].contract.split(" ", 1)[1]
        assert row.reason.startswith(clause), (code, row.reason, clause)


def test_every_emitted_exemption_code_has_a_table_row() -> None:
    """No exemption can be emitted by the package without a table row."""

    emitted: set[str] = set()
    for path in package_files():
        if path.parts[-2] != "validation":
            continue
        source = package_source(path)
        emitted |= posthoc_exempting_reasons(source)
        if path.name != "exemptions.py":
            emitted |= core_exempting_reasons(source)
    declared = {row.name for row in EXEMPTION_TABLE}
    assert emitted <= declared, sorted(emitted - declared)


def test_table_modules_emit_no_decisions() -> None:
    """The table declares exemptions; it never constructs one."""

    for name in (
        "_exemption_schema.py",
        "_exemption_table.py",
        "_exemption_rows_ops.py",
        "_exemption_rows_decisions.py",
    ):
        source = (_REPO_ROOT / "torchlens" / "validation" / name).read_text(encoding="utf-8")
        assert "PosthocPerturbDecision(" not in source
        assert ".exempted(" not in source
        assert not posthoc_exempting_reasons(source)


# ---------------------------------------------------------------------------
# Paired tests that rows cite from this module.
# ---------------------------------------------------------------------------


def _fake_layer(**fields: Any) -> Any:
    return SimpleNamespace(**fields)


def test_scalar_setitem_full_overwrite_is_exempted() -> None:
    """A scalar written over the whole destination admits the destination edge."""

    destination = torch.zeros(4, 4)
    layer = _fake_layer(
        func_name="__setitem__",
        parent_arg_positions={"args": {0: "destination"}, "kwargs": {}},
    )
    decision = _posthoc_overwrite_decision(layer, ["destination"], (destination, slice(None), 5.0))
    assert decision.exempt
    assert decision.reason == "scalar_destination_overwrite"


def test_replay_warnings_filter_is_scoped_to_the_context() -> None:
    """Warnings are silenced inside the replay context and live again outside it."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with replay_warnings_ignored():
            warnings.warn("replay noise", UserWarning, stacklevel=1)
        with pytest.raises(UserWarning):
            warnings.warn("replay noise", UserWarning, stacklevel=1)


# ---------------------------------------------------------------------------
# Gate predicates: branches the mutation campaign found unobserved.
# ---------------------------------------------------------------------------


def _embedding_layer(index: torch.Tensor, rows: int) -> Any:
    return _fake_layer(
        func_name="embedding",
        saved_args=(torch.zeros(rows, 3), index),
        saved_kwargs={},
        parent_arg_positions={"args": {0: "weight", 1: "index"}, "kwargs": {}},
    )


def test_index_domain_degenerate_requires_exactly_one_perturbed_parent() -> None:
    layer = _embedding_layer(torch.tensor([0, 0]), rows=1)
    assert _check_index_domain_degenerate(None, layer, ["index"]) is True
    assert _check_index_domain_degenerate(None, layer, ["index", "weight"]) is False
    assert _check_index_domain_degenerate(None, layer, []) is False


def test_index_domain_degenerate_refuses_a_non_index_parent() -> None:
    layer = _embedding_layer(torch.tensor([0, 0]), rows=1)
    assert _check_index_domain_degenerate(None, layer, ["weight"]) is False


def test_index_domain_degenerate_needs_an_integer_index_and_a_bound() -> None:
    float_index = _embedding_layer(torch.tensor([0.0, 0.0]), rows=1)
    assert _check_index_domain_degenerate(None, float_index, ["index"]) is False
    no_bound = _fake_layer(
        func_name="gather",
        saved_args=(torch.zeros(2, 2), None, torch.tensor([[0]])),
        saved_kwargs={},
        parent_arg_positions={"args": {2: "index"}, "kwargs": {}},
    )
    assert _check_index_domain_degenerate(None, no_bound, ["index"]) is False


def test_index_domain_degenerate_is_exact_at_the_two_row_boundary() -> None:
    one_row = _embedding_layer(torch.tensor([0, 0]), rows=1)
    assert _check_index_domain_degenerate(None, one_row, ["index"]) is True
    two_rows_live = _embedding_layer(torch.tensor([0, 1]), rows=2)
    assert _check_index_domain_degenerate(None, two_rows_live, ["index"]) is False
    two_rows_one_live = _embedding_layer(torch.tensor([-1, 1, 7]), rows=2)
    assert _check_index_domain_degenerate(None, two_rows_one_live, ["index"]) is False
    two_rows_all_out = _embedding_layer(torch.tensor([-1, 2, 7]), rows=2)
    assert _check_index_domain_degenerate(None, two_rows_all_out, ["index"]) is True


def _interpolate_layer(args: tuple[Any, ...], kwargs: dict[str, Any], positions: dict) -> Any:
    return _fake_layer(
        func_name="interpolate",
        saved_args=args,
        saved_kwargs=kwargs,
        parent_arg_positions=positions,
    )


def test_interpolate_positional_scale_factor_must_be_present_and_set() -> None:
    x = torch.zeros(1, 1, 4)
    present = _interpolate_layer((x, None, 2.0), {}, {"args": {2: "scale"}, "kwargs": {}})
    assert _check_interpolate_exempt(None, present, ["scale"]) is True
    none_value = _interpolate_layer((x, None, None), {}, {"args": {2: "scale"}, "kwargs": {}})
    assert _check_interpolate_exempt(None, none_value, ["scale"]) is False
    short_args = _interpolate_layer((x, None), {}, {"args": {2: "scale"}, "kwargs": {}})
    assert _check_interpolate_exempt(None, short_args, ["scale"]) is False
    assert _check_interpolate_exempt(None, present, ["scale", "x"]) is False


def test_interpolate_keyword_scale_factor_needs_a_matching_parent_and_value() -> None:
    x = torch.zeros(1, 1, 4)
    keyed = _interpolate_layer(
        (x,), {"scale_factor": 2.0}, {"args": {0: "x"}, "kwargs": {"scale_factor": "scale"}}
    )
    assert _check_interpolate_exempt(None, keyed, ["scale"]) is True
    assert _check_interpolate_exempt(None, keyed, ["x"]) is False
    unset = _interpolate_layer(
        (x,), {"scale_factor": None}, {"args": {0: "x"}, "kwargs": {"scale_factor": "scale"}}
    )
    assert _check_interpolate_exempt(None, unset, ["scale"]) is False


def test_norm_zero_weight_proof_refuses_an_empty_weight() -> None:
    """An element-less weight proves nothing: ``all(weight == 0)`` is vacuous."""

    x = torch.zeros(2, 0, 3)
    layer = _fake_layer(
        func_name="batch_norm",
        saved_args=(x, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)),
        saved_kwargs={},
        out=torch.zeros(2, 0, 3),
        parent_arg_positions={"args": {0: "x"}, "kwargs": {}},
    )
    assert _check_norm_zero_weight_annihilates(layer, ["x"]) is False
