"""Checkpoint live-ref guard: cross-member parameter reads refuse typed (A-CKPT).

TorchLens records WHICH parameter a run used, not its bytes: a parameter read
resolves through a live handle to the model and returns TODAY'S weights, not
the weights at capture time. Two Bundle members captured from one mutated
model therefore reported identical weights on five public reads
(``weight_norm_diff``, ``diff_pair``, ``aggregate``, ``out``, ``grad``) --
confident wrong numbers -- and after a save/load round trip the same reads
degraded silently to NaN / empty / ``None`` instead of refusing.

The fix (foldB MEMO s4.2, D7): every cross-member parameter value /
difference / trajectory read refuses BEFORE tensor lookup with the stable
code ``checkpoint_series_live_params``, keyed on the CLAIM (a cross-member,
cross-time parameter value), never on Python object identity. The one guard
site is the ``_TensorBearing._tensor_dict`` funnel for Param members. Beside
the refusal, ``Param.value_basis`` is a derived, read-time, persisted-nowhere
disclosure of where a parameter value comes from: ``live_ref`` or
``absent(not_persisted)`` (``snapshot`` arrives only with capture-time
parameter snapshots, R8(b)).

Spellings here are DOCUMENTED-UNSTABLE pending the naming session; the code
``checkpoint_series_live_params`` is the stable branch surface.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.data_classes.param import Param, ParamValueBasis
from torchlens.errors import CheckpointSeriesLiveParamsError

_CODE = "checkpoint_series_live_params"
_WEIGHT = "0.weight"


def _toy_model() -> nn.Sequential:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    with torch.no_grad():
        # Keep the ReLU alive for the seeded input so the SGD step in the
        # stepped_pair fixture provably moves the first weight.
        model[0].bias.fill_(2.0)
    return model


def _toy_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4)


@pytest.fixture(scope="module")
def stepped_pair() -> Any:
    """One model object captured twice with a real SGD step between captures.

    The exact checkpoint-series shape from the measured defect: the SAME live
    model backs both members, so live-handle reads would report identical
    weights for both.
    """

    model = _toy_model()
    x = _toy_input()
    snapshot_before = model[0].weight.detach().clone()
    trace_before = tl.trace(model, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
    model(x).sum().backward()
    optimizer.step()
    trace_after = tl.trace(model, x)
    snapshot_after = model[0].weight.detach().clone()
    try:
        yield {
            "model": model,
            "x": x,
            "trace_before": trace_before,
            "trace_after": trace_after,
            "snapshot_before": snapshot_before,
            "snapshot_after": snapshot_after,
        }
    finally:
        trace_before.cleanup()
        trace_after.cleanup()


@pytest.fixture()
def ckpt_bundle(stepped_pair: dict[str, Any]) -> Any:
    return tl.bundle({"ck0": stepped_pair["trace_before"], "ck64": stepped_pair["trace_after"]})


def _assert_checkpoint_refusal(excinfo: pytest.ExceptionInfo[Any], *members: str) -> None:
    """Assert the refusal carries the code, the member names, and teaches."""

    exc = excinfo.value
    assert exc.fields["code"] == _CODE
    for member in members:
        assert member in exc.fields["members"]
        assert repr(member) in str(exc)
    message = str(exc)
    # The mechanism, in one plain sentence.
    assert "not its bytes" in message
    # Both remedies.
    assert "snapshot parameters at capture" in message
    assert "immutable checkpoint" in message
    # Close with what still works.
    assert "ordering is still valid" in message
    # Never these (memo wording rules).
    assert "models differ" not in message
    assert "relationship too weak" not in message
    assert "reload first" not in message
    assert "0x" not in message
    assert exc.fields["remedy"]


# ---------------------------------------------------------------------------
# The defect mechanism, proven with an independent oracle (manual snapshots)
# ---------------------------------------------------------------------------


def test_live_handles_prove_the_wrong_answer_shape(stepped_pair: dict[str, Any]) -> None:
    """Both members' live reads serve the SAME post-step bytes; capture-time truth moved."""

    moved = torch.linalg.vector_norm(
        stepped_pair["snapshot_after"] - stepped_pair["snapshot_before"]
    ).item()
    assert moved > 0.0, "the SGD step must actually move the weight"
    value_before = stepped_pair["trace_before"].params[_WEIGHT].value
    value_after = stepped_pair["trace_after"].params[_WEIGHT].value
    assert value_before is value_after, "both members resolve one live parameter object"
    assert torch.equal(value_before.detach(), stepped_pair["snapshot_after"])


# ---------------------------------------------------------------------------
# s4.2 test 3: bare-trace basis leg (no Bundle, no relation row)
# ---------------------------------------------------------------------------


def test_bare_trace_basis_discloses_live_ref(stepped_pair: dict[str, Any]) -> None:
    """A live capture's parameter value basis is live_ref, with the teaching gloss."""

    basis = stepped_pair["trace_before"].params[_WEIGHT].value_basis
    assert isinstance(basis, ParamValueBasis)
    assert basis.basis == "live_ref"
    assert basis.reason is None
    assert str(basis) == "live_ref"
    assert not basis.is_immutable
    assert "may have moved since capture" in basis.description


def test_bare_trace_at_capture_read_is_disclosed_not_silent(
    stepped_pair: dict[str, Any],
) -> None:
    """Capture, mutate, read: the value is TODAY'S bytes and the basis says so."""

    param = stepped_pair["trace_before"].params[_WEIGHT]
    assert torch.equal(param.value.detach(), stepped_pair["snapshot_after"])
    assert param.value_basis.basis == "live_ref"


# ---------------------------------------------------------------------------
# s4.2 test 4: round-trip legs (refusal survives save/load; basis is typed)
# ---------------------------------------------------------------------------


def test_loaded_basis_is_absent_not_persisted_never_bare_none(tmp_path: Any) -> None:
    """After save/load the basis reports absent(not_persisted) beside value None."""

    model = _toy_model()
    trace = tl.trace(model, _toy_input())
    path = tmp_path / "one.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    param = loaded.params[_WEIGHT]
    assert param.value is None  # documented live-handle contract, unchanged
    basis = param.value_basis
    assert basis is not None
    assert basis.basis == "absent"
    assert basis.reason == "not_persisted"
    assert str(basis) == "absent(not_persisted)"
    assert not basis.is_immutable


def test_refusal_survives_member_save_load_before_nan_empty_degradation(
    stepped_pair: dict[str, Any], tmp_path: Any
) -> None:
    """Loaded members refuse typed BEFORE the historical nan/empty degradation."""

    path0 = tmp_path / "ck0.tlspec"
    path1 = tmp_path / "ck64.tlspec"
    tl.save(stepped_pair["trace_before"], path0)
    tl.save(stepped_pair["trace_after"], path1)
    loaded = tl.bundle({"ck0": tl.load(path0), "ck64": tl.load(path1)})
    view = loaded.params[_WEIGHT]
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        _ = view.weight_norm_diff
    _assert_checkpoint_refusal(excinfo, "ck0", "ck64")
    with pytest.raises(CheckpointSeriesLiveParamsError):
        view.diff_pair()
    # The absent leg names the degradation shape the refusal replaces.
    assert "nothing to compare" in str(excinfo.value)


def test_refusal_survives_bundle_save_load(ckpt_bundle: Any, tmp_path: Any) -> None:
    """A bundle.save round trip keeps the refusal armed on the loaded Bundle."""

    path = tmp_path / "series.tlspec"
    ckpt_bundle.save(path)
    loaded = tl.load(path)
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        loaded.params[_WEIGHT].aggregate("std")
    _assert_checkpoint_refusal(excinfo, "ck0", "ck64")


# ---------------------------------------------------------------------------
# s4.2 test 5: no-overfire legs
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_ordinary_bundle_activation_reads_do_not_trigger() -> None:
    """A same-checkpoint bundle with no version claim triggers nothing."""

    model = _toy_model()
    x = _toy_input()
    bundle = tl.bundle({"a": tl.trace(model, x), "b": tl.trace(model, x)})
    matrix = bundle.compare_at("relu_1_2")
    assert tuple(matrix.shape) == (2, 2)
    view = bundle.node("relu_1_2")
    assert tuple(view.diff_pair().shape) == (2, 2)
    assert view.aggregate("mean").shape == view.members["a"].out.shape
    assert bundle.diff_pair("a", "b")


def test_version_row_still_orders_members(ckpt_bundle: Any) -> None:
    """A version-axis relation row still ORDERS members; only value claims refuse."""

    related = ckpt_bundle.relate(
        {"kind": "successor_of", "from": "ck64", "to": "ck0", "params": {}}
    )
    rows = related.member_relations
    assert len(rows) == 1
    assert rows[0].kind == "successor_of"
    assert rows[0].named_members() == ("ck64", "ck0")
    # The ordering claim stands; the parameter value claim still refuses.
    with pytest.raises(CheckpointSeriesLiveParamsError):
        _ = related.params[_WEIGHT].weight_norm_diff


@pytest.mark.smoke
def test_single_member_param_view_keeps_live_handle_contract(
    stepped_pair: dict[str, Any],
) -> None:
    """One resolved member is not a cross-member claim: the live read stands."""

    bundle = tl.bundle({"only": stepped_pair["trace_before"]})
    view = bundle.params[_WEIGHT]
    out = view.out
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, stepped_pair["snapshot_after"])  # live bytes, disclosed via basis


def test_direct_param_value_read_contract_untouched(stepped_pair: dict[str, Any]) -> None:
    """Param.value keeps its documented live-handle meaning on bare traces."""

    param = stepped_pair["trace_before"].params[_WEIGHT]
    assert isinstance(param.value, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# s4.2 test 6: bypass matrix -- every public read hits the one guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda view: view.weight_norm_diff, id="weight_norm_diff"),
        pytest.param(lambda view: view.diff_pair(), id="diff_pair"),
        pytest.param(lambda view: view.diff_pair(other="ck0"), id="diff_pair_row"),
        pytest.param(lambda view: view.aggregate("mean"), id="aggregate_mean"),
        pytest.param(lambda view: view.aggregate("std"), id="aggregate_std"),
        pytest.param(lambda view: view.out, id="out"),
        pytest.param(lambda view: view.grad, id="grad"),
        pytest.param(lambda view: view._tensor_dict("out"), id="funnel_tensor_dict"),
    ],
)
def test_bypass_matrix_every_read_hits_the_guard(ckpt_bundle: Any, read: Any) -> None:
    """direct SuperParam read, diff_pair, aggregate, out/grad: one guard, one code."""

    view = ckpt_bundle.params[_WEIGHT]
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        read(view)
    _assert_checkpoint_refusal(excinfo, "ck0", "ck64")
    assert excinfo.value.fields["param_address"] == _WEIGHT


@pytest.mark.smoke
def test_bundle_at_routes_to_the_same_guard(ckpt_bundle: Any) -> None:
    """Bundle.at(param label) reads funnel through the identical refusal."""

    view = ckpt_bundle.at(_WEIGHT)
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        _ = view.out
    assert excinfo.value.fields["code"] == _CODE


def test_refusal_fires_before_tensor_lookup(ckpt_bundle: Any, monkeypatch: Any) -> None:
    """The guard runs BEFORE any member tensor is resolved."""

    view = ckpt_bundle.params[_WEIGHT]
    monkeypatch.setattr(
        type(view),
        "_get_tensor",
        lambda self, member, field: pytest.fail("tensor lookup ran before the guard"),
    )
    with pytest.raises(CheckpointSeriesLiveParamsError):
        _ = view.out


# ---------------------------------------------------------------------------
# s4.2 test 7: gate precedence -- the guard is claim-keyed, outcome-independent
# ---------------------------------------------------------------------------


def test_halted_member_still_refuses_param_claim(stepped_pair: dict[str, Any]) -> None:
    """A HALTED member does not license (or reorder) the parameter claim."""

    model = stepped_pair["model"]
    x = stepped_pair["x"]
    halted = tl.trace(model, x, halt=tl.func("relu"))
    assert halted.outcome.status.name == "HALTED"
    bundle = tl.bundle({"h": halted, "c": stepped_pair["trace_before"]})
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        _ = bundle.params[_WEIGHT].weight_norm_diff
    _assert_checkpoint_refusal(excinfo, "h", "c")


def test_n_gate_precedence_unchanged_on_halted_member(
    stepped_pair: dict[str, Any], tmp_path: Any
) -> None:
    """N4 (halted runnable save) still fires its OWN code; this lane reorders nothing."""

    halted = tl.trace(stepped_pair["model"], stepped_pair["x"], halt=tl.func("relu"))
    with pytest.raises(Exception) as excinfo:
        tl.save(halted, tmp_path / "halted.tlspec", level="runnable")
    fields = getattr(excinfo.value, "fields", {})
    assert fields.get("code") != _CODE


# ---------------------------------------------------------------------------
# s4.2 test 8: future activation -- the R8(b) evidence door
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_immutable_evidence_fixture_passes_the_guard(ckpt_bundle: Any, monkeypatch: Any) -> None:
    """When every member basis is immutable (R8(b) simulation), the read proceeds."""

    monkeypatch.setattr(Param, "value_basis", property(lambda self: ParamValueBasis("snapshot")))
    view = ckpt_bundle.params[_WEIGHT]
    diffs = view.weight_norm_diff
    assert set(diffs) == {"ck0", "ck64"}


@pytest.mark.smoke
def test_until_r8b_the_public_read_still_refuses(ckpt_bundle: Any) -> None:
    """Without R8(b) no immutable basis exists, so the read refuses -- row shape unchanged."""

    related = ckpt_bundle.relate(
        {"kind": "forked_from", "from": "ck64", "to": "ck0", "params": {"at_step": 64}}
    )
    row_payload = related.member_relations[0].to_payload()
    assert set(row_payload) == {"kind", "from", "to", "params"}  # no evidence key minted here
    with pytest.raises(CheckpointSeriesLiveParamsError):
        _ = related.params[_WEIGHT].out


# ---------------------------------------------------------------------------
# Basis vocabulary is closed
# ---------------------------------------------------------------------------


def test_basis_vocabulary_is_closed() -> None:
    """Only live_ref / absent / snapshot are constructible basis tokens."""

    assert ParamValueBasis("live_ref").basis == "live_ref"
    assert ParamValueBasis("absent", "not_persisted").reason == "not_persisted"
    assert ParamValueBasis("snapshot").is_immutable
    with pytest.raises(ValueError):
        ParamValueBasis("weights")


def test_weight_norm_diff_docstring_no_longer_promises_diffs(ckpt_bundle: Any) -> None:
    """The five wrong reads refuse identically; nan is never returned silently."""

    view = ckpt_bundle.params[_WEIGHT]
    for attribute in ("weight_norm_diff", "out", "grad"):
        with pytest.raises(CheckpointSeriesLiveParamsError):
            getattr(view, attribute)
