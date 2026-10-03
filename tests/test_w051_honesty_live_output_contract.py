"""Lane W051-HONESTY: live ``run()`` output-contract honesty (AUD-HONESTY H1/H2/H3).

The review-5.1 audit found that every tuple/dict/HF-container model settled
``unverifiable`` + poisoned on a DEFAULT-capture live ``run()`` with a remedy-less
``output_structure_mismatch`` (H2; closed for good by W051-CAPT2, which registers
the final-output ContainerSpec on every capture), that a declared container carrying an opaque
tensor-holding leaf crashed the live provider with a bare ``ValueError`` (H1), and
that a refresh graph change leaked the projector's bare ``ValueError`` while
``on_divergence=RETURN_DIVERGED`` was ignored (H3). These pins are the live-provider
parity matrix the audit asked for: GRU / LSTM / tuple / dict rows plus the
divergence-policy rows. Consumers branch on ``RunnableErrorCode`` + the closed
``details["reason"]`` vocabulary, never on message text.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.ir.container import ContainerSpec, DictKey, HFKey, TupleIndex, declared_leaf_slots
from torchlens.options import CaptureOptions
from torchlens.runnable import DivergencePolicy, PathFaithfulness, RunnableErrorCode


class _TupleOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.lin(x)
        return y, torch.relu(y)


class _DictOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.lin(x)
        return {"y": y, "r": torch.relu(y)}


class _Branch(nn.Module):
    """Value-dependent control flow: a new input flips the executed branch."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return torch.relu(y) if y.sum() > 0 else torch.tanh(y)


def _failed_check(report: Any, name: str) -> Any:
    checks = [c for c in report.contract_checks if c.name == name and not c.passed]
    assert len(checks) == 1, [c.name for c in report.contract_checks]
    return checks[0]


_CONTAINER_ROWS: list[tuple[str, Any, tuple[int, ...]]] = [
    ("gru", lambda: nn.GRU(8, 8, batch_first=True), (2, 5, 8)),
    ("lstm", lambda: nn.LSTM(8, 8, batch_first=True), (2, 5, 8)),
    ("tuple", _TupleOut, (2, 8)),
    ("dict", _DictOut, (2, 8)),
]


@pytest.mark.parametrize(
    ("name", "factory", "shape"), _CONTAINER_ROWS, ids=lambda v: v if isinstance(v, str) else ""
)
def test_default_capture_container_output_settles_verified_live(name, factory, shape) -> None:
    """H2 (closed by W051-CAPT2): the final-output ContainerSpec is registered on
    EVERY capture, so a declared tuple/dict/GRU/LSTM output on a DEFAULT capture
    settles VERIFIED and returns the exact container type the model returns --
    the op records still carry ``container_spec=None`` (the opt-in metadata
    contract), the live provider reads the model-output snapshot."""

    torch.manual_seed(0)
    model = factory().eval()  # keep a strong ref: the live Trace holds only a weakref
    x = torch.randn(*shape)
    trace = tl.trace(model, x)
    assert all(trace.ops[label].container_spec is None for label in trace.output_layers)
    result = trace.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False
    with torch.no_grad():
        live = model(x)
    assert type(result.output) is type(live)


@pytest.mark.smoke_cells(
    "test_container_contract_capture_settles_verified_live[dict--]",
    "test_container_contract_capture_settles_verified_live[lstm--]",
)
@pytest.mark.parametrize(
    ("name", "factory", "shape"), _CONTAINER_ROWS, ids=lambda v: v if isinstance(v, str) else ""
)
def test_container_contract_capture_settles_verified_live(name, factory, shape) -> None:
    """H2 over-trigger control + the remedy WORKS: the same rows captured with
    ``capture_container_structure=True`` settle VERIFIED and return the exact
    container type the model returns."""

    torch.manual_seed(0)
    model = factory().eval()
    x = torch.randn(*shape)
    trace = tl.trace(model, x, capture=CaptureOptions(capture_container_structure=True))
    result = trace.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.poisoned is False
    with torch.no_grad():
        live = model(x)
    assert type(result.output) is type(live)


@pytest.mark.smoke
def test_opaque_root_output_names_its_reason() -> None:
    """corr2_5 parity: an unordered-set output keeps the UNVERIFIABLE verdict and now
    names the ``opaque_root`` reason (not the unrecorded-contract remedy)."""

    class _SetOut(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> Any:
            return {self.lin(x)}

    x = torch.randn(2, 4)
    model = _SetOut()
    with pytest.warns(UserWarning):
        trace = tl.trace(model, x)
    result = trace.run(inputs=x, on_divergence="return_diverged")
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    check = _failed_check(result.report, "live_output_reconstruction")
    assert dict(check.diagnostic.details)["reason"] == "opaque_root"


@pytest.mark.smoke
def test_declared_leaf_slots_counts_every_leaf_slot() -> None:
    """The opaque-leaf arm's measuring stick: slots = declared components without a
    child spec, recursively; literal children consume nothing."""

    inner = ContainerSpec(kind="tuple", length=2)
    literal = ContainerSpec(kind="literal", literal_value=None)
    spec = ContainerSpec(
        kind="hf_model_output",
        length=3,
        keys=("logits", "past", "flag"),
        child_specs=((HFKey("past"), inner), (HFKey("flag"), literal)),
    )
    assert declared_leaf_slots(spec) == 3
    assert declared_leaf_slots(ContainerSpec(kind="dict", length=2, keys=("a", "b"))) == 2
    assert declared_leaf_slots(ContainerSpec(kind="opaque")) == 0
    assert declared_leaf_slots(inner) == 2
    assert declared_leaf_slots(literal) == 0
    _ = (DictKey, TupleIndex)  # component types exercised through the specs above


def test_rebuild_arity_error_names_the_dry_slot() -> None:
    """H1 opaque-leaf arm in the codec: running dry names the declared slot and the
    opaque-leaf cause instead of the bare historical message."""

    from torchlens.ir.container import rebuild_container_from_spec

    spec = ContainerSpec(kind="hf_model_output", length=2, keys=("logits", "past_key_values"))
    with pytest.raises(ValueError, match="Not enough leaves") as excinfo:
        rebuild_container_from_spec(spec, [torch.zeros(1)])
    assert "past_key_values" in str(excinfo.value)
    assert "opaque non-tensor object" in str(excinfo.value)


@pytest.mark.smoke
def test_live_graph_change_raise_policy_is_structured() -> None:
    """H3 (RAISE): a refresh graph change surfaces through the divergence spine --
    ``fields["code"] == call_structure_mismatch``, DIVERGED, the failed contract check
    attached, the pinned phrase kept -- and never as a bare untyped ValueError.

    Interim lineage bridge: until ``PathDivergenceError`` gains ``ValueError`` in its
    bases (recorded OUT-OF-FENCE by this lane; the historical ``except ValueError``
    callers pin that lineage), the raised object is the projector's ValueError carrying
    the SAME structured fields; once the lineage change lands the same code path raises
    the typed error and this pin keeps holding.
    """

    torch.manual_seed(0)
    model = _Branch().eval()
    x = torch.randn(3, 8)
    trace = tl.trace(model, x)
    flipped = torch.full((3, 8), -5.0)
    with pytest.raises(ValueError, match="computational graph changed") as excinfo:
        trace.run(inputs=flipped)
    exc = excinfo.value
    fields = exc.fields  # type: ignore[attr-defined]
    assert fields["code"] == RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
    assert fields["path_faithfulness"] is PathFaithfulness.DIVERGED
    assert fields["contract_check"].name == "live_refresh_graph_signature"
    assert fields["first_mismatch"].code is RunnableErrorCode.CALL_STRUCTURE_MISMATCH
    assert dict(fields["first_mismatch"].details)["reason"] == "refresh_graph_changed"
    assert fields["remedy"]
    assert isinstance(exc, PathDivergenceError) == issubclass(PathDivergenceError, ValueError)
    # The source trace is untouched: a second, matching-input run still verifies.
    again = trace.run(inputs=x)
    assert again.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_live_graph_change_return_diverged_policy_is_honored() -> None:
    """H3 (RETURN_DIVERGED): the opt-in returns a poisoned DIVERGED result through the
    one provider finalizer instead of raising; ``output`` is None (no faithful
    refreshed output exists) and the first mismatch is the graph-change check."""

    torch.manual_seed(0)
    model = _Branch().eval()
    x = torch.randn(3, 8)
    trace = tl.trace(model, x)
    flipped = torch.full((3, 8), -5.0)
    result = trace.run(inputs=flipped, on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert result.report.poisoned is True
    assert result.output is None
    assert result.report.first_mismatch is not None
    assert result.report.first_mismatch.code is RunnableErrorCode.CALL_STRUCTURE_MISMATCH
    assert "computational graph changed" in result.report.first_mismatch.message
    assert result.trace is not trace
    # Monotone poison: the returned fork refuses faithful consumers.
    from torchlens.errors import PoisonedRunError

    with pytest.raises(PoisonedRunError):
        tl.save(result.trace, "/nonexistent/never_written.tlspec")
    # The SOURCE trace is untouched: a matching-input run still verifies.
    again = trace.run(inputs=x)
    assert again.report.path_faithfulness is PathFaithfulness.VERIFIED
