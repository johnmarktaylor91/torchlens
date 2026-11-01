"""Torch as the reference backend: its protocol adapter and conformance roster.

The roster's oracles (op inventories derived from each model definition and
NumPy-computed intermediates) are accepted only when torch passes them, so
these tests drive every config-built row through the torch protocol object.
Two rows expose torch behaviour the shared rules name: a closure constant
(torch records no source op for it today, and validation fails on the
unattributed argument) and a forward that draws randomness (torch capture is
RNG-neutral by design). Those rows are pinned as they behave, never skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens.backends.registry import BackendUnsupportedError, get_backend_spec
from torchlens.conformance._cases import _captured_op_kinds
from torchlens.conformance._rosters import torch as torch_roster
from torchlens.validation.status import ValidationStatus

ROWS = {row.family: row for row in torch_roster.config_built_roster()}
CLOSURE_ROW = "closure_constant"
RANDOM_ROW = "random_forward"
REPLAYABLE_ROWS = sorted(set(ROWS) - {CLOSURE_ROW, RANDOM_ROW})


def _protocol() -> Any:
    return get_backend_spec("torch").protocol()


def _capture_row(family: str) -> tuple[Any, Any, Any]:
    protocol = _protocol()
    model, example = ROWS[family].builder()
    reference = protocol.reference_output(model, example, None)
    return protocol.capture(model, example, None), reference, protocol


def test_roster_has_the_probe_rows_and_the_two_rule_rows() -> None:
    """At least eight config-built rows, each with an inventory; rule rows flagged."""

    rows = torch_roster.config_built_roster()
    assert len(rows) >= 8
    assert len({row.family for row in rows}) == len(rows)
    assert all(row.realism == "config_built" and row.expected_op_kinds for row in rows)
    flags = {family: (const, rand) for family, _b, _o, const, rand in torch_roster._CONFIG_BUILT}
    assert flags[CLOSURE_ROW] == (True, False)
    assert flags[RANDOM_ROW] == (False, True)


def test_ecosystem_rows_need_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretrained rows join only under the ecosystem switch, with checkpoint evidence."""

    monkeypatch.delenv(torch_roster.ECOSYSTEM_ENV, raising=False)
    assert all(row.realism == "config_built" for row in torch_roster.roster())
    monkeypatch.setenv(torch_roster.ECOSYSTEM_ENV, "1")
    pretrained = [row for row in torch_roster.roster() if row.realism == "pretrained"]
    assert len({row.family for row in pretrained}) >= 2
    assert all(row.checkpoint_evidence for row in pretrained)


@pytest.mark.parametrize("family", REPLAYABLE_ROWS)
def test_row_inventory_intermediates_and_validation(family: str) -> None:
    """Torch captures the declared ops, matches NumPy, and replay-validates."""

    trace, reference, protocol = _capture_row(family)
    assert _captured_op_kinds(trace) == ROWS[family].expected_op_kinds
    expected = torch_roster._expected_intermediates().get(family, {})
    for kind, want in expected.items():
        ops = [op for op in trace.layer_list if str(op.func_name) == kind]
        wants = want if isinstance(want, list) else [want]
        assert len(ops) == len(wants), kind
        for op, value in zip(ops, wants):
            np.testing.assert_allclose(op.out.detach().numpy(), value, rtol=1e-4, atol=1e-5)
    status = protocol.validate(trace, reference)
    assert isinstance(status, ValidationStatus)
    assert status.state == "passed", status.message


def test_closure_constant_gap_fails_validation_by_name() -> None:
    """Torch records no source op for a closure tensor; the tripwire names it.

    SPEC section 2 expects a ``constant`` source op. Until torch records one,
    the inventory misses it and validation fails with
    ``unattributed_tensor_args`` on the consuming op; when torch gains the
    rule, this test flips and is updated with that change.
    """

    trace, reference, protocol = _capture_row(CLOSURE_ROW)
    kinds = _captured_op_kinds(trace)
    assert "constant" not in kinds and kinds != ROWS[CLOSURE_ROW].expected_op_kinds
    status = protocol.validate(trace, reference)
    assert status.state == "failed"
    failed = [d for d in status.decisions if d.get("decision") == "failed"]
    assert [(d["func_name"], d["reason"]) for d in failed] == [
        ("matmul", "unattributed_tensor_args")
    ]


def test_literal_producers_are_parentless_ops() -> None:
    """Torch records ``ones`` and ``full`` as ops with no parents (rule 2's shape)."""

    for family, kind in (("reshape_transpose", "ones"), (CLOSURE_ROW, "full")):
        trace, _reference, _protocol = _capture_row(family)
        producers = [op for op in trace.layer_list if str(op.func_name) == kind]
        assert len(producers) == 1 and not tuple(producers[0].parents or ())


def test_random_forward_capture_is_rng_neutral() -> None:
    """Torch capture restores the global RNG and draws from its own seed."""

    protocol = _protocol()
    model, example = ROWS[RANDOM_ROW].builder()
    torch.manual_seed(0)
    before = torch.get_rng_state()
    trace = protocol.capture(model, example, None)
    assert torch.equal(torch.get_rng_state(), before)
    assert isinstance(trace.random_seed, int)


def test_capture_refuses_unknown_trace_keyword_before_running() -> None:
    """A keyword that is not a ``tl.trace`` parameter refuses typed, model untouched."""

    calls: list[int] = []

    class Counting(torch.nn.Module):
        def forward(self, value: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            calls.append(1)
            return value * 2.0

    with pytest.raises(BackendUnsupportedError) as info:
        _protocol().capture(Counting(), torch.ones(2), None, not_a_trace_option=1)
    assert info.value.fields["code"] == "trace_option_unsupported"
    assert info.value.fields["option"] == "not_a_trace_option"
    assert calls == []


def test_capture_passes_model_kwargs_and_caller_capture_options() -> None:
    """Forward kwargs reach the model; a caller ``capture=`` replaces the default."""

    class Scaled(torch.nn.Module):
        def forward(self, value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return value * scale

    protocol = _protocol()
    trace = protocol.capture(Scaled(), torch.ones(2), {"scale": 3.0})
    assert torch.equal(trace.output_ops[0].out, torch.full((2,), 3.0))
    plain = protocol.capture(Scaled(), torch.ones(2), None, capture=tl.CaptureOptions())
    assert protocol.validate(plain, torch.ones(2)).state != "passed"


def test_host_copy_keeps_shape_and_widens_bfloat16() -> None:
    """0-d stays 0-d; bfloat16 becomes float32 (NumPy has no bfloat16)."""

    protocol = _protocol()
    scalar = protocol.host_copy(torch.tensor(2.5))
    assert scalar.shape == () and scalar == 2.5
    widened = protocol.host_copy(torch.ones(2, 3, dtype=torch.bfloat16))
    assert widened.shape == (2, 3) and widened.dtype == np.float32


def test_protocol_members_are_the_reference_values() -> None:
    """Torch is eager, stable, and refuses only registered codes."""

    from torchlens.backends._backend_protocol import REFUSAL_CODES, protocol_problems

    protocol = _protocol()
    assert protocol_problems(protocol, expected_name="torch") == ()
    assert (protocol.materialization, protocol.maturity) == ("eager", "stable")
    assert protocol.refusal_codes() <= REFUSAL_CODES
    assert protocol.realize_once([torch.ones(1)]) is None
    assert protocol.is_tensor(torch.ones(1)) and not protocol.is_tensor(np.ones(1))
