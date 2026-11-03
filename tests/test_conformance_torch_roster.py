"""Torch as the reference backend: its protocol adapter and conformance roster.

The roster's oracles (op inventories derived from each model definition and
NumPy-computed intermediates) are accepted only when torch passes them, so
these tests drive every config-built row through the torch protocol object.
Two rows exercise the shared rules: a closure constant (torch records it as a
``constant`` source op with a saved payload, SPEC section 2) and a forward
that draws randomness (torch capture is RNG-neutral by design). Those rows are
pinned as they behave, never skipped.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch

import torchlens as tl
from tests.conformance._cases import _captured_op_kinds
from tests.conformance._rosters import ECOSYSTEM_ENV, torch as torch_roster
from torchlens._state import pause_logging
from torchlens.backends.registry import BackendUnsupportedError, get_backend_spec
from torchlens.options import CaptureOptions
from torchlens.validation.status import ValidationStatus

ROWS = {row.family: row for row in torch_roster.config_built_roster()}
CLOSURE_ROW = "closure_constant"
RANDOM_ROW = "random_forward"
REPLAYABLE_ROWS = sorted(set(ROWS) - {RANDOM_ROW})


def _protocol() -> Any:
    return get_backend_spec("torch").protocol()


#: Warning torch emits for a tensor argument with no recorded producer.
UNATTRIBUTED_WARNING = "no graph/source provenance"


def _capture_row(family: str) -> tuple[Any, Any, Any, Any]:
    """Capture one row; the model is returned so it outlives replay validation.

    Replay from saved arguments reads the model's parameters through weak
    references, so a caller that drops the model before ``validate`` gets
    ``ParamReleasedError`` instead of a verdict.
    """

    protocol = _protocol()
    model, example = ROWS[family].builder()
    reference = protocol.reference_output(model, example, None)
    return protocol.capture(model, example, None), reference, protocol, model


def test_roster_has_the_probe_rows_and_the_two_rule_rows() -> None:
    """At least eight config-built rows, each with an inventory; rule rows flagged."""

    rows = torch_roster.config_built_roster()
    assert len(rows) >= 8
    assert len({row.family for row in rows}) == len(rows)
    assert all(row.group == "config_built" and row.expected_op_kinds for row in rows)
    flags = {row.family: (row.has_constants, row.random_init) for row in rows}
    assert flags[CLOSURE_ROW] == (True, False)
    assert flags[RANDOM_ROW] == (False, True)
    assert sum(bool(row.expected_intermediates) for row in rows) >= 8


def test_ecosystem_rows_need_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretrained rows join only under the ecosystem switch, with checkpoint evidence."""

    monkeypatch.delenv(ECOSYSTEM_ENV, raising=False)
    assert all(row.group == "config_built" for row in torch_roster.roster())
    monkeypatch.setenv(ECOSYSTEM_ENV, "1")
    pretrained = [row for row in torch_roster.roster() if row.realism == "pretrained"]
    assert len({row.family for row in pretrained}) >= 2
    assert all(row.checkpoint_evidence and row.group == "ecosystem" for row in pretrained)


@pytest.mark.parametrize("family", REPLAYABLE_ROWS)
def test_row_inventory_intermediates_and_validation(family: str) -> None:
    """Torch captures the declared ops, matches NumPy, and replay-validates."""

    trace, reference, protocol, _model = _capture_row(family)
    assert _captured_op_kinds(trace) == ROWS[family].expected_op_kinds
    for kind, want in (ROWS[family].expected_intermediates or {}).items():
        ops = [op for op in trace._layer_list if str(op.func_name) == kind]
        wants = want if isinstance(want, list) else [want]
        assert len(ops) == len(wants), kind
        for op, value in zip(ops, wants, strict=True):
            np.testing.assert_allclose(op.out.detach().numpy(), value, rtol=1e-4, atol=1e-5)
    status = protocol.validate(trace, reference)
    assert isinstance(status, ValidationStatus)
    assert status.state == "passed", status.message


def _constant_ops(trace: Any) -> list[Any]:
    return [op for op in trace._layer_list if str(op.func_name) == "constant"]


def test_closure_constant_is_a_constant_source_op_and_validates() -> None:
    """A closure tensor is a parentless ``constant`` source op the consumer lists as parent.

    SPEC section 2 rule 1: no provenance warning, the saved payload is the
    closure's value, the consuming ``matmul`` names it as a parent, and replay
    validation passes with nothing failed.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace, reference, protocol, _model = _capture_row(CLOSURE_ROW)
    assert _captured_op_kinds(trace) == ROWS[CLOSURE_ROW].expected_op_kinds
    (constant,) = _constant_ops(trace)
    assert tuple(constant.parents or ()) == ()
    assert (
        constant.graph.is_internal_source and not constant.graph.is_buffer and not constant.is_input
    )
    assert constant.layer_label.startswith("constant_")
    np.testing.assert_array_equal(constant.out.detach().numpy(), torch_roster.W44)
    (matmul,) = [op for op in trace._layer_list if str(op.func_name) == "matmul"]
    assert constant.layer_label in {str(parent).split(":")[0] for parent in matmul.parents}
    assert not tuple(matmul.call.unattributed_tensor_args or ())
    status = protocol.validate(trace, reference)
    assert status.state == "passed", status.message
    assert not [d for d in status.decisions if d.get("decision") == "failed"]


def test_closure_constant_payload_obeys_save() -> None:
    """SPEC 2 rule 5: ``save=`` filters a constant's payload like any activation."""

    model, example = ROWS[CLOSURE_ROW].builder()
    kept = tl.trace(model, example, save=tl.func("constant"))
    (constant,) = _constant_ops(kept)
    assert constant.has_saved_activation
    dropped = tl.trace(model, example, save=tl.func("matmul"))
    (constant,) = _constant_ops(dropped)
    assert not constant.has_saved_activation


def test_closure_constant_needs_a_pre_forward_tensor() -> None:
    """A tensor first built mid-forward and smuggled in untagged is no constant.

    The narrow rule: only the exact object the pre-forward snapshot pinned,
    reachable only from a forward callable, becomes a ``constant``; a tensor
    the forward stores in a global and reads back is a capture gap and still
    fails validation by name.
    """

    holder: dict[str, torch.Tensor] = {}

    class _Smuggles(torch.nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            with pause_logging():
                holder["t"] = v * 3.0
            return torch.relu(v + holder["t"])

    with pytest.warns(UserWarning, match=UNATTRIBUTED_WARNING):
        trace = tl.trace(_Smuggles(), torch.randn(2, 4))
    assert not _constant_ops(trace)


def test_literal_producers_are_parentless_ops() -> None:
    """Torch records ``ones`` and ``full`` as ops with no parents (rule 2's shape)."""

    ones_trace, _reference, _protocol, _model = _capture_row("reshape_transpose")
    closure_trace, _reference, _protocol, _model = _capture_row(CLOSURE_ROW)
    for trace, kind in ((ones_trace, "ones"), (closure_trace, "full")):
        producers = [op for op in trace._layer_list if str(op.func_name) == kind]
        assert len(producers) == 1 and not tuple(producers[0].parents or ())


def test_random_forward_capture_is_rng_neutral() -> None:
    """Torch capture restores the global RNG and draws from its own seed."""

    protocol = _protocol()
    model, example = ROWS[RANDOM_ROW].builder()
    torch.manual_seed(0)
    before = torch.get_rng_state()
    trace = protocol.capture(model, example, None)
    assert torch.equal(torch.get_rng_state(), before)
    assert isinstance(trace.options.seed, int)


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
    plain = protocol.capture(Scaled(), torch.ones(2), None, capture=CaptureOptions())
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


@pytest.mark.parametrize("backend", [None, "torch"], ids=["default", "explicit_torch"])
def test_grad_options_refuses_with_the_capability_code(backend: Any) -> None:
    """SPEC 1.3: torch refuses ``grad_options=`` with the flag's code on ``exc.code``.

    The option is the ``intermediate_derived_grads`` capability's entry, so the
    refusal is that flag's contracted code, raised before the model runs.
    """

    calls: list[int] = []

    class _Counts(torch.nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return v * 2.0

    with pytest.raises(BackendUnsupportedError) as excinfo:
        tl.trace(
            _Counts(),
            torch.randn(2, 4),
            backend=backend,
            capture=tl.options.CaptureOptions(grad_options={"derived": True}),
        )
    assert excinfo.value.code == "intermediate_derived_grads_unsupported"
    assert excinfo.value.fields["code"] == excinfo.value.code
    assert not calls


def test_after_load_cell_reloads_at_the_replay_level(tmp_path: Any) -> None:
    """Only ``c2_validation_after_load`` reads a reload saved with re-execution payloads.

    The default-level reload the other C2 cells read drops saved arguments;
    the replay reload keeps them and validates. An adapter without a replay
    level validates the default reload unchanged.
    """

    from types import SimpleNamespace

    from tests.conformance._adapters import TorchReferenceAdapter
    from tests.conformance._protocol_cases import load_for_replay

    adapter = TorchReferenceAdapter()
    trace, reference, protocol, _model = _capture_row("mlp")
    path = tmp_path / "c2.tlspec"
    adapter.save(trace, path)
    loaded = adapter.load(path)
    assert all(op.call.saved_args is None for op in loaded._layer_list)
    replay = load_for_replay(adapter, trace, path, loaded)
    assert replay is not loaded and any(op.call.saved_args is not None for op in replay._layer_list)
    status = protocol.validate(replay, reference)
    assert status.state == "passed", status.message
    assert load_for_replay(SimpleNamespace(), trace, path, loaded) is loaded
