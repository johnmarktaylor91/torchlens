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

from typing import Any

import pytest
import torch

from tests.conformance._rosters import ECOSYSTEM_ENV, torch as torch_roster
from torchlens.backends.registry import BackendUnsupportedError, get_backend_spec

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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_protocols/torch.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _io/scrub.py.
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
