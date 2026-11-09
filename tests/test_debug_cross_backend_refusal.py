"""``tl.debug.compare`` and ``tl.debug.first_divergence`` refuse a cross-backend pair.

Both tools pair ops by literal label, and labels and op order do not correspond across
backends, so a label join between, say, a PyTorch trace and an MLX trace reports a false
divergence (the first torch op as ``only-a``) or value-compares two unrelated ops whose
labels coincide. A label join across backends (``align="label"``) raises a typed refusal
before any value is read; the default ``align="auto"`` pairs module calls instead, and refuses a
backend whose module recording is not admitted. The other backend is faked by setting the private provenance field on
a torch trace, so these tests run on Linux without MLX.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.debug import _graph, _xb_values
from torchlens.debug._compare_guard import refuse_uncomparable_pair
from torchlens.errors import BackendUnsupportedError

pytest.importorskip("pandas")

_TOOLS: dict[str, Callable[..., Any]] = {
    "compare": tl.debug.compare,
    "first_divergence": tl.debug.first_divergence,
    "compare_rows": _graph._compare_rows,
}


def _model() -> nn.Module:
    """A small eval-mode MLP with fixed weights."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()


def _pair(shift: float = 0.0) -> tuple[tl.Trace, tl.Trace]:
    """Two torch traces of one model; the second input is shifted by ``shift``."""

    model, x = _model(), torch.ones(1, 4)
    return tl.trace(model, x), tl.trace(model, x + shift)


def _as_backend(trace: tl.Trace, backend: str) -> tl.Trace:
    """Stamp ``backend`` on the trace's provenance (the private field the guard reads)."""

    trace._provenance_backend = backend  # type: ignore[assignment]
    assert trace.provenance.backend == backend
    return trace


def _label_join(tool: str) -> Callable[..., Any]:
    """The tool with the label join forced (the pandas-free core always joins by label)."""

    if tool == "compare_rows":
        return _TOOLS[tool]
    return lambda first, second: _TOOLS[tool](first, second, align="label")


@pytest.mark.parametrize("tool", sorted(_TOOLS))
@pytest.mark.parametrize("order", ["torch-first", "other-first"])
def test_cross_backend_label_join_refuses_in_both_orders(tool: str, order: str) -> None:
    """A label join refuses across backends, whichever trace comes first."""

    torch_trace, other = _pair()
    _as_backend(other, "mlx")
    first, second = (torch_trace, other) if order == "torch-first" else (other, torch_trace)
    with pytest.raises(BackendUnsupportedError) as caught:
        _label_join(tool)(first, second)
    error = caught.value
    assert error.code == "backend_unsupported"
    assert error.fields["code"] == "backend_unsupported"
    assert (error.fields["backend_a"], error.fields["backend_b"]) == (
        str(first.provenance.backend),
        str(second.provenance.backend),
    )
    message = str(error)
    assert "do not correspond across backends" in message
    assert "same backend" in message
    assert "pair module calls" in message


@pytest.mark.parametrize("tool", ["compare", "first_divergence"])
@pytest.mark.parametrize("order", ["torch-first", "other-first"])
def test_unadmitted_backend_refuses_module_pairing(tool: str, order: str) -> None:
    """The default align="auto" pairs modules, and refuses a backend not yet admitted."""

    torch_trace, other = _pair()
    _as_backend(other, "paddle")
    first, second = (torch_trace, other) if order == "torch-first" else (other, torch_trace)
    with pytest.raises(BackendUnsupportedError, match="admitted today: mlx, torch") as caught:
        _TOOLS[tool](first, second)
    assert caught.value.fields["code"] == "backend_unsupported"


@pytest.mark.parametrize("tool", sorted(_TOOLS))
def test_refusal_happens_before_any_value_is_read(
    tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No op is listed and no payload is decoded once the backends differ."""

    def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("value comparison started on a cross-backend pair")

    monkeypatch.setattr(_graph, "_compared_site_ops", _must_not_run)
    monkeypatch.setattr(_graph, "_safe_out", _must_not_run)
    monkeypatch.setattr(_graph, "get_payload_codec", _must_not_run)
    monkeypatch.setattr(_xb_values, "read_values", _must_not_run)
    torch_trace, other = _pair()
    with pytest.raises(BackendUnsupportedError):
        _TOOLS[tool](torch_trace, _as_backend(other, "paddle"))


def test_same_backend_behaviour_is_unchanged() -> None:
    """Two torch runs still compare op by op: equal runs agree, shifted runs diverge."""

    same_a, same_b = _pair()
    assert tl.debug.first_divergence(same_a, same_b) is None
    frame = tl.debug.compare(same_a, same_b).to_pandas()
    assert frame.attrs["value_diverged"] == 0
    assert frame.attrs["matched"] == len(frame) > 0

    base, shifted = _pair(shift=1.0)
    row = tl.debug.first_divergence(base, shifted)
    assert row is not None and row["status"] == "present-in-both"
    assert row["allclose"] is False
    assert tl.debug.compare(base, shifted).to_pandas().attrs["value_diverged"] > 0


def test_guard_admits_two_traces_of_one_non_torch_backend() -> None:
    """The guard compares backend names only: one shared backend, whatever it is, passes."""

    first, second = _pair()
    assert refuse_uncomparable_pair(_as_backend(first, "mlx"), _as_backend(second, "mlx")) is None


def test_refusal_holds_after_save_and_load(tmp_path: Path) -> None:
    """Loaded traces keep their backend, compare as before, and refuse a cross-backend pair."""

    first, second = _pair()
    first.save(str(tmp_path / "a.tlspec"))
    second.save(str(tmp_path / "b.tlspec"))
    loaded_a = tl.load(str(tmp_path / "a.tlspec"))
    loaded_b = tl.load(str(tmp_path / "b.tlspec"))
    assert loaded_a.provenance.backend == loaded_b.provenance.backend == "torch"
    assert tl.debug.first_divergence(loaded_a, loaded_b) is None
    assert tl.debug.compare(loaded_a, loaded_b).to_pandas().attrs["value_diverged"] == 0

    _as_backend(loaded_b, "paddle")
    for tool in _TOOLS.values():
        for pair in ((loaded_a, loaded_b), (loaded_b, loaded_a), (first, loaded_b)):
            with pytest.raises(BackendUnsupportedError, match="same backend"):
                tool(*pair)
