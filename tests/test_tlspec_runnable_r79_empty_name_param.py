"""Save-side canonical-label preflight for empty-name state (r79, r78 free LOW).

Round-78 (free) found a save/load door asymmetry: ``self._parameters[""] =
nn.Parameter(...)`` bypasses ``register_parameter``'s empty-name validation,
captures fine, and SAVED a runnable artifact whose weight tensor entry carried
label ``""`` -- which ``tl.load`` categorically refuses wholesale
(``TorchLensIOError: Runnable weight tensor entry 0 requires a canonical
label``). Fail-closed and typed at load, no verdict risk, but the save door
accepted what the load door refuses: a stillborn artifact.

r79 adds the save-side mirror of the load door's canonical-label predicate
(non-``str`` or empty refuses -- nothing wider) at the two payload families the
load check governs: runnable weight entries and non-persistent buffer entries.
The save now refuses typed (``RunnablePreflightError``,
``sparse_preflight_failed``) before any artifact bytes can exist.

Zero-collateral pins: a normal named param still saves, loads, and verifies;
the WEIGHTLESS empty-name lane (``include_weights=False``) is deliberately
unchanged -- load accepts it today, so refusing it at save would create the
reverse asymmetry; the empty-name buffer pokes keep refusing at save.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class EmptyNameParam(nn.Module):
    """Pathological carrier: dict-poked param with the empty name."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self._parameters[""] = nn.Parameter(torch.randn(4, 4), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the empty-name param on-path."""

        return x + self._parameters[""]


class NamedParam(nn.Module):
    """Zero-collateral control: ordinary registered param."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Parameter(torch.randn(4, 4), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the named param on-path."""

        return x + self.w


class EmptyNamePersistentBuffer(nn.Module):
    """Pathological carrier: dict-poked persistent buffer with the empty name."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self._buffers[""] = torch.randn(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the empty-name buffer on-path."""

        return x + self._buffers[""]


class EmptyNameNonPersistentBuffer(nn.Module):
    """Pathological carrier: dict-poked NON-persistent buffer with the empty name."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self._buffers[""] = torch.randn(4, 4)
        self._non_persistent_buffers_set.add("")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the empty-name non-persistent buffer on-path."""

        return x + self._buffers[""]


def _x() -> torch.Tensor:
    """Return a fixed probe input."""

    torch.manual_seed(7)
    return torch.randn(2, 4, 4)


@pytest.mark.smoke
def test_r79_empty_name_param_refuses_at_save(tmp_path: Path) -> None:
    """RED-now-fixed: the empty-name param save refuses typed, never stillborn.

    Pre-fix the save SUCCEEDED and only ``tl.load`` refused ("requires a
    canonical label"). The save-side mirror must refuse with
    ``RunnablePreflightError`` (``sparse_preflight_failed``) naming the
    canonical-label requirement, and no artifact may be left on disk.
    """

    x = _x()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(EmptyNameParam().eval(), x.clone(), capture=_CAPTURE)

    path = tmp_path / "empty.tlspec"
    with pytest.raises(RunnablePreflightError, match="canonical label") as excinfo:
        trace.save(path, level="runnable", include_weights=True)
    assert excinfo.value.fields["code"] == RunnableErrorCode.SPARSE_PREFLIGHT_FAILED.value
    assert not path.exists()


@pytest.mark.smoke
def test_r79_named_param_still_saves_and_loads(tmp_path: Path) -> None:
    """Zero collateral: a normal named param round-trips and verifies."""

    x = _x()
    path = tmp_path / "named.tlspec"
    trace = tl.trace(NamedParam().eval(), x.clone(), capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r79_weightless_empty_name_lane_unchanged(tmp_path: Path) -> None:
    """Zero collateral: the weightless empty-name lane keeps its current parity.

    ``include_weights=False`` produces no weight tensor entries, the load door
    accepts the artifact today, and the r79 mirror is deliberately scoped to
    the two families the load check governs -- refusing here would create the
    REVERSE save/load asymmetry.
    """

    x = _x()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(EmptyNameParam().eval(), x.clone(), capture=_CAPTURE)

    path = tmp_path / "weightless.tlspec"
    trace.save(path, level="runnable", include_weights=False)
    tl.load(path)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls",
    [EmptyNamePersistentBuffer, EmptyNameNonPersistentBuffer],
    ids=["persistent", "non_persistent"],
)
def test_r79_empty_name_buffer_pokes_refuse_at_save(
    tmp_path: Path, model_cls: type[nn.Module]
) -> None:
    """Posture pin: empty-name buffer pokes refuse typed at save, never stillborn.

    Both dict-poked buffer variants already refuse through the producer
    preflight; the r79 non-persistent-family mirror is belt-and-suspenders
    behind it. Pin the door outcome so neither lane can regress to producing
    an artifact the load door would refuse.
    """

    x = _x()
    trace = tl.trace(model_cls().eval(), x.clone(), capture=_CAPTURE)

    path = tmp_path / "buffer.tlspec"
    with pytest.raises(RunnablePreflightError):
        trace.save(path, level="runnable", include_weights=True)
    assert not path.exists()
