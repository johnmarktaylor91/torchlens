"""Typed run-door refusal for bool and out-of-range ``run(seed=...)`` (r79).

Round-78 found two values that escaped the r77 seed door
(``seed is not None and not isinstance(seed, int)``) and reached raw torch:

- ``bool`` (hon1): an ``int`` subclass, so it passed the type check, but the
  run path uses ``default_generator.manual_seed``, which rejects bool with a
  raw ``RuntimeError: manual_seed expected a long, but got bool``.
- an ``int`` outside torch's accepted ``[-0x8000_0000_0000_0000,
  0xFFFF_FFFF_FFFF_FFFF]`` long range (Sol): pybind overflow, raw
  ``RuntimeError: Overflow when unpacking long``.

The r79 door mirrors the codebase's own capture-seed convention
(``isinstance(seed, int) and not isinstance(seed, bool)``) plus the empirical
torch range, raising typed ``RunPreconditionError`` (``context_field_invalid``)
BEFORE any provider work. The same canonical guard (``validate_run_seed``) is
mirrored at the sibling ``manual_seed`` sites (``_seed_run_generators`` in the
executor and ``_generator_for_slot`` in random-state initialization) so no path
reaches raw torch. Rejected calls are transactional: global torch RNG is
bit-identical afterwards and the same loaded trace still verifies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable_state import (
    TORCH_MANUAL_SEED_MAX,
    TORCH_MANUAL_SEED_MIN,
    validate_run_seed,
)
from torchlens.errors import RunPreconditionError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class _Small(nn.Module):
    """Minimal runnable carrier."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Affine map."""

        return self.linear(x)


def _saved(tmp_path: Path, x: torch.Tensor) -> Path:
    """Capture and save one runnable artifact."""

    path = tmp_path / "seed.tlspec"
    trace = tl.trace(_Small().eval(), x, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return path


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_seed",
    [True, False, 2**64, -(2**64) - 1, TORCH_MANUAL_SEED_MAX + 1, TORCH_MANUAL_SEED_MIN - 1],
    ids=["true", "false", "2p64", "neg_2p64_m1", "max_plus_one", "min_minus_one"],
)
def test_r79_bool_and_out_of_range_seed_refuse_typed(tmp_path: Path, bad_seed: object) -> None:
    """RED-now-fixed: bool / out-of-range ``seed`` raises typed, not raw torch.

    Pre-fix ``seed=True`` escaped as ``RuntimeError: manual_seed expected a
    long, but got bool`` and ``seed=2**64`` as ``RuntimeError: Overflow when
    unpacking long``. The door must refuse with ``RunPreconditionError``
    carrying the ``context_field_invalid`` code, leave the global torch RNG
    bit-identical, and keep the loaded trace runnable afterwards.
    """

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    loaded = tl.load(_saved(tmp_path, x))

    rng_before = torch.get_rng_state()
    with pytest.raises(RunPreconditionError) as excinfo:
        loaded.run(inputs=x.clone(), seed=bad_seed)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == RunnableErrorCode.CONTEXT_FIELD_INVALID.value
    assert torch.equal(torch.get_rng_state(), rng_before)

    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
@pytest.mark.parametrize(
    "good_seed",
    [0, None, 17, TORCH_MANUAL_SEED_MIN, TORCH_MANUAL_SEED_MAX],
    ids=["zero", "none", "seventeen", "range_min", "range_max"],
)
def test_r79_valid_seed_still_verifies(tmp_path: Path, good_seed: int | None) -> None:
    """Zero collateral: valid seeds -- including both range boundaries -- verify."""

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    path = _saved(tmp_path, x)

    result = tl.load(path).run(inputs=x.clone(), seed=good_seed)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r79_validate_run_seed_guard_boundaries() -> None:
    """Unit pin for the ONE canonical guard shared by all three manual_seed sites.

    ``validate_run_seed`` is called at the run door, in the executor's
    ``_seed_run_generators``, and in random-state ``_generator_for_slot`` --
    this pins its accept/reject boundary directly so the mirrors cannot drift.
    """

    assert validate_run_seed(None) is None
    assert validate_run_seed(0) == 0
    assert validate_run_seed(TORCH_MANUAL_SEED_MIN) == TORCH_MANUAL_SEED_MIN
    assert validate_run_seed(TORCH_MANUAL_SEED_MAX) == TORCH_MANUAL_SEED_MAX

    for bad in (True, False, 1.5, "x", TORCH_MANUAL_SEED_MAX + 1, TORCH_MANUAL_SEED_MIN - 1):
        with pytest.raises(RunPreconditionError) as excinfo:
            validate_run_seed(bad)
        assert excinfo.value.fields["code"] == RunnableErrorCode.CONTEXT_FIELD_INVALID.value

    # The accepted boundary values must be exactly what torch itself accepts.
    generator = torch.Generator()
    state_before = generator.get_state()
    generator.manual_seed(TORCH_MANUAL_SEED_MIN)
    generator.manual_seed(TORCH_MANUAL_SEED_MAX)
    with pytest.raises(RuntimeError):
        generator.manual_seed(TORCH_MANUAL_SEED_MAX + 1)
    with pytest.raises(RuntimeError):
        generator.manual_seed(TORCH_MANUAL_SEED_MIN - 1)
    generator.set_state(state_before)
