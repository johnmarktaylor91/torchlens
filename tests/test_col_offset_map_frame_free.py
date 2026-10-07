"""The column-offset map build enters no Python frame (r16 tap-budget fix).

``_build_col_offset_map`` is often first called inside an
``intervention_ready`` capture, where the RNG monitor's profile hook snapshots
NumPy RNG state on every Python frame entry. The old ``dis``-based build
entered several helper frames per instruction, so one cold build of a large
code object cost seconds under the hook (98% of the full-tier time of
``test_tap_records_without_modifying_output``). These tests pin the build to
zero frame entries, its result to the ``dis``-derived reference map, and the
RNG monitor's witnessing of user frames to unchanged.
"""

from __future__ import annotations

import dis
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.utils import introspection
from torchlens.utils.rng import host_nondeterminism_monitor

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="column offsets need co_positions (Python 3.11+)"
)


def _reference_col_offset_map(code: types.CodeType) -> dict[int, int | None]:
    """Return the ``dis``-derived map: each column spread over its cache region."""

    offset_map: dict[int, int | None] = {}
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        positions = instruction.positions
        col_offset = None if positions is None else positions.col_offset
        next_offset = (
            instructions[index + 1].offset if index + 1 < len(instructions) else len(code.co_code)
        )
        for offset in range(instruction.offset, max(next_offset, instruction.offset + 2), 2):
            offset_map[offset] = col_offset
    return offset_map


def _walk_codes(code: types.CodeType) -> Iterator[types.CodeType]:
    """Yield ``code`` and every code object nested in its constants."""

    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _walk_codes(const)


def _sample_codes() -> list[types.CodeType]:
    """Return a varied code-object sample: method calls, loops, closures, big bodies."""

    def method_calls(x: torch.Tensor) -> torch.Tensor:
        """Exercise method calls, attribute loads, and binary ops."""
        return x.sum() + x.mean() if x.numel() > 1 else x.abs().max()

    roots = [
        method_calls.__code__,
        torch.nn.Module._call_impl.__code__,
        torch.nn.functional.relu.__code__,
        dis._get_instructions_bytes.__code__,
        _reference_col_offset_map.__code__,
    ]
    return [code for root in roots for code in _walk_codes(root)]


def _runs_beneath(frame: types.FrameType, code: types.CodeType) -> bool:
    """Return whether ``frame`` executes strictly beneath a frame running ``code``."""

    caller = frame.f_back
    while caller is not None:
        if caller.f_code is code:
            return True
        caller = caller.f_back
    return False


@contextmanager
def _count_frames_entered_beneath(code: types.CodeType) -> Iterator[list[str]]:
    """Record every Python frame entered strictly beneath a frame running ``code``."""

    entered: list[str] = []
    previous = sys.getprofile()

    def profile(frame: types.FrameType, event: str, arg: Any) -> None:
        if event == "call" and _runs_beneath(frame, code):
            entered.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        yield entered
    finally:
        sys.setprofile(previous)


def test_col_offset_map_matches_dis_reference() -> None:
    """The ``co_positions`` map equals the ``dis`` map, cache regions included."""

    codes = _sample_codes()
    assert len(codes) >= 5
    for code in codes:
        built = introspection._build_col_offset_map(code)
        assert built == _reference_col_offset_map(code), code.co_qualname
        assert sorted(built) == list(range(0, len(code.co_code), 2)), code.co_qualname


def test_col_offset_map_build_enters_no_python_frame() -> None:
    """Building the map for a large code object enters zero Python frames.

    The ``dis``-based build entered thousands for ``Module._call_impl``.
    """

    code = torch.nn.Module._call_impl.__code__
    assert len(code.co_code) > 400
    build_code = introspection._build_col_offset_map.__code__
    with _count_frames_entered_beneath(build_code) as entered:
        offset_map = introspection._build_col_offset_map(code)
    assert offset_map
    assert entered == []


class _ReluBranch(torch.nn.Module):
    """Model whose forward frame the RNG monitor must still witness."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU then scale."""

        return torch.relu(x) * 2


def test_cold_map_build_in_hooked_capture_triggers_no_rng_frame_snapshots() -> None:
    """A cold map build inside an intervention_ready capture costs no RNG snapshots.

    Positive control: the monitor still snapshots the user model's forward frame
    in the same window, so the witness itself is untouched.
    """

    original_build = introspection._build_col_offset_map
    original_snapshot = host_nondeterminism_monitor._snapshot_numpy_frame_rngs
    state = {"in_build": 0, "builds": 0}
    snapshots_in_build: list[str] = []
    snapshotted_codes: set[types.CodeType] = set()

    def counting_build(code: types.CodeType) -> dict[int, int | None]:
        state["in_build"] += 1
        state["builds"] += 1
        try:
            return original_build(code)
        finally:
            state["in_build"] -= 1

    build_code = original_build.__code__

    def counting_snapshot(self: Any, frame: types.FrameType) -> None:
        if state["in_build"] and _runs_beneath(frame, build_code):
            snapshots_in_build.append(frame.f_code.co_name)
        snapshotted_codes.add(frame.f_code)
        original_snapshot(self, frame)

    model = _ReluBranch()
    tap = tl.tap(tl.func("relu"))
    introspection._clear_col_offset_cache()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(introspection, "_build_col_offset_map", counting_build)
        patcher.setattr(
            host_nondeterminism_monitor, "_snapshot_numpy_frame_rngs", counting_snapshot
        )
        hooked = tl.trace(
            model,
            torch.tensor([[-1.0, 2.0]]),
            capture=tl.options.CaptureOptions(intervention_ready=True, hooks=tap),
        )

    assert tap.values()
    assert hooked.output_layers
    assert state["builds"] > 0, "the cleared cache must force cold map builds"
    assert snapshots_in_build == []
    assert _ReluBranch.forward.__code__ in snapshotted_codes
