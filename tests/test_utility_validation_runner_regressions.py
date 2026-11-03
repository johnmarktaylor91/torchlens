"""Regression locks for three utility/validation-runner bugs.

Covers (audit IDs from the private maintainer task list):

1. **FUNC-CALL-LOC-LEAK** — ``FuncCallLocation`` must not retain a strong
   reference to the captured frame's function object when source loading is
   enabled, while still honoring the lazy source-load contract.
2. **HASH-COLLISION** — ``make_short_barcode_from_input`` must be a stable,
   strong, process-independent hash (the pre-fix implementation used Python's
   per-process-salted ``hash()`` plus lossy base64/decimal truncation).
3. **VALIDATE-STATE-RESTORE** — ``validate_forward_pass`` must restore the
   model's registered state bit-exact even when the forward pass raises
   (including in-place buffer mutation before the raise), on both the
   ground-truth call and the traced call.

These tests are tripwire-preserving: they assert state restoration and hash
stability only; they do not alter what validation checks.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import weakref
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
import torch.nn as nn

import torchlens._utils.hashing as tl_hashing
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens._utils.hashing import make_short_barcode_from_input
from torchlens.errors import TraceCaptureError
from torchlens.validation import validate_forward_pass

# ---------------------------------------------------------------------------
# Bug 1: FUNC-CALL-LOC-LEAK
# ---------------------------------------------------------------------------


def _make_victim_func() -> tuple:
    """Create a throwaway closure-bearing function for leak tests.

    The function closes over a large payload so that retaining it would be
    a real memory leak, and is created dynamically so that deleting the
    returned reference is sufficient for it to be collected.

    Returns
    -------
    tuple
        ``(func, weak_ref, payload_weak_ref, call_line_number)`` where
        ``call_line_number`` points at the marker line inside the body.
    """

    big_payload = bytearray(1024)

    def _victim(x: int) -> int:
        """Victim docstring for leak tests."""
        return x + len(big_payload)  # TL-LAZY-SOURCE-MARKER

    line_number = _victim.__code__.co_firstlineno + 2
    return _victim, weakref.ref(_victim), line_number


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_func_call_location_pickles_before_and_after_lazy_load() -> None:
    """Pickle round-trips work pre- and post-load and never carry the func."""

    victim, _, line_number = _make_victim_func()
    loc = FuncCallLocation(
        file=__file__,
        line_number=line_number,
        func_name="_victim",
        num_context_lines_requested=2,
        _frame_func_obj=victim,
        source_loading_enabled=True,
    )
    del victim

    # Pickle BEFORE the lazy load: the clone must still load lazily.
    early_clone = pickle.loads(pickle.dumps(loc))
    assert early_clone._frame_func_obj is None
    assert "TL-LAZY-SOURCE-MARKER" in early_clone.call_line

    # Pickle AFTER the lazy load: cached source survives.
    assert "TL-LAZY-SOURCE-MARKER" in loc.call_line
    late_clone = pickle.loads(pickle.dumps(loc))
    assert late_clone._frame_func_obj is None
    assert late_clone.source_context == loc.source_context


# ---------------------------------------------------------------------------
# Bug 2: HASH-COLLISION
# ---------------------------------------------------------------------------

_CROSS_PROCESS_PAYLOAD = ["ab", "c", 123, (4, 5), "pos0_tensortorch.Size([2, 3])"]

_SUBPROCESS_SNIPPET = """\
import importlib.util
spec = importlib.util.spec_from_file_location("tl_hashing_standalone", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.make_short_barcode_from_input({payload!r}), end="")
"""


def _barcode_in_subprocess(hash_seed: str) -> str:
    """Compute the reference barcode in a fresh process with a given seed.

    Loads ``torchlens/_utils/hashing.py`` standalone (stdlib-only imports) so
    the subprocess does not pay the full ``import torch`` cost.

    Parameters
    ----------
    hash_seed:
        Value for ``PYTHONHASHSEED`` in the child process.

    Returns
    -------
    str
        The barcode printed by the child process.
    """

    script = _SUBPROCESS_SNIPPET.format(
        path=str(Path(tl_hashing.__file__)), payload=_CROSS_PROCESS_PAYLOAD
    )
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout


def test_short_barcode_deterministic_across_processes() -> None:
    """Barcodes are PYTHONHASHSEED-independent (pre-fix ``hash()`` was salted)."""

    in_process = make_short_barcode_from_input(_CROSS_PROCESS_PAYLOAD)
    seed_zero = _barcode_in_subprocess("0")
    seed_other = _barcode_in_subprocess("424242")

    assert seed_zero == in_process
    assert seed_other == in_process


# ---------------------------------------------------------------------------
# Bug 3: VALIDATE-STATE-RESTORE
# ---------------------------------------------------------------------------

_INITIAL_BUFFER = torch.tensor([3.0, -1.5])


ValidationFailureSite = Literal["ground_truth", "traced", "never"]


class _BufferMutatingFailModel(nn.Module):
    """Model that mutates a registered buffer in-place, then raises.

    ``fail_during`` selects whether the deep-copied ground-truth run or the
    original traced run raises inside ``validate_forward_pass``.
    """

    running_total: torch.Tensor

    def __init__(self, fail_during: ValidationFailureSite) -> None:
        """Register the mutable buffer and configure the failing call.

        Parameters
        ----------
        fail_during:
            Validation run site that raises.
        """

        super().__init__()
        self.register_buffer("running_total", _INITIAL_BUFFER.clone())
        self.fail_during = fail_during
        self.num_calls = 0
        self._is_ground_truth_copy = False

    def __deepcopy__(self, memo: dict[int, Any]) -> _BufferMutatingFailModel:
        """Return a validation copy marked as the ground-truth runner.

        Parameters
        ----------
        memo:
            Deep-copy memo table.

        Returns
        -------
        _BufferMutatingFailModel
            Copy with cloned registered state and a ground-truth marker.
        """

        clone = type(self)(fail_during=self.fail_during)
        memo[id(self)] = clone
        clone.load_state_dict(self.state_dict())
        clone._is_ground_truth_copy = True
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mutate the buffer in-place; raise on the configured invocation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Doubled input (only on non-failing invocations).
        """

        self.num_calls += 1
        # In-place mutation BEFORE the raise: if the state snapshot held live
        # references instead of clones, this would corrupt the snapshot too.
        self.running_total.add_(x.sum())
        should_fail_ground_truth = self.fail_during == "ground_truth" and self._is_ground_truth_copy
        should_fail_traced = self.fail_during == "traced" and not self._is_ground_truth_copy
        if should_fail_ground_truth or should_fail_traced:
            raise RuntimeError("intentional mid-pass failure")
        return x * 2.0


# Kept by shrink-r4 (per-id subset rule): sole coverage of _plain_attr_fidelity.py, _validation_argument_references.py.
@pytest.mark.parametrize(
    ("failure_site", "expected_original_calls"),
    [("traced", 0)],
)
def test_validate_forward_pass_restores_buffer_bit_exact_on_exception(
    failure_site: ValidationFailureSite,
    expected_original_calls: int,
) -> None:
    """Registered buffer state is restored bit-exact when forward raises.

    Parametrized over the failing validation site: the deep-copied
    ground-truth run and the isolated traced validation run. Both must leave the
    original model exactly as it was before
    ``validate_forward_pass`` was called, with the original exception
    propagating.
    """

    model = _BufferMutatingFailModel(fail_during=failure_site)
    pre_call_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    # The ground-truth forward runs outside any capture, so the model's own
    # exception propagates; the traced run is a capture, so it raises
    # TraceCaptureError chained from it.
    if failure_site == "ground_truth":
        with pytest.raises(RuntimeError, match="intentional mid-pass failure") as excinfo:
            validate_forward_pass(model, torch.ones(2), seed=1)
        assert not isinstance(excinfo.value, TraceCaptureError)
    else:
        with pytest.raises(TraceCaptureError, match="intentional mid-pass failure") as excinfo:
            validate_forward_pass(model, torch.ones(2), seed=1)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    assert model.num_calls == expected_original_calls
    post_call_state = model.state_dict()
    assert set(post_call_state) == set(pre_call_state)
    for name, expected in pre_call_state.items():
        assert torch.equal(post_call_state[name], expected), name
    assert torch.equal(model.running_total, _INITIAL_BUFFER)
