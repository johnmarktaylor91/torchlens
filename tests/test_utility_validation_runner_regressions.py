"""Regression locks for three utility/validation-runner bugs.

Covers (audit IDs from ``.project-context/todos.md``):

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

import gc
import importlib.util
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

import torchlens.utils.hashing as tl_hashing
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.utils.hashing import make_short_barcode_from_input
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


def test_func_call_location_releases_function_object_with_source_loading() -> None:
    """No strong ref to the frame func obj survives construction (lazy path)."""

    victim, victim_ref, line_number = _make_victim_func()
    loc = FuncCallLocation(
        file=__file__,
        line_number=line_number,
        func_name="_victim",
        num_context_lines_requested=2,
        _frame_func_obj=victim,
        source_loading_enabled=True,
    )

    del victim
    gc.collect()

    # The leak regression: pre-fix, _frame_func_obj kept the function (and
    # its closure/payload) alive until _load_source() ran — which normal
    # usage (accessing only .file) never triggered.
    assert victim_ref() is None
    assert loc._frame_func_obj is None

    # Metadata snapshot still captured at construction. (Under
    # ``from __future__ import annotations`` the signature renders with
    # quoted annotations, so match leniently.)
    assert "x:" in (loc.func_signature or "")
    assert "int" in (loc.func_signature or "")
    assert "Victim docstring" in (loc.func_docstring or "")


def test_func_call_location_lazy_source_contract_preserved() -> None:
    """Non-lazy ``.file`` does not trigger loading; lazy props still work."""

    victim, victim_ref, line_number = _make_victim_func()
    loc = FuncCallLocation(
        file=__file__,
        line_number=line_number,
        func_name="_victim",
        num_context_lines_requested=2,
        _frame_func_obj=victim,
        source_loading_enabled=True,
    )
    del victim
    gc.collect()
    assert victim_ref() is None

    # Normal usage path: plain attribute access must NOT trigger source load.
    assert loc.file == __file__
    assert loc.line_number == line_number
    assert loc._source_loaded is False

    # Lazy source load still works correctly AFTER the function obj is gone.
    assert "TL-LAZY-SOURCE-MARKER" in loc.call_line
    assert loc._source_loaded is True
    assert "TL-LAZY-SOURCE-MARKER" in loc.source_context
    assert loc.code_context is not None
    assert loc.num_context_lines == len(loc.code_context)
    assert "--->" in loc.code_context_labeled


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


def test_func_call_location_source_loading_disabled_unchanged() -> None:
    """``source_loading_enabled=False`` keeps the no-source contract (D17)."""

    victim, victim_ref, line_number = _make_victim_func()
    loc = FuncCallLocation(
        file=__file__,
        line_number=line_number,
        func_name="_victim",
        num_context_lines_requested=2,
        _frame_func_obj=victim,
        source_loading_enabled=False,
    )
    del victim
    gc.collect()

    assert victim_ref() is None
    assert loc._frame_func_obj is None
    assert loc._source_loaded is True
    assert loc.code_context is None
    assert loc.source_context == "None"
    assert loc.func_signature is None
    assert loc.func_docstring is None


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

    Loads ``torchlens/utils/hashing.py`` standalone (stdlib-only imports) so
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


def test_short_barcode_deterministic_across_calls() -> None:
    """Repeated calls with identical input return identical barcodes."""

    payload = ["layer_type", 7, (1, 2, 3), "kw_stride_2"]
    assert make_short_barcode_from_input(payload) == make_short_barcode_from_input(payload)


def test_short_barcode_deterministic_across_processes() -> None:
    """Barcodes are PYTHONHASHSEED-independent (pre-fix ``hash()`` was salted)."""

    in_process = make_short_barcode_from_input(_CROSS_PROCESS_PAYLOAD)
    seed_zero = _barcode_in_subprocess("0")
    seed_other = _barcode_in_subprocess("424242")

    assert seed_zero == in_process
    assert seed_other == in_process


def test_short_barcode_separator_prevents_concatenation_collisions() -> None:
    """Adjacent-element concatenations must not collide."""

    assert make_short_barcode_from_input(["ab", "c"]) != make_short_barcode_from_input(["a", "bc"])
    assert make_short_barcode_from_input(["1", 2]) != make_short_barcode_from_input([12])
    assert make_short_barcode_from_input([""]) != make_short_barcode_from_input(["", ""])


def test_short_barcode_no_collisions_at_scale() -> None:
    """Tens of thousands of distinct structured inputs yield distinct barcodes."""

    barcodes = {
        make_short_barcode_from_input([f"pos{i % 7}_tensortorch.Size([{i}, {j}])", i * 31 + j])
        for i in range(200)
        for j in range(200)
    }
    assert len(barcodes) == 200 * 200


def test_short_barcode_length_and_alphabet() -> None:
    """Barcodes honor ``barcode_len`` and stay within the hex alphabet."""

    for barcode_len in (8, 16, 32):
        barcode = make_short_barcode_from_input(["abc", 1], barcode_len=barcode_len)
        assert len(barcode) == barcode_len
        assert all(ch in "0123456789abcdef" for ch in barcode)


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

    def __deepcopy__(self, memo: dict[int, Any]) -> "_BufferMutatingFailModel":
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


@pytest.mark.parametrize(
    ("failure_site", "expected_original_calls"),
    [("ground_truth", 0), ("traced", 1)],
)
def test_validate_forward_pass_restores_buffer_bit_exact_on_exception(
    failure_site: ValidationFailureSite,
    expected_original_calls: int,
) -> None:
    """Registered buffer state is restored bit-exact when forward raises.

    Parametrized over the failing validation site: the deep-copied
    ground-truth run and the original traced run. Both must leave the original
    model exactly as it was before
    ``validate_forward_pass`` was called, with the original exception
    propagating.
    """

    model = _BufferMutatingFailModel(fail_during=failure_site)
    pre_call_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    with pytest.raises(RuntimeError, match="intentional mid-pass failure"):
        validate_forward_pass(model, torch.ones(2), random_seed=1)

    assert model.num_calls == expected_original_calls
    post_call_state = model.state_dict()
    assert set(post_call_state) == set(pre_call_state)
    for name, expected in pre_call_state.items():
        assert torch.equal(post_call_state[name], expected), name
    assert torch.equal(model.running_total, _INITIAL_BUFFER)


def test_validate_forward_pass_snapshot_is_isolated_from_inplace_mutation() -> None:
    """The pre-call snapshot must be clones, not live references.

    Pre-fix, ``validate_forward_pass`` snapshotted ``model.state_dict()``
    directly; ``state_dict`` values are references, so in-place mutation
    during the forward also mutated the snapshot, making any later restore a
    no-op. The clone requirement is what makes the bit-exact restore in the
    exception test meaningful; this test asserts it directly via the
    module-level helper.
    """

    from torchlens.user_funcs import _clone_state_dict_with_metadata

    model = _BufferMutatingFailModel(fail_during="never")
    snapshot = _clone_state_dict_with_metadata(model)

    model.running_total.add_(100.0)

    assert torch.equal(snapshot["running_total"], _INITIAL_BUFFER)
    assert not torch.equal(snapshot["running_total"], model.running_total)
