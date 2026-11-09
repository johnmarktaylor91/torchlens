"""SDPA capability probes survive a caller's forced SDPA backend.

The SDPA probes behind ``HAS_SDPA_GQA`` and ``HAS_SDPA_EMPTY_ROW_ZERO`` run lazily,
inside the first capture of a process, on small CPU tensors. A caller who wraps that
first capture in ``sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`` leaves the probes'
CPU calls no backend (the memory-efficient kernel has no CPU implementation), which
used to raise from inside ``tl.trace``. Forcing that backend on CPU reproduces it.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from torchlens._utils import _kernel_probes, _torch_compat as tc

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # torch 2.1-2.2: no sdpa_kernel to force a backend with
    SDPBackend = sdpa_kernel = None  # type: ignore[assignment,misc]

requires_sdpa_kernel = pytest.mark.skipif(
    sdpa_kernel is None, reason="torch.nn.attention.sdpa_kernel postdates torch 2.2"
)


def _forced_efficient() -> contextlib.AbstractContextManager[object]:
    """Force the memory-efficient SDPA backend, which refuses every CPU call."""

    assert sdpa_kernel is not None and SDPBackend is not None
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def _assert_forced_backend_refuses_cpu() -> None:
    """Precondition: under the forced backend a plain CPU SDPA call is refused."""

    query = torch.ones(1, 1, 1, 1)
    with pytest.raises(RuntimeError, match="No viable backend|No available kernel"):
        F.scaled_dot_product_attention(query, query, query)


@requires_sdpa_kernel
def test_sdpa_probes_answer_unchanged_under_a_refusing_forced_backend() -> None:
    """Both SDPA probes return their unrestricted answers and leave the caller's restriction on."""

    with _forced_efficient():
        _assert_forced_backend_refuses_cpu()
        gqa = _kernel_probes.probe_sdpa_gqa()
        empty_row_zero = _kernel_probes.probe_sdpa_empty_row_zero()
        assert not torch.backends.cuda.math_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
    assert gqa is tc.HAS_SDPA_GQA
    assert empty_row_zero is tc.HAS_SDPA_EMPTY_ROW_ZERO


@requires_sdpa_kernel
def test_sdpa_probes_degrade_without_raising_when_no_backend_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the math pin unavailable, a backend refusal is answered, never raised."""

    monkeypatch.setattr(_kernel_probes, "_sdpa_math_backend", contextlib.nullcontext)
    with _forced_efficient():
        _assert_forced_backend_refuses_cpu()
        gqa = _kernel_probes.probe_sdpa_gqa()
        empty_row_zero = _kernel_probes.probe_sdpa_empty_row_zero()
    assert gqa is tc.HAS_SDPA_GQA  # read from SDPA's documented signature
    assert empty_row_zero is False  # unknown: the conservative older-torch answer


def test_sdpa_probes_still_surface_an_unrelated_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the no-backend refusal is absorbed; any other failure still raises."""

    def _broken(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("unrelated kernel failure")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _broken)
    with pytest.raises(RuntimeError, match="unrelated kernel failure"):
        _kernel_probes.probe_sdpa_gqa()
    with pytest.raises(RuntimeError, match="unrelated kernel failure"):
        _kernel_probes.probe_sdpa_empty_row_zero()


_FIRST_CAPTURE_SCRIPT = textwrap.dedent(
    """
    import sys

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import torchlens as tl

    assert "torchlens._utils._torch_compat" not in sys.modules, "probes already ran"
    model = torch.nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        trace = tl.trace(model, x)
        assert not torch.backends.cuda.math_sdp_enabled()
    assert "torchlens._utils._torch_compat" in sys.modules
    assert torch.equal(trace.output, model(x))
    print("first capture ok")
    """
)


@pytest.mark.heavy
@requires_sdpa_kernel
def test_first_capture_of_a_process_under_a_refusing_forced_backend() -> None:
    """A fresh process's first ``tl.trace`` inside a refusing forced backend succeeds."""

    result = subprocess.run(
        [sys.executable, "-c", _FIRST_CAPTURE_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "first capture ok" in result.stdout
