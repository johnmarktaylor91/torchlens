"""Per-backend ``fp32_precision`` policies (torch >= 2.9) vs the ambient record.

torch 2.9 added per-backend ``torch.backends.*.fp32_precision`` knobs. The legacy
getters the v2 ambient record reads (``cudnn.allow_tf32``,
``cuda.matmul.allow_tf32``, ``get_float32_matmul_precision()``) RAISE when the
per-backend state has no legacy equivalent, which used to crash every
``intervention_ready`` capture and ``tl.debug.check_determinism``. Capture must
succeed; a policy the record cannot represent must refuse runnable save typed
(``execution_context_unavailable``), never persist a record that replays under a
different TF32/bf16 policy; and a run transaction must restore the caller's
exact per-backend policy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode
from torchlens.utils import _torch_compat as tc

pytestmark = pytest.mark.skipif(
    not tc.HAS_FP32_PRECISION_CONTROLS,
    reason="per-backend fp32_precision API needs torch >= 2.9",
)

_CAPTURE = CaptureOptions(intervention_ready=True)


def _set(path: str, value: str) -> None:
    holder: object = torch.backends
    for part in path.split("."):
        holder = getattr(holder, part)
    setattr(holder, "fp32_precision", value)


_SCENARIOS: dict[str, tuple[Callable[[], None], tuple[str, ...]]] = {
    "cudnn_conv_tf32_rnn_ieee": (
        lambda: (_set("cudnn.conv", "tf32"), _set("cudnn.rnn", "ieee")),
        ("cudnn_allow_tf32",),
    ),
    "cuda_matmul_new_api_tf32": (
        lambda: _set("cuda.matmul", "tf32"),
        ("float32_matmul_precision", "cuda_matmul_allow_tf32"),
    ),
    "mkldnn_matmul_bf16": (
        lambda: _set("mkldnn.matmul", "bf16"),
        ("float32_matmul_precision",),
    ),
    "mkldnn_conv_tf32": (
        # The legacy getters all read cleanly here, but no legacy field can
        # express a CPU oneDNN conv TF32 policy: still unrepresentable.
        lambda: _set("mkldnn.conv", "tf32"),
        ("fp32_precision[mkldnn.conv]=tf32",),
    ),
}


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.head = nn.Linear(4 * 6 * 6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)).flatten(1))


@pytest.fixture(autouse=True)
def _restore_fp32_policy() -> Iterator[None]:
    """Put the process back to its exact starting fp32 policy after each test."""

    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    policy = tc.snapshot_fp32_precision_controls()
    yield
    # The legacy setters also own torch's hidden legacy flags (which the legacy
    # getters cross-check against the knobs), so restore them before the knobs.
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    tc.restore_fp32_precision_controls(policy)
    assert tc.read_legacy_fp32_controls()[1] == ()


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _ConvNet().eval(), torch.randn(1, 3, 8, 8)


@pytest.mark.smoke
@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_unrepresentable_policy_captures_and_refuses_runnable_save(
    scenario: str, tmp_path: Path
) -> None:
    from torchlens.errors import RunnablePreflightError

    setup, expected = _SCENARIOS[scenario]
    setup()
    model, x = _model_and_input()

    trace = tl.trace(model, x, capture=_CAPTURE)

    ambient = trace._runnable.capture_ambient
    assert ambient[tc.AMBIENT_FP32_UNREPRESENTABLE_KEY] == expected
    for field in expected:
        if not field.startswith("fp32_precision["):
            assert ambient[field] is None
    path = tmp_path / f"{scenario}.tlspec"
    with pytest.raises(RunnablePreflightError) as excinfo:
        tl.save(trace, str(path), level="runnable")
    diagnostics = excinfo.value.fields["diagnostics"]
    codes = {diag.code for diag in diagnostics}
    assert RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE in codes
    assert not path.exists()
    # Analysis capture and an analysis save stay available.
    tl.save(trace, str(tmp_path / f"{scenario}-analysis.tlspec"))


@pytest.mark.smoke
def test_legacy_set_policy_stays_representable_and_verified(tmp_path: Path) -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")
    assert tc.read_legacy_fp32_controls()[1] == ()
    model, x = _model_and_input()

    trace = tl.trace(model, x, capture=_CAPTURE)
    assert tc.AMBIENT_FP32_UNREPRESENTABLE_KEY not in trace._runnable.capture_ambient
    path = tmp_path / "legacy.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)

    report = tl.load(str(path)).run(inputs=x.clone(), seed=0).report
    assert report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_replay_restores_the_callers_exact_mixed_policy(tmp_path: Path) -> None:
    model, x = _model_and_input()
    path = tmp_path / "clean.tlspec"
    tl.save(tl.trace(model, x, capture=_CAPTURE), str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))

    _set("cudnn.conv", "tf32")
    _set("cudnn.rnn", "ieee")
    _set("mkldnn.conv", "tf32")
    caller_policy = tc.snapshot_fp32_precision_controls()

    report = loaded.run(inputs=x.clone(), seed=0).report

    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert tc.snapshot_fp32_precision_controls() == caller_policy


@pytest.mark.smoke
def test_check_determinism_runs_under_mixed_policy() -> None:
    _set("cudnn.conv", "tf32")
    _set("cudnn.rnn", "ieee")
    model, x = _model_and_input()

    report = tl.debug.check_determinism(model, x)

    assert report.verdict is not None
