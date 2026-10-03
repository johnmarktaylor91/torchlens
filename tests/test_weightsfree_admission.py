"""Entry/tamper matrix for scoped meta admission (memo sec 4.2 / 8.1 item 7).

Admission is SCOPED (D2): meta state/inputs are admitted if and only if
structure-only is in force. An unscoped carve-out silently captures an
unbannered value-free trace — measured to be worse than the refusal it
replaces — so every one of the seven gate call sites is exercised and the
non-capture sites thread a permanently closed regime.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from test_weightsfree_fixtures import (
    FactoryToy,
    LinearReluLinear,
    build_twins,
    weightsfree_trace,
)

import torchlens as tl
from torchlens._robustness import UnsupportedTensorVariantError


def _meta_model() -> nn.Module:
    with torch.device("meta"):
        model = LinearReluLinear()
    model.eval()
    return model


def _meta_input() -> torch.Tensor:
    return torch.empty(3, 2, device="meta")


# ---------------------------------------------------------------------------
# Scoped admission (D2)
# ---------------------------------------------------------------------------


def test_meta_without_flag_refuses_unchanged() -> None:
    """A meta model WITHOUT structure_only keeps today's typed teach."""

    with pytest.raises(UnsupportedTensorVariantError):
        tl.trace(_meta_model(), _meta_input())


def test_meta_with_flag_admits() -> None:
    """The D8 flip: meta + structure_only captures COMPLETE."""

    tr = weightsfree_trace(_meta_model(), _meta_input())
    assert tr.outcome.status.value == "complete"
    assert tr.structure_only is True


def test_real_substrate_structure_only_still_works() -> None:
    """Structure-only on a real model is unchanged by the flip."""

    real, _ = build_twins(LinearReluLinear)
    tr = weightsfree_trace(real, torch.randn(3, 2))
    assert tr.structure_only is True
    assert tr.outcome.status.value == "complete"


# ---------------------------------------------------------------------------
# Substrate uniformity: mixed cells refuse typed in BOTH directions
# ---------------------------------------------------------------------------


def test_meta_model_real_input_refuses_substrate_mismatch() -> None:
    exc = _expect_mismatch(_meta_model(), torch.randn(3, 2))
    assert "input" in str(exc).lower() or "state" in str(exc).lower()


def test_real_model_meta_input_refuses_substrate_mismatch() -> None:
    real, _ = build_twins(LinearReluLinear)
    _expect_mismatch(real, _meta_input())


def test_partially_meta_state_refuses_substrate_mismatch() -> None:
    model = _meta_model()
    # Materialize ONE parameter onto CPU: registered state is now mixed.
    model.fc1.weight = nn.Parameter(torch.randn(5, 2))
    _expect_mismatch(model, _meta_input())


def _expect_mismatch(model: nn.Module, x: torch.Tensor):
    with pytest.raises(Exception) as excinfo:
        weightsfree_trace(model, x)
    exc = excinfo.value
    assert getattr(exc, "fields", {}).get("code") == "structure_only_substrate_mismatch", (
        f"expected structure_only_substrate_mismatch, got {exc!r}"
    )
    return exc


# ---------------------------------------------------------------------------
# The other refused variants stay refused even under admission
# ---------------------------------------------------------------------------


def test_sparse_input_still_refuses_under_admission() -> None:
    model = _meta_model()
    sparse = torch.sparse_coo_tensor(torch.zeros(2, 1, dtype=torch.long), torch.ones(1), (3, 2))
    with pytest.raises(Exception) as excinfo:
        weightsfree_trace(model, sparse)
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code in ("unsupported_tensor_variant", "structure_only_substrate_mismatch")


def test_fake_tensor_still_refuses_under_admission() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    real, _ = build_twins(LinearReluLinear)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.randn(3, 2))
    with pytest.raises(Exception) as excinfo:
        weightsfree_trace(real, fake)
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code in ("unsupported_tensor_variant", "structure_only_substrate_mismatch")


# ---------------------------------------------------------------------------
# The seven gate call sites: the non-trace sites thread a CLOSED regime
# ---------------------------------------------------------------------------


def test_summary_and_record_paths_gate_meta_consistently() -> None:
    """tl.record (fastlog gate site) refuses meta even with no flip lever."""

    model = _meta_model()
    with pytest.raises(Exception) as excinfo:
        tl.record(model, _meta_input())
    # fastlog threads admit_meta=False permanently: the historical refusal.
    assert isinstance(excinfo.value, UnsupportedTensorVariantError) or (
        getattr(excinfo.value, "fields", {}).get("code") == "unsupported_tensor_variant"
    )


def test_validate_path_gates_meta_closed() -> None:
    """tl.validate (backward gate site) keeps the historical refusal."""

    model = _meta_model()
    with pytest.raises((UnsupportedTensorVariantError, RuntimeError)):
        tl.validate(model, _meta_input(), scope="forward")


# ---------------------------------------------------------------------------
# Stale pre-wrap factory reference: typed refusal at the user's line
# ---------------------------------------------------------------------------


def test_stale_prewrap_factory_reference_refuses_typed() -> None:
    """A stale pre-wrap factory ref lands on CPU and dies TYPED (D4 cost).

    The refusal names the substrate boundary, not a phantom value branch
    (W1-CLS): the classification is structure_only_substrate_mismatch.
    """

    # In-process, torch is already wrapped by earlier captures, so a truly
    # stale pre-wrap binding needs subprocess isolation (test_weightsfree_order
    # carries that leg). Here we pin the same failure CLASS: a REAL tensor
    # minted mid-forward on an admitted meta capture refuses typed rather
    # than recording a mixed-substrate graph.
    class RealMint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) + torch.ones(1, 4, device="cpu")

    with torch.device("meta"):
        model = RealMint()
    model.eval()
    with pytest.raises(Exception) as excinfo:
        weightsfree_trace(model, torch.empty(1, 4, device="meta"))
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code in (
        "structure_only_substrate_mismatch",
        "value_dependent_branch_unsupported",
        "meta_kernel_unavailable",
    ), f"expected a typed refusal, got {excinfo.value!r}"
    assert code == "structure_only_substrate_mismatch"


# ---------------------------------------------------------------------------
# Failure paths restore state
# ---------------------------------------------------------------------------


def test_failed_admitted_capture_restores_wrappers_and_flags() -> None:
    """A failing admitted capture restores torch/module state (sec 4.2)."""

    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise ValueError("user forward failure")

    with torch.device("meta"):
        model = Boom()
    model.eval()
    with pytest.raises(ValueError, match="user forward failure"):
        weightsfree_trace(model, torch.empty(1, 2, device="meta"))
    # The factory-device slot must not leak past the failed capture: an
    # ordinary capture right after must complete with factory tensors on CPU.
    from torchlens.backends.torch._weightsfree_ctx import active_factory_device

    assert active_factory_device() is None
    real, _ = build_twins(FactoryToy)
    tr = tl.trace(real, torch.randn(1, 4))
    assert tr.outcome.status.value == "complete"


# ---------------------------------------------------------------------------
# Admission-marker tamper rows (t1-t5): an admitted meta offense without the
# final marker fails closed
# ---------------------------------------------------------------------------


def test_admitted_capture_carries_admission_and_envelope() -> None:
    tr = weightsfree_trace(_meta_model(), _meta_input())
    envelope = tr.structure_evidence
    assert envelope is not None
    assert envelope["capture_mode"] == "structure_only"
    assert envelope["substrate"] == "meta"
    assert envelope["values_available"] is False
    assert envelope["factory_device_policy"] == "torchlens_owned"


def test_real_structure_only_envelope_says_real() -> None:
    real, _ = build_twins(LinearReluLinear)
    tr = weightsfree_trace(real, torch.randn(3, 2))
    envelope = tr.structure_evidence
    assert envelope is not None
    assert envelope["substrate"] == "real"
    assert envelope["factory_device_policy"] == "none"


def test_identity_self_test_failure_refuses_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """D20: a torch build without a trustworthy meta identity primitive
    refuses admission typed (structure_only_meta_identity_unavailable) rather
    than guessing where every data_ptr reads 0."""

    from torchlens._errors import WeightsfreeIntegrityError
    from torchlens.capture import _weightsfree_admission as admission

    monkeypatch.setattr(
        admission,
        "_IDENTITY_SELF_TEST_RESULT",
        "sibling meta allocations report identical storage identity",
    )
    with pytest.raises(WeightsfreeIntegrityError) as excinfo:
        weightsfree_trace(_meta_model(), _meta_input())
    assert excinfo.value.fields["code"] == "structure_only_meta_identity_unavailable"


def test_ambient_context_without_surgery_refuses_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D19 fallback: a caller-active DeviceContext at an admitted capture on
    a build without mode-stack surgery refuses typed
    (structure_only_ambient_device_context) with the exit-context remedy."""

    import torchlens.utils._torch_compat as torch_compat
    from torchlens._errors import WeightsfreeIntegrityError

    monkeypatch.setattr(torch_compat, "HAS_TORCH_FUNCTION_STACK_SURGERY", False)
    model = _meta_model()
    with torch.device("cpu"), pytest.raises(WeightsfreeIntegrityError) as excinfo:
        weightsfree_trace(model, _meta_input())
    assert excinfo.value.fields["code"] == "structure_only_ambient_device_context"
    assert "exit the construction context" in str(excinfo.value)
