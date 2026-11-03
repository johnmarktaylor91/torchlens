"""Stable codes on the two refusals the nightly stress matrix found code-less.

``test_typed_refusal_names_condition`` requires every ``TorchLensError`` a stress
cell raises to carry ``fields["code"]``. Two did not: ``TraceCaptureError`` (a
failed forward, every ``hook_raises_plain`` and ``cc_raise_mid_capture`` cell) and
``LiveModeLabelError`` (every ``hook_live_final_label`` cell). The sparse-mm cells
failed for a third reason: the buffer-write tracker could not snapshot a sparse
buffer, so the capture failed inside TorchLens before the forward ran. A sparse
registered buffer now refuses typed at entry; a sparse tensor built inside forward
captures through ``torch.sparse.mm``. These tests provoke each code through its
production raise site and pin the sparse capture.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.arg_positions import FUNC_ARG_SPECS, extract_tensors_and_params
from torchlens._constants import ORIG_TORCH_FUNCS
from torchlens._ir.selector_eval import evaluate
from torchlens._robustness import UnsupportedTensorVariantError
from torchlens._utils._sparse_parts import sparse_component_tensors, sparse_with_values
from torchlens.backends.torch._state_bytes import _state_bytes_equal, _whole_storage_uint8
from torchlens.errors import ArtifactError, TraceCaptureError
from torchlens.intervention.errors import LiveModeLabelError
from torchlens.intervention.hooks import live_selector_matches_site
from torchlens.types import FailureOrigin
from torchlens.validation._perturb import _perturb_layer_outs

_LIVE_SUBJECT = SimpleNamespace(_layer_label_raw="relu_1")


class _PlainUserError(Exception):
    """A user exception with no TorchLens ancestry."""


class _ForwardRaises(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise _PlainUserError("the model's own code raised on purpose")


class _ForwardRaisesInsideTorchLens(nn.Module):
    """A forward whose plain exception is raised from a TorchLens frame."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sparse_component_tensors(x)  # strided input: refuses inside torchlens
        return x


class _SparseAdjacency(nn.Module):
    """The stress matrix's ``mech_sparse_mm`` shape: a sparse COO buffer in ``sparse.mm``."""

    def __init__(self) -> None:
        super().__init__()
        dense = torch.eye(3) + torch.diag(torch.ones(2), 1)
        self.register_buffer("adjacency", dense.to_sparse())
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.adjacency, self.linear(x))


class _SparseInForward(nn.Module):
    """The entry refusal's remedy: a dense buffer made sparse inside forward."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.eye(3) + torch.diag(torch.ones(2), 1))
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.adjacency.to_sparse(), self.linear(x))


class _SampledAddmm(nn.Module):
    """``torch.sparse.sampled_addmm`` with all three tensor operands from distinct ops."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("pattern", torch.eye(3) + torch.diag(torch.ones(2), 1))
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``to_sparse(layout=...)`` is the recorded spelling; ``Tensor.to_sparse_csr``
        # is in torch's override-ignored set and is not decorated.
        pattern = self.pattern.to_sparse(layout=torch.sparse_csr)
        sampled = torch.sparse.sampled_addmm(pattern, self.left(x), self.right(x).t())
        return sampled.to_dense()


class _Spdiags(nn.Module):
    """``torch.sparse.spdiags``: computed diagonals placed at a buffer of offsets."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offsets", torch.tensor([0, -1, 1]))
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        band = torch.sparse.spdiags(self.linear(x), self.offsets, (3, 3))
        return band.to_dense() @ x


class _Spsolve(nn.Module):
    """``torch.sparse.spsolve``: a CSR system matrix and a computed right-hand side."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("system", torch.eye(3) * 2 + torch.diag(torch.ones(2), 1))
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.spsolve(
            self.system.to_sparse(layout=torch.sparse_csr), self.linear(x).squeeze(-1)
        )


def _hook_raises(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    raise _PlainUserError("stress hook raises a plain exception on purpose")


def _hook_scales(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    return out * 0.5


def test_forward_raising_carries_forward_raised() -> None:
    with pytest.raises(TraceCaptureError) as excinfo:
        tl.trace(_ForwardRaises(), [torch.randn(2, 4)])
    error = excinfo.value
    assert error.fields["code"] == "forward_raised"
    assert error.origin is FailureOrigin.USER_OP
    assert isinstance(error.__cause__, _PlainUserError)


def test_hook_raising_plain_carries_forward_raised() -> None:
    """The stress row's repro: a capture-time hook raising a plain exception."""

    with pytest.raises(TraceCaptureError) as excinfo:
        tl.trace(
            nn.Linear(4, 4),
            [torch.randn(2, 4)],
            intervene=tl.when(tl.func("linear"), _hook_raises),
        )
    assert excinfo.value.fields["code"] == "forward_raised"
    assert isinstance(excinfo.value.__cause__, _PlainUserError)


def test_torchlens_frame_raising_carries_capture_internal_error() -> None:
    with pytest.raises(TraceCaptureError) as excinfo:
        tl.trace(_ForwardRaisesInsideTorchLens(), [torch.randn(2, 4)])
    error = excinfo.value
    assert error.fields["code"] == "capture_internal_error"
    assert error.origin is FailureOrigin.TORCHLENS
    assert isinstance(error.__cause__, ValueError)


def test_trace_capture_error_defaults_to_forward_raised() -> None:
    assert TraceCaptureError("message only").fields["code"] == "forward_raised"
    assert (
        TraceCaptureError("x", code="capture_internal_error").fields["code"]
        == "capture_internal_error"
    )


def test_live_finalized_label_hook_refusal_carries_code() -> None:
    """The stress row's repro: a capture-time hook aimed by a finalized label."""

    with pytest.raises(LiveModeLabelError) as excinfo:
        tl.trace(
            nn.Linear(4, 4),
            [torch.randn(2, 4)],
            intervene=tl.when(tl.label("buffer_1:1"), _hook_scales),
        )
    assert excinfo.value.fields["code"] == "label_selector_not_live"
    assert "finalized postprocess label" in str(excinfo.value)


def test_live_evaluation_of_finalized_label_carries_code() -> None:
    with pytest.raises(LiveModeLabelError) as excinfo:
        evaluate(tl.label("relu_1_3"), _LIVE_SUBJECT, lifecycle="live")
    assert excinfo.value.fields["code"] == "label_selector_not_live"


def test_hooks_fallback_refusal_carries_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hooks-side raise fires when evaluation declines without raising."""

    import torchlens._ir.selector_eval as selector_eval

    monkeypatch.setattr(selector_eval, "evaluate", lambda *args, **kwargs: False)
    with pytest.raises(LiveModeLabelError) as excinfo:
        live_selector_matches_site(tl.label("relu_1_3"), _LIVE_SUBJECT)
    assert excinfo.value.fields["code"] == "label_selector_not_live"


def test_live_mode_label_error_defaults_its_code() -> None:
    assert LiveModeLabelError("message only").fields["code"] == "label_selector_not_live"
    explicit = LiveModeLabelError("x", code="rerun_staged_label_unmatchable")
    assert explicit.fields["code"] == "rerun_staged_label_unmatchable"


def test_sparse_buffer_refuses_typed_at_entry() -> None:
    model = _SparseAdjacency().eval()
    with pytest.raises(UnsupportedTensorVariantError) as info:
        tl.trace(model, [torch.randn(3, 4)])
    assert info.value.fields["code"] == "unsupported_tensor_variant"
    assert "model.adjacency" in str(info.value)
    assert model.adjacency.layout is torch.sparse_coo


def test_sparse_in_forward_model_captures() -> None:
    torch.manual_seed(0)
    model = _SparseInForward().eval()
    x = torch.randn(3, 4)
    with torch.no_grad():
        expected = model(x)
    trace = tl.trace(model, [x])
    torch.testing.assert_close(trace[trace.output_layers[0]].out, expected)
    labels = [layer.layer_label for layer in trace.layers]
    assert any(label.startswith("mm_") for label in labels), labels


def test_torch_sparse_ops_are_in_the_wrap_inventory() -> None:
    """``torch.sparse`` re-exports C builtins the overridable crawl never visits."""
    for name in ("mm", "addmm", "sum", "softmax", "log_softmax", "sampled_addmm"):
        assert ("torch.sparse", name) in ORIG_TORCH_FUNCS, name


def test_whole_storage_bytes_cover_sparse_components() -> None:
    sparse = (torch.eye(3) + torch.diag(torch.ones(2), 1)).to_sparse()
    before = _whole_storage_uint8(sparse).clone()
    assert before.dtype is torch.uint8
    assert torch.equal(_whole_storage_uint8(sparse), before)
    sparse._values().mul_(2)
    assert not torch.equal(_whole_storage_uint8(sparse), before)


def test_sparse_component_tensors_per_layout() -> None:
    dense = torch.eye(3) + torch.diag(torch.ones(2), 1)
    assert len(sparse_component_tensors(dense.to_sparse())) == 2
    assert len(sparse_component_tensors(dense.to_sparse_csr())) == 3
    assert len(sparse_component_tensors(dense.to_sparse_csc())) == 3
    with pytest.raises(ValueError, match="no known component decomposition"):
        sparse_component_tensors(dense)


def test_state_bytes_equal_reads_sparse_components() -> None:
    sparse = (torch.eye(3) + torch.diag(torch.ones(2), 1)).to_sparse()
    clone = sparse.detach().clone()
    assert _state_bytes_equal(sparse, clone)
    sparse._values().mul_(2)
    assert not _state_bytes_equal(sparse, clone)
    assert not _state_bytes_equal(sparse, clone.to_sparse_csr())


def test_perturbing_a_sparse_parent_keeps_its_structure() -> None:
    sparse = (torch.eye(3) + torch.diag(torch.ones(2), 1)).to_sparse()
    perturbed = _perturb_layer_outs(sparse, torch.randn(3, 4))
    assert perturbed.layout is torch.sparse_coo
    assert perturbed.shape == sparse.shape
    assert torch.equal(perturbed._indices(), sparse._indices())
    assert not torch.equal(perturbed._values(), sparse._values())
    csr = sparse.to_sparse_csr()
    rebuilt = sparse_with_values(csr, csr.values() * 3)
    assert rebuilt.layout is torch.sparse_csr
    torch.testing.assert_close(rebuilt.to_dense(), csr.to_dense() * 3)


def test_sparse_in_forward_model_validates() -> None:
    torch.manual_seed(0)
    model = _SparseInForward().eval()
    assert tl.validate(model, [torch.randn(3, 4)], scope="forward", seed=0)


def _single_op(trace: Any, func_name: str) -> Any:
    ops = [op for op in trace.ops if op.func_name == func_name]
    assert len(ops) == 1, [op.func_name for op in trace.ops]
    return ops[0]


def test_sparse_sampled_addmm_captures_all_operands_and_validates() -> None:
    torch.manual_seed(0)
    model = _SampledAddmm().eval()
    x = torch.randn(3, 4)
    trace = tl.trace(model, [x])
    assert len(_single_op(trace, "sampled_addmm").parents) == 3
    assert tl.validate(model, [x], scope="forward", seed=0)


def test_sparse_spdiags_captures_diagonals_and_offsets_and_validates() -> None:
    torch.manual_seed(0)
    model = _Spdiags().eval()
    x = torch.randn(3, 4)
    trace = tl.trace(model, [x])
    assert len(_single_op(trace, "spdiags").parents) == 2
    assert tl.validate(model, [x], scope="forward", seed=0)


@pytest.mark.skipif(
    not (hasattr(torch.sparse, "spsolve") and torch.cuda.is_available()),
    reason="torch.sparse.spsolve has only a CUDA kernel (torch>=2.5)",
)
def test_sparse_spsolve_captures_both_operands_and_validates() -> None:
    torch.manual_seed(0)
    model = _Spsolve().cuda().eval()
    x = torch.randn(3, 4, device="cuda")
    try:
        with torch.no_grad():
            model(x)
    except RuntimeError as error:  # a CUDA build without the cuDSS solver backend
        pytest.skip(f"this torch build cannot run spsolve: {error}")
    trace = tl.trace(model, [x])
    assert len(_single_op(trace, "spsolve").parents) == 2
    assert tl.validate(model, [x], scope="forward", seed=0)


def test_spsolve_arg_spec_extracts_both_operands() -> None:
    """The static row matches the binding's ``(A, B)`` names; the docstring says input/other."""
    a = torch.eye(2).to_sparse_csr()
    b = torch.ones(2)
    spec = FUNC_ARG_SPECS["spsolve"]
    assert extract_tensors_and_params(spec, (a, b), {}) == ([a, b], [])
    assert extract_tensors_and_params(spec, (), {"A": a, "B": b, "left": True}) == ([a, b], [])
    if hasattr(torch.sparse, "spsolve"):
        # A wrong keyword name raises TypeError before dispatch; a build with no CPU
        # kernel raises after the keywords parsed.
        with contextlib.suppress(NotImplementedError, RuntimeError):
            torch.sparse.spsolve(A=a, B=b)


def test_sampled_addmm_and_spdiags_bind_their_spec_keywords() -> None:
    pattern = torch.eye(2).to_sparse_csr()
    mat1, mat2 = torch.ones(2, 3), torch.ones(3, 2)
    spec = FUNC_ARG_SPECS["sampledaddmm"]
    kwargs = {"input": pattern, "mat1": mat1, "mat2": mat2}
    assert extract_tensors_and_params(spec, (), kwargs) == ([pattern, mat1, mat2], [])
    torch.testing.assert_close(torch.sparse.sampled_addmm(**kwargs).to_dense(), torch.eye(2) * 4)
    diagonals, offsets = torch.ones(1, 2), torch.tensor([0])
    spec = FUNC_ARG_SPECS["spdiags"]
    kwargs = {"diagonals": diagonals, "offsets": offsets}
    assert extract_tensors_and_params(spec, (), kwargs) == ([diagonals, offsets], [])
    torch.testing.assert_close(
        torch.sparse.spdiags(**kwargs, shape=(2, 2)).to_dense(), torch.eye(2)
    )


def test_portable_save_of_a_sparse_activation_refuses_with_a_code(tmp_path) -> None:
    trace = tl.trace(_SparseInForward().eval(), [torch.randn(3, 4)])
    with pytest.raises(ArtifactError) as info:
        trace.save(tmp_path / "sparse.tlspec", level="portable")
    assert info.value.fields["code"] == "artifact_tensor_unsupported"
    assert "sparse_coo layout" in str(info.value)
    assert info.value.fields["remedy"]
