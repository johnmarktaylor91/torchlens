"""Layout-specific sparse conversions are recorded ops with provenance.

``Tensor.to_sparse_{csr,csc,bsr,bsc}`` are in torch's ``get_ignored_functions()`` (unlike
``to_sparse``), so they must be in TorchLens's decorated set for a sparse tensor built in
forward to keep its graph provenance. These rows came from the r18 capture-gap file, which
the coverage-redundancy shrink deleted after the rows were added.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture.arg_positions import FUNC_ARG_SPECS, _normalize_func_name
from torchlens._constants import get_orig_torch_funcs


def _decorated_normalized_names() -> set[str]:
    """Normalized torch function names in TorchLens's decorated (first-wrap) set."""

    return {_normalize_func_name(func_name.strip("_")) for _, func_name in get_orig_torch_funcs()}


# ---------------------------------------------------------------------------
# Layout-specific sparse conversions (``Tensor.to_sparse_{coo,csr,csc,bsr,bsc}``)
# ---------------------------------------------------------------------------
# All five are in torch's ``get_ignored_functions()`` (``to_sparse`` itself is overridable).
# ``to_sparse_coo`` is a Python wrapper that calls ``self.to_sparse()``, so it records as a
# ``to_sparse`` op and needs no row. The four compressed ones are C methods: before they
# were re-added to ``IGNORED_FUNCS`` a sparse tensor built in
# forward with ``x.to_sparse_csr()`` reached its consumer as an unattributed literal: the
# "no graph/source provenance" UserWarning (an error under the suite's filter) and an
# unattributed arg0 on ``sparse.mm`` / ``sampled_addmm``. ``to_sparse(layout=...)`` recorded
# fine, which is what each converted tensor is checked against below.

_SPARSE_CONVERSIONS: dict[str, dict[str, object]] = {
    "to_sparse_coo": {},
    "to_sparse_csr": {},
    "to_sparse_csc": {},
    "to_sparse_bsr": {"blocksize": (2, 2)},
    "to_sparse_bsc": {"blocksize": (2, 2)},
}


class _SparseConvertModel(nn.Module):
    """Builds a sparse tensor with the method spelling, then consumes it."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method
        self.kwargs = _SPARSE_CONVERSIONS[method]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = torch.relu(x)
        sparse = getattr(masked, self.method)(**self.kwargs)
        if self.method in ("to_sparse_coo", "to_sparse_csr"):
            return torch.sparse.mm(sparse, x) + 1
        return sparse.to_dense() * 2 + x


def _sparse_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 4)


_COMPRESSED_CONVERSIONS = tuple(name for name in _SPARSE_CONVERSIONS if name != "to_sparse_coo")


def _recorded_conversion_name(method: str) -> str:
    """``to_sparse_coo`` delegates to ``to_sparse``, which is the op it records as."""

    return "to_sparse" if method == "to_sparse_coo" else method


@pytest.mark.parametrize("method", _COMPRESSED_CONVERSIONS)
def test_sparse_conversion_is_decorated_with_static_spec(method: str) -> None:
    """Each conversion is in the decorated set and has a static self-only arg-spec."""

    normalized = _normalize_func_name(method)
    assert normalized in _decorated_normalized_names()
    spec = FUNC_ARG_SPECS[normalized]
    assert spec.positions == (0,)


@pytest.mark.parametrize("method", tuple(_SPARSE_CONVERSIONS))
def test_sparse_conversion_captured_with_provenance(method: str) -> None:
    """The conversion is a recorded op whose parent is the dense tensor it converted."""

    log = tl.trace(_SparseConvertModel(method), _sparse_input())
    converted = [op for op in log.ops if op.func_name == _recorded_conversion_name(method)]
    assert len(converted) == 1, f"{method} not captured as an op"
    (op,) = converted
    relu = [candidate for candidate in log.ops if candidate.func_name == "relu"]
    assert len(relu) == 1
    assert tuple(op.parents) == (relu[0].layer_label,)
    assert op.children, f"{method} output has no recorded consumer"


@pytest.mark.parametrize("method", tuple(_SPARSE_CONVERSIONS))
def test_sparse_conversion_validates_and_replays(method: str) -> None:
    """Forward validation passes and the live replay reproduces a plain forward."""

    model = _SparseConvertModel(method)
    x = _sparse_input()
    assert tl.validate(model, x, scope="forward", validate_metadata=True) is True

    trace = tl.trace(model, x)
    try:
        fresh = torch.randn(4, 4)
        result = trace.run(inputs=fresh)
        try:
            assert torch.equal(result.output, model(fresh))
        finally:
            result.trace.cleanup()
    finally:
        trace.cleanup()
