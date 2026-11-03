"""Regression guards for the torch-2.13 arg-spec capture-gap fix (R18CG).

Two related findings surfaced while reconciling ``tests/test_arg_spec_coverage.py`` against
torch 2.13:

* ``torch.align_as`` (the named-tensor ``Tensor.align_as`` method) was REMOVED in torch 2.13. It
  is therefore version-varying: still decorated on the pinned torch<=2.12 CI legs, absent on
  2.13. Its static arg-spec is retained in ``arg_positions.py`` for the older legs, but it is
  dropped from the high-confidence "must be decorated on every leg" set.

* ``torch.rand_like`` / ``torch.randn_like`` / ``torch.randint_like`` are real public torch
  factories that were NOT wrapped: they are in torch's ``get_ignored_functions()`` (not
  overridable via ``__torch_function__``) but were absent from ``torchlens._constants.IGNORED_FUNCS``
  -- unlike their non-``_like`` siblings ``rand`` / ``randn`` / ``randint``. A forward that called
  them recorded the output as an unattributed literal (the r45 ``.mH`` capture-gap class). The fix
  adds the three factories to ``IGNORED_FUNCS`` so they decorate and capture; their arg-specs
  already existed (``_FACTORY_SOURCE_SPEC`` / the ``randintlike`` entry).
* ``Tensor.to_sparse_{csr,csc,bsr,bsc}``: the same class (ignored by torch, unlike
  ``to_sparse``); see the section at the end of this file.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture.arg_positions import FUNC_ARG_SPECS, _normalize_func_name
from torchlens._constants import get_orig_torch_funcs

_RANDOM_LIKE_FACTORIES = ("rand_like", "randn_like", "randint_like")


def _decorated_normalized_names() -> set[str]:
    """Normalized torch function names in TorchLens's decorated (first-wrap) set."""

    return {_normalize_func_name(func_name.strip("_")) for _, func_name in get_orig_torch_funcs()}


def test_align_as_static_spec_retained_and_version_varying() -> None:
    """align_as is version-varying: static spec retained, never asserted decorated on 2.13.

    Its static arg-spec must remain in the table (the pinned torch<=2.12 legs still decorate
    ``Tensor.align_as`` and would fail ``test_every_decorated`` without it), while on torch>=2.13
    where the method is gone it must simply be undecorated -- the retained spec is then inert.
    """

    assert "alignas" in FUNC_ARG_SPECS
    assert FUNC_ARG_SPECS["alignas"].tensor_kwargs == ("self", "other")
    if not hasattr(torch.Tensor, "align_as"):
        assert _normalize_func_name("align_as") not in _decorated_normalized_names()


def test_randint_like_static_spec_covers_self_tensor() -> None:
    """randint_like's static spec extracts its ``self`` reference tensor at position 0.

    ATen schema: ``randint_like(Tensor self, SymInt high, *, ...)`` -- the primary tensor input is
    ``self`` at position 0, so capture attributes the op to its source tensor.
    """

    spec = FUNC_ARG_SPECS["randintlike"]

    assert 0 in spec.positions
    assert "self" in spec.tensor_kwargs


@pytest.mark.parametrize("factory_name", _RANDOM_LIKE_FACTORIES)
def test_random_like_factory_is_decorated(factory_name: str) -> None:
    """Each random ``*_like`` factory is in TorchLens's decorated (first-wrap) set."""

    assert _normalize_func_name(factory_name) in _decorated_normalized_names()


@pytest.mark.parametrize("factory_name", _RANDOM_LIKE_FACTORIES)
def test_random_like_factory_output_is_captured_not_literal(factory_name: str) -> None:
    """A forward using a random ``*_like`` factory captures it as an op with provenance.

    Before the fix the factory output fed downstream ops as an unattributed literal (TorchLens
    raised the promoted "no graph/source provenance" UserWarning under the suite's filter, or the
    op was simply absent). After the fix it is captured as a real op attributed to its source.

    NOTE: the factory is resolved via ``getattr(torch, ...)`` INSIDE ``forward`` (call time), not
    pre-bound at test scope. TorchLens wraps by replacing the ``torch.<fn>`` attribute on the first
    trace; a reference captured before that first wrap would still point at the original callable
    and bypass capture -- which is exactly how real user code (``torch.rand_like(x)``) behaves.
    """

    class _Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            factory = getattr(torch, factory_name)
            if factory_name == "randint_like":
                drawn = factory(x, 0, 5).float()
            else:
                drawn = factory(x)
            return x + drawn

    log = tl.trace(_Model(), torch.randn(2, 3))
    normalized = factory_name.replace("_", "")
    captured = [label for label in log.layer_labels if normalized in label.replace("_", "").lower()]
    assert captured, f"{factory_name} not captured as an op (unwrapped -> unattributed literal)"


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
