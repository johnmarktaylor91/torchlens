"""Capture of tensor-producing torch functions that ``__torch_function__`` ignores.

Every function here is in torch's ``get_ignored_functions()``, so the overridable crawl skips it
and it decorates only through a row in ``torchlens._constants.IGNORED_FUNCS``. Unwrapped, each
one's output reached its consumer with no graph or source provenance (a UserWarning that the
suite's filter turns into an error) and validation failed. The checks: the function is in the
decorated set, it records as an op with recorded parents and a recorded consumer, and forward
validation plus live replay agree with a plain forward.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture.arg_positions import FUNC_ARG_SPECS, _normalize_func_name
from torchlens._constants import get_orig_torch_funcs
from torchlens.options import CaptureOptions


def _decorated_pairs() -> set[tuple[str, str]]:
    """``(namespace, name)`` pairs in TorchLens's decorated (first-wrap) set."""

    return set(get_orig_torch_funcs())


def _complex_of(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x, x * 2)


# ---------------------------------------------------------------------------
# Private Tensor methods. Their ``torch.*`` twins are crawled and decorated; the method
# spellings are C methods in the ignored list and were not.
# ---------------------------------------------------------------------------
_PRIVATE_METHOD_FORWARDS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "_to_sparse": lambda m, x: m._to_sparse().to_dense() * 2 + x,
    "_to_sparse_csr": lambda m, x: m._to_sparse_csr().to_dense() * 2 + x,
    "_to_sparse_csc": lambda m, x: m._to_sparse_csc().to_dense() * 2 + x,
    "_to_sparse_bsr": lambda m, x: m._to_sparse_bsr((2, 2)).to_dense() * 2 + x,
    "_to_sparse_bsc": lambda m, x: m._to_sparse_bsc((2, 2)).to_dense() * 2 + x,
    "_conj": lambda m, x: torch.view_as_real(_complex_of(m)._conj().resolve_conj()).sum(-1) + x,
    "_conj_physical": lambda m, x: torch.view_as_real(_complex_of(m)._conj_physical()).sum(-1) + x,
    "_neg_view": lambda m, x: m._neg_view().resolve_neg() * 2 + x,
    "_lazy_clone": lambda m, x: m._lazy_clone() * 2 + x,
    # Distinct mat operands: one tensor passed as both mats does not re-trace to the same graph,
    # and the decorated ``torch._addmm_activation`` twin shows the same on the unchanged base.
    "_addmm_activation": lambda m, x: m._addmm_activation(x, x * 2) + x,
    "_is_all_true": lambda m, x: x + (m > -1)._is_all_true().float(),
    "_is_any_true": lambda m, x: x + (m > 100)._is_any_true().float(),
}
_PRIVATE_METHODS = tuple(name for name in _PRIVATE_METHOD_FORWARDS if hasattr(torch.Tensor, name))


class _PrivateMethodModel(nn.Module):
    """Calls one private Tensor method on a recorded tensor and consumes the result."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _PRIVATE_METHOD_FORWARDS[self.method](torch.relu(x), x)


def _input_4x4() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 4)


def _assert_validates_and_replays(
    model: nn.Module, x: torch.Tensor, seed: int | None = None
) -> None:
    """Forward validation passes and a live replay on fresh input matches a plain forward.

    With ``seed`` the capture and the run use that seed, and the plain forward runs right after
    ``torch.manual_seed(seed)``: a run seeds the CPU generator the same way before its forward.
    """

    assert tl.validate(model, x, scope="forward", validate_metadata=True) is True
    capture = CaptureOptions() if seed is None else CaptureOptions(seed=seed)
    trace = tl.trace(model, x, capture=capture)
    try:
        fresh = torch.randn_like(x)
        if seed is None:
            result = trace.run_transaction(inputs=fresh)
        else:
            result = trace.run_transaction(inputs=fresh, seed=seed)
        try:
            if seed is not None:
                torch.manual_seed(seed)
            assert torch.equal(result.output, model(fresh))
        finally:
            result.trace.cleanup()
    finally:
        trace.cleanup()


@pytest.mark.parametrize("method", _PRIVATE_METHODS)
def test_private_tensor_method_is_decorated(method: str) -> None:
    """The method spelling is decorated, not only its ``torch.*`` twin."""

    assert ("torch.Tensor", method) in _decorated_pairs()


@pytest.mark.parametrize("method", _PRIVATE_METHODS)
def test_private_tensor_method_captured_with_provenance(method: str) -> None:
    """The private method records as one op with recorded parents and a consumer."""

    log = tl.trace(_PrivateMethodModel(method), _input_4x4())
    try:
        recorded = [op for op in log.ops if op.func_name == method]
        assert len(recorded) == 1, f"{method} not captured: {[op.func_name for op in log.ops]}"
        (op,) = recorded
        op_labels = {candidate.layer_label for candidate in log.ops}
        assert any(parent in op_labels for parent in op.parents), op.parents
        assert op.children, f"{method} output has no recorded consumer"
    finally:
        log.cleanup()


@pytest.mark.parametrize("method", _PRIVATE_METHODS)
def test_private_tensor_method_validates_and_replays(method: str) -> None:
    """Forward validation passes and the live replay reproduces a plain forward."""

    _assert_validates_and_replays(_PrivateMethodModel(method), _input_4x4())


# ---------------------------------------------------------------------------
# ``get_rng_state``: the returned state tensor is a no-parent factory op. TorchLens's own RNG
# snapshots (per op with ``save_rng``, around validation and replay) never record.
# ---------------------------------------------------------------------------
_RNG_STATE_SPELLINGS: dict[str, Callable[[], torch.Tensor]] = {
    "torch": lambda: torch.get_rng_state(),
    "torch.random": lambda: torch.random.get_rng_state(),
}


class _RngStateModel(nn.Module):
    """Reads the CPU RNG state, draws, and feeds the state bytes on."""

    def __init__(self, spelling: str) -> None:
        super().__init__()
        self.spelling = spelling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = _RNG_STATE_SPELLINGS[self.spelling]()
        noise = torch.randn_like(x)
        return x + noise + state[: x.numel()].float().view_as(x) / 255.0


class _DropoutModel(nn.Module):
    """A plain stochastic model: no RNG-state read of its own."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x)) + torch.rand_like(x)


@pytest.mark.parametrize("namespace", tuple(_RNG_STATE_SPELLINGS))
def test_get_rng_state_is_decorated_factory(namespace: str) -> None:
    """Both spellings are decorated, and the static spec has no tensor operand."""

    assert (namespace, "get_rng_state") in _decorated_pairs()
    spec = FUNC_ARG_SPECS[_normalize_func_name("get_rng_state")]
    assert spec.positions == () and not spec.tensor_kwargs


@pytest.mark.parametrize("spelling", tuple(_RNG_STATE_SPELLINGS))
def test_get_rng_state_captured_as_source_op(spelling: str) -> None:
    """The state read is one parentless op whose output feeds a recorded consumer."""

    torch.manual_seed(0)
    log = tl.trace(_RngStateModel(spelling), _input_4x4())
    try:
        recorded = [op for op in log.ops if op.func_name == "get_rng_state"]
        assert len(recorded) == 1, [op.func_name for op in log.ops]
        (op,) = recorded
        assert tuple(op.parents) == ()
        assert op.children, "the state tensor has no recorded consumer"
    finally:
        log.cleanup()


@pytest.mark.parametrize("spelling", tuple(_RNG_STATE_SPELLINGS))
def test_get_rng_state_validates_and_replays(spelling: str) -> None:
    """Validation passes and the replay reads the same state a plain forward reads."""

    _assert_validates_and_replays(_RngStateModel(spelling), _input_4x4(), seed=7)


def test_torchlens_rng_snapshots_record_no_rng_state_op() -> None:
    """With per-op RNG saving on, a plain stochastic model shows no RNG-state op."""

    model = _DropoutModel().train()
    log = tl.trace(model, _input_4x4(), capture=CaptureOptions(save_rng=True))
    try:
        names = [op.func_name for op in log.ops]
        assert "get_rng_state" not in names, names
        assert "dropout" in names
    finally:
        log.cleanup()
    assert tl.validate(model, _input_4x4(), scope="forward") is True
