"""Capture of tensor-producing torch functions that ``__torch_function__`` ignores.

Every function here is in torch's ``get_ignored_functions()``, so the overridable crawl skips it
and it decorates only through a row in ``torchlens._constants.IGNORED_FUNCS``. Unwrapped, each
one's output reached its consumer with no graph or source provenance (a UserWarning that the
suite's filter turns into an error) and validation failed. The checks: the function is in the
decorated set, it records as an op with recorded parents and a recorded consumer, and forward
validation plus live replay agree with a plain forward.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

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


# ---------------------------------------------------------------------------
# The rest of the ignored list that returns a tensor and escaped (ignored-funcs-triage).
# View-function methods replay a view's chain onto a new base; ``_rev_view_func_unsafe``
# exists only for views made under original-view tracking.
# ---------------------------------------------------------------------------
def _view_of(m: torch.Tensor) -> torch.Tensor:
    """A transpose view of ``m`` that keeps its forward and reverse view functions."""

    with torch.autograd._force_original_view_tracking(True):
        return m.t()


_VIEW_FUNC_FORWARDS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "_view_func": lambda m, x: _view_of(m)._view_func(x * 2) * 2 + x,
    "_view_func_unsafe": lambda m, x: _view_of(m)._view_func_unsafe(x * 2) * 2 + x,
    "_rev_view_func_unsafe": lambda m, x: _view_of(m)._rev_view_func_unsafe(x * 2) * 2 + x,
}
_VIEW_FUNCS = tuple(name for name in _VIEW_FUNC_FORWARDS if hasattr(torch.Tensor, name))


class _ViewFuncModel(nn.Module):
    """Replays a recorded view's chain onto a recorded new base and consumes the result."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _VIEW_FUNC_FORWARDS[self.method](torch.relu(x), x)


@pytest.mark.parametrize("method", _VIEW_FUNCS)
def test_view_func_method_captured_validates_and_replays(method: str) -> None:
    """The view-function method is decorated, records with the new base as a parent, validates."""

    assert ("torch.Tensor", method) in _decorated_pairs()
    model, x = _ViewFuncModel(method), _input_4x4()
    log = tl.trace(model, x)
    try:
        (op,) = [op for op in log.ops if op.func_name == method]
        new_base = [p for p in op.parents if p.startswith("mul")]
        assert new_base and op.children, (op.parents, op.children)
    finally:
        log.cleanup()
    _assert_validates_and_replays(model, x)


class _MkldnnRnnLayerModel(nn.Module):
    """One LSTM layer through ``torch.mkldnn_rnn_layer`` with parameter weights."""

    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.w_ih = nn.Parameter(torch.randn(8, 4, generator=gen) * 0.3)
        self.w_hh = nn.Parameter(torch.randn(8, 2, generator=gen) * 0.3)
        self.b_ih = nn.Parameter(torch.randn(8, generator=gen) * 0.1)
        self.b_hh = nn.Parameter(torch.randn(8, generator=gen) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(1, 2)
        weights = (self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        # mode 2 is LSTM; hidden 2, one layer, biases, not bidirectional or batch-first, eval.
        out = torch.mkldnn_rnn_layer(
            torch.relu(x).unsqueeze(1),
            *weights,
            state,
            state,
            False,
            [],
            2,
            2,
            1,
            True,
            False,
            False,
            False,
        )
        return out[0].sum() * 2 + x


@pytest.mark.parametrize("name", tuple(n for n in ("mkldnn_rnn_layer",) if hasattr(torch, n)))
def test_mkldnn_rnn_layer_captured_validates_and_replays(name: str) -> None:
    """The fused RNN layer records with its input and weights as parents, validates, replays."""

    assert ("torch", name) in _decorated_pairs()
    model, x = _MkldnnRnnLayerModel(), _input_4x4()
    log = tl.trace(model, x)
    try:
        (op,) = [op for op in log.ops if op.func_name == name]
        assert len(op.parents) >= 5 and op.children, (op.parents, op.children)
    finally:
        log.cleanup()
    _assert_validates_and_replays(model, x)


class _DepTokenModel(nn.Module):
    """Makes a dependency token, zeroes it, and threads it through a range constraint."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = torch._make_dep_token().zero_()
        constrained = torch._functional_sym_constrain_range(3, 0, 10, token + x.sum())
        return x * (constrained + 1)


def test_dep_token_functions_captured_validate_and_replay() -> None:
    """``_make_dep_token`` is a parentless factory op; the constraint op takes the token."""

    assert ("torch", "_make_dep_token") in _decorated_pairs()
    assert ("torch", "_functional_sym_constrain_range") in _decorated_pairs()
    spec = FUNC_ARG_SPECS[_normalize_func_name("_make_dep_token")]
    assert spec.positions == () and not spec.tensor_kwargs
    model, x = _DepTokenModel(), _input_4x4()
    log = tl.trace(model, x)
    try:
        by_name = {op.func_name: op for op in log.ops}
        assert tuple(by_name["_make_dep_token"].parents) == ()
        constrained = by_name["_functional_sym_constrain_range"]
        assert constrained.parents and constrained.children, constrained.parents
    finally:
        log.cleanup()
    _assert_validates_and_replays(model, x)


# ---------------------------------------------------------------------------
# ``torch.load`` in a forward is an exclusion on record (Lead 3, 4 Nov 2026): replay never
# resolves a file load, so the loaded tensor stays sourceless, and the provenance warning
# says so by name for both spellings.
# ---------------------------------------------------------------------------
def _saved_tensor_bytes() -> bytes:
    """A 4x4 tensor serialized with ``torch.save``."""

    buf = io.BytesIO()
    torch.save(torch.arange(16.0).view(4, 4), buf)
    return buf.getvalue()


_LOAD_SPELLINGS: dict[str, Callable[..., Any]] = {
    "torch": lambda f: torch.load(f, weights_only=True),
    "torch.serialization": lambda f: torch.serialization.load(f, weights_only=True),
}


class _LoadModel(nn.Module):
    """Loads a saved tensor inside forward and adds it to the input."""

    def __init__(self, spelling: str) -> None:
        super().__init__()
        self.spelling = spelling
        self.payload = _saved_tensor_bytes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LOAD_SPELLINGS[self.spelling](io.BytesIO(self.payload)) + x


@pytest.mark.parametrize("spelling", tuple(_LOAD_SPELLINGS))
def test_torch_load_in_forward_warning_names_load(spelling: str) -> None:
    """The provenance warning names ``torch.load`` and says replay will not reload files."""

    assert (spelling, "load") in _decorated_pairs()
    with pytest.warns(UserWarning, match="no graph/source provenance") as record:
        log = tl.trace(_LoadModel(spelling), _input_4x4())
    log.cleanup()
    messages = " ".join(str(w.message) for w in record)
    assert "torch.load" in messages and "replay will not reload files" in messages, messages
