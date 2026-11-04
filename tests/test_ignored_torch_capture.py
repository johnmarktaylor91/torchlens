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
import torchlens._state as _state
from torchlens._capture.arg_positions import FUNC_ARG_SPECS, _normalize_func_name
from torchlens._constants import get_orig_torch_funcs
from torchlens.options import CaptureOptions
from torchlens.validation.core import validate_saved_outs
from torchlens.validation.diagnostics import get_validation_failure


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
    "_addmm_activation": lambda m, x: m._addmm_activation(x, x * 2) + x,
    "_is_all_true": lambda m, x: x + (m > -1)._is_all_true().float(),
    "_is_any_true": lambda m, x: x + (m > 100)._is_any_true().float(),
}
_PRIVATE_METHODS = tuple(name for name in _PRIVATE_METHOD_FORWARDS if hasattr(torch.Tensor, name))
# Further cases of a method above, by test id: ``(method, forward)``. One tensor as both mats of
# ``_addmm_activation`` sits beside the distinct-mats case.
_PRIVATE_METHOD_EXTRA_CASES: dict[
    str, tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
] = {
    "_addmm_activation-same-mats": (
        "_addmm_activation",
        lambda m, x: m._addmm_activation(x, x) + x,
    ),
}
_PRIVATE_METHOD_CASES = [
    pytest.param(name, _PRIVATE_METHOD_FORWARDS[name], id=name) for name in _PRIVATE_METHODS
] + [
    pytest.param(method, forward, id=case_id)
    for case_id, (method, forward) in _PRIVATE_METHOD_EXTRA_CASES.items()
    if hasattr(torch.Tensor, method)
]


class _PrivateMethodModel(nn.Module):
    """Calls one private Tensor method on a recorded tensor and consumes the result."""

    def __init__(self, forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.call = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(torch.relu(x), x)


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


@pytest.mark.parametrize(("method", "forward"), _PRIVATE_METHOD_CASES)
def test_private_tensor_method_captured_with_provenance(
    method: str, forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> None:
    """The private method records as one op with recorded parents and a consumer."""

    log = tl.trace(_PrivateMethodModel(forward), _input_4x4())
    try:
        recorded = [op for op in log.ops if op.func_name == method]
        assert len(recorded) == 1, f"{method} not captured: {[op.func_name for op in log.ops]}"
        (op,) = recorded
        op_labels = {candidate.layer_label for candidate in log.ops}
        assert any(parent in op_labels for parent in op.parents), op.parents
        assert op.children, f"{method} output has no recorded consumer"
    finally:
        log.cleanup()


@pytest.mark.parametrize(("method", "forward"), _PRIVATE_METHOD_CASES)
def test_private_tensor_method_validates_and_replays(
    method: str, forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> None:
    """Forward validation passes and the live replay reproduces a plain forward.

    Each case starts from a cold dynamic arg-spec cache, so the first trace takes the crawl a
    process's first call takes (one tensor as both mats once lost a parent there).
    """

    _state._dynamic_arg_specs.pop(_normalize_func_name(method), None)
    _assert_validates_and_replays(_PrivateMethodModel(forward), _input_4x4())


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
# Philox in-place fills. ``x._philox_normal_(key)`` and ``x._philox_uniform_(key)`` fill their
# receiver from a counter-based key tensor (torch 2.13+, CUDA kernels only). The key is a recorded
# parent, and the fill is a pure function of the key, so a replay with the same key reproduces it
# bit for bit and a replay with a new key reproduces a plain forward on that key.
# ---------------------------------------------------------------------------
_PHILOX_FILLS = tuple(
    name for name in ("_philox_normal_", "_philox_uniform_") if hasattr(torch.Tensor, name)
)


class _PhiloxFillModel(nn.Module):
    """Fills a fresh tensor from ``key`` (folded first when ``fold_key``) and consumes it."""

    def __init__(self, method: str, fold_key: bool) -> None:
        super().__init__()
        self.method = method
        self.fold_key = fold_key

    def forward(self, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if self.fold_key:
            key = torch._philox_key_fold_in(key, 3)
        return getattr(torch.empty_like(x), self.method)(key) * 2 + x


def _philox_inputs(seed: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(4, 4, device="cuda")
    return x, torch.tensor([seed, offset], dtype=torch.uint64, device="cuda")


@pytest.mark.parametrize("method", _PHILOX_FILLS)
def test_philox_fill_is_decorated(method: str) -> None:
    """The in-place method spelling is decorated, not only its ``torch.*`` twin."""

    assert ("torch.Tensor", method) in _decorated_pairs()


@pytest.mark.cuda
@pytest.mark.parametrize("fold_key", (False, True), ids=("input_key", "folded_key"))
@pytest.mark.parametrize("method", _PHILOX_FILLS)
def test_philox_fill_captured_with_key_parent(method: str, fold_key: bool) -> None:
    """The fill records as one op whose parents are its receiver and the key's producer."""

    log = tl.trace(_PhiloxFillModel(method, fold_key), _philox_inputs(7, 0))
    try:
        by_name = {op.func_name: op for op in log.ops}
        assert method in by_name, f"{method} not captured: {list(by_name)}"
        assert [op.func_name for op in log.ops].count(method) == 1
        fill = by_name[method]
        receiver = by_name["empty_like"].layer_label
        assert receiver in fill.parents, fill.parents
        key_parents = [parent for parent in fill.parents if parent != receiver]
        assert len(key_parents) == 1, fill.parents
        if fold_key:
            assert key_parents == [by_name["_philox_key_fold_in"].layer_label]
        assert fill.children, f"{method} output has no recorded consumer"
    finally:
        log.cleanup()


@pytest.mark.cuda
@pytest.mark.parametrize("fold_key", (False, True), ids=("input_key", "folded_key"))
@pytest.mark.parametrize("method", _PHILOX_FILLS)
def test_philox_fill_validates_and_replays(method: str, fold_key: bool) -> None:
    """Validation passes; seeded and key-driven replays reproduce the fill exactly."""

    model = _PhiloxFillModel(method, fold_key)
    inputs = _philox_inputs(7, 0)
    assert tl.validate(model, inputs, scope="forward", validate_metadata=True) is True
    trace = tl.trace(model, inputs, capture=CaptureOptions(seed=5))
    try:
        for replay_inputs in (inputs, _philox_inputs(11, 4)):
            result = trace.run_transaction(inputs=replay_inputs, seed=5)
            try:
                assert torch.equal(result.output, model(*replay_inputs))
            finally:
                result.trace.cleanup()
    finally:
        trace.cleanup()


@pytest.mark.cuda
@pytest.mark.parametrize("method", _PHILOX_FILLS)
def test_planted_key_blind_philox_fill_fails_perturbation(method: str) -> None:
    """A fill whose replay ignores its key fails: only the destination slot is excused."""

    model, inputs = _PhiloxFillModel(method, fold_key=True), _philox_inputs(7, 0)
    torch.manual_seed(0)
    ground_truth = model(*inputs)
    torch.manual_seed(0)
    trace = tl.trace(model, inputs, capture=CaptureOptions(save_args=True, seed=0), save="all")
    try:
        fill = next(op for op in trace._layer_list if op.func_name == method)
        key_label = fill.parent_arg_positions["args"][1]
        frozen = fill.out.detach().clone()

        def key_blind_fill(*args: object, **kwargs: object) -> torch.Tensor:
            return frozen.clone()

        key_blind_fill.__name__ = fill.func.__name__
        fill.func = key_blind_fill
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None and failure.check_name == "perturbation"
        assert key_label in failure.extra["perturbed_parents"]
    finally:
        trace.cleanup()
