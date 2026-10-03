"""grind-r8 cluster 4.0.2: the WRONG-TRACE capture-cache family (R39 x3 + R35).

Three cache-correctness holes all served stale/wrong data under ``cache=True``:

* ``_callable_code_digest`` omitted ``__closure__`` cells, so two
  factory-configured forwards sharing one code object collided on one key and
  the cache served the WRONG activations (fable probe).
* The input hash reduced the forward inputs to their TENSOR leaves only, so a
  changed non-tensor input (``use_relu=False``), a changed kwarg NAME, or a
  restructured container hit the original capture's entry (opus end-to-end).
* The module-namespace container-slot memo invalidated on ``len(namespace)``
  only, so replacing a value at constant length kept a stale slot list and a
  stamped tensor escaped the session-end metadata clear (sol probe).

Plus the R35 crash family in the same digest authority: the hand-rolled
uint8-reinterpret views did not resolve lazy conj/neg dispatch bits, so
``tl.trace(m, x.conj(), cache=True)`` died with a bare RuntimeError. The
transport + reinterpret now live in ONE authority
(``torchlens._transport.digest_byte_view``).
"""

from __future__ import annotations

import types

import torch
from torch import nn

import torchlens as tl
from torchlens._capture_fingerprint import (
    _callable_code_digest,
    _forward_input_fragment,
    _hash_tensor_content,
)
from torchlens._capture_state_helpers import _capture_cache_key
from torchlens._runnable_state import runnable_tensor_byte_digest
from torchlens._transport import digest_byte_view, to_cpu_contiguous


def _cache_capture(tmp_path):
    return tl.options.CaptureOptions(cache=True, cache_dir=tmp_path / "cache")


class _CacheModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _FlagModel(nn.Module):
    """Model whose traced program depends on a NON-TENSOR forward input."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, use_relu: bool = True) -> torch.Tensor:
        if use_relu:
            return torch.relu(self.lin(x))
        return torch.sigmoid(self.lin(x))


class _DoubleModel(nn.Module):
    """Dtype-agnostic model for the lazy-conj input capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


# ---------------------------------------------------------------------------
# R39-1: closure cells participate in the callable code digest
# ---------------------------------------------------------------------------


def _make_scaler(k: int):
    def scaler(x):
        return x * k

    return scaler


def test_closure_cells_change_the_code_digest() -> None:
    """Same code object, different closure cell values -> different digests."""

    assert _callable_code_digest(_make_scaler(1)) != _callable_code_digest(_make_scaler(2))
    assert _callable_code_digest(_make_scaler(3)) == _callable_code_digest(_make_scaler(3))


def test_self_referential_closure_digest_terminates() -> None:
    """A closure cell holding its own function must not recurse forever."""

    def factory():
        def rec(x):
            return rec

        return rec

    assert isinstance(_callable_code_digest(factory()), str)


def _make_flag_forward(use_relu: bool):
    def forward(self, x):
        if use_relu:
            return torch.relu(self.lin(x))
        return torch.sigmoid(self.lin(x))

    return forward


def test_factory_configured_instance_forward_is_a_cache_miss(tmp_path) -> None:
    """Closure-configured instance forwards must not collide on one entry."""

    model = _CacheModel()
    model.forward = types.MethodType(_make_flag_forward(True), model)
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False

    twin = _CacheModel()
    twin.load_state_dict(model.state_dict())
    twin.forward = types.MethodType(_make_flag_forward(False), twin)
    second = tl.trace(twin, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a factory-configured forward with a different closure value must miss"
    )
    labels = list(second.layer_labels)
    assert any("sigmoid" in label for label in labels)
    assert not any("relu" in label for label in labels)


# ---------------------------------------------------------------------------
# R39-2: non-tensor forward inputs participate in the cache key
# ---------------------------------------------------------------------------


def test_non_tensor_forward_input_is_a_cache_miss(tmp_path) -> None:
    """``use_relu=False`` must not be served the ``use_relu=True`` capture."""

    model = _FlagModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, (x, True), capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    assert any("relu" in label for label in first.layer_labels)

    second = tl.trace(model, (x, False), capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a changed non-tensor forward input must be a cache miss"
    )
    labels = list(second.layer_labels)
    assert any("sigmoid" in label for label in labels)
    assert not any("relu" in label for label in labels)


def test_cache_key_sees_non_tensor_inputs_and_kwarg_names() -> None:
    """Key-level checks for the axes the tensor-leaf hash was blind to."""

    model = nn.Linear(2, 2)
    x = torch.randn(1, 2)
    base = _capture_cache_key(model, (x, True), {}, {})
    assert base != _capture_cache_key(model, (x, False), {}, {})
    assert _capture_cache_key(model, (x,), {"a": x}, {}) != _capture_cache_key(
        model, (x,), {"b": x}, {}
    )
    assert _capture_cache_key(model, (x,), {"f": 1}, {}) != _capture_cache_key(
        model, (x,), {"f": 2}, {}
    )
    # Determinism: an unchanged call keys identically.
    assert base == _capture_cache_key(model, (x, True), {}, {})


def test_forward_input_fragment_frames_structure() -> None:
    """Container shape changes fragment identity; equal trees are stable."""

    x = torch.arange(4.0)
    assert _forward_input_fragment((x, [1, 2])) == _forward_input_fragment((x, [1, 2]))
    assert _forward_input_fragment((x, [1, 2])) != _forward_input_fragment((x, (1, 2)))
    assert _forward_input_fragment({"k": x}) != _forward_input_fragment({"j": x})
    # Over-ceiling depth never matches (conservative always-miss).
    deep: list = [x]
    for _ in range(70):
        deep = [deep]
    assert _forward_input_fragment(deep) != _forward_input_fragment(deep)


# ---------------------------------------------------------------------------
# R39-3: module-namespace slot walk sees same-length value replacement
# ---------------------------------------------------------------------------


def test_namespace_slot_walk_sees_same_length_replacement() -> None:
    """A tensor REPLACING a scalar at constant namespace length is visited."""

    from torchlens.backends.torch.model_prep import _clear_session_tensor_metadata

    namespace_module = types.ModuleType("tl_test_namespace_module")
    namespace_module.stash = 123
    visited: list[torch.Tensor] = []
    _clear_session_tensor_metadata(namespace_module, set(), 0, visited.append)
    assert visited == []

    replacement = torch.randn(3)
    namespace_module.stash = replacement
    _clear_session_tensor_metadata(namespace_module, set(), 0, visited.append)
    assert any(item is replacement for item in visited), (
        "the session clear must see a tensor that replaced a same-length slot"
    )


# ---------------------------------------------------------------------------
# R35: lazy conj/neg resolve in the one digest authority
# ---------------------------------------------------------------------------


def test_digest_sites_accept_lazy_conj_and_neg() -> None:
    """Every uint8-reinterpret digest site survives lazy dispatch bits."""

    conj = torch.randn(3, dtype=torch.complex64).conj()
    assert isinstance(_hash_tensor_content(conj), str)
    assert isinstance(tl.hash.content(conj), str)
    assert isinstance(runnable_tensor_byte_digest(conj), str)
    neg = torch.randn(3).conj().resolve_conj() * -1
    neg_view = torch.randn(3, dtype=torch.complex64).conj().imag
    assert isinstance(runnable_tensor_byte_digest(neg_view), str)
    assert isinstance(runnable_tensor_byte_digest(neg), str)


def test_conj_digest_equals_resolved_twin() -> None:
    """The lazy-bit tensor digests as its LOGICAL value."""

    source = torch.randn(4, dtype=torch.complex64)
    lazy = source.conj()
    resolved = source.conj().resolve_conj()
    assert runnable_tensor_byte_digest(lazy) == runnable_tensor_byte_digest(resolved)
    assert _hash_tensor_content(lazy) == _hash_tensor_content(resolved)
    assert bytes(digest_byte_view(lazy)) == bytes(digest_byte_view(resolved))
    assert not to_cpu_contiguous(lazy).is_conj()


def test_lazy_conj_input_capture_with_cache_does_not_crash(tmp_path) -> None:
    """End-to-end R35 repro: ``tl.trace(m, x.conj(), cache=True)``."""

    model = _DoubleModel()
    x = torch.randn(2, 3, dtype=torch.complex64).conj()
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is True
