"""grind-p3 T5.1: a model IMPLEMENTATION change must invalidate the capture cache.

The capture-cache key fingerprinted only tensor content (``state_dict`` values,
training flags, non-persistent buffers) plus the call configuration. Two models
with identical parameters but DIFFERENT forward code therefore collided on the
same key: editing ``forward`` between runs silently served the STALE cached
trace of the old implementation. The key now folds in a model-implementation
signature (module tree structure, class qualnames, and per-class ``forward``
code-object digests, including instance-level ``forward`` overrides), so an
implementation change is a cache miss and a recapture.
"""

from __future__ import annotations

import os
import types

import torch
from torch import nn

import torchlens as tl


class _CacheModel(nn.Module):
    """Tiny deterministic model whose forward the tests mutate."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _cache_capture(tmp_path):
    return tl.options.CaptureOptions(cache=True, cache_dir=tmp_path / "cache")


def _op_labels(trace) -> list[str]:
    return list(trace.layer_labels)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_state_helpers.py.
def test_instance_forward_override_is_a_cache_miss(tmp_path) -> None:
    """An instance-level ``forward`` override also invalidates the cache."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.provenance.cache_hit is False

    def tanh_forward(self, inputs):  # noqa: ANN001 - test shim
        return torch.tanh(self.lin(inputs))

    model.forward = types.MethodType(tanh_forward, model)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    finally:
        del model.forward

    assert second.provenance.cache_hit is False
    assert any("tanh" in label for label in _op_labels(second))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_fingerprint.py.
def test_attribute_fragments_are_bounded_and_address_free() -> None:
    """Fragment rules: primitives by value, opaque objects by type, bounded."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    assert _attribute_state_fragment(5) == 5
    assert _attribute_state_fragment("mode") == "mode"
    assert _attribute_state_fragment(torch.float32) == ("torch-value", "torch.float32")
    # Tensor attributes key by CONTENT, not identity.
    tensor_a = torch.ones(3)
    tensor_b = torch.ones(3)
    assert _attribute_state_fragment(tensor_a) == _attribute_state_fragment(tensor_b)

    # Opaque objects key by TYPE identity only -- never an address repr.
    class _Opaque:
        pass

    fragment = _attribute_state_fragment(_Opaque())
    assert fragment == _attribute_state_fragment(_Opaque())
    assert "0x" not in repr(fragment)
    # Depth ceiling terminates pathological nesting.
    nested: list = [1]
    for _ in range(10):
        nested = [nested]
    assert "<attr-depth-ceiling>" in repr(_attribute_state_fragment(nested))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_state_helpers.py.
def test_edited_hook_implementation_is_a_cache_miss(tmp_path) -> None:
    """Same registration slot, different hook CODE -> different key."""

    model = _CacheModel()
    x = torch.randn(1, 4)

    handle = model.register_forward_hook(lambda mod, args, out: out * 2)
    try:
        first = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert first.provenance.cache_hit is False
    finally:
        handle.remove()

    handle = model.register_forward_hook(lambda mod, args, out: out * 3)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert second.provenance.cache_hit is False, (
            "an edited hook implementation must not hit the old hook's cached trace"
        )
    finally:
        handle.remove()


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_frozen_parameters_are_a_cache_miss(tmp_path) -> None:
    """Freezing params (requires_grad_(False)) changes captured grad metadata."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.provenance.cache_hit is False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.provenance.cache_hit is False


# Kept by shrink-r4 (per-id subset rule): sole coverage of wrapped.py.
def test_sibling_package_forward_is_not_torchlens_instrumentation() -> None:
    """A ``torchlens_contrib`` install must not be classified as TorchLens'.

    grind-r2 b4-fable R39-3: the bare prefix match claimed any sibling path
    that string-extends the package dir, so a user forward override defined
    in ``.../site-packages/torchlens_contrib/model.py`` was EXCLUDED from the
    implementation signature and its edits silently hit the stale cache.
    """

    from torchlens._capture_state_helpers import _is_torchlens_instrumentation

    package_dir = os.path.dirname(os.path.abspath(tl.__file__))

    def _function_with_filename(filename: str):
        code = compile("def shim(x):\n    return x\n", filename, "exec")
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102 - test-owned source
        return namespace["shim"]

    sibling = _function_with_filename(package_dir + "_contrib/model.py")
    assert _is_torchlens_instrumentation(sibling) is False

    interior = _function_with_filename(package_dir + "/wrapped.py")
    assert _is_torchlens_instrumentation(interior) is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_fingerprint.py.
def test_unhashable_tensor_attribute_never_matches() -> None:
    """A content-unreadable tensor poisons the key instead of keying stably."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    sparse = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2))
    first = _attribute_state_fragment(sparse)
    second = _attribute_state_fragment(sparse)
    assert first != second, (
        "content-unreadable tensors must mint never-matching fragments, "
        "never a stable content-blind one"
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_fingerprint.py.
def test_meta_tensor_attribute_keys_by_metadata() -> None:
    """Meta tensors have no bytes: metadata IS their content, keyed stably."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    a = torch.empty(3, device="meta")
    b = torch.empty(3, device="meta")
    assert _attribute_state_fragment(a) == _attribute_state_fragment(b)
    assert _attribute_state_fragment(a) != _attribute_state_fragment(torch.empty(4, device="meta"))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture_fingerprint.py.
def test_over_ceiling_containers_never_key_equal() -> None:
    """Containers past the item ceiling always miss instead of truncating.

    grind-r4 b4-opus F39-A': the dict/sequence/set fragments truncated at 256
    items, so two containers identical up to the cut but differing PAST it
    keyed identically and served the wrong cached trace (hit=True, no
    warning). Over-ceiling containers now mint never-matching tokens.
    """

    from torchlens._capture_state_helpers import (
        _ATTRIBUTE_FRAGMENT_ITEM_CEILING as ceiling,
        _attribute_state_fragment,
    )

    n = ceiling + 40
    seq_base = list(range(n))
    seq_changed = list(seq_base)
    seq_changed[ceiling + 10] = -1
    assert _attribute_state_fragment(seq_base) != _attribute_state_fragment(seq_changed)

    dict_base = {f"k{i}": i for i in range(n)}
    dict_changed = dict(dict_base)
    dict_changed[f"k{ceiling + 10}"] = -1
    assert _attribute_state_fragment(dict_base) != _attribute_state_fragment(dict_changed)

    set_base = {f"m{i:04d}" for i in range(n)}
    set_changed = (set_base - {f"m{ceiling + 10:04d}"}) | {"zzzz-tail-swap"}
    assert _attribute_state_fragment(set_base) != _attribute_state_fragment(set_changed)

    # The documented trade: even EQUAL over-ceiling containers never match
    # (conservative always-miss, cache utility traded for correctness) ...
    assert _attribute_state_fragment(seq_base) != _attribute_state_fragment(list(seq_base))
    # ... while in-ceiling containers keep stable equal-content fragments.
    assert _attribute_state_fragment(list(range(10))) == _attribute_state_fragment(list(range(10)))
