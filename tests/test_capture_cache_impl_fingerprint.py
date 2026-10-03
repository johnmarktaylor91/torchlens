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

import types

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.utils._torch_compat import get_cpu_float8_deterministic_fill_support


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


def test_changed_class_forward_is_a_cache_miss(tmp_path) -> None:
    """Same state_dict, changed ``forward`` -> cache miss, fresh capture."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    assert any("relu" in label for label in _op_labels(first))

    original_forward = _CacheModel.forward
    try:

        def sigmoid_forward(self, inputs):  # noqa: ANN001 - test shim
            return torch.sigmoid(self.lin(inputs))

        _CacheModel.forward = sigmoid_forward
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    finally:
        _CacheModel.forward = original_forward

    assert second.capture_cache_hit is False, (
        "a changed forward implementation must not hit the stale cached trace"
    )
    labels = _op_labels(second)
    assert any("sigmoid" in label for label in labels)
    assert not any("relu" in label for label in labels)


def test_instance_forward_override_is_a_cache_miss(tmp_path) -> None:
    """An instance-level ``forward`` override also invalidates the cache."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False

    def tanh_forward(self, inputs):  # noqa: ANN001 - test shim
        return torch.tanh(self.lin(inputs))

    model.forward = types.MethodType(tanh_forward, model)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    finally:
        del model.forward

    assert second.capture_cache_hit is False
    assert any("tanh" in label for label in _op_labels(second))


def test_unchanged_model_still_hits(tmp_path) -> None:
    """The fingerprint stays stable for an unchanged model (no false misses)."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is True


def test_submodule_structure_change_is_a_cache_miss(tmp_path) -> None:
    """Swapping a parameter-free submodule class changes the implementation."""

    class _Act(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    class _OtherAct(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    class _Wrapper(nn.Module):
        def __init__(self, act: nn.Module) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.act = act

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.lin(x))

    torch.manual_seed(0)
    first_model = _Wrapper(_Act())
    torch.manual_seed(0)
    second_model = _Wrapper(_OtherAct())
    # Identical tensor content: only the activation submodule CLASS differs.
    second_model.load_state_dict(first_model.state_dict())

    x = torch.randn(1, 4)
    first = tl.trace(first_model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(second_model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False
    assert any("sigmoid" in label for label in _op_labels(second))


class _AttrLoop(nn.Module):
    """Model whose traced program depends on a PLAIN instance attribute."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        for _ in range(self.k):
            y = torch.relu(y)
        return y


def test_changed_plain_instance_attribute_is_a_cache_miss(tmp_path) -> None:
    """Loop(5) must never be served Loop(1)'s cached trace (grind-r2 F39-1).

    The content fingerprint covers ``state_dict`` + training flags +
    non-persistent buffers and the implementation fingerprint covers module
    tree + class identity + ``forward`` code; before the fix nothing covered
    instance ``__dict__`` config, so ``Loop(5)`` hit ``Loop(1)``'s entry
    (hit=True, 3 ops instead of 7, no warning).
    """

    x = torch.randn(1, 4)
    torch.manual_seed(0)
    first = tl.trace(_AttrLoop(1), x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    torch.manual_seed(0)
    second = tl.trace(_AttrLoop(5), x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a changed plain instance attribute must not hit the stale cached trace"
    )
    assert len(second.layer_labels) > len(first.layer_labels)
    torch.manual_seed(0)
    third = tl.trace(_AttrLoop(5), x, capture=_cache_capture(tmp_path))
    assert third.capture_cache_hit is True, (
        "an unchanged attribute inventory must still re-hit (no false misses)"
    )


def test_mutated_numeric_instance_attribute_is_a_cache_miss(tmp_path) -> None:
    """The malignant variant: same op count, numerically wrong served values."""

    class _Scaled(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = 1.0
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) * self.scale

    model = _Scaled()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    model.scale = 2.0
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False
    assert torch.allclose(second.layer_list[-1].out, first.layer_list[-1].out * 2.0)


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


def test_registered_forward_hook_is_a_cache_miss(tmp_path) -> None:
    """A user nn.Module hook is real model behavior: register/edit must miss.

    grind-r2 b4-fable R39 finding 1: the config key's ``hooks`` entry covers
    only the TorchLens ``hooks=`` kwarg; an ``m.register_forward_hook(...)``
    between ``cache=True`` runs hit the pre-hook cached trace and served
    wrong activations with no warning.
    """

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False

    handle = model.register_forward_hook(lambda mod, args, out: out * 2)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert second.capture_cache_hit is False, (
            "a newly registered forward hook must not hit the hook-free cached trace"
        )
    finally:
        handle.remove()

    # Removing the hook restores the original inventory: the first entry re-hits.
    third = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert third.capture_cache_hit is True


def test_registered_forward_pre_hook_is_a_cache_miss(tmp_path) -> None:
    """Pre-hooks mutate module inputs and must also invalidate the key."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False

    handle = model.lin.register_forward_pre_hook(lambda mod, args: (args[0] + 1,))
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert second.capture_cache_hit is False
    finally:
        handle.remove()


def test_edited_hook_implementation_is_a_cache_miss(tmp_path) -> None:
    """Same registration slot, different hook CODE -> different key."""

    model = _CacheModel()
    x = torch.randn(1, 4)

    handle = model.register_forward_hook(lambda mod, args, out: out * 2)
    try:
        first = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert first.capture_cache_hit is False
    finally:
        handle.remove()

    handle = model.register_forward_hook(lambda mod, args, out: out * 3)
    try:
        second = tl.trace(model, x, capture=_cache_capture(tmp_path))
        assert second.capture_cache_hit is False, (
            "an edited hook implementation must not hit the old hook's cached trace"
        )
    finally:
        handle.remove()


def test_input_requires_grad_flip_is_a_cache_miss(tmp_path) -> None:
    """requires_grad changes grad_fn/backward metadata: the key must see it.

    grind-r2 b4-fable R39-2: shape + dtype + CPU bytes were the whole tensor
    hash, so freezing params or flipping input requires_grad between
    cache=True runs hit the stale entry with wrong tensor_requires_grad /
    grad_fn metadata.
    """

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(model, x.clone().requires_grad_(True), capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "an input requires_grad flip must not hit the no-grad cached trace"
    )


def test_frozen_parameters_are_a_cache_miss(tmp_path) -> None:
    """Freezing params (requires_grad_(False)) changes captured grad metadata."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False


def test_tensor_content_hash_covers_device_and_requires_grad() -> None:
    """Unit coverage for the hash axes (CUDA device flip untestable on CPU CI)."""

    from torchlens._capture_state_helpers import _hash_tensor_content

    base = torch.ones(3)
    flagged = torch.ones(3).requires_grad_(True)
    assert _hash_tensor_content(base) != _hash_tensor_content(flagged)
    assert _hash_tensor_content(base) == _hash_tensor_content(torch.ones(3))


def test_sibling_package_forward_is_not_torchlens_instrumentation() -> None:
    """A ``torchlens_contrib`` install must not be classified as TorchLens'.

    grind-r2 b4-fable R39-3: the bare prefix match claimed any sibling path
    that string-extends the package dir, so a user forward override defined
    in ``.../site-packages/torchlens_contrib/model.py`` was EXCLUDED from the
    implementation signature and its edits silently hit the stale cache.
    """

    from torchlens._capture_state_helpers import (
        _TORCHLENS_PACKAGE_DIR,
        _is_torchlens_instrumentation,
    )

    def _function_with_filename(filename: str):
        code = compile("def shim(x):\n    return x\n", filename, "exec")
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102 - test-owned source
        return namespace["shim"]

    sibling = _function_with_filename(_TORCHLENS_PACKAGE_DIR + "_contrib/model.py")
    assert _is_torchlens_instrumentation(sibling) is False

    interior = _function_with_filename(_TORCHLENS_PACKAGE_DIR + "/wrapped.py")
    assert _is_torchlens_instrumentation(interior) is True


def test_tensor_content_hash_frames_logical_dtype_bf16_vs_fp32() -> None:
    """bf16 and fp32 tensors with equal values must NOT collide (r3 R35-1).

    ``_hash_tensor_content`` upcasts bf16 to float32 for numpy transport and
    framed the POST-upcast dtype, so a bfloat16 attribute/state tensor hashed
    identically to its float32 twin and ``cache=True`` served the WRONG
    cached trace across the dtype change. The digest now frames the logical
    (pre-upcast) dtype, like ``op.py::_tensor_content_hash``.
    """

    from torchlens._capture_state_helpers import _hash_tensor_content

    values = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    as_bf16 = values.to(torch.bfloat16)
    as_fp32 = as_bf16.to(torch.float32)  # exact same numeric payload post-upcast
    assert _hash_tensor_content(as_bf16) != _hash_tensor_content(as_fp32)
    # Determinism within one dtype is unchanged.
    assert _hash_tensor_content(as_bf16) == _hash_tensor_content(as_bf16.clone())


def test_bf16_state_flip_is_a_cache_miss(tmp_path) -> None:
    """End-to-end: casting a buffer bf16<->fp32 must be a capture-cache miss."""

    class _BufModel(nn.Module):
        def __init__(self, dtype: torch.dtype) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor([2.0], dtype=dtype))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale.to(x.dtype)

    x = torch.randn(1, 4)
    first = tl.trace(_BufModel(torch.bfloat16), x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(_BufModel(torch.float32), x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a bf16 buffer and its fp32 twin must not share a capture-cache key"
    )


@pytest.mark.skipif(
    not get_cpu_float8_deterministic_fill_support(),
    reason="CPU Float8 empty-fill under deterministic mode postdates the torch 2.1 floor",
)
def test_float8_attribute_tensors_key_by_content() -> None:
    """Exotic-dtype tensors hash by CONTENT via the uint8-view byte path.

    r3 b1/b7 (partial reopen of the p4 R39 attr-blind class): the hash-failure
    fallback degraded to a content-blind ("tensor-meta", shape, dtype, device)
    fragment, so two DIFFERENT-content float8 attribute tensors produced
    identical key fragments -- a false cache HIT through the fix's own
    "false hits never" docstring.
    """

    float8 = getattr(torch, "float8_e4m3fn", None)
    if float8 is None:
        pytest.skip("torch build without float8_e4m3fn")

    from torchlens._capture_state_helpers import _attribute_state_fragment

    a = torch.tensor([1.0, 2.0]).to(float8)
    b = torch.tensor([3.0, 4.0]).to(float8)
    assert _attribute_state_fragment(a) != _attribute_state_fragment(b)
    # Equal content still keys equal: float8 keeps positive cache utility.
    assert _attribute_state_fragment(a) == _attribute_state_fragment(a.clone())


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


def test_meta_tensor_attribute_keys_by_metadata() -> None:
    """Meta tensors have no bytes: metadata IS their content, keyed stably."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    a = torch.empty(3, device="meta")
    b = torch.empty(3, device="meta")
    assert _attribute_state_fragment(a) == _attribute_state_fragment(b)
    assert _attribute_state_fragment(a) != _attribute_state_fragment(torch.empty(4, device="meta"))


class _UnderscoreLoop(nn.Module):
    """Loop count held in a LEADING-UNDERSCORE attribute (r3 R39-1 probe)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self._n):
            x = self.lin(x)
        return x


def test_underscore_instance_attribute_is_a_cache_miss(tmp_path) -> None:
    """A changed ``self._n`` must miss: the traced program depends on it.

    r3 b4-opus-R39-1 (REOPENED wrong-trace-on-hit): the plain-attribute
    filter skipped every leading-underscore name, so ``_UnderscoreLoop(5)``
    with identical weights hit ``_UnderscoreLoop(1)``'s cached trace and
    silently returned a 1-layer graph for a 5-layer forward.
    """

    x = torch.randn(1, 4)
    one = _UnderscoreLoop(1)
    five = _UnderscoreLoop(5)
    five.load_state_dict(one.state_dict())  # identical weights, different program

    first = tl.trace(one, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    assert sum("linear" in op.label for op in first.ops) == 1

    second = tl.trace(five, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a changed underscore instance attribute must not hit the stale cached trace"
    )
    # Five passes of the (recurrently grouped) linear layer, not one.
    assert sum("linear" in op.label for op in second.ops) == 5


def test_underscore_attribute_unchanged_model_still_hits(tmp_path) -> None:
    """Same underscore attrs, same weights -> the second capture still hits."""

    x = torch.ones(1, 4)
    model = _UnderscoreLoop(3)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is True


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


def test_over_ceiling_dict_tail_difference_is_a_cache_miss(tmp_path) -> None:
    """End-to-end F39-A' probe: a config value past the 256-item cut steers
    the traced program; hitting the stale entry serves the WRONG trace."""

    class _DictLoop(nn.Module):
        def __init__(self, num_layers: int) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            config = {f"pad_{i}": i for i in range(280)}
            config["num_layers"] = num_layers  # inserted past the cut
            self.config = config

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(self.config["num_layers"]):
                x = torch.relu(self.lin(x))
            return x

    x = torch.randn(1, 4)
    shallow = _DictLoop(1)
    first = tl.trace(shallow, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    assert first["relu_1_2"].num_passes == 1

    deep = _DictLoop(5)
    deep.load_state_dict(shallow.state_dict())
    second = tl.trace(deep, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is False, (
        "a config difference past the container item ceiling must not hit the stale cached trace"
    )
    assert second["relu_1_2"].num_passes == 5


def test_depth_ceiling_difference_never_keys_equal() -> None:
    """Depth-ceiling sibling of F39-A': a difference BELOW the depth ceiling
    must not key equal through a stable ceiling token."""

    from torchlens._capture_state_helpers import _attribute_state_fragment

    def nest(leaf: object, levels: int) -> list:
        value: list = [leaf]
        for _ in range(levels):
            value = [value]
        return value

    deep_a = nest({"num_layers": 1}, 8)
    deep_b = nest({"num_layers": 5}, 8)
    assert _attribute_state_fragment(deep_a) != _attribute_state_fragment(deep_b)
