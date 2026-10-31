"""Regression tests for r18h introspection.py + hashing.py hardening.

Covers the round-18 A1 findings in the introspect/hash group:

* H5 -- ``dir()`` cached by ``type(item)`` dropped per-instance tensors on a
  second same-typed object (order-dependent capture gap).
* M6 -- ``make_short_barcode_from_input`` untyped ``str()`` + raw NUL join
  produced deterministic pre-hash collisions.
* M7/F8 -- ``compute_graph_shape_hash`` / ``compute_raw_event_shape_hash``
  sorted parent indices, making the shape hash blind to operand/edge order
  (feeds public ``torchlens.fingerprint.trace()`` and the re-trace reproducibility
  tripwire). Fixed consistent with the operand-order-sensitive refresh graph
  signature (commit 74898ada).
* M8 -- ``"grad" in name`` substring filter suppressed unrelated attributes such
  as ``upgrade``; replaced with exact-name matching.
* M9 -- ``iter_accessible_attributes`` promised warning suppression during
  attribute access but let warnings escape.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import torchlens._utils.hashing as hashing
import torchlens._utils.introspection as introspection
from torchlens import _state

# ---------------------------------------------------------------------------
# H5 -- per-instance dir() cache soundness (order-dependent capture gap)
# ---------------------------------------------------------------------------


class _Box:
    """Plain default-``__dir__`` object holding a tensor as an instance attr."""


def test_h5_second_same_typed_object_tensor_not_dropped() -> None:
    """Tensors on a second object of an already-seen type are still found."""

    _state._dir_cache.clear()
    first = _Box()
    first.first_field = torch.zeros(2)
    second = _Box()
    second.second_field = torch.ones(3)

    found = introspection.get_vars_of_type_from_obj([first, second], torch.Tensor)

    assert len(found) == 2, "second same-typed object's per-instance tensor was dropped"


def test_h5_capture_gap_is_order_independent() -> None:
    """Both orderings of the two objects surface both tensors."""

    _state._dir_cache.clear()
    a = _Box()
    a.a_field = torch.zeros(2)
    b = _Box()
    b.b_field = torch.ones(2)

    forward = introspection.get_vars_of_type_from_obj([a, b], torch.Tensor)
    _state._dir_cache.clear()
    reverse = introspection.get_vars_of_type_from_obj([b, a], torch.Tensor)

    assert len(forward) == 2
    assert len(reverse) == 2


def test_h5_nn_module_per_instance_params_found() -> None:
    """Custom-``__dir__`` objects (nn.Module) surface per-instance params."""

    _state._dir_cache.clear()
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(3, 3)
    params = introspection.get_vars_of_type_from_obj([m1, m2], nn.Parameter, allow_repeats=True)
    assert len(params) == 4, "per-instance registered parameters were missed"


# ---------------------------------------------------------------------------
# M6 -- type-tagged, injection-safe barcode
# ---------------------------------------------------------------------------


def test_m6_type_tag_disambiguates_coincident_str() -> None:
    """``1`` and ``"1"`` must not collide despite identical ``str()``."""

    assert hashing.make_short_barcode_from_input([1]) != hashing.make_short_barcode_from_input(
        ["1"]
    )


def test_m6_separator_cannot_be_forged() -> None:
    """A value containing the raw separator cannot masquerade as two elements."""

    assert hashing.make_short_barcode_from_input(
        ["a\x00b"]
    ) != hashing.make_short_barcode_from_input(["a", "b"])


def test_m6_is_deterministic() -> None:
    """Repeated calls on equal input give the same barcode."""

    assert hashing.make_short_barcode_from_input([1, "x", (2, 3)]) == (
        hashing.make_short_barcode_from_input([1, "x", (2, 3)])
    )


# ---------------------------------------------------------------------------
# M7/F8 -- operand-order-sensitive shape hashes
# ---------------------------------------------------------------------------


def _mk_layer(label: str, parents: list[str], func: str = "sub") -> SimpleNamespace:
    """Build a minimal layer stand-in for ``compute_graph_shape_hash``."""

    return SimpleNamespace(
        layer_label=label,
        parents=list(parents),
        module=None,
        layer_type="function",
        func_name=func,
        container_path=None,
        container_spec=None,
        is_input=False,
        is_output=False,
        is_buffer=False,
    )


def _graph(parents_of_c: list[str]) -> SimpleNamespace:
    """Two input layers feeding a noncommutative op with the given parent order."""

    return SimpleNamespace(
        layer_list=[
            _mk_layer("a", [], func="input"),
            _mk_layer("b", [], func="input"),
            _mk_layer("c", parents_of_c),
        ]
    )


def test_m7_graph_shape_hash_is_operand_order_sensitive() -> None:
    """(a, b) and (b, a) parents must hash differently."""

    assert hashing.compute_graph_shape_hash(_graph(["a", "b"])) != (
        hashing.compute_graph_shape_hash(_graph(["b", "a"]))
    )


def _mk_event(label: str, parent_labels: list[str]) -> SimpleNamespace:
    """Build a minimal raw capture event stand-in for the raw-event hash."""

    return SimpleNamespace(
        label_raw=label,
        kind="op",
        layer_type="function",
        function=SimpleNamespace(func_name="sub", func_qualname="torch.sub"),
        output=SimpleNamespace(
            tensor=SimpleNamespace(shape=(2,), dtype="float32"),
            container_path=None,
            container_spec=None,
        ),
        parents=[SimpleNamespace(parent_label_raw=p) for p in parent_labels],
        modules=(),
    )


def _events(parents_of_c: list[str]) -> SimpleNamespace:
    """Raw capture-events stand-in with an operand-order-variable op."""

    return SimpleNamespace(
        op_events=[
            _mk_event("a", []),
            _mk_event("b", []),
            _mk_event("c", parents_of_c),
        ]
    )


def test_m7_raw_event_shape_hash_is_operand_order_sensitive() -> None:
    """The sibling raw-event hash must also honor parent edge order."""

    assert hashing.compute_raw_event_shape_hash(_events(["a", "b"])) != (
        hashing.compute_raw_event_shape_hash(_events(["b", "a"]))
    )


# ---------------------------------------------------------------------------
# M8 -- exact grad-attr matching (no substring over-match)
# ---------------------------------------------------------------------------


class _WrapWithUpgrade:
    """Object whose tensor lives on an attribute containing the text 'grad'."""


def test_m8_attr_containing_grad_substring_not_suppressed() -> None:
    """A tensor at ``.upgrade`` must be discovered ('grad' is a substring)."""

    _state._dir_cache.clear()
    obj = _WrapWithUpgrade()
    obj.upgrade = torch.zeros(4)
    obj.gradient = torch.zeros(5)

    found = introspection.get_vars_of_type_from_obj(obj, torch.Tensor)
    assert len(found) == 2, "attributes merely containing 'grad' were over-filtered"


def test_m8_true_grad_attrs_still_skipped() -> None:
    """The real grad attributes stay excluded from the tensor crawl."""

    assert "grad" in introspection._ATTR_SKIP_SET
    assert "grad_fn" in introspection._ATTR_SKIP_SET
    # A crawled (non-matching) tensor must not leak its .grad into results.
    _state._dir_cache.clear()
    leaf = torch.zeros(3, requires_grad=True)
    (leaf * 2).sum().backward()

    class _Holder:
        """Non-tensor holder crawled while searching for a sentinel type."""

    holder = _Holder()
    holder.value = leaf
    # Search for a type the tensor is NOT, forcing the tensor to be crawled.
    found = introspection.get_vars_of_type_from_obj(holder, nn.Linear)
    assert found == []


# ---------------------------------------------------------------------------
# M9 -- iter_accessible_attributes warning-suppression contract
# ---------------------------------------------------------------------------


class _Noisy:
    """Object with a property that warns on access."""

    @property
    def noisy(self) -> int:
        """Emit a warning on access, as deprecated tensor properties do."""

        warnings.warn("noisy attribute access", UserWarning, stacklevel=2)
        return 1


def test_m9_warnings_suppressed_during_attribute_access() -> None:
    """No warning from attribute access escapes to the caller."""

    obj = _Noisy()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        collected = dict(introspection.iter_accessible_attributes(obj))

    assert collected.get("noisy") == 1
    assert [w for w in recorded if "noisy attribute access" in str(w.message)] == []


# ---------------------------------------------------------------------------
# Mutation proofs -- re-introduce each tightened defect and prove the test kills
# it. These run the ORIGINAL buggy logic inline; they never patch source.
# ---------------------------------------------------------------------------


def _old_barcode(things_to_hash: list[Any]) -> str:
    """The pre-fix M6 encoding (untyped str + raw NUL join)."""

    import hashlib

    joined = "\x00".join([str(x) for x in things_to_hash])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def test_mutation_m6_old_encoding_would_collide() -> None:
    """Mutation proof: the pre-fix barcode collides where the fix does not."""

    assert _old_barcode([1]) == _old_barcode(["1"])
    assert _old_barcode(["a\x00b"]) == _old_barcode(["a", "b"])


def _shape_hash_sorted(trace: SimpleNamespace) -> str:
    """The pre-fix M7 graph-shape hash that sorted parent indices."""

    import hashlib
    import json

    order_by_label = {layer.layer_label: i for i, layer in enumerate(trace.layer_list)}
    records = []
    for index, layer in enumerate(trace.layer_list):
        parent_indices = sorted(order_by_label[p] for p in layer.parents if p in order_by_label)
        records.append(
            {
                "index": index,
                "layer_type": layer.layer_type,
                "func_name": str(layer.func_name),
                "parent_indices": parent_indices,
                "_address_normalized": None,
                "container_path": [],
                "container_cardinality": None,
                "is_input": bool(layer.is_input),
                "is_output": bool(layer.is_output),
                "is_buffer": bool(layer.is_buffer),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_mutation_m7_sorted_parents_would_lose_operand_order() -> None:
    """Mutation proof: sorting parents makes (a,b) and (b,a) collide."""

    assert _shape_hash_sorted(_graph(["a", "b"])) == _shape_hash_sorted(_graph(["b", "a"]))


def test_mutation_h5_type_cached_full_dir_would_drop_tensor() -> None:
    """Mutation proof: type-caching the first instance's dir() drops the 2nd.

    Reproduces the pre-fix behaviour inline: a per-type cache seeded from the
    first object's ``dir()`` misses a second same-typed object's distinct
    instance attribute.
    """

    cache: dict[type, list[str]] = {}

    def buggy_names(item: object) -> list[str]:
        """Old logic: cache filtered dir(item) keyed by type, reuse blindly."""

        t = type(item)
        if t not in cache:
            cache[t] = [
                a
                for a in dir(item)
                if not a.startswith("__") and a not in introspection._ATTR_SKIP_SET
            ]
        return cache[t]

    first = _Box()
    first.first_field = torch.zeros(2)
    second = _Box()
    second.second_field = torch.ones(2)

    first_hits = [
        n for n in buggy_names(first) if isinstance(getattr(first, n, None), torch.Tensor)
    ]
    second_hits = [
        n for n in buggy_names(second) if isinstance(getattr(second, n, None), torch.Tensor)
    ]
    assert first_hits == ["first_field"]
    # The bug: the second object's tensor attribute is invisible via the cache.
    assert second_hits == []
