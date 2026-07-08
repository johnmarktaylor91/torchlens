"""Regression tests for the cert7 (round-7-prep) intervention hardening fixes.

Each test pins a silent-corruption / inappropriate-crash bug found by the cert6
inspector in the immediate neighborhood of the round-6 fixes -- the recurrence
pattern where a fix closed one instance of a class but left siblings open:

* BLOCKER -- ``_serialize_value``/``_deserialize_value`` detected all six internal
  sentinel-wrapper keys via unguarded ``key in value`` presence checks. A plain user
  dict whose only key literally equalled one of those strings was misread on load --
  five crashed loudly, but ``{"__opaque_audit__": ...}`` was *silently* reconstructed
  as a ``HelperSpec`` (data corruption). The fix escapes any dict carrying a reserved
  wrapper key (the ``__dunder__`` namespace) through the general ``__dict_items__``
  encoding so a genuine user key can never be mistaken for a wrapper tag.
* MAJOR -- ``Bundle.most_changed`` (and its sibling ``_distance_value``) reimplemented
  the round-6 scalar/vector routing guard asymmetrically (``is_scalar_like(base)``
  only), so a scalar baseline vs a genuine vector member fell into the scalar fallback
  and hard-crashed instead of routing to the intended vector metric. Both sites now use
  the symmetric ``is_scalar_like(a) and is_scalar_like(b)`` guard used by ``_diff_row``.
* MINOR -- ``"source_trace"`` was listed in both the ``share`` and ``reconstruct`` sets
  of ``_build_op_log_fork_policy``; the dead ``share`` membership was dropped.
"""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.bundle import _distance_value
from torchlens.intervention.save import (
    SaveLevel,
    _SerializedState,
    _deserialize_value,
    _is_reserved_wrapper_key,
    _serialize_value,
)


def _new_state() -> _SerializedState:
    """Return a fresh empty serialization state."""

    return _SerializedState(tensor_entries=[], tensor_refs={})


def _roundtrip(value: object) -> object:
    """Serialize then deserialize ``value`` through the spec value codec."""

    serialized = _serialize_value(value, SaveLevel.AUDIT, _new_state())
    return _deserialize_value(serialized, {})


# ---------------------------------------------------------------------------
# BLOCKER: reserved sentinel-key collision must never silently corrupt a dict.
# ---------------------------------------------------------------------------

# The six wrapper tags ``_deserialize_value`` keys off, plus ``__tuple_key__`` used
# by the dict-key codec. A user dict whose key equals any of these must round-trip
# unchanged (or fail loudly) -- never silently become a spec object.
_SENTINEL_KEYS = [
    "__tensor_ref__",
    "__helper__",
    "__callable__",
    "__output_path_component__",
    "__opaque_audit__",
    "__dict_items__",
    "__tuple_key__",
]


def test_user_dict_with_sentinel_key_roundtrips_unchanged() -> None:
    """BLOCKER: a user dict keyed by any reserved sentinel survives round-trip."""

    for key in _SENTINEL_KEYS:
        # A dict value (the shape that silently corrupted for ``__opaque_audit__``).
        with_dict = {key: {"foo": "bar", "n": 3}}
        loaded = _roundtrip(with_dict)
        assert loaded == with_dict, f"{key!r} dict-value payload was corrupted"
        assert type(loaded) is dict
        assert type(loaded[key]) is dict

        # A scalar value (the shape that crashed loudly for the crash-class keys).
        with_scalar = {key: 42}
        assert _roundtrip(with_scalar) == with_scalar, f"{key!r} scalar payload corrupted"


def test_sentinel_key_mixed_and_nested_dicts_roundtrip() -> None:
    """BLOCKER: sentinel keys survive alongside plain keys and when nested."""

    mixed = {"__opaque_audit__": {"x": 1}, "plain": "ok", "__helper__": [1, 2, 3]}
    assert _roundtrip(mixed) == mixed

    nested = {"outer": {"__callable__": {"deep": "value"}}}
    assert _roundtrip(nested) == nested

    # A dict that mixes a reserved key with a genuine non-string key.
    combined = {"__dict_items__": "s", 7: "int-key"}
    assert _roundtrip(combined) == combined


def test_plain_all_string_dict_keeps_plain_on_disk_format() -> None:
    """BLOCKER: ordinary dicts keep the unescaped plain-object encoding."""

    plain = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    serialized = _serialize_value(plain, SaveLevel.AUDIT, _new_state())
    # No ``__dict_items__`` escaping for the common all-plain-keys case.
    assert serialized == plain
    assert _deserialize_value(serialized, {}) == plain


def test_reserved_wrapper_key_predicate() -> None:
    """BLOCKER: the reserved-namespace predicate flags dunders, not plain keys."""

    for key in _SENTINEL_KEYS:
        assert _is_reserved_wrapper_key(key)
    assert _is_reserved_wrapper_key("__anything_future__")
    # Plain keys (including single/partial underscores) stay unescaped.
    for key in ("plain", "_leading", "trailing_", "__", "a", "__x"):
        assert not _is_reserved_wrapper_key(key)
    assert not _is_reserved_wrapper_key(3)


# ---------------------------------------------------------------------------
# MAJOR: symmetric scalar/vector routing guard in the bundle distance helpers.
# ---------------------------------------------------------------------------


def test_distance_value_scalar_vs_vector_routes_to_vector_metric() -> None:
    """MAJOR: a scalar ref vs a vector candidate uses the vector metric, not the
    scalar fallback -- and thus fails via the *intended* metric guard."""

    scalar = torch.tensor(3.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    try:
        _distance_value(scalar, vector, "cosine")
    except ValueError as exc:
        # Symmetric guard routed to cosine (the vector metric). The OLD asymmetric
        # guard would have called ``relative_l1_scalar`` and raised its message.
        assert "cosine_distance" in str(exc)
        assert "relative_l1_scalar" not in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected a vector-metric ValueError for scalar-vs-vector")


def test_distance_value_both_scalar_uses_scalar_fallback() -> None:
    """MAJOR: genuinely scalar-like operands still use the scalar fallback."""

    # 0-d ref vs 1-element vector: both scalar-like -> relative_l1_scalar, no crash,
    # no truncation. |3 - 5| / |3| == 2/3.
    value = _distance_value(torch.tensor(3.0), torch.tensor([5.0]), "cosine")
    assert isinstance(value, float)
    assert abs(value - (2.0 / 3.0)) < 1e-6


def test_most_changed_happy_path_unbroken_by_symmetric_guard() -> None:
    """MAJOR: the symmetric guard leaves the normal same-shape path working."""

    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model_a = torch.nn.Linear(4, 4)
    model_b = torch.nn.Linear(4, 4)
    log_a = tl.trace(model_a, x, intervention_ready=True)
    log_b = tl.trace(model_b, x, intervention_ready=True)
    bundle = tl.bundle({"a": log_a, "b": log_b}, baseline="a")

    rows = bundle.most_changed()
    assert isinstance(rows, list)
    # Non-empty, sorted descending, finite scores -- no crash, no truncation.
    assert rows
    scores = [score for _, score in rows]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(score, float) and score == score for score in scores)


# ---------------------------------------------------------------------------
# MINOR: ``source_trace`` no longer duplicated across share/reconstruct sets.
# ---------------------------------------------------------------------------


def test_op_log_fork_policy_source_trace_is_reconstruct_only() -> None:
    """MINOR: ``source_trace`` resolves to RECONSTRUCT and is not double-listed."""

    from torchlens.intervention.types import (
        LAYER_PASS_LOG_FIELD_FORK_POLICY,
        ForkFieldPolicy,
    )

    assert LAYER_PASS_LOG_FIELD_FORK_POLICY["source_trace"] is ForkFieldPolicy.FORK_RECONSTRUCT
