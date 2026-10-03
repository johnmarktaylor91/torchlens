"""AUD-CODE 4.4 regression pins for OpAccessor / LayerAccessor.

``OpAccessor.__getitem__(int)`` is the 0-based POSITION while the write side
is the 1-based PASS INDEX (the storage key three finalization callers write
through). The in-fence part of the fix names the write basis explicitly
(``set_pass``) and pins the asymmetry so the coordinated caller change can
flip these expectations deliberately, never by accident. The duplicated
``layer.layer_label`` in ``LayerAccessor._resolve_substring`` is gone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchlens.data_classes._layer_accessors import LayerAccessor, OpAccessor


def _op(label: str) -> SimpleNamespace:
    return SimpleNamespace(
        label=f"{label}:1",
        label_short=f"{label}:1",
        layer_label=label,
        layer_label_short=label,
        _label_raw=label,
        raw_label=label,
    )


def test_set_pass_is_the_explicit_one_based_write() -> None:
    a, b = _op("a"), _op("b")
    accessor = OpAccessor()
    accessor.set_pass(1, a)
    accessor.set_pass(2, b)
    assert accessor[0] is a and accessor[1] is b
    assert accessor.keys() == [1, 2]
    assert len(accessor) == 2


@pytest.mark.parametrize("bad", [0, -1, True, "1"])
def test_set_pass_refuses_a_non_pass_index(bad: object) -> None:
    with pytest.raises(ValueError, match="1-based pass index"):
        OpAccessor().set_pass(bad, _op("a"))  # type: ignore[arg-type]


def test_setitem_int_is_pinned_as_the_pass_index_basis() -> None:
    """KNOWN ASYMMETRY (AUD-CODE 4.4): ops[1] = op is read back as ops[0].

    Flipping this to the read basis is a coordinated change with the
    finalization callers (backends/_finalize.py, backends/jax/backend.py,
    postprocess/finalization.py); this pin makes the flip deliberate.
    """

    a, c = _op("a"), _op("c")
    accessor = OpAccessor({1: a})
    accessor[1] = c
    assert accessor[0] is c
    assert accessor.keys() == [1]


def test_layer_accessor_resolves_short_label_once() -> None:
    layer = SimpleNamespace(layer_label="linear_1_1", layer_label_short="linear_1", is_buffer=False)
    accessor = LayerAccessor({"linear_1_1": layer})
    assert accessor._resolve_substring("linear_1_1") is layer
    assert accessor._resolve_substring("linear_1") is layer
    assert accessor._resolve_substring("nope") is None
