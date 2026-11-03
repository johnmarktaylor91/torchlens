"""Static FIELD_ORDER / FieldPolicy / portable-state contract tests."""

from __future__ import annotations

from torchlens import _constants as constants
from torchlens._data_classes.buffer import Buffer
from torchlens._data_classes.grad_fn import GradFn

# Static contract checks over constants only: cheap enough for the commit-level
# gate, which is exactly where a schema drift has to become visible (the
# 00bc67d3 KEEP->DROP flips were invisible to -m smoke for want of a marker).


def test_known_glossary_aliases_have_backed_field_order_names() -> None:
    """Back-compat aliases are reconciled to current FIELD_ORDER names."""

    assert "activation_memory" in constants.OP_LOG_FIELD_ORDER
    assert "activation_memory" in constants.LAYER_LOG_FIELD_ORDER
    assert "activation_memory" in constants.BUFFER_LOG_FIELD_ORDER
    assert "memory" not in constants.OP_LOG_FIELD_ORDER
    assert "memory" not in constants.LAYER_LOG_FIELD_ORDER

    assert "overwrite_index" in constants.BUFFER_LOG_FIELD_ORDER
    assert "buffer_overwrite_index" not in constants.BUFFER_LOG_FIELD_ORDER
    assert "buffer_pass" not in constants.BUFFER_LOG_FIELD_ORDER
    assert hasattr(Buffer, "overwrite_index")
    assert not hasattr(Buffer, "buffer_overwrite_index")

    assert "has_op" in constants.GRAD_FN_LOG_FIELD_ORDER
    assert "is_intervening" not in constants.GRAD_FN_LOG_FIELD_ORDER
    assert "has_op" in GradFn._PORTABLE_STATE_SPEC
