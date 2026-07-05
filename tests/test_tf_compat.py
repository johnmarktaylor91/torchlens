"""Tests for TensorFlow private compatibility probes."""

from __future__ import annotations

import pytest

from torchlens.backends.registry import BackendUnsupportedError
from torchlens.backends.tf import _tf_compat as tfc

pytestmark = pytest.mark.smoke


def test_op_callbacks_absence_marks_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing TensorFlow op callbacks raise the public unsupported-backend error."""

    monkeypatch.setattr(tfc, "HAS_TF_OP_CALLBACKS", True)
    tfc._warned_missing_capabilities.discard("HAS_TF_OP_CALLBACKS")
    monkeypatch.setattr(tfc, "_import_op_callbacks_module", lambda: None)

    with pytest.warns(UserWarning, match="HAS_TF_OP_CALLBACKS"):
        with pytest.raises(BackendUnsupportedError):
            tfc.get_op_callbacks_module()

    assert tfc.HAS_TF_OP_CALLBACKS is False
