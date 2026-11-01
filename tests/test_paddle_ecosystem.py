"""Pretrained ``paddle.vision`` roster rows capture and validate (graduation item G3).

Needs the checkpoints (downloaded by Paddle on first use), so it runs only under
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.registry import get_backend_spec  # noqa: E402
from torchlens.conformance._rosters import paddle as paddle_roster  # noqa: E402

pytestmark = [
    pytest.mark.backend_paddle,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("TORCHLENS_CONFORMANCE_ECOSYSTEM") != "1",
        reason="ecosystem rows need TORCHLENS_CONFORMANCE_ECOSYSTEM=1 (checkpoint download)",
    ),
]

_ROWS = {row.family: row for row in paddle_roster.roster() if row.realism == "pretrained"}


@pytest.mark.parametrize("family", sorted(_ROWS))
def test_pretrained_row_captures_matches_and_validates(family: str) -> None:
    """Output equals the eager forward and replay validation passes."""

    model, image = _ROWS[family].builder()
    reference = model(image).numpy()
    trace = tl.trace(model, (image,), backend="paddle")

    output = [op for op in trace.layer_list if op.func_name == "functional.linear"][-1]
    np.testing.assert_allclose(output.out.numpy(), reference, rtol=1e-4, atol=1e-4)
    factory = get_backend_spec("paddle").protocol
    assert factory is not None
    status = factory().validate(trace, reference)
    assert status.state == "passed", status.message
