"""Two pretrained Flax families load completely and pass the JAX c1 cells (G3).

The rows download pinned checkpoints, so this runs only under
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1``. The independent output check against
the PyTorch Transformers models is ``tests/test_parity_b_jcheck_reference.py``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

from tests.conformance import _jax_checkpoints as checkpoints  # noqa: E402
from tests.conformance._rosters import jax as jax_roster  # noqa: E402

pytestmark = [
    pytest.mark.backend_jax,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("TORCHLENS_CONFORMANCE_ECOSYSTEM") != "1",
        reason="ecosystem rows need TORCHLENS_CONFORMANCE_ECOSYSTEM=1 (checkpoint download)",
    ),
    pytest.mark.rare,
]

#: family -> (pin, builder, tensors in the checkpoint file)
_FAMILIES: dict[str, tuple[checkpoints.FlaxCheckpoint, Any, int]] = {
    "bert_uncased_l2_h256_flax": (checkpoints.BERT_L2_H256, checkpoints.build_bert, 39),
    "vit_base_patch16_in21k_flax": (checkpoints.VIT_B16_IN21K, checkpoints.build_vit, 200),
}


def _ecosystem_rows() -> tuple[Any, ...]:
    """The JAX roster's ecosystem rows."""
    return jax_roster.ecosystem_roster()


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_every_checkpoint_tensor_fills_exactly_one_parameter(family: str) -> None:
    """The verified file's tensors and the model's parameters join one to one."""
    pin, builder, count = _FAMILIES[family]
    flat = checkpoints.load_flat_params(pin)
    assert len(flat) == count
    model, _ = builder()
    params = {
        "/".join(str(part) for part in path): variable
        for path, variable in nnx.iter_graph(model)
        if isinstance(variable, nnx.Param)
    }
    assert set(params) == set(flat)
    for path, variable in params.items():
        assert (jax.numpy.asarray(variable.value) == flat[path]).all(), path


def test_a_missing_tensor_is_refused() -> None:
    """Dropping one checkpoint tensor fails the load instead of leaving a random weight."""
    flat = dict(checkpoints.load_flat_params(checkpoints.BERT_L2_H256))
    flat.pop("pooler/dense/bias")
    model, _ = checkpoints.build_bert()
    with pytest.raises(checkpoints.CheckpointMismatchError, match="model-only"):
        checkpoints.assign_exact(model, flat)


def test_ecosystem_rows_pass_every_c1_cell() -> None:
    """Capture, parity, repeat, graph, payload, validation and constants cells all pass."""
    from tests.conformance._adapters import resolve_adapter
    from tests.conformance._cases import run_c1_cases

    rows = _ecosystem_rows()
    assert [row.family for row in rows] == list(_FAMILIES)
    results = run_c1_cases(resolve_adapter("jax"), rows)
    per_row = [result for result in results if result.family in _FAMILIES]
    assert {result.family for result in per_row} == set(_FAMILIES)
    failed = [(r.family, r.case_id, r.outcome, r.detail) for r in per_row if r.outcome != "passed"]
    assert not failed, failed
