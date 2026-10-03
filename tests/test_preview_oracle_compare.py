"""Preview-backend replay-oracle comparison drift tests (R13-8b).

The tf/mlx/paddle validation oracles each own a payload comparison helper.
These pure-numpy tests pin the cross-backend doctrine without requiring the
preview frameworks: identical NaN patterns are agreement (``equal_nan=True``,
matching torch's ``tensor_nanequal``), and tolerance bands are sized to each
dtype's own ULP scale (fp16 must not inherit bf16's 10x coarser band).
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.backends.mlx.validation import _payloads_close as mlx_payloads_close
from torchlens.backends.tf.validation import _payloads_close as tf_payloads_close


def test_tf_oracle_treats_identical_nan_patterns_as_agreement() -> None:
    """TF was the one sibling omitting equal_nan: NaN payloads false-FAILED."""

    payload = np.array([1.0, np.nan, -2.5], dtype=np.float32)
    assert tf_payloads_close(payload, payload.copy())
    # NaN-vs-number still fails elementwise.
    disagreeing = np.array([1.0, 7.0, -2.5], dtype=np.float32)
    assert not tf_payloads_close(payload, disagreeing)
    # And the tolerance band still rejects real drift.
    assert not tf_payloads_close(
        np.array([1.0, 2.0], dtype=np.float32), np.array([1.01, 2.0], dtype=np.float32)
    )


def test_mlx_oracle_gives_fp16_its_own_tolerance_band() -> None:
    """fp16 (eps 9.8e-4) was lumped with bf16 (eps 7.8e-3) under one 1e-2 band.

    A ~0.9% relative drift is ~9 fp16 ULPs -- far beyond faithful-replay noise
    -- and used to pass. It must now fail for fp16 while few-ULP fp16 noise
    still passes and float32 keeps its tighter band.
    """

    base16 = np.full((8,), 1.0, dtype=np.float16)
    corrupted16 = (base16.astype(np.float64) * 1.009).astype(np.float16)
    assert not mlx_payloads_close(base16, corrupted16)

    one_ulp16 = np.nextafter(base16, np.float16(2.0))
    assert mlx_payloads_close(base16, one_ulp16)

    base32 = np.full((8,), 1.0, dtype=np.float32)
    assert not mlx_payloads_close(base32, base32 * np.float32(1.0001))
    assert mlx_payloads_close(base32, base32 * np.float32(1.0 + 5e-6))

    # NaN doctrine unchanged: identical patterns agree.
    nan16 = np.array([np.nan, 1.0], dtype=np.float16)
    assert mlx_payloads_close(nan16, nan16.copy())


def test_paddle_oracle_source_matches_nan_doctrine() -> None:
    """paddle/validation.py's compare uses equal_nan like its own backend oracle.

    The helper needs live paddle tensors to execute, so this pins the source:
    the float branch must carry ``equal_nan=True`` (the drift was validation.py
    lacking it while paddle/backend.py had it).
    """

    import inspect

    import torchlens.backends.paddle.validation as paddle_validation

    source = inspect.getsource(paddle_validation)
    float_branch = source.split("np.issubdtype(left.dtype, np.floating)")[1]
    assert "equal_nan=True" in float_branch.split("return")[1]


@pytest.mark.smoke
def test_blas_layout_band_rejects_beyond_64_ulp() -> None:
    """R13-8a: the benign-layout band is 64 ULP (16x the observed ~4 ULP).

    This band downgrades an attestation mismatch to not_applicable instead of
    raising, so looseness here silently launders corruption: the former 256
    blessed 4x more divergence than the derivation supports. Faithful
    reduction-order noise (few ULP) stays inside; a 128-ULP divergence -- which
    the old band accepted -- must now raise through the strict path.
    """

    import torch

    from torchlens._runnable_path_faithfulness import _within_layout_reduction_tolerance

    eps = torch.finfo(torch.float32).eps
    archived = torch.full((64,), 1.0, dtype=torch.float32)

    within = archived * (1.0 + 4 * eps)
    assert _within_layout_reduction_tolerance(within, archived)

    beyond = archived * (1.0 + 128 * eps * 2.0)  # 128 ULP against the (|x|+1) band
    assert not _within_layout_reduction_tolerance(beyond, archived)
