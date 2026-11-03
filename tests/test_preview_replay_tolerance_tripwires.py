"""Tripwire-strengthening tests for the MLX and Paddle replay oracles.

Two masked-corruption classes are locked out here:

* The MLX oracle used ``atol == rtol`` per dtype band, so every element whose
  magnitude sat below the band's decimal tolerance could be replaced by zero
  (or sign-flipped) and still read EQUAL.  The absolute term must only absorb
  jitter at the bottom of the representable range, never bless corruption of
  small normal values.
* The Paddle oracle applied one fp32-shaped decimal pair (rtol 1e-5 /
  atol 1e-6) to EVERY float dtype: fp64 corruption thousands of times larger
  than fp64 round-off read as agreement, while a legitimate 1-ULP fp16
  replay difference false-FAILED.

These tests exercise the pure-numpy comparison cores directly, so they run
without the optional backend runtimes installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.backends.mlx.validation import _payloads_close as mlx_payloads_close
from torchlens.backends.paddle.validation import _arrays_close as paddle_arrays_close


class TestMLXPayloadsClose:
    """MLX replay oracle: absolute term must not bless sub-band corruption."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/mlx/_capture_records.py.
    def test_all_zero_replay_of_low_magnitude_fp16_fails(self) -> None:
        """The fp16 band's absolute term must not swallow zeroed elements."""

        saved = np.full((8,), 4e-4, dtype=np.float16)
        assert not mlx_payloads_close(np.zeros_like(saved), saved)


class TestMLXValidationOracleEpsDerived:
    """The mlx validation oracle shares the eps-derived band with its siblings.

    Its former decimal ladder gave every wide float the single-precision
    relative tolerance, so relative double-precision corruption hundreds of
    millions of its own ULPs wide passed validation on mlx only.
    """

    @pytest.mark.parametrize(
        ("saved", "replay", "agree"),
        [
            pytest.param(
                np.full((4,), 1.0 + 1.0j, dtype=np.complex128),
                np.full((4,), 1.0 + 1.0j, dtype=np.complex128) * (1.0 + 1e-6),
                False,
                id="complex-component-corruption",
            ),
            pytest.param(
                np.ones((8,), dtype=np.float64),
                np.ones((8,), dtype=np.float64) * (1.0 + 1e-6),
                False,
                id="double-relative-corruption",
            ),
            pytest.param(
                np.ones((4,), dtype=np.float16),
                np.nextafter(np.ones((4,), dtype=np.float16), np.float16(2.0), dtype=np.float16),
                True,
                id="half-one-ulp-storage-rounding",
            ),
            pytest.param(
                np.full((8,), 4e-4, dtype=np.float16),
                np.zeros((8,), dtype=np.float16),
                False,
                id="half-low-magnitude-zeroed",
            ),
        ],
    )
    def test_eps_derived_band_rejects_corruption_and_admits_storage_rounding(
        self, saved: np.ndarray, replay: np.ndarray, agree: bool
    ) -> None:
        """Component-eps complex and double relative corruption fail; the half
        band admits one storage ULP and still rejects zeroed small elements."""

        assert mlx_payloads_close(saved.copy(), saved)
        assert mlx_payloads_close(replay, saved) is agree


class TestPaddleArraysClose:
    """Paddle replay oracle: tolerances must be dtype-aware in both directions."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/paddle/validation.py.
    def test_shape_and_dtype_mismatch_fail(self) -> None:
        """Shape or dtype disagreement is corruption, never coerced."""

        saved = np.ones((4,), dtype=np.float32)
        assert not paddle_arrays_close(np.ones((5,), dtype=np.float32), saved)
        assert not paddle_arrays_close(np.ones((4,), dtype=np.float64), saved)

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/paddle/validation.py.
    def test_integer_and_bool_stay_exact(self) -> None:
        """Integer (incl. bf16-as-uint16 transport) and bool payloads compare exactly."""

        ints = np.arange(4, dtype=np.uint16)
        assert paddle_arrays_close(ints.copy(), ints)
        assert not paddle_arrays_close(ints + 1, ints)
        bools = np.array([True, False])
        assert paddle_arrays_close(bools.copy(), bools)
        assert not paddle_arrays_close(~bools, bools)


@pytest.mark.backend_paddle
class TestPaddlePayloadsCloseLive:
    """The paddle-tensor wrapper must route through the dtype-aware core."""

    def test_fp64_corruption_fails_on_live_tensors(self) -> None:
        """fp64 corruption below the old decimal pair FAILS on real tensors."""

        paddle = pytest.importorskip("paddle")
        from torchlens.backends.paddle.validation import _payloads_close

        saved = paddle.full((8,), 5e-7, dtype="float64")
        zeros = paddle.zeros((8,), dtype="float64")
        assert not _payloads_close(zeros, saved)
        assert _payloads_close(saved.clone(), saved)


class TestComplexPayloadConsistency:
    """R17 consistency sweep: complex payloads take ONE policy on every
    backend (component-eps band) instead of bit-exact on tf/paddle vs
    tolerant on mlx/jax."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/tf/validation.py.
    def test_tf_complex_round_off_jitter_passes_corruption_fails(self) -> None:
        from torchlens.backends.tf.validation import _payloads_close as tf_close

        saved = np.full((4,), 1.0 + 1.0j, dtype=np.complex128)
        jitter = saved * (1.0 + 4.0 * float(np.finfo(np.float64).eps))
        assert tf_close(jitter, saved)
        assert not tf_close(saved * (1.0 + 1e-6), saved)


class TestTinygradScalarOracle:
    """R17 sweep sibling (R13 unswept port): the tinygrad payload oracle
    kept the dtype-blind 1e-6/1e-5 decimal pair with no NaN doctrine —
    byte-identical NaN replay FAILED only there, and fp64 corruption below
    the decimal pair PASSED."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_validation_shared.py, backends/tinygrad/__init__.py, backends/tinygrad/_captures.py ....
    def test_identical_nan_pattern_passes(self) -> None:
        from torchlens.backends.tinygrad.backend import _payload_values_close

        assert _payload_values_close(float("nan"), float("nan"), "dtypes.float")
        assert _payload_values_close([1.0, float("nan")], [1.0, float("nan")], "dtypes.float")

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_validation_shared.py.
    def test_equal_infinities_pass_opposite_fail(self) -> None:
        from torchlens.backends.tinygrad.backend import _payload_values_close

        assert _payload_values_close(float("inf"), float("inf"), "dtypes.float")
        assert not _payload_values_close(float("inf"), float("-inf"), "dtypes.float")
        assert not _payload_values_close(float("inf"), 1.0, "dtypes.float")

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_validation_shared.py.
    def test_fp64_corruption_below_old_decimal_pair_fails(self) -> None:
        from torchlens.backends.tinygrad.backend import _payload_values_close

        assert not _payload_values_close(1.0 + 1e-6, 1.0, "dtypes.double")
        assert _payload_values_close(
            1.0 + 4.0 * float(np.finfo(np.float64).eps), 1.0, "dtypes.double"
        )

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_validation_shared.py.
    def test_fp16_and_bfloat16_storage_bands(self) -> None:
        from torchlens.backends.tinygrad.backend import _payload_values_close

        one_ulp_16 = float(np.nextafter(np.float16(1.0), np.float16(2.0)))
        assert _payload_values_close(one_ulp_16, 1.0, "dtypes.half")
        # bfloat16: 4-ULP headroom at eps 2**-7; a >4-ULP delta fails.
        assert _payload_values_close(1.0 + 2.0 * 2.0**-7, 1.0, "dtypes.bfloat16")
        assert not _payload_values_close(1.0 + 8.0 * 2.0**-7, 1.0, "dtypes.bfloat16")

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/tinygrad/backend.py.
    def test_int_and_bool_stay_exact(self) -> None:
        from torchlens.backends.tinygrad.backend import _payload_values_close

        assert _payload_values_close(3, 3, "dtypes.int")
        assert not _payload_values_close(3, 4, "dtypes.int")
        assert _payload_values_close(True, True, "dtypes.bool")


class TestSharedDerivationSingleSource:
    """The derivation itself lives in exactly one reviewed module."""

    def test_no_stray_local_copies_remain(self) -> None:
        """SIBLING SWEEP GUARD: no backend module may re-grow a private
        ``_float_replay_tolerances`` definition or inline eps ladder."""

        import pathlib

        import torchlens

        backends = pathlib.Path(torchlens.__file__).parent / "backends"
        offenders = []
        for path in backends.rglob("*.py"):
            if path.name == "_validation_shared.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "def _float_replay_tolerances" in text or "def float_replay_tolerances" in text:
                offenders.append(str(path))
        assert not offenders, offenders
