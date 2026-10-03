"""Tripwire-strengthening tests for the remaining preview comparison oracles.

Finding R13 (round-2 hunts): five preview-backend replay/gradient oracles kept
the dtype-blind fp32 decimal pair (``rtol=1e-5, atol=1e-6``) that the paddle
and MLX validation oracles already shed in b7a07864.  The pair is wrong in
both directions:

* fp64 corruption at ~1e-6 relative is ~4.5e9 fp64 ULPs yet read as agreement;
* the ``atol=1e-6`` floor blessed TOTAL corruption (zeroed / sign-flipped
  replays) of every element below 1e-6;
* fp16 payloads false-failed (rtol 1e-5 is ~1/100 of one fp16 ULP).

These tests pin the ported per-dtype derivation (accumulating dtypes keep the
legacy fp32 strictness rescaled by the eps ratio; storage dtypes get a 4-ULP
band; ``atol = rtol * finfo.tiny``) at every remaining site:

* ``tf/validation.py::_payloads_close`` (TF replay validation oracle),
* ``tf/derived_grads.py::_tf_values_close`` (TF gradient divergence refusal),
* ``jax/backend.py::_values_close`` (JAX replay oracle; pure derivation
  helper tested framework-free),
* ``paddle/backend.py::_paddle_values_close`` (paddle replay divergence gate),
* ``mlx/backend.py::_mlx_values_close`` (MLX GradOptions divergence refusal).

The comparison cores are numpy-level, so everything but the jax live check
runs without the optional backend runtimes installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from torchlens.backends.mlx.backend import _mlx_values_close
from torchlens.backends.paddle.backend import _paddle_values_close
from torchlens.backends.tf.derived_grads import _tf_values_close
from torchlens.backends.tf.validation import _payloads_close as tf_payloads_close


class _StubPaddleTensor:
    """Minimal stand-in for a paddle tensor: shape, dtype text, ``.numpy()``."""

    def __init__(self, array: np.ndarray, dtype_text: str | None = None) -> None:
        self._array = array
        self.shape = tuple(array.shape)
        self.dtype = dtype_text if dtype_text is not None else str(array.dtype)

    def numpy(self) -> np.ndarray:
        return self._array


def _paddle_close(left: np.ndarray, right: np.ndarray, dtype_text: str | None = None) -> bool:
    """Compare two arrays through the paddle backend oracle via stub tensors."""

    return _paddle_values_close(
        _StubPaddleTensor(left, dtype_text), _StubPaddleTensor(right, dtype_text)
    )


_NUMPY_ORACLES: tuple[tuple[str, Any], ...] = (
    ("tf_validation", tf_payloads_close),
    ("tf_derived_grads", _tf_values_close),
    ("mlx_backend", _mlx_values_close),
    ("paddle_backend", _paddle_close),
)


@pytest.mark.parametrize(("site", "close"), _NUMPY_ORACLES, ids=[s for s, _ in _NUMPY_ORACLES])
class TestDtypeHonestBands:
    """All four numpy-level oracles must use per-dtype ULP-derived bands."""

    @pytest.mark.smoke_cells("test_fp64_relative_corruption_fails[tf_derived_grads]")
    def test_fp64_relative_corruption_fails(self, site: str, close: Any) -> None:
        """fp64 corruption ~4.5e9 ULP must FAIL (the old blanket rtol blessed it)."""

        saved = np.ones((8,), dtype=np.float64)
        assert not close(saved * (1.0 + 1e-6), saved)

    def test_fp64_reorder_scale_noise_passes(self, site: str, close: Any) -> None:
        """fp64 differences at fp64 round-off scale stay a PASS."""

        saved = np.ones((8,), dtype=np.float64)
        replay = saved * (1.0 + 4.0 * float(np.finfo(np.float64).eps))
        assert close(replay, saved)

    @pytest.mark.smoke_cells("test_fp64_small_value_corruption_fails[tf_validation]")
    def test_fp64_small_value_corruption_fails(self, site: str, close: Any) -> None:
        """fp64 elements below the old 1e-6 atol floor must not be zeroable."""

        saved = np.full((8,), 5e-7, dtype=np.float64)
        assert not close(np.zeros_like(saved), saved)

    def test_fp32_small_value_corruption_fails(self, site: str, close: Any) -> None:
        """fp32 elements below the old 1e-6 atol floor must not be zeroable."""

        saved = np.full((8,), 5e-7, dtype=np.float32)
        assert not close(np.zeros_like(saved), saved)

    def test_fp32_sign_flip_of_small_values_fails(self, site: str, close: Any) -> None:
        """Sign-flipped low-magnitude fp32 replays must FAIL, not vanish in atol."""

        saved = np.array([4e-7, -3e-7, 2e-7, -1e-7], dtype=np.float32)
        assert not close(-saved, saved)

    def test_fp32_relative_band_unchanged(self, site: str, close: Any) -> None:
        """The fp32 relative band keeps the legacy 1e-5 strictness."""

        saved = np.ones((8,), dtype=np.float32)
        assert close(saved * np.float32(1.0 + 5e-6), saved)
        assert not close(saved * np.float32(1.0 + 1e-4), saved)

    @pytest.mark.smoke_cells("test_fp16_one_ulp_rounding_passes[mlx_backend]")
    def test_fp16_one_ulp_rounding_passes(self, site: str, close: Any) -> None:
        """A one-ULP fp16 storage-rounding difference is agreement, not corruption.

        The old dtype-blind pair (rtol 1e-5, ~1/100 of one fp16 ULP)
        false-failed every non-bitwise fp16 agreement.
        """

        saved = np.ones((4,), dtype=np.float16)
        replay = np.nextafter(saved, np.float16(2.0), dtype=np.float16)
        assert close(replay, saved)

    def test_fp16_zeroed_replay_fails(self, site: str, close: Any) -> None:
        """Zeroed fp16 activations still FAIL under the dtype-aware band."""

        saved = np.full((8,), 1e-2, dtype=np.float16)
        assert not close(np.zeros_like(saved), saved)

    def test_identical_payloads_pass(self, site: str, close: Any) -> None:
        """Exact agreement stays a PASS."""

        saved = np.array([4e-7, -3e-7, 0.0, 1.5], dtype=np.float64)
        assert close(saved.copy(), saved)

    def test_nan_pattern_agreement_passes_nan_vs_number_fails(self, site: str, close: Any) -> None:
        """equal_nan doctrine unchanged: identical NaN patterns agree."""

        saved = np.array([np.nan, 1.0], dtype=np.float64)
        assert close(saved.copy(), saved)
        assert not close(np.array([0.0, 1.0], dtype=np.float64), saved)

    def test_shape_mismatch_fails(self, site: str, close: Any) -> None:
        """Shape disagreement is corruption, never coerced."""

        saved = np.ones((4,), dtype=np.float32)
        assert not close(np.ones((5,), dtype=np.float32), saved)


class TestNonFloatExactness:
    """Integer/bool payloads keep exact comparison at every fixed site."""

    def test_tf_validation_integer_exact(self) -> None:
        ints = np.arange(4, dtype=np.int64)
        assert tf_payloads_close(ints.copy(), ints)
        assert not tf_payloads_close(ints + 1, ints)

    def test_mlx_backend_bool_exact(self) -> None:
        bools = np.array([True, False])
        assert _mlx_values_close(bools.copy(), bools)
        assert not _mlx_values_close(~bools, bools)

    @pytest.mark.smoke
    def test_paddle_bf16_uint16_transport_compares_exactly(self) -> None:
        """bf16-as-uint16 transport has no numpy finfo: exact bit comparison.

        The paddle dtype TEXT says bfloat16 (a float), but ``.numpy()``
        transports the payload as ``uint16``; deriving a band from the wrong
        dtype's finfo (or the old decimal pair's implicit float cast) is
        forbidden -- this mirrors paddle/validation.py's transport handling.
        """

        saved = (np.arange(4, dtype=np.uint16) + 1) << 7
        assert _paddle_close(saved.copy(), saved, dtype_text="paddle.bfloat16")
        assert not _paddle_close(np.zeros_like(saved), saved, dtype_text="paddle.bfloat16")
        assert not _paddle_close(saved + 1, saved, dtype_text="paddle.bfloat16")


class TestJaxToleranceDerivation:
    """The JAX oracle's band derivation must be per-dtype ULP-honest."""

    def test_derivation_matches_the_ported_error_model(self) -> None:
        """fp64/fp32/fp16 rows match the paddle/mlx-validation derivation."""

        from torchlens.backends._validation_shared import (
            float_replay_tolerances as _float_replay_tolerances,
        )

        eps32 = float(np.finfo(np.float32).eps)

        rtol64, atol64 = _float_replay_tolerances(np.finfo(np.float64))
        # Old blanket rtol=1e-5 was ~4.5e10 fp64 ULPs; the honest band is
        # the legacy fp32 strictness rescaled into fp64's own eps.
        assert rtol64 == pytest.approx(1e-5 * float(np.finfo(np.float64).eps) / eps32)
        assert rtol64 < 1e-6
        assert atol64 == pytest.approx(rtol64 * float(np.finfo(np.float64).tiny))

        rtol32, atol32 = _float_replay_tolerances(np.finfo(np.float32))
        assert rtol32 == pytest.approx(1e-5)
        assert atol32 == pytest.approx(1e-5 * float(np.finfo(np.float32).tiny))

        rtol16, atol16 = _float_replay_tolerances(np.finfo(np.float16))
        assert rtol16 == pytest.approx(4.0 * float(np.finfo(np.float16).eps))
        assert atol16 == pytest.approx(rtol16 * float(np.finfo(np.float16).tiny))

    def test_complex_component_derivation(self) -> None:
        """Complex dtypes derive from their component real finfo."""

        from torchlens.backends._validation_shared import (
            float_replay_tolerances as _float_replay_tolerances,
        )

        rtol_c64, atol_c64 = _float_replay_tolerances(np.finfo(np.complex64))
        rtol_f32, atol_f32 = _float_replay_tolerances(np.finfo(np.float32))
        assert rtol_c64 == rtol_f32
        assert atol_c64 == atol_f32

    @pytest.mark.backend_jax
    def test_live_jax_fp64_corruption_fails(self) -> None:
        """With jax installed, the full oracle refuses fp64 corruption.

        ``jax.config`` is process-global: an unscoped ``update("jax_enable_x64",
        True)`` stuck enabled for the rest of the test session (order-dependent
        under pytest-randomly), promoting every later jax trace's float32
        arrays to float64 and cascading into unrelated capture/validation
        failures across the whole jax suite. Restore the prior value so this
        test's x64 probe never escapes its own scope.
        """

        jax = pytest.importorskip("jax")
        previous_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            import jax.numpy as jnp

            from torchlens.backends.jax.backend import _values_close

            saved = jnp.ones((8,), dtype=jnp.float64)
            assert not _values_close(saved * (1.0 + 1e-6), saved)
            assert _values_close(saved + 0.0, saved)
        finally:
            jax.config.update("jax_enable_x64", previous_x64)


class TestJaxFiniteDifferenceStep:
    """r4 sweep: the fdiff second oracle's flat 1e-4 step sat BELOW fp16 and
    bf16 spacing at unit scale, so the probe never moved the input and the
    check was vacuous. The step is now dtype-derived (cbrt(eps) for
    storage-rounding dtypes) and an unmoved probe fails CLOSED."""

    @pytest.mark.backend_jax
    def test_fp16_probe_actually_moves_the_input(self) -> None:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.ones((4,), dtype=jnp.float16)

        def loss(v):
            return jnp.sum(v * 2.0)

        grad = jax.grad(lambda v: jnp.sum(v * 2.0).astype(jnp.float32))(value)
        # Pre-fix: value +/- 1e-4 rounds back to value in fp16 (spacing at
        # 1.0 is ~9.77e-4), observed reads 0, and the true gradient of 2s
        # FAILED the check. The dtype-derived step must confirm it.
        assert _finite_difference_directional_check(value=value, grad=grad, scalar_loss=loss)

    @pytest.mark.backend_jax
    def test_bf16_probe_actually_moves_the_input(self) -> None:
        pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.ones((4,), dtype=jnp.bfloat16)

        def loss(v):
            return jnp.sum(v * 2.0)

        grad = jnp.full((4,), 2.0, dtype=jnp.bfloat16)
        assert _finite_difference_directional_check(value=value, grad=grad, scalar_loss=loss)

    @pytest.mark.backend_jax
    def test_wrong_gradient_still_fails_on_fp16(self) -> None:
        """The de-vacuumed probe keeps its teeth: a fabricated gradient is
        refused, proving the fp16 path is no longer trivially unsatisfiable
        OR trivially satisfiable."""

        pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.ones((4,), dtype=jnp.float16)

        def loss(v):
            return jnp.sum(v * 2.0)

        wrong = jnp.full((4,), 7.0, dtype=jnp.float16)
        assert not _finite_difference_directional_check(value=value, grad=wrong, scalar_loss=loss)

    @pytest.mark.backend_jax
    def test_unmoved_probe_fails_closed(self) -> None:
        """A magnitude so large the step underflows spacing refuses rather
        than certifying an unprobed gradient."""

        pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.full((4,), 65000.0, dtype=jnp.float16)

        def loss(v):
            return jnp.sum(v.astype(jnp.float32) * 2.0).astype(jnp.float16)

        grad = jnp.full((4,), 2.0, dtype=jnp.float16)
        assert not _finite_difference_directional_check(value=value, grad=grad, scalar_loss=loss)

    @pytest.mark.backend_jax
    def test_small_derivative_sign_flip_is_not_blessed_by_an_atol_floor(self) -> None:
        """F13-A (b): the fixed atol=5e-3 floor blessed ANY tap whose true
        directional derivative sat below 5e-3 -- including a SIGN-FLIPPED
        candidate gradient (routine for post-softmax / normalized taps).
        The derived error model must refuse it (red-capable: pre-fix this
        exact sign flip PASSED)."""

        pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.ones((4,), dtype=jnp.float32)

        def loss(v):
            return jnp.sum(v) * jnp.asarray(1e-4, dtype=jnp.float32)

        flipped = jnp.full((4,), -1e-4, dtype=jnp.float32)
        assert not _finite_difference_directional_check(value=value, grad=flipped, scalar_loss=loss)

    @pytest.mark.backend_jax
    def test_mixed_magnitude_tap_probes_every_element(self) -> None:
        """F13-A (a): the step is scaled per element by max(1, |value|), so a
        tensor spanning magnitudes probes ALL elements. Pre-fix the ABSOLUTE
        scalar step froze the 1e6-magnitude element (fp32 spacing there is
        ~0.0625 >> 1e-2) while others moved, the mixed freeze slipped the
        all(...)-and-all(...) gate, and a CORRECT gradient false-FAILED
        because the frozen element carried the signal (red-capable)."""

        pytest.importorskip("jax")
        import jax.numpy as jnp

        from torchlens.backends.jax.backend import _finite_difference_directional_check

        value = jnp.asarray([1.0e6, 1.0, 1.0, 1.0], dtype=jnp.float32)
        weights = jnp.asarray([1.0, 1e-8, 1e-8, 1e-8], dtype=jnp.float32)

        def loss(v):
            return jnp.sum(v * weights)

        assert _finite_difference_directional_check(value=value, grad=weights, scalar_loss=loss)
