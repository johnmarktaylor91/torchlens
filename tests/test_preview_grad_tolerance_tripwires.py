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
* ``mlx/_grad_oracle.py::_mlx_values_close`` (MLX GradOptions divergence refusal).

The comparison cores are numpy-level, so everything but the jax live check
runs without the optional backend runtimes installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from torchlens.backends.mlx._grad_oracle import _mlx_values_close
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

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/mlx/_grad_oracle.py, backends/paddle/backend.py, backends/paddle/validation.py ....
    def test_fp16_one_ulp_rounding_passes(self, site: str, close: Any) -> None:
        """A one-ULP fp16 storage-rounding difference is agreement, not corruption.

        The old dtype-blind pair (rtol 1e-5, ~1/100 of one fp16 ULP)
        false-failed every non-bitwise fp16 agreement.
        """

        saved = np.ones((4,), dtype=np.float16)
        replay = np.nextafter(saved, np.float16(2.0), dtype=np.float16)
        assert close(replay, saved)

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/mlx/_grad_oracle.py, backends/paddle/backend.py, backends/tf/derived_grads.py ....
    def test_shape_mismatch_fails(self, site: str, close: Any) -> None:
        """Shape disagreement is corruption, never coerced."""

        saved = np.ones((4,), dtype=np.float32)
        assert not close(np.ones((5,), dtype=np.float32), saved)


class TestNonFloatExactness:
    """Integer/bool payloads keep exact comparison at every fixed site."""

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/tf/validation.py.
    def test_tf_validation_integer_exact(self) -> None:
        ints = np.arange(4, dtype=np.int64)
        assert tf_payloads_close(ints.copy(), ints)
        assert not tf_payloads_close(ints + 1, ints)

    # Kept by shrink-r4 (per-id subset rule): sole coverage of backends/mlx/_grad_oracle.py.
    def test_mlx_backend_bool_exact(self) -> None:
        bools = np.array([True, False])
        assert _mlx_values_close(bools.copy(), bools)
        assert not _mlx_values_close(~bools, bools)


class TestJaxToleranceDerivation:
    """The JAX oracle's band derivation must be per-dtype ULP-honest."""

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
