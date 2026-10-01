import traceback

import jax
import jax.numpy as jnp

import torchlens as tl
from torchlens.validation.invariants import check_metadata_invariants


@jax.custom_vjp
def custom_square(x):
    return x * x


def fwd(x):
    return custom_square(x), x


def bwd(residual, grad):
    return (2 * residual * grad,)


custom_square.defvjp(fwd, bwd)


def uses_custom_vjp(params, x):
    del params
    return custom_square(x) + 1.0


trace = tl.trace(uses_custom_vjp, ({}, jnp.ones((2, 3), dtype=jnp.float32)), backend="jax")

try:
    check_metadata_invariants(trace)
    print("metadata invariants: OK")
except Exception:
    traceback.print_exc()
