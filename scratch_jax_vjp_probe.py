from collections import Counter

import jax
import jax.numpy as jnp

import torchlens as tl
from torchlens.backends.jax.backend import (
    JAXBackend,
    JaxRegionCapture,
    _jax_equation_ops,
    _jax_op_capture_kind,
    _jax_region_ops,
)


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

backend = JAXBackend()
eq_ok = backend._validate_jax_equations(trace)
region_ok = backend._validate_jax_regions(trace)
print("equations ok:", eq_ok)
print("regions ok:", region_ok)

captures = tuple(getattr(trace, "jax_equation_captures", ()))
equation_ops = _jax_equation_ops(trace)
print("len(captures)=", len(captures), "len(equation_ops)=", len(equation_ops))

capture_kind_counts = Counter(c.kind for c in captures)
op_kind_counts = Counter(_jax_op_capture_kind(op) for op in equation_ops)
print("capture_kind_counts", capture_kind_counts)
print("op_kind_counts", op_kind_counts)

region_captures = tuple(getattr(trace, "jax_region_captures", ()))
region_captures = tuple(c for c in region_captures if isinstance(c, JaxRegionCapture))
region_ops = _jax_region_ops(trace)
print("len(region_captures)=", len(region_captures), "len(region_ops)=", len(region_ops))

for op in trace.layer_list:
    print(op.label, op.layer_type, op.func_name)
