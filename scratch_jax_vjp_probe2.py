import jax
import jax.numpy as jnp

import torchlens as tl
from torchlens.backends.jax.backend import (
    JAXBackend,
    _control_parent_labels,
    _projection_has_boundary_parent,
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
print("regions ok:", backend._validate_jax_regions(trace))

boundary_op = next(op for op in trace.layer_list if op.layer_type == "jax_region")
proj_op = next(op for op in trace.layer_list if op.layer_type == "jax_region_out")
print(
    "boundary label",
    boundary_op.label,
    "layer_label",
    boundary_op.layer_label,
    "num_passes",
    boundary_op.num_passes,
)
print("proj label", proj_op.label, "parents", proj_op.parents)
print("control parents of proj", _control_parent_labels(proj_op))
print("has boundary parent?", _projection_has_boundary_parent(proj_op, boundary_op))
