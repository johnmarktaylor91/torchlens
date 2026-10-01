import traceback
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

import torchlens as tl
from torchlens.validation.invariants import check_metadata_invariants


@jax.jit
def user_scan(carry0: Any, xs: Any) -> Any:
    def body(carry: Any, item: Any) -> tuple[Any, Any]:
        carry_next = carry + item
        return carry_next, carry_next

    return lax.scan(body, carry0, xs)[1]


def model(params: dict[str, Any], xs: Any) -> Any:
    return user_scan(params["carry0"], xs)


trace = tl.trace(
    model,
    ({"carry0": jnp.asarray(0.0, dtype=jnp.float32)}, jnp.arange(4, dtype=jnp.float32)),
    backend="jax",
)

try:
    check_metadata_invariants(trace)
    print("metadata invariants: OK")
except Exception:
    traceback.print_exc()

print("validate_forward_pass:", trace.validate_forward_pass([]))
