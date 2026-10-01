import traceback
from typing import Any

import jax.numpy as jnp
from jax import lax

import torchlens as tl
from torchlens.validation.invariants import check_metadata_invariants


def uses_scan(params: dict[str, Any], x: Any) -> Any:
    def body(carry: Any, item: Any) -> tuple[Any, Any]:
        new_carry = carry + item
        return new_carry, new_carry

    return lax.scan(body, x[0], x)[1]


trace = tl.trace(
    uses_scan,
    ({}, jnp.arange(4, dtype=jnp.float32).reshape(4, 1)),
    backend="jax",
)

try:
    check_metadata_invariants(trace)
    print("metadata invariants: OK")
except Exception:
    traceback.print_exc()

print("validate_forward_pass:", trace.validate_forward_pass([]))

for op in trace.layer_list:
    print(op.label, op.site_key, op.annotations.get("jax_source_path"))
