import traceback
from typing import Any

import jax.numpy as jnp
from jax import lax

import torchlens as tl
from torchlens.validation.invariants import check_metadata_invariants


def uses_while(params: dict[str, Any], x: Any) -> Any:
    def condition(state: Any) -> Any:
        return state[0] < params["limit"]

    def body(state: Any) -> Any:
        return (state[0] + 1, state[1] + params["step"])

    return lax.while_loop(condition, body, (jnp.asarray(0, dtype=jnp.int32), x))[1]


trace = tl.trace(
    uses_while,
    (
        {"limit": jnp.asarray(3, dtype=jnp.int32), "step": jnp.ones((2, 3))},
        jnp.zeros((2, 3), dtype=jnp.float32),
    ),
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
