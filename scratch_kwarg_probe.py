import sys

backend = sys.argv[1] if len(sys.argv) > 1 else "tinygrad"

import torchlens as tl  # noqa: E402

if backend == "tinygrad":
    from tinygrad import Tensor

    def model(x):
        return (x + 1.0).relu()

    x = Tensor([1.0, 2.0])
elif backend == "jax":
    import jax.numpy as jnp

    def model(params, x):
        del params
        return jnp.tanh(x)

    x = ({}, jnp.ones((2, 3), dtype=jnp.float32))
else:
    raise SystemExit(f"unknown backend {backend}")

for kwargs in ({"layers_to_save": ["relu"]}, {"save_grads": True}):
    try:
        tl.trace(model, x, backend=backend, **kwargs)
        print(backend, kwargs, "-> NO EXCEPTION")
    except Exception as exc:
        print(backend, kwargs, "->", type(exc).__name__, str(exc)[:200])
