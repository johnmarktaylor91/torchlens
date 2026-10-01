import sys

import torchlens as tl

backend = sys.argv[1] if len(sys.argv) > 1 else "tinygrad"

if backend == "tinygrad":
    from tinygrad import Tensor

    def model(x):
        return (x + 1.0).relu()

    args: object = Tensor([1.0, 2.0])
elif backend == "jax":
    import jax.numpy as jnp

    def model(params, x):
        del params
        return jnp.tanh(x)

    args = ({}, jnp.ones((2, 3), dtype=jnp.float32))
else:
    raise SystemExit(f"unknown backend {backend}")

field_names = frozenset(tl.options.CaptureOptions().as_dict())
print("layers_to_save in fields:", "layers_to_save" in field_names)
print("save_grads in fields:", "save_grads" in field_names)

for capture_kwargs in ({"layers_to_save": ["relu"]}, {"save_grads": True}):
    capture = tl.options.CaptureOptions(**capture_kwargs)
    try:
        tl.trace(model, args, backend=backend, capture=capture)
        print(backend, capture_kwargs, "-> NO EXCEPTION")
    except Exception as exc:
        print(backend, capture_kwargs, "->", type(exc).__name__, str(exc)[:200])
