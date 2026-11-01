"""Per-backend ``BackendProtocol`` objects, kept import-light.

One module per backend (``torch``, ``tf``, ``jax``, ``mlx``, ``paddle``,
``tinygrad``). Each imports only ``_backend_protocol`` at module level and
its framework lazily, so ``register_backend_spec`` can validate the protocol
object at import time without pulling a backend package (whose ``__init__``
imports the data classes that are themselves mid-import then).
"""
