"""Per-backend conformance rosters (``config_built`` and ``ecosystem`` rows).

One module per backend (``paddle.py``, ...); each exposes ``roster()``. The
modules import their framework only inside row builders.
"""
