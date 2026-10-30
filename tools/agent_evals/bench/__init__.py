"""Agent benchmark: coding agents reimplement interpretability methods, scored by oracles.

Each task directory under ``oracles/`` holds a library-neutral spec, a plain-PyTorch reference, a
deterministic grader and held-out inputs. ``registry`` loads them, ``arms`` and ``guards`` build
the per-arm environments, ``runner`` drives one subject-agent run under the caps, ``telemetry``
turns its transcript and code into one record, and ``scheduler`` orders the runs. See
``README.md`` for the task contract and the command line.
"""
