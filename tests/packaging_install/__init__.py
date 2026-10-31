"""Clean-environment wheel and sdist install tests (slow; packaging marker).

Not named ``tests/packaging``: the tests directory is on ``sys.path``, so a
``packaging`` package here would shadow the ``packaging`` distribution torchlens
imports.
"""
