"""Keep validation verdicts independent of the caller's warning filters.

Validation re-executes recorded ops (replay, perturbed replay, evidence
probes) and judges them only by the values they return. A warning that such a
re-execution emits is a duplicate of one the forward pass already raised, but
under an ``error`` filter it arrives as an exception, which the replay paths
read as "this op raised for these inputs": a verdict that then depends on the
process's filters. ``replay_warnings_ignored`` runs a re-execution with
warnings ignored so its returned values alone decide the verdict.
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def replay_warnings_ignored() -> Iterator[None]:
    """Ignore warnings emitted by a validation re-execution.

    Yields
    ------
    None
        Control while the re-execution runs; the caller's filters are restored
        on exit.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
