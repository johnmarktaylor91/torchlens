"""Names of torch functions whose outputs depend on random draws.

Shared by the precision report (cross-dtype RNG consumption is not comparable
for these) and the replay engine (an unseeded draw keeps its captured value
rather than re-drawing).
"""

from __future__ import annotations

#: Function names (trailing underscores normalized away) whose outputs depend
#: on random draws; cross-dtype RNG consumption is not comparable for these.
_STOCHASTIC_FUNC_NAMES = frozenset(
    {
        "alpha_dropout",
        "bernoulli",
        "binomial",
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "exponential",
        "feature_alpha_dropout",
        "feature_dropout",
        "geometric",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "rrelu",
        "rrelu_with_noise",
        "uniform",
    }
)
