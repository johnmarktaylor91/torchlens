"""Exception types for receptive-field queries and validation."""

from __future__ import annotations

from ..backends._errors import BackendUnsupportedError
from ..errors import TorchLensError


class ReceptiveFieldError(TorchLensError):
    """Base class for receptive-field failures.

    Examples
    --------
    >>> issubclass(ReceptiveFieldError, tl.errors.TorchLensError)
    True
    >>> try:
    ...     tl.receptive_field.register_rf_rule()
    ... except ReceptiveFieldError as err:
    ...     print(type(err).__name__)
    ReceptiveFieldConfigurationError
    """


class ReceptiveFieldConfigurationError(ReceptiveFieldError, ValueError):
    """Raised when a receptive-field call is misconfigured by its caller.

    Covers invalid public arguments and rule-registration mistakes (bad
    tolerances, unknown table levels, malformed rule names, out-of-range
    display options). Subclasses :class:`ValueError` so existing callers
    catching the historical raw errors keep working (SF-07 typed stratum).

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     tl.receptive_field.verify(trace, empirical_adjoint_atol=-1.0)
    ... except ReceptiveFieldConfigurationError as err:
    ...     print(type(err).__name__)
    ReceptiveFieldConfigurationError
    """


class AmbiguousInputError(ReceptiveFieldError, ValueError):
    """Raised when a query has more than one reachable model input.

    Examples
    --------
    >>> class Pair(nn.Module):
    ...     def forward(self, a, b):
    ...         return a + b
    >>> trace = tl.trace(Pair(), (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4)))
    >>> try:
    ...     _ = trace["add_1_1"].receptive_field.size
    ... except AmbiguousInputError as err:
    ...     print(type(err).__name__)
    AmbiguousInputError
    """


class NoInfluencePathError(ReceptiveFieldError, ValueError):
    """Raised when two graph points have no directed influence path.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     trace["conv2d_1_1"].receptive_field.at((4, 4), source=trace["conv2d_2_3"])
    ... except NoInfluencePathError as err:
    ...     print(type(err).__name__)
    NoInfluencePathError
    """


class AmbiguousTargetError(ReceptiveFieldError, ValueError):
    """Raised when a projective query has more than one reachable target.

    A query with no ``target=`` whose other targets are caches (an output role
    with a ``past_key_values``, ``presents`` or ``mems`` segment, or one
    containing ``cache``) answers the primary output instead: the ``logits``
    role, else the first non-cache output. Two real outputs still raise.

    Examples
    --------
    >>> class TwoOut(nn.Module):
    ...     def forward(self, t):
    ...         return t * 2.0, t + 1.0
    >>> trace = tl.trace(TwoOut(), torch.randn(1, 3))
    >>> try:
    ...     _ = trace["input_1"].projective_field.size
    ... except AmbiguousTargetError as err:
    ...     print(type(err).__name__)
    AmbiguousTargetError
    """


class AmbiguousPassError(ReceptiveFieldError, ValueError):
    """Raised when a layer-level query has multiple executed passes.

    Examples
    --------
    >>> class Twice(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.shared = nn.Linear(3, 3)
    ...     def forward(self, t):
    ...         return self.shared(self.shared(t))
    >>> trace = tl.trace(Twice(), torch.randn(2, 3))
    >>> layer = next(layer for layer in trace.layers if layer.num_passes == 2)
    >>> try:
    ...     _ = layer.receptive_field
    ... except AmbiguousPassError as err:
    ...     print(type(err).__name__)
    AmbiguousPassError
    """


class AmbiguousCallError(ReceptiveFieldError, ValueError):
    """Raised when a module-level query has multiple executed calls.

    Examples
    --------
    >>> class Twice(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.shared = nn.Linear(3, 3)
    ...     def forward(self, t):
    ...         return self.shared(self.shared(t))
    >>> trace = tl.trace(Twice(), torch.randn(2, 3))
    >>> try:
    ...     _ = trace.modules["shared"].receptive_field
    ... except AmbiguousCallError as err:
    ...     print(type(err).__name__)
    AmbiguousCallError
    """


class ReceptiveFieldUnavailableError(ReceptiveFieldError, RuntimeError):
    """Raised when a receptive-field result requires unavailable capture state.

    Examples
    --------
    >>> view = tl.trace(model, x)["conv2d_2_3"].receptive_field
    >>> try:
    ...     view.probe_grad((0, 0, 4, 4))
    ... except ReceptiveFieldUnavailableError as err:
    ...     print(type(err).__name__)
    ReceptiveFieldUnavailableError
    """


class ReceptiveFieldValidationError(ReceptiveFieldError, AssertionError):
    """Raised when receptive-field cross-validation detects a violation.

    Examples
    --------
    >>> issubclass(ReceptiveFieldValidationError, tl.errors.ReceptiveFieldError)
    True
    >>> issubclass(ReceptiveFieldValidationError, AssertionError)
    True
    """


__all__ = [
    "AmbiguousCallError",
    "AmbiguousInputError",
    "AmbiguousTargetError",
    "AmbiguousPassError",
    "BackendUnsupportedError",
    "NoInfluencePathError",
    "ReceptiveFieldConfigurationError",
    "ReceptiveFieldError",
    "ReceptiveFieldUnavailableError",
    "ReceptiveFieldValidationError",
]
