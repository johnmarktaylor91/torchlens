"""Pre-flight detection of model wrappers TorchLens cannot instrument.

Some third-party wrappers redirect attribute ASSIGNMENT on the model object
to their wrapped components. TorchLens prepares a model for capture by
assigning instrumented forwards onto its modules; when the wrapper forwards
those assignments elsewhere, the instrumentation never lands and capture
dies partway through with an internal ``AttributeError`` -- an undebuggable
crash for the exact user most likely to hit it (a TransformerLens 3.x
convert whose first torchlens command is tracing their bridged model). This
module turns that crash into a typed refusal that teaches the remedy at the
point of failure (mikit fix F10).

Detection is STRUCTURAL and narrow: only the known offender class is
refused, by MRO scan, with no probe writes to the user's model (the bridge
forwards assignments to the wrapped HF modules, so even a sentinel-attribute
probe would mutate user state).

Internal module; spellings DOCUMENTED-UNSTABLE pending naming ratification.
"""

from __future__ import annotations

from typing import Any

from .errors._base import CompatibilityError

__all__ = ["UninstrumentableModelWrapperError", "check_model_wrapper"]


class UninstrumentableModelWrapperError(CompatibilityError, RuntimeError):
    """Raised when a model wrapper redirects the assignments capture needs.

    Public code may branch on this class (reachable as a
    :class:`torchlens.errors.CompatibilityError` subclass); the message
    carries the remedy.
    """


def _is_transformer_lens_bridge(model: Any) -> bool:
    """Return whether ``model`` is a transformer_lens ``TransformerBridge``.

    Exact structural match over the MRO: a class named ``TransformerBridge``
    whose module lives in the ``transformer_lens`` namespace. Plain
    ``HookedTransformer`` models are NOT matched -- they trace fully and are
    the same-object oracle's own subject.
    """

    for klass in type(model).__mro__:
        if klass.__name__ != "TransformerBridge":
            continue
        module = getattr(klass, "__module__", "") or ""
        if module == "transformer_lens" or module.startswith("transformer_lens."):
            return True
    return False


def check_model_wrapper(model: Any) -> None:
    """Refuse capture of wrapper objects instrumentation cannot land on.

    Parameters
    ----------
    model:
        Model about to be captured.

    Raises
    ------
    UninstrumentableModelWrapperError
        When the model is a known assignment-redirecting wrapper.
    """

    if not _is_transformer_lens_bridge(model):
        return
    raise UninstrumentableModelWrapperError(
        "torchlens cannot capture a transformer_lens TransformerBridge: the bridge "
        "redirects attribute assignment to its wrapped components, so TorchLens's "
        "forward instrumentation never lands and capture would die with an internal "
        "AttributeError.\n"
        "Trace a PRISTINE copy of the underlying model instead:\n"
        "  - reload the HF model fresh (AutoModelForCausalLM.from_pretrained(...)) and "
        "trace that, or\n"
        "  - trace a transformer_lens HookedTransformer directly (it captures fully).\n"
        "Note: transformer_lens's bridge/boot machinery MUTATES the HF model it wraps "
        "IN PLACE, so a model that has already passed through the bridge should be "
        "reloaded fresh, not reused. See docs/migration/from_transformerlens.md.",
        code="model_wrapper_uninstrumentable",
        remedy=("Trace a pristine reloaded HF model or a HookedTransformer instead of the bridge"),
    )
