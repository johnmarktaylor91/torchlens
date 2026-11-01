"""TensorFlow / Keras runtime probes used by the default backend specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from packaging.version import InvalidVersion, Version


def _tf_runtime_supported(tf: object, keras: object) -> bool:
    """Return whether installed TensorFlow/Keras versions are supported.

    Parameters
    ----------
    tf:
        Imported TensorFlow module.
    keras:
        Imported Keras module.

    Returns
    -------
    bool
        True for the Keras-3 multi-backend surface (TF >= 2.16 ships it).

    Notes
    -----
    r-b4 R26-6b: feature-probed, not version-parsed. The historical
    ``Version(...) >= 2.16 and >= 3`` gate returned ``False`` on
    ``InvalidVersion``, so an odd/custom build string silently made
    ``backend="tf"`` unavailable. ``keras.ops`` ships only in Keras 3 and the
    multi-backend selector ``keras.backend.backend`` is the exact surface the
    TF preview consumes; a parseable version pair is still honored as a
    fallback signal when the structural probe is inconclusive.
    """

    if hasattr(keras, "ops") and callable(
        getattr(getattr(keras, "backend", None), "backend", None)
    ):
        return True
    try:
        tf_version = Version(str(getattr(tf, "__version__", "0")))
        keras_version = Version(str(getattr(keras, "__version__", "0")))
    except InvalidVersion:
        return False
    return tf_version >= Version("2.16") and keras_version >= Version("3")


def _is_keras_object(value: object) -> bool:
    """Return whether ``value`` appears to come from standalone Keras.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True when the type module is a Keras module.
    """

    return type(value).__module__.split(".", maxsplit=1)[0] == "keras"


def _is_tf_concrete_function(value: object, tf: object) -> bool:
    """Return whether ``value`` is a TensorFlow ``ConcreteFunction``.

    Parameters
    ----------
    value:
        Candidate object.
    tf:
        Imported ``tensorflow`` module.

    Returns
    -------
    bool
        True for TensorFlow concrete functions.
    """

    tf_runtime = cast(Any, tf)
    concrete_function_type = getattr(tf_runtime.types.experimental, "ConcreteFunction", None)
    return bool(concrete_function_type is not None and isinstance(value, concrete_function_type))


def _has_saved_model_signatures(value: object) -> bool:
    """Return whether ``value`` looks like a loaded SavedModel object.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True when a non-empty ``signatures`` mapping is present.
    """

    signatures = getattr(value, "signatures", None)
    return isinstance(signatures, Mapping) and bool(signatures)
