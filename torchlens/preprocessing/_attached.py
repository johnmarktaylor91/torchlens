"""Model-attached authorities and transform recognition (registry gaps).

Three gaps in the authority registry close here:

- torchvision weights discovery: a model built with ``weights=...`` keeps no
  handle to its weights, and the preset cannot be inferred from the model
  class (``IMAGENET1K_V1`` and ``V2`` share a class and differ in resize), so
  :func:`attach_authority` records the caller's own authority on the model
  and discovery reads it first;
- an explicit capture ``transform=`` that IS a recognized authority (the
  object ``weights.transforms()`` returns, an HF image processor, a
  registered adapter) stamps that authority's own record
  (:func:`record_for_transform`), not the opaque ``user_transform``;
- open_clip: models carry their preprocessing config on the visual tower
  (``model.visual.preprocess_cfg``), which :func:`discover_open_clip` reads.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from torchlens.data_classes.trace import ResolvedPreprocessing

    from ._records import Resolution

#: Host attributes written with ``object.__setattr__`` so an ``nn.Module``
#: authority (a torchvision preset) never registers as a submodule.
_ATTACHED_ATTR = "_torchlens_preprocessing_authority"
_WEIGHTS_ATTR = "_torchlens_weights"

ModelT = TypeVar("ModelT")


def attach_authority(model: ModelT, authority: Any) -> ModelT:
    """Record a model's preprocessing authority so discovery finds it.

    ``torchvision.models.resnet18(weights=w)`` keeps no handle to ``w``;
    ``attach_authority(model, w)`` stores it, and every later
    ``torchlens.preprocessing.resolve(model=model)`` -- and so PIL-input
    autorouting and :func:`torchlens.preprocessing.check_card` -- resolves
    the attached authority as model metadata. A torchvision weights preset
    is also exposed to output decoding (its category names).

    Parameters
    ----------
    model:
        Model object to annotate (returned unchanged otherwise).
    authority:
        A recognized authority: torchvision weights preset or the object its
        ``transforms()`` returns, HF image processor, timm data config,
        open_clip preprocess ``Compose``, or a declaration mapping.

    Returns
    -------
    ModelT
        The same model, for chaining.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_authority_unrecognized`` when no adapter recognizes
        the authority, ``preprocessing_attach_opaque`` when it resolves to an
        opaque callable that declares nothing.
    """

    from torchlens._errors import InvalidArgumentError

    from ._authorities import _probe_torchvision_weights, resolve

    resolution = resolve(authority)
    if resolution.preprocessing.source == "unknown":
        raise InvalidArgumentError(
            f"attach_authority: {type(authority).__name__} declares no "
            "preprocessing, so attaching it would record nothing.",
            code="preprocessing_attach_opaque",
            remedy=(
                "attach your loader's own preprocessing object (weights preset, "
                "processor, data config, preprocess Compose) or a declaration mapping"
            ),
            received_type=type(authority).__name__,
        )
    object.__setattr__(model, _ATTACHED_ATTR, authority)
    if _probe_torchvision_weights(authority):
        object.__setattr__(model, _WEIGHTS_ATTR, authority)
    return model


def discover_attached(model: Any) -> Resolution | None:
    """Resolve an authority recorded by :func:`attach_authority`, if any.

    Parameters
    ----------
    model:
        Candidate model object.

    Returns
    -------
    Resolution | None
        The attached authority re-stamped as model metadata, or ``None``.
    """

    from ._authorities import _as_model_metadata, resolve

    authority = getattr(model, "__dict__", {}).get(_ATTACHED_ATTR)
    if authority is None:
        return None
    return _as_model_metadata(resolve(authority))


def discover_open_clip(model: Any) -> Resolution | None:
    """Resolve an open_clip model's own ``visual.preprocess_cfg`` (no network).

    The transform is open_clip's own (``image_transform_v2`` in eval mode),
    parsed through the Compose adapter and re-stamped as model metadata.

    Parameters
    ----------
    model:
        Candidate model object.

    Returns
    -------
    Resolution | None
        The resolution, or ``None`` when the model carries no open_clip
        config or open_clip is not importable.
    """

    config = getattr(getattr(model, "visual", None), "preprocess_cfg", None)
    if not isinstance(config, Mapping) or not {"size", "mean", "std"} <= set(config):
        return None
    try:
        from open_clip.transform import PreprocessCfg, image_transform_v2
    except ImportError:
        return None
    from ._authorities import _adapt_compose, _as_model_metadata

    transform = image_transform_v2(PreprocessCfg(**dict(config)), is_train=False)
    resolution = _as_model_metadata(_adapt_compose(transform))
    record = resolution.preprocessing
    record.source = "open_clip_preprocess_cfg"
    record.identifier = type(model).__name__
    record.description = f"open_clip visual.preprocess_cfg of {type(model).__name__}"
    record.config["preprocess_cfg"] = {
        key: list(value) if isinstance(value, tuple) else value for key, value in config.items()
    }
    return resolution


def probe_torchvision_preset(authority: Any) -> bool:
    """Match the preset objects ``weights.transforms()`` returns."""

    return type(authority).__module__.startswith("torchvision.transforms._presets")


def adapt_torchvision_preset(authority: Any) -> Resolution:
    """Normalize a torchvision preset object (``weights.transforms()``)."""

    from ._authorities import _declared_from_torchvision_preset, _record

    declared = _declared_from_torchvision_preset(authority)
    return _record(
        "torchvision_preset",
        type(authority).__name__,
        declared=declared,
        description=f"torchvision preset {type(authority).__name__}",
        transform=authority,
    )


def record_for_transform(transform: Any) -> ResolvedPreprocessing | None:
    """Return the authority record for a capture ``transform=``, if recognized.

    Only objects an adapter identifies by kind count: registered adapters,
    torchvision presets and HF image processors. A ``Compose`` is NOT
    recognized here -- a hand-written pipeline looks exactly like a
    library's -- and neither is a declaration mapping or weights enum (not
    transforms); those return ``None`` so the capture stamps the honest
    ``user_transform`` record.

    Parameters
    ----------
    transform:
        The transform the capture applied.

    Returns
    -------
    ResolvedPreprocessing | None
        The recognized authority's own record, unchanged, or ``None``.
    """

    from ._authorities import _REGISTERED_ADAPTERS, _probe_hf_processor

    for adapter in _REGISTERED_ADAPTERS:
        if adapter.probe(transform):
            return adapter.adapt(transform).preprocessing
    if probe_torchvision_preset(transform):
        return adapt_torchvision_preset(transform).record
    if _probe_hf_processor(transform):
        from ._authorities import _adapt_hf_processor

        return _adapt_hf_processor(transform).record
    return None


__all__ = [
    "adapt_torchvision_preset",
    "attach_authority",
    "discover_attached",
    "discover_open_clip",
    "probe_torchvision_preset",
    "record_for_transform",
]
