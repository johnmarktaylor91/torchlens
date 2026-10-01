"""Authority adapters + the neutral resolver (tvscope B2, memo D8).

An AUTHORITY is the preprocessing metadata shipped by the user's own loader:
a torchvision ``Weights`` preset, a Hugging Face image processor, a timm data
config, an open_clip ``Compose``, or an explicit declaration mapping. Each
adapter normalizes one authority kind into the neutral
:class:`~torchlens.preprocessing._records.DeclaredPreprocessing` schema; the
registry is the small protocol future loaders extend (one probe + one adapt
function per type).

Resolution law (memo D8): explicit authority objects at the public entry;
auto-discovery ONLY where the user's own model object genuinely carries
metadata (a planted torchvision weights handle, a timm ``default_cfg``, an HF
``name_or_path``); an explicit offline policy with fetch disclosure; network
failure yields ``unknown``, never a lower tier's silent guess; torchvision
weights are NEVER inferred from the model class.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._records import DeclaredPreprocessing, Resolution, unknown_resolution

__tl_layer__ = "L5"

#: PIL resample integers -> interpolation names (stable since PIL 9).
_PIL_RESAMPLE_NAMES = {
    0: "nearest",
    1: "lanczos",
    2: "bilinear",
    3: "bicubic",
    4: "box",
    5: "hamming",
}

#: Accepted explicit-declaration keys -> neutral field names.
_EXPLICIT_ALIASES = {
    "modality": "modality",
    "resize_size": "resize_size",
    "resize": "resize_size",
    "size": "resize_size",
    "crop_size": "crop_size",
    "crop": "crop_size",
    "interpolation": "interpolation",
    "antialias": "antialias",
    "channel_order": "channel_order",
    "value_range": "value_range",
    "mean": "mean",
    "image_mean": "mean",
    "std": "std",
    "image_std": "std",
}


def _float_tuple(value: Any) -> Any:
    """Coerce stat-like sequences to float tuples, passing scalars through.

    Parameters
    ----------
    value:
        Mean/std-like value from an adapter.

    Returns
    -------
    Any
        ``tuple[float, ...]`` for sequences, ``float`` for scalars, ``None``
        passed through.
    """

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return float(value)


def _interpolation_name(value: Any) -> str | None:
    """Normalize an interpolation spelling to a lower-case name.

    Parameters
    ----------
    value:
        A torchvision ``InterpolationMode``, PIL resample int, or string.

    Returns
    -------
    str | None
        Lower-case mode name, or ``None`` when undeclared.
    """

    if value is None:
        return None
    if isinstance(value, int):
        return _PIL_RESAMPLE_NAMES.get(value, f"resample_{value}")
    name = getattr(value, "value", value)
    return str(name).lower()


@dataclass(frozen=True)
class AuthorityAdapter:
    """One registered authority kind.

    Attributes
    ----------
    name:
        Stable adapter name (rides the provenance record's config).
    probe:
        Cheap predicate: does this object look like this authority kind?
    adapt:
        Normalizer: authority object -> :class:`Resolution`.
    """

    name: str
    probe: Callable[[Any], bool]
    adapt: Callable[[Any], Resolution]


def _record(
    source: str,
    identifier: str,
    *,
    declared: DeclaredPreprocessing | None,
    description: str,
    transform: Callable[[Any], Any] | None = None,
) -> Resolution:
    """Assemble one EXPLICIT-authority resolution with a provenance record.

    Every builtin adapter mints ``verified=False`` +
    ``resolution_method="explicit"`` here (an explicitly supplied authority
    is authoritative by standing, not by model attachment); the discovery
    tiers re-stamp through :func:`_as_model_metadata`. Adapters needing
    extra provenance disclosures mutate ``record.config`` after assembly.

    Parameters
    ----------
    source:
        Provenance source token (e.g. ``"torchvision_weights"``).
    identifier:
        Authority identifier (weights name, processor path, config id).
    declared:
        Normalized comparable fields, or ``None``.
    description:
        One-line human summary.
    transform:
        The authority's own transform, when it ships one.

    Returns
    -------
    Resolution
        Assembled resolution.
    """

    from torchlens.data_classes.trace import ResolvedPreprocessing

    config: dict[str, Any] = {"resolution_method": "explicit"}
    if declared is not None:
        config.update(declared.to_config())
    else:
        config["declares"] = "nothing"
    return Resolution(
        record=ResolvedPreprocessing(
            source=source,
            identifier=identifier,
            verified=False,
            config=config,
            description=description,
        ),
        declared=declared,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# torchvision Weights presets
# ---------------------------------------------------------------------------


def _probe_torchvision_weights(authority: Any) -> bool:
    """Match torchvision ``WeightsEnum`` members (public ``transforms`` + ``meta``)."""

    return (
        callable(getattr(authority, "transforms", None))
        and isinstance(getattr(authority, "meta", None), Mapping)
        and hasattr(authority, "url")
    )


def _declared_from_torchvision_preset(preset: Any) -> DeclaredPreprocessing | None:
    """Read the public preset attributes into the neutral schema.

    A classification preset exposes ``resize_size`` / ``crop_size`` / ``mean``
    / ``std`` / ``interpolation`` / ``antialias`` publicly. A detection preset
    (e.g. ``ObjectDetection``) exposes NONE of them -- the model normalizes
    internally and a ``[0, 1]`` float batch is the CORRECT input -- so the
    honest return is ``None`` (declares nothing), never a guess.

    Parameters
    ----------
    preset:
        The instance returned by ``weights.transforms()``.

    Returns
    -------
    DeclaredPreprocessing | None
        Declared fields, or ``None`` when the preset declares nothing.
    """

    mean = getattr(preset, "mean", None)
    std = getattr(preset, "std", None)
    resize = getattr(preset, "resize_size", None)
    crop = getattr(preset, "crop_size", None)
    if mean is None and std is None and resize is None and crop is None:
        return None
    return DeclaredPreprocessing(
        resize_size=_squeeze_size(resize),
        crop_size=_squeeze_size(crop),
        interpolation=_interpolation_name(getattr(preset, "interpolation", None)),
        antialias=_coerce_antialias(getattr(preset, "antialias", None)),
        channel_order="rgb",
        value_range=(0.0, 1.0),
        mean=_float_tuple(mean),
        std=_float_tuple(std),
        extras={"preset": type(preset).__name__},
    )


def _squeeze_size(value: Any) -> Any:
    """Collapse one-element size lists to their scalar (torchvision spelling)."""

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        items = tuple(int(item) for item in value)
        # equal (h, w) pairs collapse to the scalar so adapters that spell
        # square sizes differently (int vs pair) stay comparable
        if len(set(items)) == 1:
            return items[0]
        return items
    if value is None:
        return None
    return int(value)


def _coerce_antialias(value: Any) -> bool | str | None:
    """Normalize torchvision antialias spellings (bool / None / ``"warn"``).

    torchvision's classification presets default ``antialias`` to the
    LITERAL STRING ``"warn"`` on torchvision < 0.17ish (its legacy
    transitional default: antialiasing applies to PIL input but not Tensor
    input, with a one-time warning) before later releases default to the
    explicit ``True``. Collapsing that declared sentinel to ``None``
    ("undeclared") used to make a transform's self-audit against its own
    declaration register as ``unknown`` (one undeclared-on-both-sides field
    forces the whole verdict to ``unknown``, memo D4) even though both sides
    agree exactly -- declare it honestly instead of guessing a bool.
    """

    if isinstance(value, bool):
        return value
    if value == "warn":
        return "warn"
    return None


def _adapt_torchvision_weights(authority: Any) -> Resolution:
    """Normalize a torchvision ``Weights`` preset authority."""

    preset = authority.transforms()
    declared = _declared_from_torchvision_preset(preset)
    describes = "declares nothing (model normalizes internally)"
    if declared is not None:
        describes = f"resize={declared.resize_size} crop={declared.crop_size}"
    return _record(
        "torchvision_weights",
        str(authority),
        declared=declared,
        description=f"torchvision preset {authority}: {describes}",
        transform=preset,
    )


# ---------------------------------------------------------------------------
# Hugging Face image processors
# ---------------------------------------------------------------------------


def _probe_hf_processor(authority: Any) -> bool:
    """Match HF image processors (and wrapping processors) by public attrs."""

    if hasattr(authority, "image_processor"):
        return True
    return hasattr(authority, "image_mean") and hasattr(authority, "do_normalize")


def _hf_size_value(size: Any, *keys: str) -> Any:
    """Read one of HF's ``size`` spellings (dict, SizeDict, scalar)."""

    if size is None:
        return None
    if isinstance(size, Mapping):
        mapping_values = [size.get(key) for key in keys]
        if all(value is not None for value in mapping_values):
            values = tuple(int(value) for value in mapping_values if value is not None)
            return values[0] if len(set(values)) == 1 else values
        return None
    if isinstance(size, (int, float)):
        return int(size)
    # transformers 5.x SizeDict and friends: attribute-shaped size holders.
    attr_values = [getattr(size, key, None) for key in keys]
    if all(value is not None for value in attr_values):
        values = tuple(int(value) for value in attr_values if value is not None)
        return values[0] if len(set(values)) == 1 else values
    return None


def _declared_from_hf_processor(processor: Any) -> DeclaredPreprocessing:
    """Read an HF image processor's public config into the neutral schema."""

    size = getattr(processor, "size", None)
    resize = None
    if getattr(processor, "do_resize", False):
        resize = _hf_size_value(size, "shortest_edge")
        if resize is None:
            resize = _hf_size_value(size, "height", "width")
    crop = None
    if getattr(processor, "do_center_crop", False):
        crop = _hf_size_value(getattr(processor, "crop_size", None), "height", "width")
    do_rescale = getattr(processor, "do_rescale", None)
    value_range = (0.0, 1.0) if do_rescale or do_rescale is None else None
    mean = std = None
    if getattr(processor, "do_normalize", False):
        mean = _float_tuple(getattr(processor, "image_mean", None))
        std = _float_tuple(getattr(processor, "image_std", None))
    return DeclaredPreprocessing(
        resize_size=resize,
        crop_size=crop,
        interpolation=_interpolation_name(getattr(processor, "resample", None)),
        antialias=None,
        channel_order="rgb",
        value_range=value_range,
        mean=mean,
        std=std,
        extras={"processor_class": type(processor).__name__},
    )


def _adapt_hf_processor(authority: Any) -> Resolution:
    """Normalize an HF (image) processor object supplied explicitly."""

    image_processor = getattr(authority, "image_processor", authority)
    declared = _declared_from_hf_processor(image_processor)
    identifier = str(getattr(image_processor, "name_or_path", None) or type(authority).__name__)

    def transform(image: Any) -> Any:
        """Apply the processor (batch-native: one call per image list)."""

        return authority(images=image, return_tensors="pt")

    transform._tl_batch_input = True  # type: ignore[attr-defined]
    return _record(
        "hf_image_processor",
        identifier,
        declared=declared,
        description=f"HF image processor {identifier}",
        transform=transform,
    )


# ---------------------------------------------------------------------------
# timm data configs
# ---------------------------------------------------------------------------


def _probe_timm_config(authority: Any) -> bool:
    """Match timm data-config mappings (``input_size`` + ``mean``/``std``)."""

    return isinstance(authority, Mapping) and "input_size" in authority


def _declared_from_timm_config(config: Mapping[str, Any]) -> DeclaredPreprocessing:
    """Normalize a timm data config (resolve_data_config output shape)."""

    input_size = config.get("input_size")
    crop: int | tuple[int, int] | None = None
    if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
        crop = (int(input_size[1]), int(input_size[2]))
        if crop[0] == crop[1]:
            crop = crop[0]
    resize = None
    crop_pct = config.get("crop_pct")
    if crop_pct and isinstance(crop, int):
        resize = int(round(crop / float(crop_pct)))
    return DeclaredPreprocessing(
        resize_size=resize,
        crop_size=crop,
        interpolation=_interpolation_name(config.get("interpolation")),
        antialias=None,
        channel_order="rgb",
        value_range=(0.0, 1.0),
        mean=_float_tuple(config.get("mean")),
        std=_float_tuple(config.get("std")),
        extras={"crop_pct": crop_pct, "crop_mode": config.get("crop_mode")},
    )


def _adapt_timm_config(authority: Mapping[str, Any]) -> Resolution:
    """Normalize an explicit timm data-config mapping.

    The transform is built through ``timm.data.create_transform`` when timm is
    importable; declaration-only resolution otherwise (the fields, not the
    callable, are the authority).
    """

    declared = _declared_from_timm_config(authority)
    transform: Callable[[Any], Any] | None = None
    with contextlib.suppress(Exception):  # timm absent or config not transform-complete
        import timm

        transform = timm.data.create_transform(**dict(authority))
    identifier = str(authority.get("architecture", "timm-data-config"))
    return _record(
        "timm",
        identifier,
        declared=declared,
        description=(
            f"timm data config: input_size={authority.get('input_size')} "
            f"crop_pct={authority.get('crop_pct')}"
        ),
        transform=transform,
    )


# ---------------------------------------------------------------------------
# torchvision-style Compose pipelines (open_clip preprocess et al.)
# ---------------------------------------------------------------------------


def _probe_compose(authority: Any) -> bool:
    """Match Compose-shaped pipelines (an iterable ``transforms`` attribute)."""

    steps = getattr(authority, "transforms", None)
    return isinstance(steps, (list, tuple))


def declared_from_compose(pipeline: Any) -> tuple[DeclaredPreprocessing, list[str]]:
    """Parse a torchvision-style ``Compose`` into declared fields.

    This is the open_clip adapter (its ``preprocess`` is a plain Compose) and
    the audit's applied-side parser for user pipelines.

    Parameters
    ----------
    pipeline:
        Object with a ``transforms`` list of steps.

    Returns
    -------
    tuple[DeclaredPreprocessing, list[str]]
        Declared fields, plus the names of steps the parser could not
        interpret (they make the declaration PARTIAL, never wrong: unparsed
        steps are disclosed and the untouched fields stay ``None``-honest
        only when nothing parsed; callers surface the list).
    """

    fields: dict[str, Any] = {}
    unparsed: list[str] = []
    for step in getattr(pipeline, "transforms", ()):
        name = type(step).__name__
        if name == "Resize":
            fields["resize_size"] = _squeeze_size(getattr(step, "size", None))
            fields["interpolation"] = _interpolation_name(getattr(step, "interpolation", None))
            fields["antialias"] = _coerce_antialias(getattr(step, "antialias", None))
        elif name == "CenterCrop":
            fields["crop_size"] = _squeeze_size(getattr(step, "size", None))
        elif name in {"ToTensor", "PILToTensor"}:
            fields["value_range"] = (0.0, 1.0) if name == "ToTensor" else (0.0, 255.0)
        elif name == "Normalize":
            fields["mean"] = _float_tuple(getattr(step, "mean", None))
            fields["std"] = _float_tuple(getattr(step, "std", None))
        elif name.lower() in {"_convert_to_rgb", "lambda"} or "rgb" in name.lower():
            fields["channel_order"] = "rgb"
        else:
            unparsed.append(name)
    declared = DeclaredPreprocessing(
        extras={"unparsed_steps": list(unparsed)} if unparsed else {},
        **fields,
    )
    return declared, unparsed


def _adapt_compose(authority: Any) -> Resolution:
    """Normalize an explicit Compose pipeline (open_clip ``preprocess``)."""

    declared, unparsed = declared_from_compose(authority)
    resolution = _record(
        "compose_pipeline",
        type(authority).__name__,
        declared=declared,
        description=(
            "parsed transform pipeline"
            + (f" ({len(unparsed)} unparsed steps disclosed)" if unparsed else "")
        ),
        transform=authority,
    )
    if unparsed:
        resolution.record.config["unparsed_steps"] = list(unparsed)
    return resolution


# ---------------------------------------------------------------------------
# explicit declaration mappings
# ---------------------------------------------------------------------------


def _probe_explicit_mapping(authority: Any) -> bool:
    """Match plain declaration mappings (any recognized alias key)."""

    return isinstance(authority, Mapping) and any(key in _EXPLICIT_ALIASES for key in authority)


def _adapt_explicit_mapping(authority: Mapping[str, Any]) -> Resolution:
    """Normalize an explicit user declaration mapping.

    Unrecognized keys are DISCLOSED in extras, never silently dropped and
    never compared.
    """

    fields: dict[str, Any] = {}
    unrecognized: dict[str, Any] = {}
    for key, value in authority.items():
        target = _EXPLICIT_ALIASES.get(key)
        if target is None:
            unrecognized[key] = value
        elif target in {"mean", "std"}:
            fields[target] = _float_tuple(value)
        elif target in {"resize_size", "crop_size"}:
            fields[target] = _squeeze_size(value)
        elif target == "value_range":
            fields[target] = tuple(float(v) for v in value) if value is not None else None
        else:
            fields[target] = value
    declared = DeclaredPreprocessing(
        extras={"unrecognized_keys": sorted(unrecognized)} if unrecognized else {},
        **fields,
    )
    return _record(
        "explicit_declaration",
        "caller-declared",
        declared=declared,
        description="explicit caller-declared preprocessing constants",
        transform=None,
    )


#: Probe order matters: timm configs are Mappings too, so they probe before
#: the generic explicit mapping; Weights enums before processors.
_BUILTIN_ADAPTERS: tuple[AuthorityAdapter, ...] = (
    AuthorityAdapter("torchvision_weights", _probe_torchvision_weights, _adapt_torchvision_weights),
    AuthorityAdapter("hf_image_processor", _probe_hf_processor, _adapt_hf_processor),
    AuthorityAdapter("timm_config", _probe_timm_config, _adapt_timm_config),
    AuthorityAdapter("compose_pipeline", _probe_compose, _adapt_compose),
    AuthorityAdapter("explicit_mapping", _probe_explicit_mapping, _adapt_explicit_mapping),
)

_REGISTERED_ADAPTERS: list[AuthorityAdapter] = []


def register_authority_adapter(
    name: str,
    probe: Callable[[Any], bool],
    adapt: Callable[[Any], Resolution],
) -> None:
    """Register a new authority kind (the small extension protocol).

    Registered adapters probe BEFORE the builtins, most-recent first, so a
    more specific third-party authority can win over the generic mapping
    adapter.

    Parameters
    ----------
    name:
        Stable adapter name.
    probe:
        Cheap predicate over candidate authority objects.
    adapt:
        Normalizer returning a :class:`Resolution`.
    """

    _REGISTERED_ADAPTERS.insert(0, AuthorityAdapter(name, probe, adapt))


def _all_adapters() -> tuple[AuthorityAdapter, ...]:
    """Return the active probe order (registered first, then builtins)."""

    return (*_REGISTERED_ADAPTERS, *_BUILTIN_ADAPTERS)


def _offline_policy(offline: bool | None) -> tuple[bool, str]:
    """Resolve the effective offline policy and its disclosure token.

    Parameters
    ----------
    offline:
        ``None`` respects the HF offline environment switches; ``True``/
        ``False`` are explicit.

    Returns
    -------
    tuple[bool, str]
        ``(effective_offline, policy_token)``.
    """

    if offline is True:
        return True, "explicit_offline"
    if offline is False:
        return False, "explicit_online"
    env_offline = any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    return env_offline, "env_offline" if env_offline else "env_online"


def _discover_from_model(model: Any, offline: bool | None) -> Resolution:
    """Auto-discover an authority from model-attached metadata (memo D8).

    Order: no-network tiers first (a planted torchvision weights handle, a
    timm ``default_cfg``), then the network-gated HF processor fetch. A fetch
    failure yields ``unknown`` with the fetch disclosure -- never a silent
    fall-through to a TorchLens-authored recipe.

    Parameters
    ----------
    model:
        Candidate model object.
    offline:
        Offline policy (see :func:`resolve`).

    Returns
    -------
    Resolution
        Discovered authority or the honest unknown.
    """

    weights = getattr(model, "_torchlens_weights", None)
    if weights is not None and _probe_torchvision_weights(weights):
        resolution = _adapt_torchvision_weights(weights)
        return _as_model_metadata(resolution)
    if isinstance(getattr(model, "default_cfg", None), dict) or hasattr(model, "pretrained_cfg"):
        timm_resolution = _discover_timm(model)
        if timm_resolution is not None:
            return timm_resolution
    return _discover_hf(model, offline)


def _as_model_metadata(resolution: Resolution) -> Resolution:
    """Re-stamp an explicit-adapter resolution as model-attached metadata."""

    from torchlens.data_classes.trace import ResolvedPreprocessing

    record = resolution.record
    config = dict(record.config)
    config["resolution_method"] = "model_metadata"
    return Resolution(
        record=ResolvedPreprocessing(
            source=record.source,
            identifier=record.identifier,
            verified=True,
            config=config,
            description=record.description,
        ),
        declared=resolution.declared,
        transform=resolution.transform,
    )


def _discover_timm(model: Any) -> Resolution | None:
    """Resolve a timm model's own data config (no network)."""

    data_config = None
    with contextlib.suppress(Exception):  # timm absent or model not timm-shaped
        import timm

        data_config = timm.data.resolve_data_config({}, model=model)
    if data_config is None:
        return None
    resolution = _adapt_timm_config(data_config)
    return _as_model_metadata(resolution)


def _discover_hf(model: Any, offline: bool | None) -> Resolution:
    """Resolve an HF model's image processor (network-gated, disclosed)."""

    name_or_path = getattr(model, "name_or_path", None) or getattr(
        getattr(model, "config", None), "name_or_path", None
    )
    if not name_or_path:
        return unknown_resolution("model carries no preprocessing metadata")
    effective_offline, policy = _offline_policy(offline)
    fetch: dict[str, Any] = {"policy": policy, "attempted": False, "error": None}
    if effective_offline and policy == "explicit_offline":
        fetch["note"] = "fetch skipped by explicit offline policy"
        return unknown_resolution(
            f"offline policy forbids fetching the processor for {name_or_path!r}",
            fetch=fetch,
        )
    fetch["attempted"] = True
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(str(name_or_path))
    except Exception as exc:  # network failure -> unknown, NEVER a guess (D8)
        fetch["error"] = f"{type(exc).__name__}: {exc}"
        return unknown_resolution(
            f"processor resolution failed for {name_or_path!r}",
            fetch=fetch,
        )
    resolution = _adapt_hf_processor(processor)
    resolution = _as_model_metadata(resolution)
    resolution.record.config["fetch"] = fetch
    resolution.record.config["upstream"] = str(name_or_path)
    return resolution


def resolve(
    authority: Any = None,
    *,
    model: Any = None,
    offline: bool | None = None,
) -> Resolution:
    """Resolve a preprocessing authority (the neutral public entry; B2).

    Parameters
    ----------
    authority:
        Explicit authority object: torchvision ``Weights`` preset, HF (image)
        processor, timm data config, torchvision/open_clip ``Compose``, or an
        explicit declaration mapping. Explicit authorities never touch the
        network. An opaque callable resolves ``unknown`` (it declares
        nothing) with the callable preserved as the transform.
    model:
        Auto-discovery source, used only when ``authority`` is ``None`` and
        only for metadata the model object genuinely carries (memo D8);
        torchvision weights are never inferred from the model class.
    offline:
        ``None`` (default) respects ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``;
        ``True`` forbids fetches; ``False`` permits them. Only the HF
        discovery tier can fetch, and its attempt is disclosed on the record.

    Returns
    -------
    Resolution
        Authority record + declared fields + transform (when the authority
        ships one). Nothing resolving is the honest ``unknown``, never a
        TorchLens-authored recipe (memo D7/D9).
    """

    if authority is not None:
        for adapter in _all_adapters():
            if adapter.probe(authority):
                return adapter.adapt(authority)
        if callable(authority):
            resolution = unknown_resolution(
                "authority is an opaque callable: it declares no comparable fields"
            )
            return Resolution(
                record=resolution.record,
                declared=None,
                transform=authority,
            )
        from torchlens._errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"preprocessing authority of type {type(authority).__name__} matched "
            "no registered adapter (torchvision Weights preset, HF image "
            "processor, timm data config, Compose pipeline, or declaration "
            "mapping).",
            code="preprocessing_authority_unrecognized",
            remedy=(
                "pass your loader's own preprocessing object (weights preset, "
                "processor, data config, preprocess Compose) or an explicit "
                "declaration mapping; register_authority_adapter() extends the "
                "recognized kinds"
            ),
            received_type=type(authority).__name__,
        )
    if model is not None:
        return _discover_from_model(model, offline)
    return unknown_resolution("no authority object and no model supplied")
