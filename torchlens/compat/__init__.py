"""Compatibility adapter namespace reserved for TorchLens 2.0."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from .torchextractor import Extractor

_COMPAT_MODULES = {"lovely", "torchshow", "torchextractor"}


def __getattr__(name: str) -> ModuleType:
    """Import compat submodules lazily.

    Parameters
    ----------
    name:
        Compat module name.

    Returns
    -------
    ModuleType
        Imported compat module.

    Raises
    ------
    AttributeError
        If ``name`` is not a known compat submodule.
    """

    if name not in _COMPAT_MODULES:
        raise AttributeError(f"module 'torchlens.compat' has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def from_huggingface(
    model_id: str,
    *,
    local_files_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a Hugging Face Transformers model with offline-first defaults.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier or local model path.
    local_files_only:
        Whether to restrict loading to the local cache.
    **kwargs:
        Additional keyword arguments forwarded to ``AutoModel.from_pretrained``.

    Returns
    -------
    Any
        Loaded Transformers model.

    Raises
    ------
    ImportError
        If Transformers is unavailable.
    """

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Hugging Face loading requires the `hf` extra: install torchlens[hf]."
        ) from exc

    return AutoModel.from_pretrained(model_id, local_files_only=local_files_only, **kwargs)


def from_timm(model_name: str, *, pretrained: bool = False, **kwargs: Any) -> Any:
    """Load a timm model through TorchLens' HF compatibility extra.

    Parameters
    ----------
    model_name:
        timm model name.
    pretrained:
        Whether timm should load pretrained weights.
    **kwargs:
        Additional keyword arguments forwarded to ``timm.create_model``.

    Returns
    -------
    Any
        Loaded timm model.

    Raises
    ------
    ImportError
        If timm is unavailable.
    """

    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm loading requires the `hf` extra: install torchlens[hf].") from exc

    return timm.create_model(model_name, pretrained=pretrained, **kwargs)


def from_torchextractor(model: Any, layers: Any | None = None) -> Extractor:
    """Create a TorchLens extractor from torchextractor-style inputs.

    Parameters
    ----------
    model:
        PyTorch model or torchextractor extractor-like object.
    layers:
        Optional layers to extract. When omitted, ``model.layers`` or
        ``model.layer_names`` is used.

    Returns
    -------
    Extractor
        TorchLens-backed extractor facade.

    Raises
    ------
    ImportError
        If torchextractor is unavailable.
    ValueError
        If no layer list can be resolved.
    """

    try:
        import torchextractor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchextractor migration helpers require the `compat-shims` extra: "
            "install torchlens[compat-shims]."
        ) from exc

    source_model = getattr(model, "model", model)
    resolved_layers = layers
    if resolved_layers is None:
        resolved_layers = getattr(model, "layers", None)
    if resolved_layers is None:
        resolved_layers = getattr(model, "layer_names", None)
    if resolved_layers is None:
        raise ValueError(
            "from_torchextractor requires explicit layers or an extractor-like object."
        )
    return Extractor(source_model, resolved_layers)


def from_fx(graph_module: Any, layers: Any | None = None) -> dict[str, Any]:
    """Describe a torch.fx graph module for migration to TorchLens extraction.

    Parameters
    ----------
    graph_module:
        ``torch.fx.GraphModule`` or graph-module-like object.
    layers:
        Optional explicit layer lookups. Defaults to FX node names.

    Returns
    -------
    dict[str, Any]
        Contract payload with the graph module and suggested TorchLens layers.
    """

    graph = getattr(graph_module, "graph", None)
    node_names = [node.name for node in getattr(graph, "nodes", [])] if graph is not None else []
    return {
        "schema": "torchlens.fx_migration.v1",
        "model": graph_module,
        "layers": list(layers) if layers is not None else node_names,
    }


def from_ilg(model: Any, return_layers: dict[str, str] | None = None) -> Extractor:
    """Create a TorchLens extractor from torchvision IntermediateLayerGetter inputs.

    Parameters
    ----------
    model:
        Source model or ``IntermediateLayerGetter``-like object.
    return_layers:
        Optional torchvision ``return_layers`` mapping.

    Returns
    -------
    Extractor
        TorchLens-backed extractor facade.

    Raises
    ------
    ImportError
        If torchvision is unavailable.
    ValueError
        If no return-layer mapping can be resolved.
    """

    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_ilg requires torchvision: install torchlens[vision-shims]."
        ) from exc

    source_model = getattr(model, "model", model)
    resolved_layers = return_layers
    if resolved_layers is None:
        resolved_layers = getattr(model, "return_layers", None)
    if resolved_layers is None:
        raise ValueError("from_ilg requires a return_layers mapping.")
    torchlens_layers = {
        output_name: module_name for module_name, output_name in resolved_layers.items()
    }
    return Extractor(source_model, torchlens_layers)


def from_sentence_transformers(model: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """Describe a Sentence Transformers model for TorchLens migration.

    Parameters
    ----------
    model:
        Sentence Transformers model or model identifier.
    prompt:
        Optional prompt name retained for migration notes.

    Returns
    -------
    dict[str, Any]
        Contract payload with the model and suggested module layer names.

    Raises
    ------
    ImportError
        If sentence-transformers is unavailable.
    """

    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Sentence Transformers migration helpers require the `compat-shims` extra: "
            "install torchlens[compat-shims]."
        ) from exc

    modules = getattr(model, "modules", None)
    layer_names = (
        [name for name, _module in model.named_modules()] if hasattr(model, "named_modules") else []
    )
    return {
        "schema": "torchlens.sentence_transformers_migration.v1",
        "model": model,
        "prompt": prompt,
        "modules": modules,
        "layers": layer_names,
    }


__all__ = [
    "Extractor",
    "from_fx",
    "from_huggingface",
    "from_ilg",
    "from_sentence_transformers",
    "from_timm",
    "from_torchextractor",
    "lovely",
    "torchshow",
]
