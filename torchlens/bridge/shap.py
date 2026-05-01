"""SHAP bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import first_input_tensor, source_model


def explain(
    log: Any,
    *,
    background: Any | None = None,
    inputs: Any | None = None,
    explainer_class: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a SHAP explainer against the source model retained by a log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` with a live source model reference.
    background:
        Optional SHAP background data. Defaults to the first saved input tensor.
    inputs:
        Optional inputs to explain. Defaults to ``background``.
    explainer_class:
        Optional explainer class or factory. Defaults to ``shap.DeepExplainer``.
    **kwargs:
        Additional keyword arguments forwarded to the explainer constructor.

    Returns
    -------
    dict[str, Any]
        Contract payload containing SHAP values and the explainer object.

    Raises
    ------
    ImportError
        If SHAP is unavailable.
    """

    try:
        import shap as shap_module
    except ImportError as exc:
        raise ImportError(
            "SHAP bridge requires the `shap` extra: install torchlens[shap]."
        ) from exc

    model = source_model(log)
    background_data = first_input_tensor(log) if background is None else background
    input_data = background_data if inputs is None else inputs
    factory = getattr(shap_module, "DeepExplainer") if explainer_class is None else explainer_class
    explainer = factory(model, background_data, **kwargs)
    values = explainer.shap_values(input_data)
    return {
        "schema": "torchlens.shap.v1",
        "values": values,
        "explainer": explainer,
        "model": model,
    }


__all__ = ["explain"]
