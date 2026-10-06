"""SHAP bridge helpers."""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError
from ._utils import first_input_tensor, source_model


def explain(
    log: Any,
    *,
    background: Any,
    inputs: Any | None = None,
    explainer_class: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a SHAP explainer against the source model retained by a log.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` with a live source model reference.
    background:
        SHAP background (reference) data, required. SHAP values are each
        input's contribution relative to the background's expected output, so
        explaining an input against itself gives all zeros; there is no
        default.
    inputs:
        Optional inputs to explain. Defaults to the first tensor input saved
        in ``log``.
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
    TypeError
        If ``background`` is not passed.
    InvalidArgumentError
        ``bridge_shap_background_missing`` when ``background`` is None.
    """

    if background is None:
        raise InvalidArgumentError(
            "tl.bridge.shap.explain got background=None; SHAP values are relative to a "
            "background's expected output, and there is no default",
            code="bridge_shap_background_missing",
            remedy="pass background= a batch of reference inputs shaped like the model input, "
            "for example a few dataset samples or torch.zeros_like(x)",
        )
    try:
        import shap as shap_module
    except ImportError as exc:
        raise ImportError(
            "SHAP bridge requires the `shap` extra: install torchlens[shap]."
        ) from exc

    model = source_model(log)
    input_data = first_input_tensor(log) if inputs is None else inputs
    factory = getattr(shap_module, "DeepExplainer") if explainer_class is None else explainer_class
    explainer = factory(model, background, **kwargs)
    values = explainer.shap_values(input_data)
    return {
        "schema": "torchlens.shap.v1",
        "values": values,
        "explainer": explainer,
        "model": model,
    }


__all__ = ["explain"]
