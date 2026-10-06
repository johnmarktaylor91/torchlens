"""inseq bridge helpers."""

from __future__ import annotations

from typing import Any


def attribute(
    model_or_id: Any,
    inputs: Any,
    *,
    method: str = "integrated_gradients",
    generated_texts: Any | None = None,
    attribution_model: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an inseq attribution model and normalize the result.

    Parameters
    ----------
    model_or_id:
        Model object or model identifier accepted by ``inseq.load_model``.
    inputs:
        Source text or batch forwarded as inseq's ``input_texts``.
    method:
        inseq attribution method name, one of
        ``inseq.list_feature_attribution_methods()``; used only when this
        call loads the model (``attribution_model`` is None).
    generated_texts:
        Optional target text or batch, forwarded as inseq's own
        ``generated_texts=`` (the text to attribute instead of generating).
    attribution_model:
        Optional pre-built inseq attribution model.
    **kwargs:
        Additional keyword arguments forwarded to ``attribute``.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the downstream attribution object.

    Raises
    ------
    ImportError
        If inseq is unavailable.
    """

    try:
        import inseq as inseq_module
    except ImportError as exc:
        raise ImportError(
            "inseq bridge requires the `inseq` extra: install torchlens[inseq]."
        ) from exc

    attr_model = (
        inseq_module.load_model(model_or_id, method)
        if attribution_model is None
        else attribution_model
    )
    result = attr_model.attribute(inputs, generated_texts=generated_texts, **kwargs)
    return {
        "schema": "torchlens.inseq.v1",
        "attributions": result,
        "method": _method_name(attr_model, method),
        "model": attr_model,
    }


def _method_name(attr_model: Any, fallback: str) -> str:
    """Return the attribution method the inseq model actually carries.

    A caller-supplied ``attribution_model`` keeps its own method, so the
    payload reports that instead of the unused ``method`` argument.
    """

    name = getattr(getattr(attr_model, "attribution_method", None), "method_name", None)
    return name if isinstance(name, str) else fallback


__all__ = ["attribute"]
