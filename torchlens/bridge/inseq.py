"""inseq bridge helpers."""

from __future__ import annotations

from typing import Any


def attribute(
    model_or_id: Any,
    inputs: Any,
    *,
    method: str = "integrated_gradients",
    target_texts: Any | None = None,
    attribution_model: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run an inseq attribution model and normalize the result.

    Parameters
    ----------
    model_or_id:
        Model object or model identifier accepted by ``inseq.load_model``.
    inputs:
        Source text or batch forwarded to ``attribute``.
    method:
        inseq attribution method name.
    target_texts:
        Optional target text or batch.
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
    result = attr_model.attribute(inputs, target_texts=target_texts, **kwargs)
    return {
        "schema": "torchlens.inseq.v1",
        "attributions": result,
        "method": method,
        "model": attr_model,
    }


__all__ = ["attribute"]
