"""inseq bridge helpers."""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError

#: TorchLens attribution spellings whose inseq identifier differs. Every other
#: name is passed through as inseq's own identifier.
_TORCHLENS_TO_INSEQ = {"input_x_grad": "input_x_gradient"}


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
        Attribution method: an inseq identifier
        (``inseq.list_feature_attribution_methods()``) or the TorchLens
        spelling of the same method (``"input_x_grad"`` maps to inseq's
        ``"input_x_gradient"``). Used only when this call loads the model
        (``attribution_model`` is None).
    generated_texts:
        Optional target text or batch, forwarded as inseq's own
        ``generated_texts=`` (forced decoding: the text to attribute instead
        of generating). The attributed text is checked against it.
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
    InvalidArgumentError
        ``bridge_inseq_method_unknown`` when ``method`` names no inseq
        method; ``bridge_inseq_forced_decoding_ignored`` when inseq attributed
        a text other than the requested ``generated_texts`` (inseq reports an
        ignored keyword only through its logger).
    """

    try:
        import inseq as inseq_module
    except ImportError as exc:
        raise ImportError(
            "inseq bridge requires the `inseq` extra: install torchlens[inseq]."
        ) from exc

    if attribution_model is None:
        attr_model = inseq_module.load_model(model_or_id, _inseq_method(inseq_module, method))
    else:
        attr_model = attribution_model
    result = attr_model.attribute(inputs, generated_texts=generated_texts, **kwargs)
    if generated_texts is not None:
        _check_forced_decoding(result, generated_texts)
    return {
        "schema": "torchlens.inseq.v1",
        "attributions": result,
        "method": _method_name(attr_model, method),
        "model": attr_model,
    }


def _inseq_method(inseq_module: Any, method: str) -> str:
    """Map a method name to inseq's identifier, refusing names inseq lacks."""

    known = set(inseq_module.list_feature_attribution_methods())
    resolved = method if method in known else _TORCHLENS_TO_INSEQ.get(method, method)
    if resolved not in known:
        raise InvalidArgumentError(
            f"inseq bridge: {method!r} is not an inseq attribution method.",
            code="bridge_inseq_method_unknown",
            remedy=f"use one of {sorted(known)}",
            method=method,
        )
    return resolved


def _check_forced_decoding(result: Any, generated_texts: Any) -> None:
    """Refuse a result whose attributed text is not the requested text."""

    requested = [generated_texts] if isinstance(generated_texts, str) else list(generated_texts)
    info = getattr(result, "info", None)
    attributed = info.get("generated_texts") if isinstance(info, dict) else None
    if isinstance(attributed, str):
        attributed = [attributed]
    if attributed is None or list(attributed) != requested:
        raise InvalidArgumentError(
            "inseq bridge: forced decoding was not honored; inseq attributed "
            f"{attributed!r} instead of the requested generated_texts {requested!r}.",
            code="bridge_inseq_forced_decoding_ignored",
            remedy=(
                "check that the attribution model supports forced decoding and "
                "that every keyword is spelled as inseq's attribute() names it"
            ),
            requested=requested,
            attributed=attributed,
        )


def _method_name(attr_model: Any, fallback: str) -> str:
    """Return the attribution method the inseq model actually carries.

    A caller-supplied ``attribution_model`` keeps its own method, so the
    payload reports that instead of the unused ``method`` argument.
    """

    name = getattr(getattr(attr_model, "attribution_method", None), "method_name", None)
    return name if isinstance(name, str) else fallback


__all__ = ["attribute"]
