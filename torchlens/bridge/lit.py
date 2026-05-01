"""LIT bridge helpers."""

from __future__ import annotations

from typing import Any


class TorchLensLitModel:
    """Minimal LIT-compatible model wrapper for a TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to expose.
    name:
        Display name for LIT.
    base_model_cls:
        Optional LIT base class captured for contract tests and introspection.
    """

    def __init__(self, log: Any, *, name: str, base_model_cls: Any | None = None) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        log:
            TorchLens ``ModelLog``.
        name:
            Display name.
        base_model_cls:
            Optional imported LIT model base class.
        """

        self.log = log
        self.name = name
        self.base_model_cls = base_model_cls

    def input_spec(self) -> dict[str, Any]:
        """Return a small LIT input spec.

        Returns
        -------
        dict[str, Any]
            LIT-shaped input specification.
        """

        return {"inputs": {"required": False}}

    def output_spec(self) -> dict[str, Any]:
        """Return a LIT output spec.

        Returns
        -------
        dict[str, Any]
            LIT-shaped output specification.
        """

        return {"layer_labels": {"required": False}, "num_layers": {"required": False}}

    def predict(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return log metadata for each LIT request row.

        Parameters
        ----------
        inputs:
            LIT request rows.

        Returns
        -------
        list[dict[str, Any]]
            LIT prediction rows.
        """

        labels = [str(getattr(layer, "layer_label", "layer")) for layer in self.log.layer_list]
        row = {"layer_labels": labels, "num_layers": len(labels), "model_name": self.name}
        return [dict(row) for _ in inputs]


def model(log: Any, *, name: str = "torchlens", **kwargs: Any) -> dict[str, Any]:
    """Build a LIT-compatible model wrapper from a TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    name:
        Display name for the wrapped log.
    **kwargs:
        Additional metadata retained in the contract payload.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the LIT-compatible wrapper.

    Raises
    ------
    ImportError
        If LIT is unavailable.
    """

    try:
        from lit_nlp.api import model as lit_model
    except ImportError as exc:
        raise ImportError("LIT bridge requires the `lit` extra: install torchlens[lit].") from exc

    base_model_cls = getattr(lit_model, "Model", None)
    wrapper = TorchLensLitModel(log, name=name, base_model_cls=base_model_cls)
    return {
        "schema": "torchlens.lit_model.v1",
        "model": wrapper,
        "name": name,
        "metadata": dict(kwargs),
    }


__all__ = ["TorchLensLitModel", "model"]
