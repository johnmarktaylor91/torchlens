"""pytorch-grad-cam bridge helpers."""

from __future__ import annotations

from typing import Any

from torch import nn

from ._utils import first_input_tensor, resolve_one_site, source_model


def cam(
    log: Any,
    site: Any,
    *,
    inputs: Any | None = None,
    targets: Any | None = None,
    cam_class: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a pytorch-grad-cam method for a TorchLens site.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` with a live source model reference.
    site:
        Module address, module pass label, layer selector, or layer object.
    inputs:
        Optional input tensor. Defaults to the first tensor input saved in ``log``.
    targets:
        Optional pytorch-grad-cam targets.
    cam_class:
        Optional CAM class or factory. Defaults to ``pytorch_grad_cam.GradCAM``.
    **kwargs:
        Additional keyword arguments forwarded to the CAM constructor.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the CAM output and resolved target layer.

    Raises
    ------
    ImportError
        If pytorch-grad-cam is unavailable.
    """

    try:
        import pytorch_grad_cam
    except ImportError as exc:
        raise ImportError(
            "Grad-CAM bridge requires the `gradcam` extra: install torchlens[gradcam]."
        ) from exc

    model = source_model(log)
    target_layer = layer(log, site)
    input_tensor = first_input_tensor(log) if inputs is None else inputs
    factory = getattr(pytorch_grad_cam, "GradCAM") if cam_class is None else cam_class
    cam_runner = factory(model=model, target_layers=[target_layer], **kwargs)
    cam_output = _call_cam(cam_runner, input_tensor=input_tensor, targets=targets)
    return {
        "schema": "torchlens.gradcam.v1",
        "cam": cam_output,
        "target_layers": [target_layer],
        "model": model,
    }


def _call_cam(cam_runner: Any, *, input_tensor: Any, targets: Any | None) -> Any:
    """Call a CAM runner, honoring context-manager implementations.

    Parameters
    ----------
    cam_runner:
        pytorch-grad-cam object.
    input_tensor:
        Input tensor forwarded as ``input_tensor``.
    targets:
        Optional CAM targets.

    Returns
    -------
    Any
        CAM output.
    """

    if hasattr(cam_runner, "__enter__") and hasattr(cam_runner, "__exit__"):
        with cam_runner as entered:
            return entered(input_tensor=input_tensor, targets=targets)
    return cam_runner(input_tensor=input_tensor, targets=targets)


def layer(log: Any, site: Any) -> nn.Module:
    """Resolve a TorchLens site to a pytorch-grad-cam target layer.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.
    site:
        Site or module lookup.

    Returns
    -------
    nn.Module
        Live PyTorch module.
    """

    model = source_model(log)
    modules = dict(model.named_modules())
    if site == "self":
        return model
    if isinstance(site, str):
        address = site.rsplit(":", maxsplit=1)[0]
        if address in modules:
            return modules[address]

    resolved = resolve_one_site(log, site)
    candidates = list(getattr(resolved, "module_passes_exited", ()) or [])
    containing_module = getattr(resolved, "containing_module", None)
    if containing_module is not None:
        candidates.append(str(containing_module))
    for candidate in reversed(candidates):
        address = str(candidate).rsplit(":", maxsplit=1)[0]
        if address in modules:
            return modules[address]
    raise ValueError(f"Could not resolve Grad-CAM layer for site {site!r}.")


__all__ = ["cam", "layer"]
