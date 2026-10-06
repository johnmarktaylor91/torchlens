"""pytorch-grad-cam bridge helpers."""

from __future__ import annotations

import inspect
from typing import Any

from torch import nn

from ._utils import first_input_tensor, module_for_site, source_model


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
        TorchLens ``Trace`` with a live source model reference.
    site:
        Module address, module pass label, op label, layer selector, or layer
        object; resolved by :func:`layer`.
    inputs:
        Optional input tensor. Defaults to the first tensor input saved in ``log``.
    targets:
        Optional pytorch-grad-cam targets.
    cam_class:
        Optional CAM class or factory. Defaults to ``pytorch_grad_cam.GradCAM``.
    **kwargs:
        Options split by name: a keyword the CAM constructor names
        (``reshape_transform``, ...) goes to the constructor; any other
        (``aug_smooth``, ``eigen_smooth``, ...) goes to the CAM call.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the CAM output and resolved target layer.

    Raises
    ------
    ImportError
        If pytorch-grad-cam is unavailable.
    TypeError
        If an option is named by neither the constructor nor the CAM call.
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
    init_names = _named_parameters(factory)
    init_kwargs = {key: value for key, value in kwargs.items() if key in init_names}
    call_kwargs = {key: value for key, value in kwargs.items() if key not in init_names}
    cam_runner = factory(model=model, target_layers=[target_layer], **init_kwargs)
    cam_output = _call_cam(
        cam_runner, input_tensor=input_tensor, targets=targets, call_kwargs=call_kwargs
    )
    return {
        "schema": "torchlens.gradcam.v1",
        "cam": cam_output,
        "target_layers": [target_layer],
        "model": model,
    }


def _named_parameters(factory: Any) -> frozenset[str]:
    """Return the keyword names a CAM constructor or factory declares.

    Parameters
    ----------
    factory:
        CAM class or factory callable.

    Returns
    -------
    frozenset[str]
        Declared parameter names other than ``*args``/``**kwargs``; empty when
        the signature cannot be read.
    """

    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return frozenset()
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return frozenset(param.name for param in parameters if param.kind not in variadic)


def _call_cam(
    cam_runner: Any, *, input_tensor: Any, targets: Any | None, call_kwargs: dict[str, Any]
) -> Any:
    """Call a CAM runner, honoring context-manager implementations.

    Parameters
    ----------
    cam_runner:
        pytorch-grad-cam object.
    input_tensor:
        Input tensor forwarded as ``input_tensor``.
    targets:
        Optional CAM targets.
    call_kwargs:
        Call options such as ``aug_smooth`` and ``eigen_smooth``.

    Returns
    -------
    Any
        CAM output.
    """

    if hasattr(cam_runner, "__enter__") and hasattr(cam_runner, "__exit__"):
        with cam_runner as entered:
            return entered(input_tensor=input_tensor, targets=targets, **call_kwargs)
    return cam_runner(input_tensor=input_tensor, targets=targets, **call_kwargs)


def layer(log: Any, site: Any) -> nn.Module:
    """Resolve a TorchLens site to a pytorch-grad-cam target layer.

    A module address returns that module. An op label returns the outermost
    module whose output the op's tensor is (``relu_17_66`` in a resnet18 is
    the output of ``layer4``), refusing when that module runs more than once.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    site:
        Site or module lookup.

    Returns
    -------
    nn.Module
        Live PyTorch module.

    Raises
    ------
    InvalidArgumentError
        If the site maps to no module, to sibling modules, or to a module that
        runs more than once (see :func:`torchlens.bridge._utils.module_for_site`).
    """

    return module_for_site(log, site, bridge="Grad-CAM")


__all__ = ["cam", "layer"]
