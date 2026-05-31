"""Super views and accessors for non-Op Trace log families."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from ._accessor_base import SuperAccessor
from ._base import _TENSOR_FIELD_LITERAL, Super, _TensorBearing

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ...data_classes.buffer_log import Buffer
    from ...data_classes.grad_fn_call_log import GradFnCall
    from ...data_classes.grad_fn_log import GradFn
    from ...data_classes.module_log import ModuleCall, Module
    from ...data_classes.param_log import Param


class SuperModule(Super["Module"]):
    """Aligned view of a module address across bundle members."""

    @property
    def parameter_signatures(self) -> dict[str, list[tuple[str, torch.Size]]]:
        """Return sorted parameter names and shapes per bundle member.

        Returns
        -------
        dict[str, list[tuple[str, torch.Size]]]
            Per-member sorted parameter signatures.
        """

        signatures: dict[str, list[tuple[str, torch.Size]]] = {}
        for name, module in self._members.items():
            signatures[name] = sorted(
                (str(param.name), torch.Size(param.shape)) for param in module.params
            )
        return signatures

    @property
    def child_module_addresses(self) -> dict[str, list[str]]:
        """Return child module addresses per bundle member.

        Returns
        -------
        dict[str, list[str]]
            Per-member child module addresses.
        """

        return {
            name: list(getattr(module, "address_children", []))
            for name, module in self._members.items()
        }


class SuperBuffer(Super["Buffer"], _TensorBearing):
    """Aligned view of a buffer address across bundle members."""


class SuperParam(Super["Param"], _TensorBearing):
    """Aligned view of a parameter address across bundle members."""

    @property
    def weight_norm_diff(self) -> dict[str, float]:
        """Return L2 norm of parameter differences from the first member.

        Returns
        -------
        dict[str, float]
            Per-member L2 distance from the first available parameter value.
        """

        weights = self._tensor_dict("out")
        reference = next((tensor for tensor in weights.values() if tensor is not None), None)
        if reference is None:
            return {name: math.nan for name in weights}
        diffs: dict[str, float] = {}
        for name, tensor in weights.items():
            if tensor is None:
                diffs[name] = math.nan
            else:
                diffs[name] = float(torch.linalg.vector_norm(tensor.detach() - reference).item())
        return diffs

    def _get_tensor(self, member: Any, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor | None:
        """Return the parameter value for ``out`` or live parameter grad for ``grad``.

        Parameters
        ----------
        member:
            Param member.
        field:
            Tensor field to collect.

        Returns
        -------
        torch.Tensor | None
            Parameter tensor or gradient when available.
        """

        if field == "grad":
            value = getattr(member, "grad", None) if getattr(member, "has_grad", False) else None
            return value if isinstance(value, torch.Tensor) else None
        param = getattr(member, "_param_ref", None)
        return param.detach() if isinstance(param, torch.Tensor) else None


class SuperGradFn(Super["GradFn"], _TensorBearing):
    """Aligned view of a grad_fn_handle label across bundle members."""

    def _get_tensor(self, member: Any, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor | None:
        """Return the linked forward op tensor field when available.

        Parameters
        ----------
        member:
            GradFn member.
        field:
            Tensor field to collect.

        Returns
        -------
        torch.Tensor | None
            Linked op tensor or gradient when available.
        """

        op = getattr(member, "op", None)
        if op is None:
            return None
        return super()._get_tensor(op, field)


class SuperModuleCall(Super["ModuleCall"]):
    """Aligned view of a module call label across bundle members."""


class SuperGradFnCall(Super["GradFnCall"]):
    """Aligned view of a grad_fn_handle call label across bundle members."""


class SuperModuleAccessor(SuperAccessor["Module", SuperModule]):
    """Dict-like Bundle accessor returning SuperModule objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a module accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperModule)

    def _resolve_in_member(self, trace: Any, label: str) -> Module | None:
        """Resolve ``label`` to a Module within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate module address.

        Returns
        -------
        Module | None
            Matching Module, or ``None`` when unresolved.
        """

        try:
            resolved = trace.modules[label]
        except (KeyError, ValueError):
            return None
        return cast("Module", resolved) if type(resolved).__name__ == "Module" else None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return module labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Module address labels.
        """

        return [str(label) for label in trace.modules.keys()]


class SuperBufferAccessor(SuperAccessor["Buffer", SuperBuffer]):
    """Dict-like Bundle accessor returning SuperBuffer objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a buffer accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperBuffer)

    def _resolve_in_member(self, trace: Any, label: str) -> Buffer | None:
        """Resolve ``label`` to a Buffer within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate buffer address.

        Returns
        -------
        Buffer | None
            Matching Buffer, or ``None`` when unresolved.
        """

        try:
            return cast("Buffer", trace.buffers[label])
        except (KeyError, ValueError):
            return None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return buffer labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Buffer address labels.
        """

        return [str(label) for label in trace.buffers.keys()]


class SuperParamAccessor(SuperAccessor["Param", SuperParam]):
    """Dict-like Bundle accessor returning SuperParam objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a parameter accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperParam)

    def _resolve_in_member(self, trace: Any, label: str) -> Param | None:
        """Resolve ``label`` to a Param within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate parameter address.

        Returns
        -------
        Param | None
            Matching Param, or ``None`` when unresolved.
        """

        try:
            return cast("Param", trace.params[label])
        except (KeyError, ValueError):
            return None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return parameter labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Parameter address labels.
        """

        return [str(label) for label in trace.params.keys()]


class SuperGradFnAccessor(SuperAccessor["GradFn", SuperGradFn]):
    """Dict-like Bundle accessor returning SuperGradFn objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a grad_fn_handle accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperGradFn)

    def _resolve_in_member(self, trace: Any, label: str) -> GradFn | None:
        """Resolve ``label`` to a GradFn within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate grad_fn_handle label.

        Returns
        -------
        GradFn | None
            Matching GradFn, or ``None`` when unresolved.
        """

        try:
            resolved = trace.grad_fns[label]
        except (KeyError, ValueError):
            return None
        return cast("GradFn", resolved) if type(resolved).__name__ == "GradFn" else None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return grad_fn_handle labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Grad-fn labels.
        """

        return [str(label) for label in trace.grad_fns.keys()]


class SuperModuleCallAccessor(SuperAccessor["ModuleCall", SuperModuleCall]):
    """Dict-like Bundle accessor returning SuperModuleCall objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a module-call accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperModuleCall)

    def _resolve_in_member(self, trace: Any, label: str) -> ModuleCall | None:
        """Resolve ``label`` to a ModuleCall within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate module-call label.

        Returns
        -------
        ModuleCall | None
            Matching ModuleCall, or ``None`` when unresolved.
        """

        try:
            resolved = trace.modules[label]
        except (KeyError, ValueError):
            return None
        return cast("ModuleCall", resolved) if type(resolved).__name__ == "ModuleCall" else None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return module-call labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Module-call labels.
        """

        return [
            str(call_label)
            for module in trace.modules
            for call_label in getattr(module, "call_labels", [])
        ]


class SuperGradFnCallAccessor(SuperAccessor["GradFnCall", SuperGradFnCall]):
    """Dict-like Bundle accessor returning SuperGradFnCall objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a grad-fn-call accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperGradFnCall)

    def _resolve_in_member(self, trace: Any, label: str) -> GradFnCall | None:
        """Resolve ``label`` to a GradFnCall within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate grad-fn-call label.

        Returns
        -------
        GradFnCall | None
            Matching GradFnCall, or ``None`` when unresolved.
        """

        try:
            resolved = trace.grad_fns[label]
        except (KeyError, ValueError):
            return None
        return cast("GradFnCall", resolved) if type(resolved).__name__ == "GradFnCall" else None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return grad-fn-call labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Grad-fn-call labels.
        """

        return [
            str(call_label)
            for grad_fn_handle in trace.grad_fns
            for call_label in getattr(grad_fn_handle, "call_labels", [])
        ]


__all__ = [
    "SuperBuffer",
    "SuperBufferAccessor",
    "SuperGradFn",
    "SuperGradFnAccessor",
    "SuperGradFnCall",
    "SuperGradFnCallAccessor",
    "SuperModule",
    "SuperModuleAccessor",
    "SuperModuleCall",
    "SuperModuleCallAccessor",
    "SuperParam",
    "SuperParamAccessor",
]
