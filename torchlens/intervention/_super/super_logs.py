"""Super views and accessors for non-Op Trace log families."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from ._accessor_base import SuperAccessor
from ._base import _TENSOR_FIELD_LITERAL, Super, _TensorBearing

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ...data_classes.buffer_log import BufferLog
    from ...data_classes.grad_fn_call_log import GradFnCallLog
    from ...data_classes.grad_fn_log import GradFnLog
    from ...data_classes.module_log import ModuleCallLog, ModuleLog
    from ...data_classes.param_log import ParamLog


class SuperModule(Super["ModuleLog"]):
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


class SuperBuffer(Super["BufferLog"], _TensorBearing):
    """Aligned view of a buffer address across bundle members."""


class SuperParam(Super["ParamLog"], _TensorBearing):
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
            ParamLog member.
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


class SuperGradFn(Super["GradFnLog"], _TensorBearing):
    """Aligned view of a grad_fn label across bundle members."""

    def _get_tensor(self, member: Any, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor | None:
        """Return the linked forward op tensor field when available.

        Parameters
        ----------
        member:
            GradFnLog member.
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


class SuperModuleCall(Super["ModuleCallLog"]):
    """Aligned view of a module call label across bundle members."""


class SuperGradFnCall(Super["GradFnCallLog"]):
    """Aligned view of a grad_fn call label across bundle members."""


class SuperModuleAccessor(SuperAccessor["ModuleLog", SuperModule]):
    """Dict-like Bundle accessor returning SuperModule objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a module accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperModule)

    def _resolve_in_member(self, trace: Any, label: str) -> ModuleLog | None:
        """Resolve ``label`` to a ModuleLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate module address.

        Returns
        -------
        ModuleLog | None
            Matching ModuleLog, or ``None`` when unresolved.
        """

        try:
            resolved = trace.modules[label]
        except (KeyError, ValueError):
            return None
        return cast("ModuleLog", resolved) if type(resolved).__name__ == "ModuleLog" else None

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


class SuperBufferAccessor(SuperAccessor["BufferLog", SuperBuffer]):
    """Dict-like Bundle accessor returning SuperBuffer objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a buffer accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperBuffer)

    def _resolve_in_member(self, trace: Any, label: str) -> BufferLog | None:
        """Resolve ``label`` to a BufferLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate buffer address.

        Returns
        -------
        BufferLog | None
            Matching BufferLog, or ``None`` when unresolved.
        """

        try:
            return cast("BufferLog", trace.buffers[label])
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


class SuperParamAccessor(SuperAccessor["ParamLog", SuperParam]):
    """Dict-like Bundle accessor returning SuperParam objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a parameter accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperParam)

    def _resolve_in_member(self, trace: Any, label: str) -> ParamLog | None:
        """Resolve ``label`` to a ParamLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate parameter address.

        Returns
        -------
        ParamLog | None
            Matching ParamLog, or ``None`` when unresolved.
        """

        try:
            return cast("ParamLog", trace.params[label])
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


class SuperGradFnAccessor(SuperAccessor["GradFnLog", SuperGradFn]):
    """Dict-like Bundle accessor returning SuperGradFn objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a grad_fn accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperGradFn)

    def _resolve_in_member(self, trace: Any, label: str) -> GradFnLog | None:
        """Resolve ``label`` to a GradFnLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate grad_fn label.

        Returns
        -------
        GradFnLog | None
            Matching GradFnLog, or ``None`` when unresolved.
        """

        try:
            resolved = trace.grad_fns[label]
        except (KeyError, ValueError):
            return None
        return cast("GradFnLog", resolved) if type(resolved).__name__ == "GradFnLog" else None

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return grad_fn labels from one member trace.

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


class SuperModuleCallAccessor(SuperAccessor["ModuleCallLog", SuperModuleCall]):
    """Dict-like Bundle accessor returning SuperModuleCall objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a module-call accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperModuleCall)

    def _resolve_in_member(self, trace: Any, label: str) -> ModuleCallLog | None:
        """Resolve ``label`` to a ModuleCallLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate module-call label.

        Returns
        -------
        ModuleCallLog | None
            Matching ModuleCallLog, or ``None`` when unresolved.
        """

        try:
            resolved = trace.modules[label]
        except (KeyError, ValueError):
            return None
        return (
            cast("ModuleCallLog", resolved) if type(resolved).__name__ == "ModuleCallLog" else None
        )

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


class SuperGradFnCallAccessor(SuperAccessor["GradFnCallLog", SuperGradFnCall]):
    """Dict-like Bundle accessor returning SuperGradFnCall objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a grad-fn-call accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperGradFnCall)

    def _resolve_in_member(self, trace: Any, label: str) -> GradFnCallLog | None:
        """Resolve ``label`` to a GradFnCallLog within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate grad-fn-call label.

        Returns
        -------
        GradFnCallLog | None
            Matching GradFnCallLog, or ``None`` when unresolved.
        """

        try:
            resolved = trace.grad_fns[label]
        except (KeyError, ValueError):
            return None
        return (
            cast("GradFnCallLog", resolved) if type(resolved).__name__ == "GradFnCallLog" else None
        )

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
            for grad_fn in trace.grad_fns
            for call_label in getattr(grad_fn, "call_labels", [])
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
