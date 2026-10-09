"""Private lower-bound experiments, deliberately not a Trace replacement.

The selected collector records only explicit tensor module outputs. The tape
measures a minimal per-function journal without claiming replay validation.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from torch.overrides import TorchFunctionMode


class _TierRefusal(RuntimeError):
    """Private typed refusal with a machine-readable reason."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _leaves(value: Any) -> Iterator[torch.Tensor]:
    """Walk ordinary argument containers without retaining the containers."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)


class _SelectedCapture:
    """Compile exact module addresses once and snapshot every invocation.

    Tensor and tensor-first tuple outputs only. This proves the module fast
    path cost, not selector-language completeness or full capture validity.
    """

    def __init__(self, model: nn.Module, selectors: tuple[Any, ...]) -> None:
        self.model = model
        self.targets: list[tuple[str, nn.Module]] = []
        seen: set[str] = set()
        for selector in selectors:
            if getattr(selector, "selector_kind", None) != "module":
                raise _TierRefusal("selector_requires_op_capture")
            address = selector.selector_value
            if not isinstance(address, str) or any(c in address for c in "*?:"):
                raise _TierRefusal("selector_not_exact_module")
            if address in seen:
                continue
            seen.add(address)
            self.targets.append((address, model.get_submodule(address)))
        self.records: list[tuple[str, int, torch.Tensor]] = []
        self.handles: list[Any] = []
        self.counts: dict[str, int] = {}
        self.sink: Callable[[str, int, torch.Tensor], None] | None = None

    def _hook(self, address: str) -> Callable[..., None]:
        def save(module: nn.Module, args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise _TierRefusal("output_structure_unsupported")
            count = self.counts.get(address, 0) + 1
            self.counts[address] = count
            snapshot = tensor.detach().to(device="cpu", copy=True)
            if self.sink is None:
                self.records.append((address, count, snapshot))
            else:
                self.sink(address, count, snapshot)

        return save

    def __enter__(self) -> _SelectedCapture:
        if torch.is_grad_enabled():
            raise _TierRefusal("gradient_capture_requires_full_engine")
        self.records = []
        self.counts = {}
        try:
            for address, module in self.targets:
                if self.model.get_submodule(address) is not module:
                    raise _TierRefusal("module_identity_changed")
                self.handles.append(module.register_forward_hook(self._hook(address)))
        except BaseException:
            self.__exit__(BaseException, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if exc and exc[0] is None and len(self.counts) != len(self.targets):
            raise _TierRefusal("selected_module_not_executed")


class _Tape(TorchFunctionMode):
    """Preallocated minimal function tape; measures cost, not replay fidelity.

    Each record is (function identity, parent producer ids, output references).
    Weak references avoid retaining unselected tensors. Strong references are
    the explicit memory control. Identity checks prevent Python id reuse from
    becoming a false edge. No mutation versions or replay contexts are stored.
    """

    def __init__(self, capacity: int = 100000, strong: bool = False) -> None:
        super().__init__()
        self.records: list[Any] = [None] * capacity
        self.count = 0
        self.strong = strong
        self.producers: dict[int, tuple[Any, int]] = {}
        self.overflow = False

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        parents = []
        for tensor in _leaves((args, kwargs)):
            prior = self.producers.get(id(tensor))
            if prior is not None and prior[0]() is tensor:
                parents.append(prior[1])
        output = func(*args, **kwargs)
        outputs = tuple(_leaves(output))
        if not outputs:
            return output
        if self.count >= len(self.records):
            self.overflow = True
            raise _TierRefusal("tape_capacity_exceeded")
        refs = []
        for tensor in outputs:
            reference = weakref.ref(tensor)
            self.producers[id(tensor)] = (reference, self.count)
            refs.append(tensor if self.strong else reference)
        self.records[self.count] = (func, tuple(parents), tuple(refs))
        self.count += 1
        return output
