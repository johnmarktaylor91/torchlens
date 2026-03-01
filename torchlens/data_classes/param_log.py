from typing import Dict, List, Optional, Tuple, Union

import torch

from ..helper_funcs import human_readable_size


class ParamLog:
    """Metadata about a single model parameter."""

    def __init__(
        self,
        address: str,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_params: int,
        fsize: int,
        fsize_nice: str,
        trainable: bool,
        module_address: str,
        module_type: str,
        barcode: str,
        has_optimizer: Optional[bool] = None,
    ):
        self.address = address
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.num_params = num_params
        self.fsize = fsize
        self.fsize_nice = fsize_nice
        self.trainable = trainable
        self.module_address = module_address
        self.module_type = module_type
        self.barcode = barcode
        self.has_optimizer = has_optimizer

        # Reference to the actual nn.Parameter for lazy gradient access
        self._param_ref: Optional[torch.nn.Parameter] = None

        # Populated during postprocessing
        self.num_passes: int = 1
        self.tensor_log_entries: List[str] = []
        self.linked_params: List[str] = []
        self._has_grad: bool = False
        self._grad_shape: Optional[Tuple[int, ...]] = None
        self._grad_dtype: Optional[torch.dtype] = None
        self._grad_fsize: int = 0
        self._grad_fsize_nice: str = human_readable_size(0)

    @property
    def is_quantized(self) -> bool:
        return not self.dtype.is_floating_point

    def _check_param_grad(self):
        """Check if the parameter reference has a gradient and cache the result."""
        if not self._has_grad and self._param_ref is not None and self._param_ref.grad is not None:
            grad = self._param_ref.grad
            self._has_grad = True
            self._grad_shape = tuple(grad.shape)
            self._grad_dtype = grad.dtype
            self._grad_fsize = grad.nelement() * grad.element_size()
            self._grad_fsize_nice = human_readable_size(self._grad_fsize)

    @property
    def has_grad(self) -> bool:
        self._check_param_grad()
        return self._has_grad

    @has_grad.setter
    def has_grad(self, value: bool):
        self._has_grad = value

    @property
    def grad_shape(self) -> Optional[Tuple[int, ...]]:
        self._check_param_grad()
        return self._grad_shape

    @grad_shape.setter
    def grad_shape(self, value):
        self._grad_shape = value

    @property
    def grad_dtype(self) -> Optional[torch.dtype]:
        self._check_param_grad()
        return self._grad_dtype

    @grad_dtype.setter
    def grad_dtype(self, value):
        self._grad_dtype = value

    @property
    def grad_fsize(self) -> int:
        self._check_param_grad()
        return self._grad_fsize

    @grad_fsize.setter
    def grad_fsize(self, value):
        self._grad_fsize = value

    @property
    def grad_fsize_nice(self) -> str:
        self._check_param_grad()
        return self._grad_fsize_nice

    @grad_fsize_nice.setter
    def grad_fsize_nice(self, value):
        self._grad_fsize_nice = value

    def __repr__(self) -> str:
        status = "trainable" if self.trainable else "frozen"
        lines = [
            f"ParamLog: {self.address}",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  size: {self.fsize_nice}",
            f"  {status}",
            f"  has_grad: {self.has_grad}",
            f"  module: {self.module_address} ({self.module_type})",
        ]
        if self.tensor_log_entries:
            lines.append(f"  used by: {', '.join(self.tensor_log_entries)}")
        if self.linked_params:
            lines.append(f"  linked: {', '.join(self.linked_params)}")
        if self.has_optimizer is not None:
            lines.append(f"  has_optimizer: {self.has_optimizer}")
        if self.num_passes > 1:
            lines.append(f"  num_passes: {self.num_passes}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.num_params


class ParamAccessor:
    """Dict-like accessor for ParamLog objects. Supports indexing by address, short name, or ordinal position."""

    def __init__(self, param_logs: Dict[str, "ParamLog"]):
        self._dict = param_logs
        self._list = list(param_logs.values())

    def __getitem__(self, key: Union[int, str]) -> "ParamLog":
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        # Fallback: match by short name (e.g. 'weight', 'bias')
        matches = [pl for pl in self._list if pl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' — use full address")
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._list)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "{}"
        items = []
        for pl in self._list:
            status = "trainable" if pl.trainable else "frozen"
            items.append(f"'{pl.address}': ParamLog {pl.shape} {pl.dtype} {status}")
        inner = ",\n ".join(items)
        return "{" + inner + "}"
