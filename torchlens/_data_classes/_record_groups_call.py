"""The call group of an ``Op`` or ``Layer``: ``op.call.config``, ``layer.call.arg_names``, ...

How the record's function was called: argument counts, names and templates,
saved argument values, configuration, RNG and autocast state, and the
function-transform boundary facts. Stored members read the record's private
storage attribute (``_call_<leaf>``, same persisted column as before).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import OP_GROUP_VIEWS
from ._layer_views import layer_parent_arg_positions
from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpCall(_RecordGroup):
    """How one Op's function was called.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].call.num_pos_args, int)
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def arg_names(self) -> Any:
        """Names of the arguments in this Op's function call."""

        return self._record._call_arg_names

    @property
    def args_template(self) -> Any:
        """Structural template of this Op's positional arguments, used to rebuild the call."""

        return self._record._call_args_template

    @property
    def autocast_state(self) -> Any:
        """Autocast state in effect when this Op's function was called."""

        return self._record._call_autocast_state

    @property
    def call_id(self) -> Any:
        """Identifier of the function call that produced this Op; Ops from one call share it."""

        return self._record._call_call_id

    @property
    def config(self) -> Any:
        """Dict of lightweight hyperparameters captured from the call, such as `kernel_size` and
        `stride`.
        """

        return self._record._call_config

    @property
    def dropped_edge_tensor_args(self) -> Any:
        """Argument positions, such as `"arg1"`, whose tensor inputs did not become graph edges at
        capture.
        """

        return self._record._call_dropped_edge_tensor_args

    @property
    def func_id(self) -> Any:
        """Portable key identifying the captured function this Op ran: its namespace, qualified name
        and dispatch kind.
        """

        return self._record._call_func_id

    @property
    def func_transform_chain(self) -> Any:
        """Tuple of transform tags applied to the function this Op ran."""

        return self._record._call_func_transform_chain

    @property
    def func_transform_config(self) -> Any:
        """Configuration of the function transform for this transform-boundary Op."""

        return self._record._call_func_transform_config

    @property
    def func_transform_fn_name(self) -> Any:
        """Name of the function wrapped by the transform at this boundary Op."""

        return self._record._call_func_transform_fn_name

    @property
    def func_transform_fn_qualname(self) -> Any:
        """Qualified name of the function wrapped by the transform at this boundary Op."""

        return self._record._call_func_transform_fn_qualname

    @property
    def func_transform_fn_source(self) -> Any:
        """Source location of the function wrapped by the transform at this boundary Op."""

        return self._record._call_func_transform_fn_source

    @property
    def func_transform_kind(self) -> Any:
        """Kind of function transform at this boundary Op, such as `vmap` or `grad`, or `None`."""

        return self._record._call_func_transform_kind

    @property
    def has_saved_args(self) -> Any:
        """Whether this Op's argument values were saved."""

        return self._record._call_has_saved_args

    @property
    def injection_provenance(self) -> Any:
        """Record of which intervention injected this Op and where, or `None` for ordinary model
        Ops.
        """

        return self._record._call_injection_provenance

    @property
    def is_func_transform(self) -> Any:
        """Whether this Op is a function-transform boundary node."""

        return self._record._call_is_func_transform

    @property
    def is_internally_initialized(self) -> Any:
        """Alias of `is_internal_source`: whether this Op was created inside the model rather than
        fed from an input.
        """

        return self._record._call_is_internally_initialized

    @property
    def kwargs_template(self) -> Any:
        """Structural template of this Op's keyword arguments, used to rebuild the call."""

        return self._record._call_kwargs_template

    @property
    def non_tensor_args(self) -> Any:
        """Flat list of all non-tensor arguments of this Op's function call."""

        return self._record._call_non_tensor_args

    @property
    def non_tensor_kwargs(self) -> Any:
        """Non-tensor keyword arguments of this Op's function call."""

        return self._record._call_non_tensor_kwargs

    @property
    def non_tensor_pos_args(self) -> Any:
        """Non-tensor positional arguments of this Op's function call."""

        return self._record._call_non_tensor_pos_args

    @property
    def num_args_total(self) -> Any:
        """Total number of arguments in this Op's function call."""

        return self._record._call_num_args_total

    @property
    def num_kwargs(self) -> Any:
        """Number of keyword arguments in this Op's function call."""

        return self._record._call_num_kwargs

    @property
    def num_pos_args(self) -> Any:
        """Number of positional arguments in this Op's function call."""

        return self._record._call_num_pos_args

    @property
    def parent_arg_positions(self) -> Any:
        """Mapping from call argument positions, positional and keyword, to the parent Op label used
        there.
        """

        return self._record._call_parent_arg_positions

    @property
    def rng_states(self) -> Any:
        """RNG state snapshots recorded for this Op's function call."""

        return self._record._call_rng_states

    @property
    def saved_args(self) -> Any:
        """Positional argument values saved for this Op's call, or `None` when not saved."""

        return self._record._call_saved_args

    @property
    def saved_kwargs(self) -> Any:
        """Keyword argument values saved for this Op's call, or `None` when not saved."""

        return self._record._call_saved_kwargs

    @property
    def tl_authored_root(self) -> Any:
        """Whether this Op is the root of a TorchLens-authored call rather than a model call."""

        return self._record._call_tl_authored_root

    @property
    def unattributed_tensor_args(self) -> Any:
        """Tensor arguments that could not be attributed to a parent Op."""

        return self._record._call_unattributed_tensor_args


class LayerCall(_LayerGroup):
    """How one Layer's function was called.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].call.num_pos_args, int)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "call"

    @property
    def arg_names(self) -> Any:
        """Parameter names from the signature of the function this layer calls."""

        return self._record._call_arg_names

    @property
    def config(self) -> Any:
        """Non-tensor configuration of the function call, stored as a dict."""

        return self._record._call_config

    @property
    def num_args_total(self) -> Any:
        """Total number of positional and keyword arguments passed to this layer's function."""

        return self._record._call_num_args_total

    @property
    def num_kwargs(self) -> Any:
        """Number of keyword arguments passed to this layer's function."""

        return self._record._call_num_kwargs

    @property
    def num_pos_args(self) -> Any:
        """Number of positional arguments passed to this layer's function."""

        return self._record._call_num_pos_args

    @property
    def parent_arg_positions(self) -> dict[str, dict[Any, str]]:
        """Map from argument kind and position to the parent layer label, merged across ops."""

        return layer_parent_arg_positions(self._record)

    @property
    def rng_states(self) -> Any:
        """RNG states captured for this layer's function call, for single-pass layers."""

        return self._record._single_pass_or_error("func_rng_states")

    @property
    def saved_args(self) -> Any:
        """Positional arguments captured for this layer's function call, for single-pass layers."""

        return self._record._single_pass_or_error("saved_args")

    @property
    def saved_kwargs(self) -> Any:
        """Keyword arguments captured for this layer's function call, for single-pass layers."""

        return self._record._single_pass_or_error("saved_kwargs")


OP_GROUP_VIEWS["call"] = OpCall

__all__ = ["LayerCall", "OpCall"]
