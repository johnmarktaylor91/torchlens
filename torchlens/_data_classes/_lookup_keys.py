"""Shared lookup-key validation helpers for ``Trace`` access paths."""

from typing import TYPE_CHECKING

from .._errors import InvalidArgumentError
from .._utils._display_errors import candidate_keys, lookup_message

if TYPE_CHECKING:
    from .trace import Trace


def _give_user_feedback_about_lookup_key(
    self: "Trace",
    key: int | str,
    mode: str,
) -> None:
    """Raise a contextual error for an invalid user lookup key.

    Parameters
    ----------
    self:
        Model log being queried.
    key:
        Lookup key supplied by the user.
    mode:
        Either ``"get_one_item"`` for single-item access or
        ``"query_multiple"`` for multi-layer queries.
    """
    if isinstance(key, int) and (key >= len(self._layer_list) or key < -len(self._layer_list)):
        raise InvalidArgumentError(
            f"You specified the layer with index {key}, but there are only {len(self._layer_list)} "
            f"layers; please specify an index in the range "
            f"-{len(self._layer_list)} - {len(self._layer_list) - 1}",
            code="op_lookup_index_out_of_range",
            remedy=(
                f"pass an index in the range -{len(self._layer_list)} - {len(self._layer_list) - 1}"
            ),
            key=key,
        )

    if not isinstance(key, str):
        raise InvalidArgumentError(
            _get_lookup_help_str(self, key, mode),
            code="op_lookup_not_found",
            remedy="pass an in-range integer, a layer label, or a module address",
            key=repr(key),
        )

    if ":" in key and hasattr(self, "_module_logs") and key.rsplit(":", 1)[0] in self._module_logs:
        module, call_index = key.rsplit(":", 1)
        module_log = self._module_logs[module]
        module_num_calls = getattr(module_log, "num_calls")
        raise InvalidArgumentError(
            f"You specified module {module} pass {call_index}, but {module} only has "
            f"{module_num_calls} ops; specify a lower number",
            code="op_lookup_pass_out_of_range",
            remedy=f"specify a pass number of at most {module_num_calls}",
            key=key,
        )

    if key in self.layer_labels:
        layer_num_calls = self._layer_num_calls.get(key, 1)
        if layer_num_calls > 1:
            raise InvalidArgumentError(
                f"You specified output of layer {key}, but it has {layer_num_calls} ops; "
                f"please specify e.g. {key}:2 for the second pass of {key}",
                code="op_lookup_pass_required",
                remedy=f"append a pass qualifier such as {key}:2",
                key=key,
            )

    if key.rsplit(":", 1)[0] in self.layer_labels:
        layer_label, call_index = key.rsplit(":", 1)
        layer_num_calls_for_label = self._layer_num_calls.get(layer_label)
        layer_num_calls_msg: int | str = (
            layer_num_calls_for_label if layer_num_calls_for_label is not None else "unknown"
        )
        raise InvalidArgumentError(
            f"You specified layer {layer_label} pass {call_index}, but {layer_label} only has "
            f"{layer_num_calls_msg} ops. Specify a lower number",
            code="op_lookup_pass_out_of_range",
            remedy=f"specify a pass number of at most {layer_num_calls_msg}",
            key=key,
        )

    raise InvalidArgumentError(
        _get_lookup_help_str(self, key, mode),
        code="op_lookup_not_found",
        remedy="pass an in-range integer, a layer label, or a module address",
        key=key,
    )


def _get_lookup_help_str(self: "Trace", layer_label: int | str, mode: str) -> str:
    """Build the standard help text for failed Trace lookups.

    Parameters
    ----------
    self:
        Queried trace.
    layer_label:
        Original key supplied by the user.
    mode:
        ``"get_one_item"`` or ``"query_multiple"``.

    Returns
    -------
    str
        Help text describing valid lookup forms.
    """
    suggestions = self._suggest_layer_labels(str(layer_label))
    if suggestions:
        return _missing_layer_message(self, layer_label, suggestions)

    if mode not in ("get_one_item", "query_multiple"):
        raise ValueError("mode must be either get_one_item or query_multiple")
    remedy = "Use an integer position, a layer label, a module address, or a label substring."
    if mode == "get_one_item":
        remedy += " A substring must identify exactly one layer."
    return lookup_message(f"Layer {layer_label!r} not recognized.", [], remedy)


def _missing_layer_message(trace: "Trace", label: int | str, suggestions: list[str]) -> str:
    """Render a lookup miss through cached display keys."""

    return lookup_message(
        f"Layer {label!r} not found.",
        candidate_keys(suggestions, trace._layer_dict_all_keys),
        "pass an in-range integer, a layer label, or a module address.",
    )
