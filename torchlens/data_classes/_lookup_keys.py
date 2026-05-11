"""Shared lookup-key validation helpers for ``Trace`` access paths."""

import random
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .model_log import Trace


def _give_user_feedback_about_lookup_key(
    self: "Trace",
    key: Union[int, str],
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
    if (type(key) == int) and (key >= len(self.layer_list) or key < -len(self.layer_list)):
        raise ValueError(
            f"You specified the layer with index {key}, but there are only {len(self.layer_list)} "
            f"layers; please specify an index in the range "
            f"-{len(self.layer_list)} - {len(self.layer_list) - 1}."
        )

    if type(key) != str:
        raise ValueError(_get_lookup_help_str(self, key, mode))

    if hasattr(self, "_module_logs") and key.rsplit(":", 1)[0] in self._module_logs:
        module, call_index = key.rsplit(":", 1)
        module_log = self._module_logs[module]
        module_num_calls = getattr(module_log, "num_calls")
        raise ValueError(
            f"You specified module {module} pass {call_index}, but {module} only has "
            f"{module_num_calls} ops; specify a lower number."
        )

    if key in self.layer_labels:
        layer_num_calls = self.layer_num_calls.get(key, 1)
        if layer_num_calls > 1:
            raise ValueError(
                f"You specified output of layer {key}, but it has {layer_num_calls} ops; "
                f"please specify e.g. {key}:2 for the second pass of {key}."
            )

    if key.rsplit(":", 1)[0] in self.layer_labels:
        layer_label, call_index = key.rsplit(":", 1)
        layer_num_calls_for_label = self.layer_num_calls.get(layer_label)
        layer_num_calls_msg: int | str = (
            layer_num_calls_for_label if layer_num_calls_for_label is not None else "unknown"
        )
        raise ValueError(
            f"You specified layer {layer_label} pass {call_index}, but {layer_label} only has "
            f"{layer_num_calls_msg} ops. Specify a lower number."
        )

    raise ValueError(_get_lookup_help_str(self, key, mode))


def _get_lookup_help_str(self: "Trace", layer_label: Union[int, str], mode: str) -> str:
    """Build the standard help text for failed Trace lookups.

    Parameters
    ----------
    self:
        Model log being queried.
    layer_label:
        Original key supplied by the user.
    mode:
        Either ``"get_one_item"`` or ``"query_multiple"``.

    Returns
    -------
    str
        Help text describing valid lookup forms.
    """
    suggestions = self.find_layers(str(layer_label)) if hasattr(self, "find_layers") else []
    if suggestions:
        suggestion_str = ", ".join(repr(item) for item in suggestions)
        return f"Layer {layer_label!r} not found. Did you mean {suggestion_str}?"

    if not self.op_labels:
        sample_layer1 = "conv2d_1_1:1"
        sample_layer2 = "conv2d_1_1"
    else:
        sample_layer1 = random.choice(self.op_labels)
        sample_layer2 = random.choice(self.layer_labels)
    module_addrs = [ml.address for ml in self.modules if ml.address != "self"]
    if len(module_addrs) > 0:
        sample_module1 = random.choice(module_addrs)
        all_call_labels = [
            pl for ml in self.modules for pl in ml.call_labels if ml.address != "self"
        ]
        sample_module2 = random.choice(all_call_labels) if all_call_labels else "features.4:2"
    else:
        sample_module1 = "features.3"
        sample_module2 = "features.4:2"
    module_str = f"(e.g., {sample_module1}, {sample_module2})"
    if mode == "get_one_item":
        msg = (
            "e.g., 'pool' will grab the maxpool2d or avgpool2d layer, 'maxpool' will grab the "
            "'maxpool2d' layer, etc., but there must be only one such matching layer"
        )
    elif mode == "query_multiple":
        msg = (
            "e.g., 'pool' will grab all maxpool2d or avgpool2d layers, 'maxpool' will grab all "
            "'maxpool2d' layers, etc."
        )
    else:
        raise ValueError("mode must be either get_one_item or query_multiple")
    help_str = (
        f"Layer {layer_label} not recognized; please specify either "
        f"\n\n\t1) an integer giving the ordinal position of the layer "
        f"(e.g. 2 for 3rd layer, -4 for fourth-to-last), "
        f"\n\t2) the layer label (e.g., {sample_layer1}, {sample_layer2}), "
        f"\n\t3) the module address {module_str}"
        f"\n\t4) A substring of any desired layer label ({msg})."
        f"\n\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer, "
        f"and fourth layer overall in the model.)"
    )
    return help_str
