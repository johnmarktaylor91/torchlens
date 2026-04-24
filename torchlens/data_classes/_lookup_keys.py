"""Shared lookup-key validation helpers for ``ModelLog`` access paths."""

import random
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .model_log import ModelLog


def _give_user_feedback_about_lookup_key(
    self: "ModelLog",
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
        module, pass_num = key.rsplit(":", 1)
        module_log = self._module_logs[module]
        module_num_passes = getattr(module_log, "num_passes")
        raise ValueError(
            f"You specified module {module} pass {pass_num}, but {module} only has "
            f"{module_num_passes} passes; specify a lower number."
        )

    if key in self.layer_labels_no_pass:
        layer_num_passes = self.layer_num_passes.get(key, 1)
        if layer_num_passes > 1:
            raise ValueError(
                f"You specified output of layer {key}, but it has {layer_num_passes} passes; "
                f"please specify e.g. {key}:2 for the second pass of {key}."
            )

    if key.rsplit(":", 1)[0] in self.layer_labels_no_pass:
        layer_label, pass_num = key.rsplit(":", 1)
        layer_num_passes_for_label = self.layer_num_passes.get(layer_label)
        layer_num_passes_msg: int | str = (
            layer_num_passes_for_label if layer_num_passes_for_label is not None else "unknown"
        )
        raise ValueError(
            f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
            f"{layer_num_passes_msg} passes. Specify a lower number."
        )

    raise ValueError(_get_lookup_help_str(self, key, mode))


def _get_lookup_help_str(self: "ModelLog", layer_label: Union[int, str], mode: str) -> str:
    """Build the standard help text for failed ModelLog lookups.

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
    if not self.layer_labels_w_pass:
        sample_layer1 = "conv2d_1_1:1"
        sample_layer2 = "conv2d_1_1"
    else:
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
    module_addrs = [ml.address for ml in self.modules if ml.address != "self"]
    if len(module_addrs) > 0:
        sample_module1 = random.choice(module_addrs)
        all_pass_labels = [
            pl for ml in self.modules for pl in ml.pass_labels if ml.address != "self"
        ]
        sample_module2 = random.choice(all_pass_labels) if all_pass_labels else "features.4:2"
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
