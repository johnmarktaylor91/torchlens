"""Private debugging helpers for inspecting TorchLens objects."""

from __future__ import annotations

from typing import Any


def print_all_fields(obj: Any) -> None:
    """Print public, non-callable fields on a TorchLens object.

    Parameters
    ----------
    obj:
        Object whose public data attributes should be printed.

    Returns
    -------
    None
        The fields are printed directly to stdout.
    """

    fields_to_exclude = {
        "decorated_to_orig_funcs_dict",
        "func_rng_states",
        "layer_dict_all_keys",
        "layer_dict_main_keys",
        "layer_list",
        "raw_layer_dict",
        "source_model_log",
    }
    for field_name in dir(obj):
        attr = getattr(obj, field_name)
        if field_name.startswith("_") or field_name in fields_to_exclude or callable(attr):
            continue
        print(f"{field_name}: {attr}")
