"""Compatibility tests for the lazy :mod:`torchlens.utils` facade."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import torchlens.utils as utils

_EXPECTED_EXPORTS = (
    "AutocastRestore",
    "DoctorCheck",
    "DoctorReport",
    "MAX_FLOATING_POINT_TOLERANCE",
    "_ATTR_SKIP_SET",
    "_AUTOCAST_DEVICES",
    "_cuda_available",
    "_get_code_context",
    "_is_cuda_available",
    "_model_expects_single_arg",
    "_safe_copy_arg",
    "assign_to_sequence_or_dict",
    "capability_snapshot",
    "copy_arg_tree",
    "copy_tensor_payload",
    "doctor",
    "format_capability_summary",
    "ensure_iterable",
    "get_attr_values_from_tensor_list",
    "get_memory_amount",
    "get_vars_of_type_from_obj",
    "human_readable_size",
    "identity",
    "in_notebook",
    "index_nested",
    "int_list_to_compact_str",
    "is_iterable",
    "iter_accessible_attributes",
    "find_executable_save_set",
    "flop_count",
    "format_flops",
    "format_size",
    "list_modules",
    "list_ops",
    "trace_streaming",
    "log_current_autocast_state",
    "log_current_rng_states",
    "make_random_barcode",
    "make_short_barcode_from_input",
    "nested_assign",
    "nested_getattr",
    "normalize_input_args",
    "print_override",
    "peek_graph",
    "progress_bar",
    "register_op_rule",
    "remove_attributes_with_prefix",
    "remove_entry_from_list",
    "safe_copy",
    "safe_copy_args",
    "safe_copy_kwargs",
    "safe_to",
    "set_random_seed",
    "synthetic_input",
    "set_rng_from_saved_states",
    "tensor_all_nan",
    "tensor_nanequal",
    "tensor_stats_summary",
    "warn_parallel",
)

_REEXPORTED_OBJECTS = {
    name: target for name, target in utils._LAZY_EXPORTS.items() if name in _EXPECTED_EXPORTS
}


@pytest.mark.heavy
def test_utils_facade_defers_reexport_modules() -> None:
    """Importing TorchLens leaves utility re-export modules unloaded."""

    code = f"""
import sys
import torchlens

assert torchlens.__file__.startswith({str(Path(__file__).resolve().parents[1])!r})
deferred = {sorted({module_name for module_name, _ in _REEXPORTED_OBJECTS.values()})!r}
loaded = sorted(name for name in deferred if name in sys.modules)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.smoke
def test_every_previous_utils_export_resolves_to_identical_object() -> None:
    """Every frozen facade name resolves and re-exports preserve identity."""

    assert tuple(utils.__all__) == _EXPECTED_EXPORTS
    assert set(_REEXPORTED_OBJECTS) == set(_EXPECTED_EXPORTS) - {
        "DoctorCheck",
        "DoctorReport",
        "capability_snapshot",
        "doctor",
        "format_capability_summary",
        "find_executable_save_set",
        "flop_count",
        "list_modules",
        "list_ops",
        "peek_graph",
        "synthetic_input",
        "trace_streaming",
    }
    for name in _EXPECTED_EXPORTS:
        facade_object = getattr(utils, name)
        assert facade_object is getattr(utils, name)
        assert name in dir(utils)
        if name in _REEXPORTED_OBJECTS:
            module_name, attribute_name = _REEXPORTED_OBJECTS[name]
            defining_module = importlib.import_module(module_name)
            assert facade_object is getattr(defining_module, attribute_name)
