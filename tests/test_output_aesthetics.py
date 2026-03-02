"""Aesthetic testing infrastructure for torchlens.

Generates human-inspectable outputs in tests/test_outputs/:
  - aesthetic_report.txt  — comprehensive text report of all user-facing reprs/accessors
  - aesthetic-models/     — visualization PDFs exercising depth, rolling, buffers, etc.

Run:  pytest tests/test_output_aesthetics.py -v

EXTENDING THIS FILE:
- New model: add to AESTHETIC_TEXT_MODELS list
- New vis combination: add to the appropriate test_aesthetic_*_visualizations function
- New repr/accessor: add a section to _capture_model_outputs
- New data class: add its repr to the report
- New error message: add to the "Convenience Error Messages" section
- New visualization feature: add a model + combination that exercises it
"""

import io
import os
import shutil
import subprocess
import traceback
from os.path import join as opj

import pytest
import torch

from conftest import TEST_OUTPUTS_DIR, VIS_OUTPUT_DIR

import example_models
from torchlens import log_forward_pass, show_model_graph
from torchlens.postprocess import _roll_graph

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

REPORT_PATH = opj(TEST_OUTPUTS_DIR, "aesthetic_report.txt")
VIS_DIR = opj(VIS_OUTPUT_DIR, "aesthetic_test_models")


def _section(title: str, level: int = 1) -> str:
    """Return a section header string."""
    if level == 1:
        bar = "=" * 80
        return f"\n\n{bar}\n{title}\n{bar}\n"
    elif level == 2:
        bar = "-" * 60
        return f"\n{bar}\n{title}\n{bar}\n"
    else:
        return f"\n  --- {title} ---\n"


def _code(snippet: str) -> str:
    """Return a code snippet annotation."""
    return f"  # {snippet}\n"


def _capture(expr_str: str, value) -> str:
    """Format a captured value with its expression."""
    return f"{_code(expr_str)}{value}\n"


def _capture_error(expr_str: str, callable_fn) -> str:
    """Call callable_fn, capture the exception, format it."""
    try:
        callable_fn()
        return f"{_code(expr_str)}(no error raised)\n"
    except Exception as e:
        return f"{_code(expr_str)}{type(e).__name__}: {e}\n"


def _field_dump(obj, label: str, exclude=None) -> str:
    """Dump all public non-callable fields of an object."""
    if exclude is None:
        exclude = {"source_model_log", "func_rng_states", "_source_model_log"}
    lines = [f"\n  --- Field dump: {label} ---\n"]
    for field in sorted(dir(obj)):
        if field.startswith("_"):
            continue
        try:
            attr = getattr(obj, field)
        except Exception as e:
            lines.append(f"  {field}: <error: {e}>\n")
            continue
        if callable(attr):
            continue
        if field in exclude:
            continue
        lines.append(f"  {field}: {attr}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Text report: model definitions
# ---------------------------------------------------------------------------

AESTHETIC_TEXT_MODELS = [
    # (name, model_class, input_tensor, description)
    (
        "SimpleFF",
        example_models.SimpleFF,
        torch.rand(5, 5),
        "Baseline: pure functional, no modules, no params",
    ),
    (
        "NestedModules",
        example_models.NestedModules,
        torch.rand(5, 5),
        "Deep module hierarchy (3+ levels)",
    ),
    (
        "RecurrentParamsSimple",
        example_models.RecurrentParamsSimple,
        torch.rand(5, 5),
        "Multi-pass modules, shared params (loop finding)",
    ),
    (
        "SimpleBranching",
        example_models.SimpleBranching,
        torch.rand(6, 3, 224, 224),
        "Branch-and-merge topology",
    ),
    ("BufferModel", example_models.BufferModel, torch.rand(12, 12), "Buffer layers"),
    (
        "ConditionalBranching",
        example_models.ConditionalBranching,
        torch.ones(6, 3, 224, 224),
        "Conditional (if-then) branching",
    ),
    (
        "AestheticFrozenMix",
        example_models.AestheticFrozenMix,
        torch.rand(1, 8),
        "Frozen + trainable params",
    ),
    (
        "LoopingParamsDoubleNested",
        example_models.LoopingParamsDoubleNested,
        torch.rand(5, 5),
        "Double-nested loop detection with shared params",
    ),
]


def _capture_model_outputs(name: str, model, x, description: str) -> str:
    """Capture all user-facing outputs for one model."""
    out = io.StringIO()
    out.write(_section(f"Model: {name} — {description}", level=1))

    # Log the forward pass
    log = log_forward_pass(model, x, random_seed=42)

    # Build rolled graph data for models with recurrence
    if log.model_is_recurrent:
        _roll_graph(log)

    # ===== A. ModelLog Overview =====
    out.write(_section("A. ModelLog Overview", level=2))

    out.write(_capture("str(log)", str(log)))
    out.write(_capture("repr(log)", repr(log)))

    df = log.to_pandas()
    out.write(_capture("log.to_pandas()", df.to_string()))
    out.write(_capture("log.to_pandas().columns.tolist()", df.columns.tolist()))

    # ===== B. Module System =====
    out.write(_section("B. Module System", level=2))

    out.write(_capture("repr(log.modules)", repr(log.modules)))
    out.write(_capture("repr(log.root_module)", repr(log.root_module)))
    out.write(_capture('repr(log.modules["self"])', repr(log.modules["self"])))
    out.write(_capture('repr(log.modules[""])', repr(log.modules[""])))
    out.write(_capture("repr(log.modules[0])", repr(log.modules[0])))

    # Find a non-root module with params for demonstrations
    param_module_addr = None
    any_non_root_addr = None
    for ml in log.modules:
        if ml.address == "self":
            continue
        any_non_root_addr = any_non_root_addr or ml.address
        if ml.num_params > 0:
            param_module_addr = ml.address
            break
    if any_non_root_addr is None:
        any_non_root_addr = "self"
    if param_module_addr is None:
        param_module_addr = any_non_root_addr

    out.write(
        _capture(f'repr(log.modules["{param_module_addr}"])', repr(log.modules[param_module_addr]))
    )

    # Pass notation (module_addr:1)
    pass_key = f"{param_module_addr}:1"
    if pass_key in log.modules:
        out.write(
            _capture(
                f'repr(log.modules["{pass_key}"]) — ModulePassLog', repr(log.modules[pass_key])
            )
        )
    else:
        out.write(f"  # {pass_key} not in modules (single-pass module)\n")

    # Direct pass access
    mod = log.modules[param_module_addr]
    if 1 in mod.passes:
        out.write(_capture(f'log.modules["{param_module_addr}"].passes[1]', repr(mod.passes[1])))

    out.write(_capture("log.modules.to_pandas()", log.modules.to_pandas().to_string()))
    out.write(_capture("log.modules.summary()", log.modules.summary()))

    # Multi-pass module info (for recurrent models)
    if log.model_is_recurrent:
        for ml in log.modules:
            if ml.num_passes > 1 and ml.address != "self":
                out.write(
                    _capture(
                        f'repr(log.modules["{ml.address}"]) — multi-pass module ({ml.num_passes} passes)',
                        repr(ml),
                    )
                )
                # Show ModulePassLog for each pass
                for p in sorted(ml.passes.keys()):
                    out.write(
                        _capture(
                            f'repr(log.modules["{ml.address}"].passes[{p}]) — ModulePassLog pass {p}',
                            repr(ml.passes[p]),
                        )
                    )
                break

    # ===== C. Parameter System =====
    out.write(_section("C. Parameter System", level=2))

    out.write(_capture("repr(log.params)", repr(log.params)))

    if param_module_addr != "self" and log.modules[param_module_addr].num_params > 0:
        mod_params = log.modules[param_module_addr].params
        out.write(_capture(f'repr(log.modules["{param_module_addr}"].params)', repr(mod_params)))
        if len(mod_params) > 0:
            first_param = mod_params[0]
            out.write(
                _capture(
                    f'repr(log.modules["{param_module_addr}"].params["{first_param.address}"]) — by full address',
                    repr(mod_params[first_param.address]),
                )
            )
            out.write(
                _capture(
                    f'repr(log.modules["{param_module_addr}"].params["{first_param.name}"]) — by short name',
                    repr(mod_params[first_param.name]),
                )
            )
            out.write(
                _capture(
                    f'repr(log.modules["{param_module_addr}"].params[0]) — by index',
                    repr(mod_params[0]),
                )
            )

    # Show frozen vs trainable params explicitly (for AestheticFrozenMix)
    if any(not pl.trainable for pl in log.params):
        out.write(_section("C.1 Frozen vs Trainable Parameters", level=3))
        for pl in log.params:
            status = "FROZEN" if not pl.trainable else "TRAINABLE"
            out.write(f"  [{status}] {pl.address}\n")
            out.write(f"  {repr(pl)}\n\n")
        out.write(f"  Total frozen: {log.total_params_frozen}\n")
        out.write(f"  Total trainable: {log.total_params_trainable}\n")
        out.write(f"  Total params: {log.total_params}\n")

    # ===== D. TensorLog / Layer Access =====
    out.write(_section("D. TensorLog / Layer Access", level=2))

    out.write(_capture("repr(log[0]) — first layer", repr(log[0])))
    out.write(_capture("repr(log[-1]) — last layer", repr(log[-1])))
    mid = len(log) // 2
    out.write(_capture(f"repr(log[{mid}]) — middle layer", repr(log[mid])))

    # By full label
    sample_label = log.layer_labels[min(1, len(log.layer_labels) - 1)]
    out.write(_capture(f'repr(log["{sample_label}"]) — by full label', repr(log[sample_label])))

    # By label with pass and without pass
    if len(log.layer_labels_w_pass) > 1:
        sample_w_pass = log.layer_labels_w_pass[1]
        out.write(
            _capture(f'repr(log["{sample_w_pass}"]) — by label w/ pass', repr(log[sample_w_pass]))
        )
    # Find a single-pass layer to demonstrate no-pass lookup
    for lbl in log.layer_labels_no_pass:
        if lbl in log.layer_dict_all_keys:
            out.write(_capture(f'repr(log["{lbl}"]) — by label no pass', repr(log[lbl])))
            break

    # By module address (returns ModuleLog for single-pass)
    if any_non_root_addr != "self":
        result = log[any_non_root_addr]
        out.write(
            _capture(
                f'repr(log["{any_non_root_addr}"]) — by module address (returns {type(result).__name__})',
                repr(result),
            )
        )

    # Substring match
    if len(log.layer_labels) > 2:
        # Find a layer type that matches exactly one key in the lookup dict
        for entry in log.layer_list:
            candidate = entry.layer_type
            if candidate in ("input",):
                continue  # too generic
            all_key_matches = [k for k in log.layer_dict_all_keys if candidate in str(k)]
            if len(all_key_matches) == 1:
                out.write(
                    _capture(f'repr(log["{candidate}"]) — substring match', repr(log[candidate]))
                )
                break

    # ===== D.1 RolledTensorLog =====
    if log.model_is_recurrent and len(log.layer_list_rolled) > 0:
        out.write(_section("D.1 RolledTensorLog", level=3))
        # Find a rolled tensor log with multiple passes
        for rtl in log.layer_list_rolled:
            if rtl.layer_passes_total > 1:
                out.write(
                    _capture(
                        f"repr(rolled_tensor_log) — {rtl.layer_label} ({rtl.layer_passes_total} passes)",
                        repr(rtl),
                    )
                )
                break
        else:
            # Just show the first one
            out.write(_capture("repr(log.layer_list_rolled[0])", repr(log.layer_list_rolled[0])))

    # ===== E. Convenience Error Messages =====
    out.write(_section("E. Convenience Error Messages", level=2))

    out.write(_capture_error("log[9999] — out-of-range index", lambda: log[9999]))
    out.write(_capture_error('log["nonexistent"] — invalid layer name', lambda: log["nonexistent"]))

    # Multi-pass layer without pass specifier
    for label, num_p in log.layer_num_passes.items():
        if num_p > 1:
            out.write(
                _capture_error(
                    f'log["{label}"] — multi-pass layer without pass', lambda lbl=label: log[lbl]
                )
            )
            # Invalid pass number
            out.write(
                _capture_error(
                    f'log["{label}:99"] — invalid pass number', lambda lbl=label: log[f"{lbl}:99"]
                )
            )
            break

    # Multi-pass module .layers access
    for ml in log.modules:
        if ml.num_passes > 1 and ml.address != "self":
            out.write(
                _capture_error(
                    f'log.modules["{ml.address}"].layers — multi-pass per-call field',
                    lambda m=ml: m.layers,
                )
            )
            break

    # ===== F. Full Field Dumps =====
    out.write(_section("F. Full Field Dumps — Every Major Data Structure", level=2))

    # F.1 ModelLog
    out.write(_section("F.1 ModelLog field dump", level=3))
    out.write(_code(f"Model: {name}, input: {list(x.shape)}"))
    out.write(_field_dump(log, "ModelLog"))

    # F.2 TensorLog — pick a layer with params if possible
    out.write(_section("F.2 TensorLog field dump", level=3))
    tensor_for_dump = None
    for entry in log.layer_list:
        if entry.computed_with_params:
            tensor_for_dump = entry
            break
    if tensor_for_dump is None:
        tensor_for_dump = log[len(log) // 2]
    out.write(_code(f"Layer: {tensor_for_dump.layer_label}"))
    out.write(_field_dump(tensor_for_dump, f"TensorLog: {tensor_for_dump.layer_label}"))

    # F.3 RolledTensorLog
    if log.model_is_recurrent and len(log.layer_list_rolled) > 0:
        out.write(_section("F.3 RolledTensorLog field dump", level=3))
        rtl_for_dump = None
        for rtl in log.layer_list_rolled:
            if rtl.layer_passes_total > 1:
                rtl_for_dump = rtl
                break
        if rtl_for_dump is None:
            rtl_for_dump = log.layer_list_rolled[0]
        out.write(_code(f"Rolled layer: {rtl_for_dump.layer_label}"))
        out.write(_field_dump(rtl_for_dump, f"RolledTensorLog: {rtl_for_dump.layer_label}"))

    # F.4 ModuleLog
    out.write(_section("F.4 ModuleLog field dump", level=3))
    mod_for_dump = log.modules[param_module_addr]
    out.write(_code(f"Module: {mod_for_dump.address}"))
    out.write(_field_dump(mod_for_dump, f"ModuleLog: {mod_for_dump.address}"))

    # F.5 ModulePassLog
    if 1 in mod_for_dump.passes:
        out.write(_section("F.5 ModulePassLog field dump", level=3))
        mpl_for_dump = mod_for_dump.passes[1]
        out.write(_code(f"ModulePassLog: {mpl_for_dump.pass_label}"))
        out.write(_field_dump(mpl_for_dump, f"ModulePassLog: {mpl_for_dump.pass_label}"))

    # F.6 ParamLog
    if len(log.params) > 0:
        out.write(_section("F.6 ParamLog field dump", level=3))
        pl_for_dump = log.params[0]
        out.write(_code(f"Param: {pl_for_dump.address}"))
        out.write(_field_dump(pl_for_dump, f"ParamLog: {pl_for_dump.address}"))

    # F.7 ModuleAccessor
    out.write(_section("F.7 ModuleAccessor field dump", level=3))
    out.write(_field_dump(log.modules, "ModuleAccessor"))

    # F.8 ParamAccessor
    out.write(_section("F.8 ParamAccessor field dump", level=3))
    out.write(_field_dump(log.params, "ParamAccessor"))

    # ===== G. Gradient System =====
    # Only for models with params — log with save_gradients=True and run backward
    if len(log.params) > 0:
        out.write(_section("G. Gradient System (save_gradients=True + backward)", level=2))
        try:
            grad_log = log_forward_pass(model, x, save_gradients=True, random_seed=42)
            output_label = grad_log.output_layers[0]
            output_tensor = grad_log[output_label].tensor_contents
            output_tensor.sum().backward()

            out.write(_capture("grad_log.has_saved_gradients", grad_log.has_saved_gradients))
            out.write(
                _capture(
                    "grad_log.layers_with_saved_gradients",
                    grad_log.layers_with_saved_gradients,
                )
            )

            # Show gradient fields on a TensorLog that has saved grad
            for entry in grad_log.layer_list:
                if entry.has_saved_grad:
                    out.write(
                        _section(
                            f"G.1 TensorLog gradient fields — {entry.layer_label}",
                            level=3,
                        )
                    )
                    out.write(_capture("has_saved_grad", entry.has_saved_grad))
                    out.write(_capture("grad_contents", entry.grad_contents))
                    out.write(_capture("grad_shape", entry.grad_shape))
                    out.write(_capture("grad_dtype", entry.grad_dtype))
                    out.write(_capture("grad_fsize", entry.grad_fsize))
                    out.write(_capture("grad_fsize_nice", entry.grad_fsize_nice))
                    break

            # Show a TensorLog WITHOUT grad for contrast
            for entry in grad_log.layer_list:
                if not entry.has_saved_grad:
                    out.write(
                        _section(
                            f"G.2 TensorLog without grad — {entry.layer_label}",
                            level=3,
                        )
                    )
                    out.write(_capture("has_saved_grad", entry.has_saved_grad))
                    out.write(_capture("grad_contents", entry.grad_contents))
                    break

            # ParamLog gradient fields
            for pl in grad_log.params:
                if pl.has_grad:
                    out.write(_section(f"G.3 ParamLog gradient fields — {pl.address}", level=3))
                    out.write(_capture("has_grad", pl.has_grad))
                    out.write(_capture("grad_shape", pl.grad_shape))
                    out.write(_capture("grad_dtype", pl.grad_dtype))
                    out.write(_capture("grad_fsize", pl.grad_fsize))
                    out.write(_capture("grad_fsize_nice", pl.grad_fsize_nice))
                    break

            # Show frozen param without grad (if applicable)
            for pl in grad_log.params:
                if not pl.trainable:
                    out.write(
                        _section(
                            f"G.4 Frozen ParamLog — no grad — {pl.address}",
                            level=3,
                        )
                    )
                    out.write(_capture("has_grad", pl.has_grad))
                    out.write(_capture("trainable", pl.trainable))
                    break
        except Exception as e:
            out.write(f"  (gradient capture failed: {type(e).__name__}: {e})\n")
            out.write(f"  {traceback.format_exc()}\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Test: text report
# ---------------------------------------------------------------------------


def test_generate_aesthetic_report():
    """Generate the comprehensive text report."""
    report = io.StringIO()
    report.write("=" * 80 + "\n")
    report.write("TORCHLENS AESTHETIC REPORT\n")
    report.write("=" * 80 + "\n")
    report.write(
        "This report captures every user-facing repr, accessor, DataFrame,\n"
        "error message, and field dump for visual inspection.\n"
        "Regenerate: pytest tests/test_aesthetic_outputs.py::test_generate_aesthetic_report -v\n"
    )

    for name, model_cls, x, description in AESTHETIC_TEXT_MODELS:
        model = model_cls()
        report.write(_capture_model_outputs(name, model, x, description))

    with open(REPORT_PATH, "w") as f:
        f.write(report.getvalue())

    # Verify the file was written and is non-trivial
    with open(REPORT_PATH) as f:
        content = f.read()
    assert len(content) > 5000, f"Report too short ({len(content)} chars)"
    assert "ModelLog" in content
    assert "TensorLog" in content
    assert "ParamLog" in content


# ---------------------------------------------------------------------------
# LaTeX PDF report
# ---------------------------------------------------------------------------

TEX_PATH = opj(TEST_OUTPUTS_DIR, "aesthetic_report.tex")
PDF_PATH = opj(TEST_OUTPUTS_DIR, "aesthetic_report.pdf")

# Visualization configurations for the gallery section
VIS_GALLERY = [
    # (filename_stem, caption, model_class, input_shape_desc, vis_opt, depth, direction, buffers)
    (
        "deep_nested_depth1",
        "AestheticDeepNested — depth=1",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1,
        "bottomup",
        False,
    ),
    (
        "deep_nested_depth2",
        "AestheticDeepNested — depth=2",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        2,
        "bottomup",
        False,
    ),
    (
        "deep_nested_depth3",
        "AestheticDeepNested — depth=3",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        3,
        "bottomup",
        False,
    ),
    (
        "deep_nested_full",
        "AestheticDeepNested — full depth",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "deep_nested_topdown",
        "AestheticDeepNested — top-down",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1000,
        "topdown",
        False,
    ),
    (
        "deep_nested_leftright",
        "AestheticDeepNested — left-right",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1000,
        "leftright",
        False,
    ),
    (
        "shared_unrolled",
        "AestheticSharedModule — unrolled",
        "AestheticSharedModule",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "shared_rolled",
        "AestheticSharedModule — rolled",
        "AestheticSharedModule",
        "(1,8)",
        "rolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "shared_rolled_depth1",
        "AestheticSharedModule — rolled depth=1",
        "AestheticSharedModule",
        "(1,8)",
        "rolled",
        1,
        "bottomup",
        False,
    ),
    (
        "buffer_visible",
        "AestheticBufferBranch — buffers visible",
        "AestheticBufferBranch",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        True,
    ),
    (
        "buffer_hidden",
        "AestheticBufferBranch — buffers hidden",
        "AestheticBufferBranch",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "buffer_topdown",
        "AestheticBufferBranch — top-down",
        "AestheticBufferBranch",
        "(1,8)",
        "unrolled",
        1000,
        "topdown",
        True,
    ),
    (
        "buffer_leftright",
        "AestheticBufferBranch — left-right",
        "AestheticBufferBranch",
        "(1,8)",
        "unrolled",
        1000,
        "leftright",
        True,
    ),
    (
        "kitchen_sink_unrolled",
        "AestheticKitchenSink — unrolled",
        "AestheticKitchenSink",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        True,
    ),
    (
        "kitchen_sink_rolled",
        "AestheticKitchenSink — rolled",
        "AestheticKitchenSink",
        "(1,8)",
        "rolled",
        1000,
        "bottomup",
        True,
    ),
    (
        "kitchen_sink_depth1",
        "AestheticKitchenSink — depth=1",
        "AestheticKitchenSink",
        "(1,8)",
        "unrolled",
        1,
        "bottomup",
        True,
    ),
    (
        "kitchen_sink_depth2",
        "AestheticKitchenSink — depth=2",
        "AestheticKitchenSink",
        "(1,8)",
        "unrolled",
        2,
        "bottomup",
        True,
    ),
    (
        "frozen_mix_unrolled",
        "AestheticFrozenMix — unrolled",
        "AestheticFrozenMix",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "frozen_mix_topdown",
        "AestheticFrozenMix — top-down",
        "AestheticFrozenMix",
        "(1,8)",
        "unrolled",
        1000,
        "topdown",
        False,
    ),
    (
        "loop_simple_unrolled",
        "RecurrentParamsSimple — unrolled",
        "RecurrentParamsSimple",
        "(5,5)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "loop_simple_rolled",
        "RecurrentParamsSimple — rolled",
        "RecurrentParamsSimple",
        "(5,5)",
        "rolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "loop_double_nested_unrolled",
        "LoopingParamsDoubleNested — unrolled",
        "LoopingParamsDoubleNested",
        "(5,5)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "loop_double_nested_rolled",
        "LoopingParamsDoubleNested — rolled",
        "LoopingParamsDoubleNested",
        "(5,5)",
        "rolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "nested_modules_unrolled",
        "NestedModules — unrolled",
        "NestedModules",
        "(5,5)",
        "unrolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "nested_modules_rolled",
        "NestedModules — rolled",
        "NestedModules",
        "(5,5)",
        "rolled",
        1000,
        "bottomup",
        False,
    ),
    (
        "nested_modules_depth1",
        "NestedModules — depth=1",
        "NestedModules",
        "(5,5)",
        "unrolled",
        1,
        "bottomup",
        False,
    ),
    (
        "nested_modules_depth2",
        "NestedModules — depth=2",
        "NestedModules",
        "(5,5)",
        "unrolled",
        2,
        "bottomup",
        False,
    ),
]

# Gradient visualization gallery — rendered with save_gradients=True + backward()
# (filename_stem, caption, model_name, input_shape_desc, vis_opt, depth, direction)
GRADIENT_VIS_GALLERY = [
    (
        "gradient_deep_nested",
        "AestheticDeepNested — with gradient arrows",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
    ),
    (
        "gradient_deep_nested_topdown",
        "AestheticDeepNested — gradients, top-down",
        "AestheticDeepNested",
        "(1,8)",
        "unrolled",
        1000,
        "topdown",
    ),
    (
        "gradient_frozen_mix",
        "AestheticFrozenMix — gradients (frozen + trainable)",
        "AestheticFrozenMix",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
    ),
    (
        "gradient_kitchen_sink",
        "AestheticKitchenSink — gradients + nesting",
        "AestheticKitchenSink",
        "(1,8)",
        "unrolled",
        1000,
        "bottomup",
    ),
    (
        "gradient_kitchen_sink_depth1",
        "AestheticKitchenSink — gradients, depth=1",
        "AestheticKitchenSink",
        "(1,8)",
        "unrolled",
        1,
        "bottomup",
    ),
]


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters for use in normal text (not verbatim)."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _verbatim_box(title: str, content: str, color: str = "codebg") -> str:
    """Wrap content in a tcolorbox with Verbatim interior."""
    # Truncate very long content to keep PDF reasonable
    lines = content.split("\n")
    if len(lines) > 80:
        lines = lines[:75] + [f"... ({len(lines) - 75} more lines, see aesthetic_report.txt)"]
    truncated = "\n".join(lines)
    return (
        f"\\begin{{tcolorbox}}[title={{{_tex_escape(title)}}}, "
        f"colback={color}, colframe=sectioncolor, "
        f"fonttitle=\\bfseries\\small, breakable, "
        f"left=2pt, right=2pt, top=2pt, bottom=2pt]\n"
        f"\\begin{{Verbatim}}[fontsize=\\tiny, breaklines=true, breaksymbol=]\n"
        f"{truncated}\n"
        f"\\end{{Verbatim}}\n"
        f"\\end{{tcolorbox}}\n\n"
    )


def _build_latex_report() -> str:
    """Build the full LaTeX document string."""
    doc = io.StringIO()

    # Preamble
    doc.write(r"""\documentclass[10pt, letterpaper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{fancyvrb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage[breakable, skins]{tcolorbox}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}

% Colors
\definecolor{codebg}{HTML}{F5F5F0}
\definecolor{sectioncolor}{HTML}{2C3E50}
\definecolor{accentcolor}{HTML}{E74C3C}
\definecolor{linkcolor}{HTML}{2980B9}

\hypersetup{
    colorlinks=true,
    linkcolor=linkcolor,
    urlcolor=linkcolor,
}

\tcbset{
    enhanced,
    boxrule=0.5pt,
    arc=3pt,
    fonttitle=\bfseries,
}

\title{\textbf{\Huge TorchLens Aesthetic Report} \\[10pt]
    \large Visual inspection of all user-facing outputs}
\author{Auto-generated by \texttt{test\_output\_aesthetics.py}}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

""")

    # --- Section 1: Model Reports ---
    doc.write("\\section{Model Reports}\n\n")
    doc.write(
        "Each model below shows the key user-facing outputs: "
        "model summary, module hierarchy, parameter system, "
        "layer access, error messages, and field dumps.\n\n"
    )

    for name, model_cls, x, description in AESTHETIC_TEXT_MODELS:
        model = model_cls()
        log = log_forward_pass(model, x, random_seed=42)
        if log.model_is_recurrent:
            _roll_graph(log)

        doc.write(f"\\subsection{{{_tex_escape(name)} --- {_tex_escape(description)}}}\n\n")

        # A. Model Summary
        doc.write(_verbatim_box("A. Model Summary — str(log)", str(log)))

        # B. Module System
        doc.write(_verbatim_box("B. Module Accessor — repr(log.modules)", repr(log.modules)))
        doc.write(
            _verbatim_box("B. Module Summary Table — log.modules.summary()", log.modules.summary())
        )

        # C. Parameters
        if len(log.params) > 0:
            doc.write(_verbatim_box("C. Parameters — repr(log.params)", repr(log.params)))

        # Frozen vs trainable highlight
        if any(not pl.trainable for pl in log.params):
            frozen_text = ""
            for pl in log.params:
                status = "FROZEN" if not pl.trainable else "TRAINABLE"
                frozen_text += f"[{status}] {pl.address}\n{repr(pl)}\n\n"
            frozen_text += f"Total frozen: {log.total_params_frozen}\n"
            frozen_text += f"Total trainable: {log.total_params_trainable}\n"
            frozen_text += f"Total params: {log.total_params}\n"
            doc.write(_verbatim_box("C.1 Frozen vs Trainable Parameters", frozen_text))

        # D. Layer Access — first and last
        doc.write(_verbatim_box("D. First Layer — repr(log[0])", repr(log[0])))
        doc.write(_verbatim_box("D. Last Layer — repr(log[-1])", repr(log[-1])))

        # D.1 Rolled TensorLog
        if log.model_is_recurrent and len(log.layer_list_rolled) > 0:
            for rtl in log.layer_list_rolled:
                if rtl.layer_passes_total > 1:
                    doc.write(
                        _verbatim_box(
                            f"D.1 RolledTensorLog — {rtl.layer_label} ({rtl.layer_passes_total} passes)",
                            repr(rtl),
                        )
                    )
                    break

        # Multi-pass module
        if log.model_is_recurrent:
            for ml in log.modules:
                if ml.num_passes > 1 and ml.address != "self":
                    passes_text = repr(ml) + "\n\n"
                    for p in sorted(ml.passes.keys()):
                        passes_text += f"--- Pass {p} ---\n{repr(ml.passes[p])}\n\n"
                    doc.write(
                        _verbatim_box(
                            f"B.1 Multi-Pass Module — {ml.address} ({ml.num_passes} passes)",
                            passes_text,
                        )
                    )
                    break

        # E. Error Messages
        error_text = ""
        try:
            log[9999]
        except Exception as e:
            error_text += f"log[9999]:\n{type(e).__name__}: {e}\n\n"
        try:
            log["nonexistent"]
        except Exception as e:
            error_text += f'log["nonexistent"]:\n{type(e).__name__}: {e}\n\n'
        for label, num_p in log.layer_num_passes.items():
            if num_p > 1:
                try:
                    log[label]
                except Exception as e:
                    error_text += f'log["{label}"] (multi-pass):\n{type(e).__name__}: {e}\n\n'
                break
        if error_text:
            doc.write(_verbatim_box("E. Convenience Error Messages", error_text))

        # F. Field Dumps (just ModelLog headline fields + one TensorLog)
        # ModelLog field dump
        model_fields = ""
        for field in sorted(dir(log)):
            if field.startswith("_"):
                continue
            try:
                attr = getattr(log, field)
            except Exception:
                continue
            if callable(attr):
                continue
            if field in {
                "source_model_log",
                "func_rng_states",
                "layer_list",
                "layer_dict_main_keys",
                "layer_dict_all_keys",
            }:
                continue
            model_fields += f"{field}: {attr}\n"
        doc.write(_verbatim_box("F.1 ModelLog — All Fields", model_fields))

        # TensorLog field dump — pick a layer with params
        tensor_for_dump = None
        for entry in log.layer_list:
            if entry.computed_with_params:
                tensor_for_dump = entry
                break
        if tensor_for_dump is None:
            tensor_for_dump = log[len(log) // 2]
        tensor_fields = ""
        for field in sorted(dir(tensor_for_dump)):
            if field.startswith("_"):
                continue
            try:
                attr = getattr(tensor_for_dump, field)
            except Exception:
                continue
            if callable(attr):
                continue
            if field in {"source_model_log", "func_rng_states"}:
                continue
            tensor_fields += f"{field}: {attr}\n"
        doc.write(_verbatim_box(f"F.2 TensorLog — {tensor_for_dump.layer_label}", tensor_fields))

        # ModuleLog field dump
        param_module_addr = None
        for ml in log.modules:
            if ml.address != "self" and ml.num_params > 0:
                param_module_addr = ml.address
                break
        if param_module_addr is None:
            for ml in log.modules:
                if ml.address != "self":
                    param_module_addr = ml.address
                    break
        if param_module_addr:
            mod_dump = log.modules[param_module_addr]
            mod_fields = ""
            for field in sorted(dir(mod_dump)):
                if field.startswith("_"):
                    continue
                try:
                    attr = getattr(mod_dump, field)
                except Exception as e:
                    mod_fields += f"{field}: <{type(e).__name__}: {e}>\n"
                    continue
                if callable(attr):
                    continue
                mod_fields += f"{field}: {attr}\n"
            doc.write(_verbatim_box(f"F.4 ModuleLog — {mod_dump.address}", mod_fields))

        # ParamLog field dump
        if len(log.params) > 0:
            pl = log.params[0]
            pl_fields = ""
            for field in sorted(dir(pl)):
                if field.startswith("_"):
                    continue
                try:
                    attr = getattr(pl, field)
                except Exception:
                    continue
                if callable(attr):
                    continue
                pl_fields += f"{field}: {attr}\n"
            doc.write(_verbatim_box(f"F.6 ParamLog — {pl.address}", pl_fields))

        doc.write("\\newpage\n\n")

    # --- Section 2: Visualization Gallery ---
    doc.write("\\section{Visualization Gallery}\n\n")
    doc.write(
        "Each figure below shows a computational graph rendered by "
        "\\texttt{show\\_model\\_graph()} with the specified parameters.\n\n"
    )

    for stem, caption, model_name, input_desc, vis_opt, depth, direction, buffers in VIS_GALLERY:
        pdf_path = opj(VIS_DIR, f"{stem}.pdf")
        if not os.path.exists(pdf_path):
            continue
        buf_str = ", buffers=True" if buffers else ""
        param_str = (
            f"vis\\_opt={_tex_escape(vis_opt)}, "
            f"depth={depth}, "
            f"direction={_tex_escape(direction)}"
            f"{_tex_escape(buf_str)}"
        )
        doc.write("\\begin{figure}[H]\n")
        doc.write("\\centering\n")
        doc.write(
            f"\\includegraphics[max width=0.95\\textwidth, max height=0.85\\textheight]{{{pdf_path}}}\n"
        )
        doc.write(f"\\caption{{{_tex_escape(caption)} \\\\\\small\\texttt{{{param_str}}}}}\n")
        doc.write("\\end{figure}\n\n")

    # --- Section 3: Gradient Visualization Gallery ---
    doc.write("\\section{Gradient Visualization Gallery}\n\n")
    doc.write(
        "Graphs below show backward-pass gradient edges (blue arrows) rendered after "
        "\\texttt{log\\_forward\\_pass(save\\_gradients=True)} and \\texttt{.backward()}. "
        "Gradient arrows only appear in unrolled mode.\n\n"
    )

    for stem, caption, model_name, input_desc, vis_opt, depth, direction in GRADIENT_VIS_GALLERY:
        pdf_path = opj(VIS_DIR, f"{stem}.pdf")
        if not os.path.exists(pdf_path):
            continue
        param_str = (
            f"save\\_gradients=True, "
            f"vis\\_opt={_tex_escape(vis_opt)}, "
            f"depth={depth}, "
            f"direction={_tex_escape(direction)}"
        )
        doc.write("\\begin{figure}[H]\n")
        doc.write("\\centering\n")
        doc.write(
            f"\\includegraphics[max width=0.95\\textwidth, max height=0.85\\textheight]{{{pdf_path}}}\n"
        )
        doc.write(f"\\caption{{{_tex_escape(caption)} \\\\\\small\\texttt{{{param_str}}}}}\n")
        doc.write("\\end{figure}\n\n")

    doc.write("\\end{document}\n")
    return doc.getvalue()


def _ensure_vis_pdfs_exist():
    """Generate any missing visualization PDFs so the LaTeX report can include them."""
    # Map model names to (class, input)
    model_inputs = {
        "AestheticDeepNested": (example_models.AestheticDeepNested(), torch.rand(1, 8)),
        "AestheticSharedModule": (example_models.AestheticSharedModule(), torch.rand(1, 8)),
        "AestheticBufferBranch": (example_models.AestheticBufferBranch(), torch.rand(1, 8)),
        "AestheticKitchenSink": (example_models.AestheticKitchenSink(), torch.rand(1, 8)),
        "AestheticFrozenMix": (example_models.AestheticFrozenMix(), torch.rand(1, 8)),
        "RecurrentParamsSimple": (example_models.RecurrentParamsSimple(), torch.rand(5, 5)),
        "LoopingParamsDoubleNested": (example_models.LoopingParamsDoubleNested(), torch.rand(5, 5)),
        "NestedModules": (example_models.NestedModules(), torch.rand(5, 5)),
    }

    # Standard gallery
    missing = [g for g in VIS_GALLERY if not os.path.exists(opj(VIS_DIR, f"{g[0]}.pdf"))]
    for stem, caption, model_name, _, vis_opt, depth, direction, buffers in missing:
        model, x = model_inputs[model_name]
        _vis(
            model, x, stem, vis_opt=vis_opt, depth=depth, direction=direction, buffer_layers=buffers
        )

    # Gradient gallery
    grad_missing = [
        g for g in GRADIENT_VIS_GALLERY if not os.path.exists(opj(VIS_DIR, f"{g[0]}.pdf"))
    ]
    for stem, caption, model_name, _, vis_opt, depth, direction in grad_missing:
        model, x = model_inputs[model_name]
        _vis_gradient(model, x, stem, vis_opt=vis_opt, depth=depth, direction=direction)


@pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex not installed (install texlive for PDF report generation)",
)
def test_generate_pdf_report():
    """Generate the LaTeX PDF report with all outputs and visualizations.

    Requires pdflatex (system package, e.g. texlive). Skipped if not available.
    Run the visualization tests first to populate the PDFs:
        pytest tests/test_output_aesthetics.py -v
    """
    # Ensure vis PDFs exist by generating them if needed
    _ensure_vis_pdfs_exist()

    # Build LaTeX source
    tex_content = _build_latex_report()

    with open(TEX_PATH, "w") as f:
        f.write(tex_content)

    # Compile PDF (run twice for TOC)
    for _ in range(2):
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                TEST_OUTPUTS_DIR,
                TEX_PATH,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert os.path.exists(PDF_PATH), f"PDF not generated. pdflatex stderr:\n{result.stderr[-2000:]}"
    pdf_size = os.path.getsize(PDF_PATH)
    assert pdf_size > 10000, f"PDF too small ({pdf_size} bytes)"

    # Clean up LaTeX auxiliary files
    for ext in [".aux", ".log", ".out", ".toc"]:
        aux_path = opj(TEST_OUTPUTS_DIR, f"aesthetic_report{ext}")
        if os.path.exists(aux_path):
            os.remove(aux_path)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _vis(
    model, x, filename, vis_opt="unrolled", depth=1000, direction="bottomup", buffer_layers=False
):
    """Generate a single visualization PDF."""
    show_model_graph(
        model,
        x,
        vis_opt=vis_opt,
        vis_nesting_depth=depth,
        vis_outpath=opj(VIS_DIR, filename),
        save_only=True,
        vis_fileformat="pdf",
        vis_buffer_layers=buffer_layers,
        vis_direction=direction,
        random_seed=42,
    )


def _vis_gradient(model, x, filename, vis_opt="unrolled", depth=1000, direction="bottomup"):
    """Generate a visualization PDF with gradient backward arrows.

    Uses log_forward_pass(save_gradients=True) + backward() + render_graph()
    since show_model_graph() hardcodes save_gradients=False.
    """
    log = log_forward_pass(model, x, save_gradients=True, random_seed=42)
    output = log[log.output_layers[0]].tensor_contents
    output.sum().backward()
    log.render_graph(
        vis_opt=vis_opt,
        vis_nesting_depth=depth,
        vis_outpath=opj(VIS_DIR, filename),
        save_only=True,
        vis_fileformat="pdf",
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Tests: visualizations
# ---------------------------------------------------------------------------


def test_aesthetic_deep_nested_visualizations():
    """Nesting depth + direction variations."""
    model = example_models.AestheticDeepNested()
    x = torch.rand(1, 8)

    # depth=0 triggers IndexError in vis.py (edge case, not a valid user value)
    _vis(model, x, "deep_nested_depth1", depth=1)
    _vis(model, x, "deep_nested_depth2", depth=2)
    _vis(model, x, "deep_nested_depth3", depth=3)
    _vis(model, x, "deep_nested_full", depth=1000)
    _vis(model, x, "deep_nested_topdown", depth=1000, direction="topdown")
    _vis(model, x, "deep_nested_leftright", depth=1000, direction="leftright")


def test_aesthetic_shared_module_visualizations():
    """Rolled vs unrolled."""
    model = example_models.AestheticSharedModule()
    x = torch.rand(1, 8)

    _vis(model, x, "shared_unrolled")
    _vis(model, x, "shared_rolled", vis_opt="rolled")
    _vis(model, x, "shared_rolled_depth1", vis_opt="rolled", depth=1)


def test_aesthetic_buffer_branch_visualizations():
    """Buffer toggle + direction."""
    model = example_models.AestheticBufferBranch()
    x = torch.rand(1, 8)

    _vis(model, x, "buffer_visible", buffer_layers=True)
    _vis(model, x, "buffer_hidden", buffer_layers=False)
    _vis(model, x, "buffer_topdown", buffer_layers=True, direction="topdown")
    _vis(model, x, "buffer_leftright", buffer_layers=True, direction="leftright")


def test_aesthetic_kitchen_sink_visualizations():
    """Combined features."""
    model = example_models.AestheticKitchenSink()
    x = torch.rand(1, 8)

    _vis(model, x, "kitchen_sink_unrolled", buffer_layers=True)
    _vis(model, x, "kitchen_sink_rolled", vis_opt="rolled", buffer_layers=True)
    _vis(model, x, "kitchen_sink_depth1", depth=1, buffer_layers=True)
    _vis(model, x, "kitchen_sink_depth2", depth=2, buffer_layers=True)


def test_aesthetic_frozen_mix_visualizations():
    """Frozen + trainable params."""
    model = example_models.AestheticFrozenMix()
    x = torch.rand(1, 8)

    _vis(model, x, "frozen_mix_unrolled")
    _vis(model, x, "frozen_mix_topdown", direction="topdown")


def test_aesthetic_loop_visualizations():
    """Loop detection models — rolled and unrolled."""
    x_5d = torch.rand(5, 5)

    # RecurrentParamsSimple: basic loop with shared params
    model = example_models.RecurrentParamsSimple()
    _vis(model, x_5d, "loop_simple_unrolled")
    _vis(model, x_5d, "loop_simple_rolled", vis_opt="rolled")

    # LoopingParamsDoubleNested: double-nested loops
    model = example_models.LoopingParamsDoubleNested()
    _vis(model, x_5d, "loop_double_nested_unrolled")
    _vis(model, x_5d, "loop_double_nested_rolled", vis_opt="rolled")

    # NestedModules: deep module nesting with recurrence
    model = example_models.NestedModules()
    _vis(model, x_5d, "nested_modules_unrolled")
    _vis(model, x_5d, "nested_modules_rolled", vis_opt="rolled")
    _vis(model, x_5d, "nested_modules_depth1", depth=1)
    _vis(model, x_5d, "nested_modules_depth2", depth=2)


def test_aesthetic_gradient_visualizations():
    """Gradient backward arrows (blue) — requires save_gradients=True + backward().

    Note: gradient edges only render in unrolled mode.
    TODO: Consider supporting gradient arrows in rolled mode.
    """
    # DeepNested: simple nested model with gradients
    model = example_models.AestheticDeepNested()
    x = torch.rand(1, 8)
    _vis_gradient(model, x, "gradient_deep_nested")
    _vis_gradient(model, x, "gradient_deep_nested_topdown", direction="topdown")

    # FrozenMix: frozen params shouldn't have gradient arrows, trainable ones should
    model = example_models.AestheticFrozenMix()
    x = torch.rand(1, 8)
    _vis_gradient(model, x, "gradient_frozen_mix")

    # KitchenSink: gradients with nesting, branching, buffers
    model = example_models.AestheticKitchenSink()
    x = torch.rand(1, 8)
    _vis_gradient(model, x, "gradient_kitchen_sink")
    _vis_gradient(model, x, "gradient_kitchen_sink_depth1", depth=1)
