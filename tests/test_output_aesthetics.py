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
import traceback
from os.path import join as opj

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
