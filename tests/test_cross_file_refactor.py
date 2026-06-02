"""Guards against reintroducing Trace class-body method rebinding."""

import ast
import inspect

import torchlens.data_classes.trace as trace_module


EXPECTED_MODELLOG_METHODS = {
    "draw",
    "render_dagua_graph",
    "to_dagua_graph",
    "visualization_field_audit",
    "to_pandas",
    "save_new_outs",
    "validate_saved_outs",
    "validate_forward_pass",
    "check_metadata_invariants",
    "cleanup",
    "release_param_refs",
    "_postprocess",
    "_run_and_log_inputs_through_model",
    "_remove_log_entry",
    "_batch_remove_log_entries",
}


def test_modellog_has_no_class_body_attribute_rebindings() -> None:
    """Trace must define its method surface with explicit ``def`` statements."""
    source = inspect.getsource(trace_module)
    tree = ast.parse(source)
    class_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "Trace"
    ]
    assert class_defs, "Trace class not found"

    trace_class = class_defs[0]
    defined_custom_methods = {
        node.name
        for node in trace_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_custom_methods = EXPECTED_MODELLOG_METHODS - defined_custom_methods
    assert not missing_custom_methods, (
        f"Trace is missing explicit defs: {sorted(missing_custom_methods)}"
    )

    for body_node in trace_class.body:
        if isinstance(body_node, ast.Assign) and isinstance(body_node.value, ast.Name):
            targets = ", ".join(ast.unparse(target) for target in body_node.targets)
            raise AssertionError(
                f"Trace class body contains attribute rebinding '{targets} = {body_node.value.id}'"
            )
