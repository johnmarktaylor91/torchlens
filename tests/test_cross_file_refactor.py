"""Guards against reintroducing ModelLog class-body method rebinding."""

import ast
import inspect

from torchlens.data_classes import model_log as model_log_module


EXPECTED_MODELLOG_METHODS = {
    "render_graph",
    "render_dagua_graph",
    "to_dagua_graph",
    "visualization_field_audit",
    "to_pandas",
    "to_csv",
    "to_parquet",
    "to_json",
    "save_new_activations",
    "validate_saved_activations",
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
    """ModelLog must define its method surface with explicit ``def`` statements."""
    source = inspect.getsource(model_log_module)
    tree = ast.parse(source)
    class_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ModelLog"
    ]
    assert class_defs, "ModelLog class not found"

    model_log_class = class_defs[0]
    defined_methods = {
        node.name
        for node in model_log_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_methods = EXPECTED_MODELLOG_METHODS - defined_methods
    assert not missing_methods, f"ModelLog is missing explicit defs: {sorted(missing_methods)}"

    for body_node in model_log_class.body:
        if isinstance(body_node, ast.Assign) and isinstance(body_node.value, ast.Name):
            targets = ", ".join(ast.unparse(target) for target in body_node.targets)
            raise AssertionError(
                f"ModelLog class body contains attribute rebinding "
                f"'{targets} = {body_node.value.id}'"
            )
