"""Guards against reintroducing Trace class-body method rebinding."""

import ast
import inspect
from types import ModuleType

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
    """Trace method surface must come from real defs, not rebinding shims."""
    owner_by_method = {
        method_name: _method_owner(method_name) for method_name in EXPECTED_MODELLOG_METHODS
    }
    missing_custom_methods = {
        method_name for method_name, owner in owner_by_method.items() if owner is None
    }
    assert not missing_custom_methods, (
        f"Trace is missing explicit defs: {sorted(missing_custom_methods)}"
    )

    for method_name, owner in owner_by_method.items():
        assert owner is not None
        assert _is_trace_surface_module(owner), (
            f"{method_name} is defined on unexpected owner {owner.__module__}.{owner.__name__}"
        )
        assert _class_defines_method(owner, method_name), (
            f"{owner.__name__}.{method_name} is not an explicit def"
        )

    for cls in trace_module.Trace.__mro__:
        if _is_trace_surface_module(cls):
            _assert_no_class_body_rebindings(cls)


def _method_owner(method_name: str) -> type | None:
    """Return the Trace MRO class that owns ``method_name``.

    Parameters
    ----------
    method_name:
        Method name expected on ``Trace``.

    Returns
    -------
    type | None
        Owning class, or ``None`` when the method is absent.
    """

    for cls in trace_module.Trace.__mro__:
        if method_name in cls.__dict__:
            member = cls.__dict__[method_name]
            if inspect.isfunction(member):
                return cls
            return None
    return None


def _is_trace_surface_module(cls: type) -> bool:
    """Return whether ``cls`` is Trace or one of its extracted mixins.

    Parameters
    ----------
    cls:
        Class to inspect.

    Returns
    -------
    bool
        ``True`` when the class is in the allowed Trace surface modules.
    """

    module_name = cls.__module__
    return module_name == "torchlens.data_classes.trace" or module_name.startswith(
        "torchlens.data_classes._trace_"
    )


def _class_defines_method(cls: type, method_name: str) -> bool:
    """Return whether ``cls`` contains an explicit function def for ``method_name``.

    Parameters
    ----------
    cls:
        Class to inspect.
    method_name:
        Method expected in the class body.

    Returns
    -------
    bool
        ``True`` when the class AST has a matching ``def``.
    """

    class_def = _class_ast(cls)
    return any(
        isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and body_node.name == method_name
        for body_node in class_def.body
    )


def _assert_no_class_body_rebindings(cls: type) -> None:
    """Reject simple class-body attribute rebindings in ``cls``.

    Parameters
    ----------
    cls:
        Class whose body is checked.

    Returns
    -------
    None
        Raises when a rebinding shim is found.
    """

    class_def = _class_ast(cls)
    for body_node in class_def.body:
        if isinstance(body_node, ast.Assign) and isinstance(body_node.value, ast.Name):
            targets = ", ".join(ast.unparse(target) for target in body_node.targets)
            raise AssertionError(
                f"{cls.__name__} class body contains attribute rebinding "
                f"'{targets} = {body_node.value.id}'"
            )


def _class_ast(cls: type) -> ast.ClassDef:
    """Return the parsed AST node for ``cls``.

    Parameters
    ----------
    cls:
        Class to parse from its defining module.

    Returns
    -------
    ast.ClassDef
        Matching class definition AST node.
    """

    module = inspect.getmodule(cls)
    assert isinstance(module, ModuleType)
    source = inspect.getsource(module)
    tree = ast.parse(source)
    class_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    ]
    assert class_defs, f"{cls.__name__} class not found"
    return class_defs[0]
