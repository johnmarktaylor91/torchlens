"""Direct tensor gradient-hook install, admitted by a probe (a torch-only leaf of ``_torch_compat``).

``torch.Tensor.register_hook`` is Python, and during capture its reads of
``requires_grad``, ``_backward_hooks`` and ``grad_fn`` go through names capture
shadows on ``torch.Tensor``, so each read runs a TorchLens frame. The functions
:func:`detect` builds take the same steps through the C class's getsets.

``_torch_compat.get_direct_tensor_hook_install`` resolves torch's private parts,
passes them in and latches ``HAS_DIRECT_TENSOR_HOOK_INSTALL`` at first use, never
at import. :func:`detect` admits the direct install only when every part is the
expected kind, torch's ``register_hook`` body is :data:`REGISTER_HOOK_BODY`, and a
probe on scratch tensors shows both routes leave one hook dict per tensor, register
it with the tensor's node, key the hooks in registration order, fire them in that
order and refuse a tensor without gradient alike. Anything else returns ``None``,
and capture keeps ``register_hook``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

__all__ = ("REGISTER_HOOK_BODY", "DirectTensorHookInstall", "detect", "register_hook_body_matches")

Hook = Callable[[Any], Any]

#: The statements of ``torch.Tensor.register_hook`` the direct install repeats. Torch
#: spells the handle class ``hooks.RemovableHandle`` through 2.7 and imports it inside
#: the method from 2.8; both name ``torch.utils.hooks.RemovableHandle``, so the body is
#: compared after that one rewrite and with imports and the docstring dropped.
REGISTER_HOOK_BODY = """
if has_torch_function_unary(self):
    return handle_torch_function(Tensor.register_hook, (self,), self, hook)
if not self.requires_grad:
    raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
if self._backward_hooks is None:
    self._backward_hooks = OrderedDict()
    if self.grad_fn is not None:
        self.grad_fn._register_hook_dict(self)
handle = RemovableHandle(self._backward_hooks)
self._backward_hooks[handle.id] = hook
return handle
"""

_NO_GRAD_MESSAGE = "cannot register a hook on a tensor that doesn't require gradient"

# What a probe raises when a part is missing or behaves differently; any of them
# means "use register_hook", the safe direction.
_PROBE_ERRORS = (
    AttributeError,
    IndexError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
)


class DirectTensorHookInstall(NamedTuple):
    """The direct install's three entry points, built over torch's own parts."""

    takes_plain: Callable[[Any], bool]
    """Whether a tensor is a plain ``torch.Tensor`` with no ``__torch_function__`` override."""
    register: Callable[[Any, Hook], Any]
    """Register a hook as ``register_hook`` does and return torch's ``RemovableHandle``."""
    add: Callable[[Any, Hook], bool]
    """Add a hook to a plain tensor keeping no handle; ``False`` (nothing added) otherwise."""


class _HandleFree(ast.NodeTransformer):
    """Rewrite ``hooks.RemovableHandle`` to ``RemovableHandle`` (same class, older spelling)."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Return ``RemovableHandle`` for ``hooks.RemovableHandle``, else the visited node."""

        if (
            node.attr == "RemovableHandle"
            and isinstance(node.value, ast.Name)
            and node.value.id == "hooks"
        ):
            return ast.copy_location(ast.Name(id="RemovableHandle", ctx=node.ctx), node)
        return self.generic_visit(node)


def _body_dump(statements: list[ast.stmt]) -> str:
    """Return the AST dump of ``statements`` without imports and a leading docstring."""

    kept = [stmt for stmt in statements if not isinstance(stmt, (ast.Import, ast.ImportFrom))]
    if kept and isinstance(kept[0], ast.Expr) and isinstance(kept[0].value, ast.Constant):
        kept = kept[1:]
    module = _HandleFree().visit(ast.Module(body=kept, type_ignores=[]))
    return ast.dump(module, annotate_fields=False)


def register_hook_body_matches(method: Any) -> bool:
    """Return whether ``method``'s source is :data:`REGISTER_HOOK_BODY`, statement for statement.

    Parameters
    ----------
    method:
        ``torch.Tensor.register_hook`` (a test may pass another function).

    Returns
    -------
    bool
        ``False`` too when the source cannot be read (no ``.py`` shipped).
    """

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError, SyntaxError):
        return False
    func = tree.body[0] if tree.body else None
    if not isinstance(func, ast.FunctionDef):
        return False
    return _body_dump(func.body) == _body_dump(ast.parse(REGISTER_HOOK_BODY).body)


def _build(
    getsets: tuple[Any, Any, Any], has_torch_function_unary: Callable[[Any], bool], handle_cls: Any
) -> DirectTensorHookInstall:
    """Return the direct install over the ``_backward_hooks``, ``grad_fn``, ``requires_grad`` getsets."""

    hooks_of, set_hooks = getsets[0].__get__, getsets[0].__set__
    grad_fn_of, requires_grad_of = getsets[1].__get__, getsets[2].__get__
    tensor_cls = torch.Tensor

    def hooks_dict(t: Any) -> OrderedDict[int, Hook]:
        if not requires_grad_of(t):
            raise RuntimeError(_NO_GRAD_MESSAGE)
        hooks = hooks_of(t)
        if hooks is None:
            hooks = OrderedDict()
            set_hooks(t, hooks)
            grad_fn = grad_fn_of(t)
            if grad_fn is not None:
                grad_fn._register_hook_dict(t)
        return hooks

    def takes_plain(t: Any) -> bool:
        return type(t) is tensor_cls and not has_torch_function_unary(t)

    def register(t: Any, hook: Hook) -> Any:
        hooks = hooks_dict(t)
        handle = handle_cls(hooks)
        hooks[handle.id] = hook
        return handle

    def add(t: Any, hook: Hook) -> bool:
        if type(t) is not tensor_cls or has_torch_function_unary(t):
            return False
        hooks = hooks_dict(t)
        hook_id = handle_cls.next_id
        handle_cls.next_id = hook_id + 1
        hooks[hook_id] = hook
        return True

    return DirectTensorHookInstall(takes_plain, register, add)


def _parts_have_expected_kinds(
    getsets: tuple[Any, Any, Any], has_torch_function_unary: Any, handle_cls: Any
) -> bool:
    """Return whether every part is the kind the direct install calls."""

    return (
        all(isinstance(getset, types.GetSetDescriptorType) for getset in getsets)
        and callable(has_torch_function_unary)
        and isinstance(handle_cls, type)
        and type(getattr(handle_cls, "next_id", None)) is int
    )


def _recorder(order: list[str], name: str, scale: float) -> Hook:
    """Return a hook that records ``name`` and scales the gradient by ``scale``."""

    def hook(grad: Any) -> Any:
        order.append(name)
        return grad * scale

    return hook


def _fire(
    direct: DirectTensorHookInstall, hooks_of: Callable[[Any], Any], route: str, leaf: bool
) -> tuple[list[str], list[Any]]:
    """Hook one scratch tensor three times through ``route`` and run its backward once."""

    x = torch.ones(2, requires_grad=True)
    y = x if leaf else x * 2
    order: list[str] = []
    hooks = [_recorder(order, name, scale) for name, scale in (("a", 2.0), ("b", 3.0), ("c", 5.0))]
    if route == "direct":
        direct.register(y, hooks[0])
        if not direct.add(y, hooks[1]):
            raise RuntimeError("direct add refused a plain tensor")
    else:
        y.register_hook(hooks[0])
        y.register_hook(hooks[1])
    removable = y.register_hook(lambda grad: grad * 7.0)
    y.register_hook(hooks[2])
    removable.remove()
    keys = list(hooks_of(y))
    if keys != sorted(keys) or len(keys) != 3:
        raise RuntimeError("hook keys are not in registration order")
    (grad,) = torch.autograd.grad((y * 1.0).sum(), x)
    return order, grad.tolist()


def _routes_agree(direct: DirectTensorHookInstall, hooks_of: Callable[[Any], Any]) -> bool:
    """Return whether both routes behave alike on scratch tensors (leaf and non-leaf)."""

    for leaf in (False, True):
        if _fire(direct, hooks_of, "direct", leaf) != _fire(direct, hooks_of, "torch", leaf):
            return False
    try:
        direct.register(torch.ones(1), lambda grad: grad)
    except RuntimeError as error:
        return str(error) == _NO_GRAD_MESSAGE and direct.takes_plain(torch.ones(1))
    return False


def detect(
    getsets: tuple[Any, Any, Any], has_torch_function_unary: Any, handle_cls: Any
) -> DirectTensorHookInstall | None:
    """Return the direct install when torch's parts, method body and behaviour all match.

    Parameters
    ----------
    getsets:
        The C tensor class's ``_backward_hooks``, ``grad_fn`` and ``requires_grad``
        getset descriptors (``None`` where absent).
    has_torch_function_unary:
        ``torch._C._has_torch_function_unary``, or ``None`` when absent.
    handle_cls:
        ``torch.utils.hooks.RemovableHandle``, or ``None`` when absent.

    Returns
    -------
    DirectTensorHookInstall | None
        ``None`` means capture uses ``Tensor.register_hook``.
    """

    if not _parts_have_expected_kinds(getsets, has_torch_function_unary, handle_cls):
        return None
    if not register_hook_body_matches(torch.Tensor.register_hook):
        return None
    direct = _build(getsets, has_torch_function_unary, handle_cls)
    try:
        return direct if _routes_agree(direct, getsets[0].__get__) else None
    except _PROBE_ERRORS:
        return None
