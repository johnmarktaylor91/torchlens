"""Legacy tensor constructor classes whose ``__new__`` the torch backend patches.

Split out of ``_constants``; ``backends/torch/legacy_ctors.py`` installs and
restores the patches. Dependency-free basis module.
"""

from __future__ import annotations

__tl_layer__ = "L0"

# Legacy tensor constructors (2017-2019 idioms): ``torch.FloatTensor(size)``,
# ``torch.LongTensor(data)``, a module attribute bound to the class at init
# (``self.FloatTensor = torch.FloatTensor``), and ``torch.autograd.Variable(t)``.
# Their C ``tp_new`` dispatches ``aten.empty`` / ``aten.detach`` with no Python
# wrapper on the stack, so a forward that uses them left an unowned dispatch and
# failed validation completeness. The CLASS OBJECTS are patched in place (their
# ``__new__``), never the namespace attributes: identity, ``isinstance`` and
# ``x.type(torch.FloatTensor)`` keep working, and an alias bound before the
# first capture reaches the same patched class. Rows are (namespace, class
# name); ``backends/torch/legacy_ctors.py`` installs and restores them.
# ``torch.Tensor`` itself is not listed: ``torch.Tensor.__new__`` is already an
# ``IGNORED_FUNCS`` roster row.
LEGACY_TENSOR_DTYPE_CLASS_NAMES: tuple[str, ...] = (
    "FloatTensor",
    "DoubleTensor",
    "HalfTensor",
    "BFloat16Tensor",
    "LongTensor",
    "IntTensor",
    "ShortTensor",
    "CharTensor",
    "ByteTensor",
    "BoolTensor",
)
LEGACY_TENSOR_CONSTRUCTOR_SITES: tuple[tuple[str, str], ...] = (
    *(("torch", name) for name in LEGACY_TENSOR_DTYPE_CLASS_NAMES),
    *(("torch.cuda", name) for name in LEGACY_TENSOR_DTYPE_CLASS_NAMES),
    ("torch.autograd", "Variable"),
)
