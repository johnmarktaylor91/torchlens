"""``module(address) & facet``: folding a module scope into a facet selector (RC-252).

``_compose`` in :mod:`torchlens._selection.selector_base` calls ``_fold_module_facet``
before building an ``&`` composite. Kept beside ``selector_base.py`` so that module
stays within its size cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import SelectorCompositionError

if TYPE_CHECKING:
    from .selector_base import BaseSelector


def _fold_module_facet(left: Any, right: Any) -> BaseSelector | None:
    """Fold ``module(address) & facet`` (either order) into the scoped facet selector.

    Returns ``None`` when the operands are not one module selector and one
    facet selector. A facet selector already scoped to another module cannot
    also lie in this one.

    Raises
    ------
    SelectorCompositionError
        ``selector_module_facet_scope_conflict`` when the facet selector is
        scoped to a different module address.
    """

    kinds = {
        getattr(left, "selector_kind", None): left,
        getattr(right, "selector_kind", None): right,
    }
    module, facet = kinds.get("module"), kinds.get("facet")
    if module is None or facet is None or len(kinds) != 2:
        return None
    address = str(module.selector_value)
    scoped = getattr(facet, "module_address", None)
    if scoped is not None and scoped != address:
        raise SelectorCompositionError(
            f"tl.module({address!r}) & a facet selector already scoped to {scoped!r} names "
            "no site: a facet lives in one module.",
            code="selector_module_facet_scope_conflict",
            remedy="drop one of the two module scopes",
            module_address=address,
            facet_module_address=scoped,
        )
    return type(facet)(facet.name, head_index=facet.head_index, module_address=address)
