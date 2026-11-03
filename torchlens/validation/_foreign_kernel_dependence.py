"""Pre-perturbation check for the declared per-output dependence of known foreign kernels.

A known foreign kernel (``backends/torch/_foreign_data._KNOWN_FOREIGN_KERNEL_MODULES``) is
one ``autogradfunction`` boundary op per returned tensor, every op carrying the call's full
parent set. When the kernel returns more than one tensor, an output can be independent of
an argument another output depends on: FlashAttention 4's log-sum-exp (output 1) is a
function of q and k, not v. The kernel table declares that dependence per Function and
output (ruling: declare per-output dependence per kernel in a table), and this check
projects the perturbation check on it: a perturbed parent is excused for ONE output only
when every position it occupies is one the table declares that output independent of.

Fail-closed by construction. Strict (``False``) whenever the op is not a recorded
known-kernel boundary (module, Function name or exit not in the table), the output has no
row, ``multi_output_index`` is missing, a perturbed parent sits at a keyword, a nested or
unmapped position, a position outside the row's ``positions``, or any declared-dependent
position. Output 0 of FlashAttention 4 has no row, so its every edge stays tested.

Registered in ``exemptions.CUSTOM_EXEMPTION_CHECKS`` under ``autogradfunction``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backends.torch._foreign_data import (
    OutputDependence,
    known_foreign_kernel_output_dependence,
)

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def declared_output_dependence(layer: Op) -> OutputDependence | None:
    """Return the table row for this boundary op's output, or None when nothing is declared.

    Parameters
    ----------
    layer:
        An ``autogradfunction`` op record (one output of a Function boundary call).

    Returns
    -------
    OutputDependence | None
        The declared row when the op's ``transform_config`` names a known kernel's module,
        Function and exit and its ``multi_output_index`` has a row; otherwise None.
    """

    config = getattr(layer, "transform_config", None)
    output_index = getattr(layer, "multi_output_index", None)
    if not isinstance(config, dict) or not isinstance(output_index, int):
        return None
    module = config.get("function_module")
    function = config.get("function")
    exits = config.get("exits")
    if not isinstance(module, str) or not isinstance(function, str):
        return None
    if not isinstance(exits, (list, tuple)):
        return None
    return known_foreign_kernel_output_dependence(module, function, output_index, exits)


def _positions_of(layer: Op, label: str) -> list[Any] | None:
    """Return the positional keys ``label`` occupies, or None when it also fills a keyword."""

    positions = getattr(layer, "parent_arg_positions", None) or {}
    if label in (positions.get("kwargs", {}) or {}).values():
        return None
    return [key for key, parent in (positions.get("args", {}) or {}).items() if parent == label]


def check_foreign_kernel_dependence_exempt(
    source_trace: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Custom check: excuse a known kernel output's parent at a declared-independent position only.

    Parameters
    ----------
    source_trace:
        Trace being validated (unused; custom-check signature).
    layer:
        ``autogradfunction`` op record being perturbation-checked.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    bool
        True when the table declares this output independent of every position each
        perturbed parent occupies; False keeps the perturbation strict.
    """

    del source_trace
    row = declared_output_dependence(layer)
    if row is None or not layers_to_perturb:
        return False
    return all(_label_is_declared_independent(row, layer, label) for label in layers_to_perturb)


def _label_is_declared_independent(row: OutputDependence, layer: Op, label: str) -> bool:
    """Return whether every positional key ``label`` occupies is declared independent."""

    keys = _positions_of(layer, label)
    if not keys:
        return False
    return all(isinstance(key, int) and row.independent_of(key) for key in keys)
