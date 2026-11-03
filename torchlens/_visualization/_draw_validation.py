"""Draw-option validation (extracted from ``_render_dot`` per the file-size ratchet).

The two closed-vocabulary validators run at the top of ``draw()`` before
any render work. Extracted verbatim at the L5 wave-1 merge to keep
``_render_dot.py`` under its ratchet ceiling (offload preferred over a
ceiling raise); behavior unchanged.
"""

from __future__ import annotations

from .._errors import InvalidArgumentError
from .._literals import (
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisInterventionModeLiteral,
    VisNodeModeLiteral,
)
from .modes import MODE_REGISTRY
from .request import ShowContainersLiteral


def _validate_draw_options(
    node_mode: VisNodeModeLiteral,
    intervention_mode: VisInterventionModeLiteral,
    collapse: CollapseLiteral,
    fold_repeats: FoldRepeatsLiteral,
) -> None:
    """Validate the closed-vocabulary forward-render options.

    Raises
    ------
    ValueError
        If any option falls outside its supported vocabulary.
    """

    if node_mode not in MODE_REGISTRY:
        raise InvalidArgumentError(
            "Visualization node_style/node_mode must be one of 'default' or "
            f"'profiling'; received {node_mode!r}",
            code="visualization_node_style_invalid",
            remedy=(
                "pass node_style='default' or 'profiling'; domain styles moved to "
                "torchlens.experimental.node_styles.<style>_node_mode via node_spec_fn"
            ),
            argument="node_style",
        )
    if intervention_mode not in {"node_mark", "as_node"}:
        raise InvalidArgumentError(
            "vis_intervention_mode must be either 'node_mark' or 'as_node'; "
            f"received {intervention_mode!r}",
            code="visualization_intervention_mode_invalid",
            remedy="pass vis_intervention_mode='node_mark' or 'as_node'",
            argument="vis_intervention_mode",
        )
    if isinstance(collapse, float):
        if not 0.0 <= collapse <= 1.0:
            raise InvalidArgumentError(
                f"collapse float level must be in [0.0, 1.0]; received {collapse!r}",
                code="collapse_level_invalid",
                remedy="pass a collapse level between 0.0 and 1.0",
                argument="collapse",
            )
    elif collapse not in {"none", "auto", "max"}:
        raise InvalidArgumentError(
            "collapse must be 'none', 'auto', 'max', or a float in [0.0, 1.0]; "
            f"received {collapse!r}",
            code="collapse_mode_invalid",
            remedy="pass collapse='none', 'auto', 'max', or an in-range float",
            argument="collapse",
        )
    if fold_repeats not in {None, True, False}:
        raise InvalidArgumentError(
            f"fold_repeats must be None, True, or False; received {fold_repeats!r}",
            code="fold_repeats_invalid",
            remedy="pass fold_repeats=None, True, or False",
            argument="fold_repeats",
        )


# The non-``False`` ``show_containers`` vocabulary (``Trace.draw`` literal).
_SHOW_CONTAINERS_MODES = ("labels", "collapsed", "nodes")

# Removed spellings and the setting that draws what each one drew: ``"auto"`` never
# chose anything and was byte-identical to ``"collapsed"``; ``"cluster"`` equalled
# ``"labels"`` except for the dotted tuple box, which ``"nodes"`` also draws.
_REMOVED_SHOW_CONTAINERS = {"auto": "collapsed", "cluster": "labels"}


def _validate_draw_flag_options(
    show_containers: ShowContainersLiteral,
    **bool_options: object,
) -> None:
    """Validate the bool-typed public draw kwargs and ``show_containers``.

    R64-F3: strings such as ``order_siblings='yes'`` were accepted silently,
    and ``'no'``/``'false'`` truthily meant ON. Only real bools are accepted;
    ``show_containers`` additionally allows its closed string vocabulary.

    Raises
    ------
    ValueError
        If any flag is not a real bool, or ``show_containers`` falls outside
        its supported vocabulary.
    """

    for name, value in bool_options.items():
        if name == "show_legend" and value is None:
            # Tri-state: None = AUTO (L5 channel core) is a legal value on
            # this one flag; True/False keep their historical meanings.
            continue
        if not isinstance(value, bool):
            raise InvalidArgumentError(
                f"{name} must be a bool (True or False); received {value!r}",
                code="visualization_bool_option_invalid",
                remedy=f"pass {name}=True or {name}=False",
                argument=name,
            )
    if show_containers is False or show_containers in _SHOW_CONTAINERS_MODES:
        return
    problem, remedy = _show_containers_refusal(show_containers)
    raise InvalidArgumentError(
        problem,
        code="visualization_show_containers_invalid",
        remedy=remedy,
        argument="show_containers",
    )


def _show_containers_refusal(show_containers: object) -> tuple[str, str]:
    """Return the problem and remedy for a ``show_containers`` value outside the vocabulary.

    A removed spelling (``"auto"``, ``"cluster"``) names the setting that draws what it
    drew; anything else lists the vocabulary.

    Parameters
    ----------
    show_containers:
        The refused value.

    Returns
    -------
    tuple[str, str]
        Problem sentence and remedy.
    """

    if isinstance(show_containers, str) and show_containers in _REMOVED_SHOW_CONTAINERS:
        replacement = _REMOVED_SHOW_CONTAINERS[show_containers]
        extra = " (or 'nodes' for the dotted tuple box)" if replacement == "labels" else ""
        return (
            f"show_containers={show_containers!r} was removed; it drew the same picture as "
            f"{replacement!r}",
            f"pass show_containers={replacement!r}{extra}",
        )
    modes = ", ".join(repr(mode) for mode in _SHOW_CONTAINERS_MODES)
    return (
        f"show_containers must be False or one of {modes}; received {show_containers!r}",
        f"pass show_containers=False or one of {modes}",
    )
