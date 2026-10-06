"""Value validators for the grouped option dataclasses in ``options.py``.

Split out of ``torchlens/options.py`` (2026-08-26 P03 fix cycle) along the
validation seam: every function here checks or normalizes ONE resolved option
value and raises a typed, teaching refusal on violation. The construction
plumbing (explicitness tracking, frozen-field assembly, grouped/flat merging)
stays with the dataclasses in ``options.py``; this module never imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._errors import ArgumentTypeError, InvalidArgumentError
from ._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisInterventionModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
)


def _validate_node_style(node_style: VisNodeModeLiteral) -> None:
    """Validate a visualization node-style preset name.

    Parameters
    ----------
    node_style:
        Candidate node-style preset name.

    Raises
    ------
    ValueError
        If ``node_style`` is not a registered public preset.
    """

    if node_style not in {"default", "profiling"}:
        raise InvalidArgumentError(
            f"Visualization node_style={node_style!r} is not a supported preset",
            code="visualization_node_style_invalid",
            remedy=(
                "set node_style to 'default' or 'profiling'; domain styles moved to "
                "torchlens.experimental.node_styles.<style>_node_mode via node_spec_fn"
            ),
            argument="node_style",
        )


def _normalize_layout(layout: VisNodePlacementLiteral) -> VisNodePlacementLiteral:
    """Normalize visualization layout names and warn on removed backends.

    Parameters
    ----------
    layout:
        Candidate layout engine name.

    Returns
    -------
    VisNodePlacementLiteral
        Normalized layout engine name.

    Raises
    ------
    ValueError
        If ``layout`` is not supported.
    """

    if layout not in {"auto", "dot", "rank"}:
        raise InvalidArgumentError(
            f"Visualization layout={layout!r} is not supported",
            code="visualization_layout_invalid",
            remedy="set layout to 'auto', 'dot', or 'rank'",
            argument="layout",
        )
    return layout


def _validate_intervention_mode(intervention_mode: VisInterventionModeLiteral) -> None:
    """Validate an intervention visualization mode name.

    Parameters
    ----------
    intervention_mode:
        Candidate intervention visualization mode.

    Raises
    ------
    ValueError
        If ``intervention_mode`` is not a supported mode.
    """

    if intervention_mode not in {"node_mark", "as_node"}:
        raise InvalidArgumentError(
            f"Visualization intervention_mode={intervention_mode!r} is not supported",
            code="visualization_intervention_mode_invalid",
            remedy="set intervention_mode to 'node_mark' or 'as_node'",
            argument="intervention_mode",
        )


def _validate_buffer_visibility(value: BufferVisibilityLiteral) -> None:
    """Validate buffer visibility options.

    Parameters
    ----------
    value:
        Tri-state buffer visibility mode.

    Raises
    ------
    ValueError
        If ``value`` is not a supported tri-state mode. The former legacy
        bools refuse here too: pass ``'always'`` / ``'never'``.
    """

    if value in {"never", "meaningful", "always"}:
        return
    raise InvalidArgumentError(
        f"Visualization show_buffers={value!r} is not a supported visibility policy",
        code="buffer_visibility_invalid",
        remedy="set show_buffers to 'never', 'meaningful', or 'always'",
        argument="show_buffers",
    )


def _validate_collapse(value: CollapseLiteral) -> None:
    """Validate smart module-collapse mode.

    Parameters
    ----------
    value:
        Candidate collapse mode.

    Raises
    ------
    ValueError
        If ``value`` is not a supported collapse mode.
    """

    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return
        raise InvalidArgumentError(
            f"Visualization collapse={value!r} is outside the supported float range",
            code="collapse_level_invalid",
            remedy="set collapse to a float from 0.0 through 1.0 inclusive",
            argument="collapse",
        )
    if value not in {"none", "auto", "max"}:
        raise InvalidArgumentError(
            f"Visualization collapse={value!r} is not a supported collapse mode",
            code="collapse_mode_invalid",
            remedy="set collapse to 'none', 'auto', 'max', or a float in [0.0, 1.0]",
            argument="collapse",
        )


def _validate_fold_repeats(value: FoldRepeatsLiteral) -> None:
    """Validate explicit repeat-fold rendering policy.

    Parameters
    ----------
    value:
        Candidate repeat-fold policy.

    Raises
    ------
    ValueError
        If ``value`` is not ``None``, ``True``, or ``False``.
    """

    if value not in {None, True, False}:
        raise InvalidArgumentError(
            f"Visualization fold_repeats={value!r} is not a supported policy",
            code="fold_repeats_invalid",
            remedy="set fold_repeats to None, True, or False",
            argument="fold_repeats",
        )


# Visualization option fields that accept ONLY real bools. Strings such as
# ``'yes'`` or ``'no'`` used to be accepted silently, and ``'no'``/``'false'``
# truthily meant ON (R64-F3).
_VISUALIZATION_BOOL_FIELDS = (
    "save_only",
    "show_cone",
    "for_paper",
    "return_graph",
    "order_siblings",
)

# Tri-state (bool | None) flags: show_legend=None = AUTO (L5 channel core).
_VISUALIZATION_TRI_STATE_BOOL_FIELDS = ("show_legend",)


def _validate_visualization_flag_fields(values: Mapping[str, Any]) -> None:
    """Validate the bool-only and tri-state visualization flag fields."""

    for bool_field in _VISUALIZATION_BOOL_FIELDS:
        _validate_bool_option(bool_field, values[bool_field])
    for tri_state_field in _VISUALIZATION_TRI_STATE_BOOL_FIELDS:
        value = values[tri_state_field]
        if value is None or isinstance(value, bool):
            continue
        raise InvalidArgumentError(
            f"{tri_state_field} must be True, False, or None (auto); received {value!r}",
            code="visualization_bool_option_invalid",
            remedy=f"pass {tri_state_field}=True, False, or None",
            argument=tri_state_field,
        )


def _validate_bool_option(name: str, value: Any) -> None:
    """Validate a bool-only visualization option value.

    Parameters
    ----------
    name:
        Public option field name for the diagnostic.
    value:
        Candidate option value.

    Raises
    ------
    ValueError
        If ``value`` is not a real ``bool``.
    """

    if not isinstance(value, bool):
        raise InvalidArgumentError(
            f"Visualization {name}={value!r} is not a bool",
            code="visualization_bool_option_invalid",
            remedy=f"set {name} to True or False",
            argument=name,
        )


def _validate_output_device(output_device: Any) -> None:
    """Validate an ``output_device`` save option value.

    Shared by the capture entry (``tl.trace``) and ``tl.validate`` so both
    refuse the same values with the same typed error.

    Parameters
    ----------
    output_device:
        Candidate ``output_device`` value.

    Raises
    ------
    InvalidArgumentError
        (``code="output_device_invalid"``) if the value is not ``"same"``,
        ``"cpu"``, or ``"cuda"``.
    """

    if output_device not in ["same", "cpu", "cuda"]:
        raise InvalidArgumentError(
            f"output_device={output_device!r} is not supported",
            code="output_device_invalid",
            remedy="set output_device to 'same', 'cpu', or 'cuda'",
            argument="output_device",
        )


def _validate_capture_values(values: Mapping[str, Any]) -> None:
    """Validate resolved capture field values.

    Single source of truth for the invariants enforced on
    :class:`CaptureOptions`, shared by ``__init__`` and ``from_values`` so the
    flat-kwarg / ``from_values`` construction path can never accept values the
    grouped constructor rejects (the asymmetry that let bad ``jax_control_flow``
    and non-positive ``jax_max_control_flow_unroll`` slip through).

    Parameters
    ----------
    values:
        Resolved capture field values keyed by canonical field name.

    Raises
    ------
    ValueError
        If ``jax_control_flow`` or ``jax_max_control_flow_unroll`` holds an
        unsupported value.
    TypeError
        If ``jax_max_control_flow_unroll`` is not an integer.
    """

    if values["jax_control_flow"] not in {"reject", "unroll", "region"}:
        raise InvalidArgumentError(
            f"Capture option jax_control_flow={values['jax_control_flow']!r} is unsupported",
            code="jax_control_flow_invalid",
            remedy="set jax_control_flow to 'reject', 'unroll', or 'region'",
            argument="jax_control_flow",
        )
    if not isinstance(values["jax_max_control_flow_unroll"], int):
        raise ArgumentTypeError(
            "Capture option jax_max_control_flow_unroll is not an integer",
            code="jax_unroll_type_invalid",
            remedy="pass an integer greater than or equal to 1",
            argument="jax_max_control_flow_unroll",
            received_type=type(values["jax_max_control_flow_unroll"]).__name__,
        )
    if values["jax_max_control_flow_unroll"] < 1:
        raise InvalidArgumentError(
            "Capture option jax_max_control_flow_unroll is less than 1",
            code="jax_unroll_range_invalid",
            remedy="set jax_max_control_flow_unroll to an integer greater than or equal to 1",
            argument="jax_max_control_flow_unroll",
        )
    if values["distributed_witness"] not in {"none", "digest", "payload"}:
        raise InvalidArgumentError(
            f"Capture option distributed_witness={values['distributed_witness']!r} is unsupported",
            code="distributed_witness_invalid",
            remedy="set distributed_witness to 'none' or 'digest'",
            argument="distributed_witness",
        )
    if values["distributed_witness"] == "payload":
        raise InvalidArgumentError(
            "Capture option distributed_witness='payload' is reserved because payload "
            "witness blobs are not implemented",
            code="distributed_payload_witness_unsupported",
            remedy="set distributed_witness to 'digest' for byte-exact witness digests",
            argument="distributed_witness",
        )
    if not isinstance(values["structure_only"], bool):
        raise ArgumentTypeError(
            "Capture option structure_only is not a bool",
            code="structure_only_type_invalid",
            remedy="pass structure_only=True or structure_only=False",
            argument="structure_only",
            received_type=type(values["structure_only"]).__name__,
        )
