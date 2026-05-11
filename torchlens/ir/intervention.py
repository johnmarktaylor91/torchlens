"""Intervention-related IR records for live-hook capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .events import ModuleFrame


@dataclass(frozen=True, slots=True)
class FunctionEventInput:
    """Backend wrapper input bundle for function-output emission."""

    func: object
    func_name: str
    func_qualname: str | None
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    raw_output: object | None
    arg_copies: tuple[object, ...] | None
    kwarg_copies: Mapping[str, object] | None
    module_stack: tuple[ModuleFrame, ...]
    is_bottom_level_func: bool
    func_call_id: int
    expected_output_count: int


@dataclass(frozen=True, slots=True)
class FireResult:
    """Normalized result of a live intervention hook."""

    plan_id: str
    site_label: str
    fired_at_capture_index: int
    pre_hook_shape: tuple[int, ...] | None
    post_hook_shape: tuple[int, ...] | None
    pre_hook_dtype: str | None
    post_hook_dtype: str | None
    replaced: bool
    fire_record: object | None


@dataclass(frozen=True, slots=True)
class InterventionTemplateRef:
    """Reference to an intervention template used during capture."""

    template_id: str
    spec_revision: int
    template_kind: str
    template_args: tuple[object, ...]
