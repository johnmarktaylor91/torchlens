"""Name the opaque call behind a completeness-witness gap.

When the completeness witness finds aten dispatches that no wrapper accounted for, and
the call-site line those dispatches came from calls a TorchScript function
(``torch.jit.ScriptFunction``) or hands a tensor out through DLPack (``to_dlpack``),
:func:`advise_opaque_calls` emits one coded advisory that names the call and its line,
says why its internal operations are missing, and gives the spelling that captures them.

Cost and scope: it runs only on the gap path (``_finalize_census`` with at least one
diagnostic), so a capture without unaccounted dispatches pays nothing new. On the gap
path it reads each distinct call-site line once and resolves the called names in the
namespace of the module that defines that line (module globals, and for ``self.`` or
``cls.`` calls the classes of that module that define the calling function). It changes
no verdict, count or diagnostic. A scripted function stored on an instance attribute, or
defined inside a function, is not resolvable this way and gets no advisory.
"""

from __future__ import annotations

import inspect
import linecache
import os
import re
import sys
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from ..._errors import TorchLensCaptureGapWarning

__all__ = ("DLPACK_CODE", "SCRIPTED_CODE", "advise_opaque_calls")

SCRIPTED_CODE = "scripted_function_not_captured"
DLPACK_CODE = "dlpack_call_not_captured"

_GAP_REASONS = frozenset({"unowned_dispatch", "owner_not_captured"})
# Distinct call-site lines examined per capture; a gap spread over more lines is
# advised from its first lines (the witness diagnostics still list every dispatch).
_MAX_SITES = 32
_CALLED_NAME = re.compile(r"([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\(")
_DLPACK_CALL = re.compile(r"\bto_dlpack\s*\(")
_GOTCHA_POINTER = "the TorchScript and DLPack entry in docs/agent-reference/known-gotchas.md"
# Codes already emitted this session (errors.1: each warning code once per session).
_WARNED_CODES: set[str] = set()


@dataclass(frozen=True)
class _CallSite:
    """One source line that unaccounted dispatches came from."""

    file: str
    line: int
    function: str | None
    dispatch_count: int
    source: str


@dataclass(frozen=True)
class _ScriptedCall:
    """A resolved TorchScript function called on a call-site line."""

    name: str
    defined_at: str | None


def advise_opaque_calls(diagnostics: Iterable[Mapping[str, Any]]) -> None:
    """Emit the opaque-call advisories for one capture's witness diagnostics.

    Parameters
    ----------
    diagnostics:
        The completeness witness's unaccounted-dispatch records for one forward
        (``trace.completeness_diagnostics`` entries).

    Returns
    -------
    None
        Emits at most one ``scripted_function_not_captured`` and one
        ``dlpack_call_not_captured`` warning, each once per session.
    """

    pending = {SCRIPTED_CODE, DLPACK_CODE} - _WARNED_CODES
    if not pending:
        return
    scripted: list[tuple[_CallSite, _ScriptedCall]] = []
    dlpack: list[_CallSite] = []
    for site in _gap_call_sites(diagnostics):
        call = _scripted_call(site) if SCRIPTED_CODE in pending else None
        if call is not None:
            scripted.append((site, call))
        elif DLPACK_CODE in pending and _DLPACK_CALL.search(site.source):
            dlpack.append(site)
    if scripted:
        _warn_scripted(scripted)
    if dlpack:
        _warn_dlpack(dlpack)


def _gap_call_sites(diagnostics: Iterable[Mapping[str, Any]]) -> list[_CallSite]:
    """Group gap diagnostics by call-site line, in first-seen order.

    Parameters
    ----------
    diagnostics:
        Witness diagnostics for one forward.

    Returns
    -------
    list[_CallSite]
        At most ``_MAX_SITES`` lines with readable source, with their dispatch counts.
    """

    counts: dict[tuple[str, int], int] = {}
    functions: dict[tuple[str, int], str | None] = {}
    for entry in diagnostics:
        file, line = entry.get("file"), entry.get("line")
        if entry.get("reason") not in _GAP_REASONS or not isinstance(file, str):
            continue
        if not isinstance(line, int):
            continue
        key = (file, line)
        if key not in counts and len(counts) >= _MAX_SITES:
            continue
        counts[key] = counts.get(key, 0) + 1
        function = entry.get("function")
        functions.setdefault(key, function if isinstance(function, str) else None)
    sites = []
    for (file, line), count in counts.items():
        source = linecache.getline(file, line).strip()
        if source:
            sites.append(_CallSite(file, line, functions[(file, line)], count, source))
    return sites


def _scripted_call(site: _CallSite) -> _ScriptedCall | None:
    """Return the first TorchScript function the call-site line calls, if any.

    Parameters
    ----------
    site:
        A call-site line with its source text.

    Returns
    -------
    _ScriptedCall | None
        The scripted function's name and definition line, or ``None``.
    """

    module = _module_for_file(site.file)
    if module is None:
        return None
    for match in _CALLED_NAME.finditer(site.source):
        parts = re.sub(r"\s+", "", match.group(1)).split(".")
        for candidate in _resolve(module, site.function, parts):
            if isinstance(candidate, torch.jit.ScriptFunction):
                name = getattr(candidate, "name", None)
                return _ScriptedCall(
                    name if isinstance(name, str) and name else parts[-1],
                    _definition_line(candidate),
                )
    return None


def _module_for_file(file: str) -> ModuleType | None:
    """Return the imported module whose source file is ``file``.

    Parameters
    ----------
    file:
        Call-site filename as recorded by the witness.

    Returns
    -------
    ModuleType | None
        The loaded module, or ``None`` when no loaded module has that file.
    """

    target = os.path.normcase(os.path.abspath(file))
    real_target: str | None = None
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        candidate = os.path.normcase(os.path.abspath(module_file))
        if candidate == target:
            return module
        if os.path.basename(candidate) == os.path.basename(target):
            if real_target is None:
                real_target = os.path.realpath(target)
            if os.path.realpath(candidate) == real_target:
                return module
    return None


def _resolve(module: ModuleType, function: str | None, parts: list[str]) -> list[Any]:
    """Resolve a dotted called name statically in ``module``'s namespace.

    Parameters
    ----------
    module:
        Module that defines the call-site line.
    function:
        Name of the function containing the call-site line.
    parts:
        Dotted name split on ``.``, e.g. ``["self", "jit_reset"]``.

    Returns
    -------
    list[Any]
        Every object the name can denote (one per candidate class for ``self.``
        calls); attributes are read with ``inspect.getattr_static`` so no property
        or descriptor runs.
    """

    namespace = vars(module)
    head, rest = parts[0], parts[1:]
    if head in {"self", "cls"} and rest:
        roots = _classes_defining(module, function)
    elif head in namespace:
        roots = [namespace[head]]
    else:
        return []
    found = []
    for root in roots:
        value = root
        for name in rest:
            try:
                value = inspect.getattr_static(value, name)
            except AttributeError:
                break
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
        else:
            found.append(value)
    return found


def _classes_defining(module: ModuleType, function: str | None) -> list[type]:
    """Return the module's own classes, those defining ``function`` first.

    Parameters
    ----------
    module:
        Module that defines the call-site line.
    function:
        Name of the method containing the call-site line.

    Returns
    -------
    list[type]
        Classes defined in ``module`` whose body defines ``function``, or every
        class defined in ``module`` when none does.
    """

    own = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "__module__", None) == module.__name__
    ]
    defining = [cls for cls in own if function is not None and function in vars(cls)]
    return defining or own


def _definition_line(scripted: Any) -> str | None:
    """Return ``file:line`` of a scripted function's Python source, when kept.

    Parameters
    ----------
    scripted:
        A ``torch.jit.ScriptFunction``.

    Returns
    -------
    str | None
        Location of the original Python function, or ``None`` if torch did not keep it.
    """

    original = getattr(scripted, "_torchdynamo_inline", None)
    code = getattr(original, "__code__", None)
    if code is None:
        return None
    return f"{code.co_filename}:{code.co_firstlineno}"


def _more_sites(labels: list[str]) -> str:
    """Return the sentence naming further call sites with the same gap."""

    if not labels:
        return ""
    shown = ", ".join(labels[:3])
    extra = f" and {len(labels) - 3} more" if len(labels) > 3 else ""
    return f" Other call sites with the same gap: {shown}{extra}."


def _warn_scripted(found: list[tuple[_CallSite, _ScriptedCall]]) -> None:
    """Emit ``scripted_function_not_captured`` once per session."""

    _WARNED_CODES.add(SCRIPTED_CODE)
    site, call = found[0]
    defined = f" (defined at {call.defined_at})" if call.defined_at else ""
    in_function = f" in `{site.function}`" if site.function else ""
    others = [f"`{c.name}` at {s.file}:{s.line}" for s, c in found[1:]]
    warnings.warn(
        TorchLensCaptureGapWarning(
            f"The forward calls the TorchScript function `{call.name}`{defined} at "
            f"{site.file}:{site.line}{in_function}: `{site.source}`. A scripted function "
            "runs its body in the TorchScript interpreter, below the Python functions "
            f"TorchLens wraps, so the {site.dispatch_count} operation(s) it dispatched are "
            "not in the graph; the completeness witness counted them as unaccounted and "
            "the capture is marked capture_verified=False (trace.completeness_diagnostics "
            f"lists them).{_more_sites(others)} See {_GOTCHA_POINTER}. Remedy: set "
            "PYTORCH_JIT=0 in the environment before the model's code is imported, for "
            "example `PYTORCH_JIT=0 python your_script.py`; torch.jit.script then returns "
            "the plain Python function and TorchLens captures every operation inside it.",
            code="scripted_function_not_captured",
            file_path=site.file,
            line_no=site.line,
            scripted_function=call.name,
            defined_at=call.defined_at,
            dispatch_count=sum(s.dispatch_count for s, _ in found),
            other_call_sites=tuple(others),
        ),
        stacklevel=4,
    )


def _warn_dlpack(found: list[_CallSite]) -> None:
    """Emit ``dlpack_call_not_captured`` once per session."""

    _WARNED_CODES.add(DLPACK_CODE)
    site = found[0]
    in_function = f" in `{site.function}`" if site.function else ""
    others = [f"{s.file}:{s.line}" for s in found[1:]]
    warnings.warn(
        TorchLensCaptureGapWarning(
            f"The line at {site.file}:{site.line}{in_function} hands a tensor to another "
            f"library through DLPack: `{site.source}`. That library's kernels run outside "
            "the Python functions TorchLens wraps, so the "
            f"{site.dispatch_count} operation(s) dispatched from this line are not in the "
            "graph; the completeness witness counted them as unaccounted and the capture "
            "is marked capture_verified=False (trace.completeness_diagnostics lists them)."
            f"{_more_sites(others)} See {_GOTCHA_POINTER}. Remedy: no TorchLens setting "
            "captures foreign kernels; read this capture as a lower bound, or capture a "
            "version of the model that computes this step with torch operations.",
            code="dlpack_call_not_captured",
            file_path=site.file,
            line_no=site.line,
            dispatch_count=sum(s.dispatch_count for s in found),
            other_call_sites=tuple(others),
        ),
        stacklevel=4,
    )
