"""FuncCallLocation: lightweight record for a single frame in a user-visible call stack.

Each FuncCallLocation captures one stack frame where a torch function was
invoked, including the source file, line number, function name, code object
identity fields, surrounding source context, and (optionally) the function's
signature and docstring.

**Dual construction paths**:

1. **New path** (from ``_get_code_context`` in _capture/call_stack.py):
   Receives ``file``, ``line_number``, ``func_name``,
   ``num_context_lines_requested``, and ``_frame_func_obj``.  Function
   signature/docstring metadata is snapshotted immediately so the live function
   object can be released.  Source context is loaded **lazily** on first access
   via ``_load_source()``, which uses ``linecache`` (cached per-file, no
   redundant disk reads).

2. **Legacy path** (direct construction, e.g. from tests): The legacy
   source-context keyword arguments are passed directly and stored
   immediately.  No lazy loading occurs (``_source_loaded`` is True from
   init).

The sentinel object ``_SENTINEL`` distinguishes "not yet loaded" from an
actual ``None`` value in the lazy-loading placeholders.
"""

import inspect
import linecache
import os
from typing import Any, cast

from .._constants import FUNC_CALL_LOCATION_FIELD_ORDER
from .._io._field_policy import FieldPolicy
from .._io.format_contract import TLSPEC_VERSION, default_fill_state, read_tlspec_version
from .._source_links import vscode_file_line_link
from .._utils._display_records import _source_path, frame_repr, frame_str
from .field_policy import build_record_field_policy_table

# Sentinel object to distinguish "not yet loaded" from an actual None value.
# Used as the default for lazy-loading placeholders so we can tell the
# difference between "we haven't loaded this yet" and "we loaded it and
# the result was None".
_SENTINEL: Any = object()


def _snapshot_func_metadata(func_obj: Any) -> tuple[str | None, str | None]:
    """Return signature/docstring metadata without retaining ``func_obj``.

    Parameters
    ----------
    func_obj:
        Candidate callable from the captured frame locals/globals.

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        Stringified function signature and docstring, or ``None`` values when
        unavailable.
    """

    if func_obj is None or not callable(func_obj):
        return None, None
    try:
        func_signature = str(inspect.signature(func_obj))
    except (ValueError, TypeError):
        func_signature = None
    return func_signature, getattr(func_obj, "__doc__", None)


# Status of a frame's source text, as reported by ``FuncCallLocation`` reprs.
_SOURCE_TEXT_UNAVAILABLE = "source unavailable"
_SOURCE_TEXT_CHANGED = "source changed on disk since capture"
_SOURCE_TEXT_MISSING = "source file not found"


def source_fingerprint(filename: str, num_context_lines: int) -> tuple[Any, ...] | None:
    """Return the capture-time identity of a source file for lazy text reads.

    Captures without ``save_source_context`` record no source text; the text is
    read from the file when someone asks for it. This fingerprint, taken once
    per file per capture, lets that later read tell whether the file is still
    the one that ran: ``("stat", size, mtime_ns, n)`` for a file on disk,
    ``("linecache", None, None, n)`` for interactive source (notebook cells)
    that only ``linecache`` holds, ``None`` when neither is available. ``n`` is
    the requested number of context lines on each side.

    Parameters
    ----------
    filename:
        The frame's ``co_filename``.
    num_context_lines:
        Context lines requested on each side of the call line.

    Returns
    -------
    tuple[Any, ...] | None
        The fingerprint, or ``None`` when the source cannot be read later.
    """

    try:
        stat = os.stat(filename)
    except (OSError, ValueError):
        entry = linecache.cache.get(filename)
        if entry is not None and len(entry) == 4 and entry[1] is None:
            return ("linecache", None, None, num_context_lines)
        return None
    return ("stat", stat.st_size, stat.st_mtime_ns, num_context_lines)


def cached_source_fingerprint(
    filename: str, num_context_lines: int, context_cache: dict[Any, tuple[Any, ...]] | None
) -> tuple[Any, ...] | None:
    """Return a file's source fingerprint, taken at most once per capture.

    One ``os.stat`` per distinct file per capture (``context_cache`` is the
    capture's code-context cache): it lets an unsnapshotted frame read its
    text lazily and refuse it when the file later changed.
    """

    if context_cache is None:
        return source_fingerprint(filename, num_context_lines)
    key = ("__torchlens_source_fingerprint__", filename, num_context_lines)
    cached = context_cache.get(key)
    if cached is None:
        # Stored boxed in a 1-tuple: the cache holds tuples, and the
        # fingerprint itself may be ``None``.
        cached = (source_fingerprint(filename, num_context_lines),)
        context_cache[key] = cached
    return cast(tuple[Any, ...] | None, cached[0])


def with_source_fingerprint(location: Any, trace: Any) -> "FuncCallLocation":
    """Fix a freshly captured location's source text policy; return it.

    With ``save_source_context`` the text is snapshotted now; otherwise the
    file's fingerprint is stamped so the text can be read lazily, with the
    trace's ``num_context_lines`` around the line. For one-off locations built
    outside ``_get_code_context`` (which caches one fingerprint per file per
    capture). ``location`` is a ``FuncCallLocation``; it is typed ``Any``
    because ``_linecache_entry`` is a runtime-only row cell with no declared
    annotation (the record schema bindings are generated from annotations).
    """

    if location.source_loading_enabled:
        location._ensure_source_loaded()  # save_source_context: snapshot at capture
    else:
        location._linecache_entry = source_fingerprint(
            location.file, int(getattr(trace, "num_context_lines", 7))
        )
    return location


def _source_view(
    all_lines: list[str], line_number: int, num_context_lines: int
) -> tuple[list[str], str, str, str, int]:
    """Build the text fields of one frame from its file's lines.

    Returns ``(code_context, source_context, code_context_labeled, call_line,
    num_context_lines)`` for the window of ``num_context_lines`` lines on each
    side of ``line_number``.
    """

    start = max(0, line_number - num_context_lines - 1)
    end = min(len(all_lines), start + 2 * num_context_lines + 1)
    code_lines = all_lines[start:end]
    call_line_idx = line_number - 1 - start
    call_line = code_lines[call_line_idx].strip() if 0 <= call_line_idx < len(code_lines) else ""
    labeled = "\n".join(
        f"  --->  {line.rstrip()}" if i == call_line_idx else f"        {line.rstrip()}"
        for i, line in enumerate(code_lines)
    )
    return code_lines, "".join(code_lines), labeled, call_line, len(code_lines)


def _read_fingerprinted_lines(filename: str, fingerprint: Any) -> list[str] | str:
    """Return the file's current lines, or a status when they cannot be trusted.

    The lines are returned only when the file still matches ``fingerprint``
    (same size and modification time as at capture); otherwise the result is
    the status text the repr shows instead of wrong source.
    """

    if not isinstance(fingerprint, tuple) or len(fingerprint) != 4:
        return _SOURCE_TEXT_UNAVAILABLE
    if fingerprint[0] == "linecache":
        entry = linecache.cache.get(filename)
        if entry is None or len(entry) != 4:
            return _SOURCE_TEXT_UNAVAILABLE
        return list(entry[2])
    try:
        stat = os.stat(filename)
    except (OSError, ValueError):
        return _SOURCE_TEXT_MISSING
    if (stat.st_size, stat.st_mtime_ns) != (fingerprint[1], fingerprint[2]):
        return _SOURCE_TEXT_CHANGED
    linecache.checkcache(filename)
    return linecache.getlines(filename) or _SOURCE_TEXT_UNAVAILABLE


def _torch_package_dir() -> str:
    """Return the normalized directory of the installed ``torch`` package.

    Computed on each call with no cache: torch is already in ``sys.modules``
    and the path arithmetic is cheap, and this runs only on display paths, so
    holding the value as process state (a module global or a memo) buys nothing.
    """

    import torch

    return os.path.normcase(os.path.dirname(os.path.abspath(torch.__file__)))


def display_frame(code_context: Any) -> "FuncCallLocation | None":
    """Return the frame a display should name as the line that ran an op.

    ``code_context`` is ordered outermost to innermost, so the innermost frame
    (the last) ran the op. When that frame is inside torch's own Python code
    (``torch/nn/functional.py`` under a ``Linear`` call, say), the line a user
    wants is the innermost one outside torch: the model line that called into
    torch. When every frame is torch's, the innermost frame is returned.
    """

    frames = list(code_context or ())
    if not frames:
        return None
    torch_dir = _torch_package_dir() + os.sep
    for frame in reversed(frames):
        file = getattr(frame, "file", "") or ""
        if not os.path.normcase(os.path.abspath(file)).startswith(torch_dir):
            return cast("FuncCallLocation", frame)
    return cast("FuncCallLocation", frames[-1])


def short_source_path(file: str) -> str:
    """Return a short, readable form of a source path for one-line displays.

    Installed packages show from the package root (``transformers/activations.py``),
    files under the working directory show relative to it, anything else in full.
    """

    return _source_path(file)


class FuncCallLocation:
    """A location in source code where a function call occurred.

    Supports two construction paths:

    **New path** (from ``_get_code_context``): pass ``file``, ``line_number``,
    ``func_name``, ``num_context_lines_requested``, and optionally
    ``_frame_func_obj``.  Source context and signature/docstring are loaded
    lazily on first access.

    **Legacy path** (direct construction, e.g. tests): pass the legacy
    source-context keyword arguments directly. Values are stored directly
    with no lazy loading.

    **Unsnapshotted frames** (``source_loading_enabled=False``, the capture
    default): no text is stored. When the capture recorded a source
    fingerprint, the text properties read the file at access time and return
    its current lines, but only while the file still has the size and
    modification time it had at capture; when it changed or is gone they
    return the no-source values and the repr says why.
    Examples
    --------
    >>> where = FuncCallLocation(file="model.py", line_number=12, func_name="forward")
    >>> where.func_name, where.line_number
    ('forward', 12)
    >>> location = tl.trace(model, x).ops[1].code_context[0]
    >>> isinstance(location, FuncCallLocation), isinstance(location.line_number, int)
    (True, True)
    """

    _PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "file": FieldPolicy.KEEP,
        "line_number": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "code_firstlineno": FieldPolicy.KEEP,
        "func_qualname": FieldPolicy.KEEP,
        "col_offset": FieldPolicy.KEEP,
        "source_loading_enabled": FieldPolicy.KEEP,
        "_num_context_lines_requested": FieldPolicy.KEEP,
        "_frame_func_obj": FieldPolicy.DROP,
        "_source_loaded": FieldPolicy.KEEP,
        "_code_context": FieldPolicy.KEEP,
        "_source_context": FieldPolicy.KEEP,
        "_code_context_labeled": FieldPolicy.KEEP,
        "_call_line": FieldPolicy.KEEP,
        "_num_context_lines": FieldPolicy.KEEP,
        "_func_signature": FieldPolicy.KEEP,
        "_func_docstring": FieldPolicy.KEEP,
        "_linecache_entry": FieldPolicy.DROP,
    }

    _FIELD_POLICY = build_record_field_policy_table(
        FUNC_CALL_LOCATION_FIELD_ORDER,
        _PORTABLE_STATE_SPEC,
        schema_key="func_call_location",
    )

    # Reached through the class by ``_utils.introspection``, which cannot
    # import this module at module scope.
    _cached_source_fingerprint = staticmethod(cached_source_fingerprint)

    def __init__(
        self,
        file: str,
        line_number: int,
        func_name: str,
        # New-path args
        num_context_lines_requested: int = _SENTINEL,
        _frame_func_obj: Any = _SENTINEL,
        # Legacy-path args (all default to sentinel)
        func_signature: str | None = _SENTINEL,
        func_docstring: str | None = _SENTINEL,
        call_line: str = _SENTINEL,
        code_context: list[str] | None = _SENTINEL,
        source_context: str = _SENTINEL,
        code_context_labeled: str = _SENTINEL,
        source_context_lines: int = _SENTINEL,
        code_firstlineno: int = _SENTINEL,
        func_qualname: str | None = None,
        col_offset: int | None = None,
        source_loading_enabled: bool = True,
    ) -> None:
        """Initialize source-location metadata for a captured function call.

        Parameters
        ----------
        file:
            Source file path for the call.
        line_number:
            One-indexed source line for the call.
        func_name:
            Display name of the called function.
        num_context_lines_requested:
            Number of context lines to load around the call in the capture path.
        _frame_func_obj:
            Live function object used for lazy source loading when available.
        func_signature:
            Preloaded function signature for direct construction paths.
        func_docstring:
            Preloaded function docstring for direct construction paths.
        call_line:
            Preloaded source line for direct construction paths.
        code_context:
            Preloaded surrounding code lines for direct construction paths.
        source_context:
            Preloaded source context block for direct construction paths.
        code_context_labeled:
            Preloaded labeled source context for direct construction paths.
        source_context_lines:
            Number of context lines present in preloaded source context.
        code_firstlineno:
            First source line of the enclosing function.
        func_qualname:
            Qualified function name when known.
        col_offset:
            Column offset for AST-aware conditional attribution.
        source_loading_enabled:
            Whether lazy source loading should be attempted.
        """

        self.file = file
        self.line_number = line_number
        self.func_name = func_name
        self.code_firstlineno = line_number if code_firstlineno is _SENTINEL else code_firstlineno
        self.func_qualname = func_qualname
        self.col_offset = col_offset
        self.source_loading_enabled = source_loading_enabled

        # Detect which construction path: if func_signature was explicitly
        # passed, we're on the legacy path (direct construction from tests
        # or serialized data).  Otherwise, we're on the new path with lazy loading.
        legacy = func_signature is not _SENTINEL

        if not self.source_loading_enabled:
            self._num_context_lines_requested = 0
            self._frame_func_obj = None
            self._source_loaded = True
            self._initialize_no_source_state()
            return

        if legacy:
            # Legacy path: store all values directly, no lazy loading needed.
            self._func_signature = func_signature
            self._func_docstring = func_docstring if func_docstring is not _SENTINEL else None
            self._call_line = call_line if call_line is not _SENTINEL else ""
            self._code_context = code_context if code_context is not _SENTINEL else None
            self._source_context = source_context if source_context is not _SENTINEL else "None"
            self._code_context_labeled = (
                code_context_labeled if code_context_labeled is not _SENTINEL else ""
            )
            self._num_context_lines = (
                source_context_lines if source_context_lines is not _SENTINEL else 0
            )
            self._num_context_lines_requested = 0
            self._frame_func_obj = None
            self._source_loaded = True
        else:
            # New path: defer source loading until a property is accessed.
            # _source_loaded stays False until _load_source() is called.
            self._num_context_lines_requested = (
                num_context_lines_requested if num_context_lines_requested is not _SENTINEL else 7
            )
            func_obj = _frame_func_obj if _frame_func_obj is not _SENTINEL else None
            self._func_signature, self._func_docstring = _snapshot_func_metadata(func_obj)
            self._frame_func_obj = None
            self._source_loaded = False
            # Placeholders (set by _load_source)
            self._code_context = _SENTINEL
            self._source_context = _SENTINEL
            self._code_context_labeled = _SENTINEL
            self._call_line = _SENTINEL
            self._num_context_lines = _SENTINEL

    def _initialize_no_source_state(self) -> None:
        """Populate the canonical no-source state used when source is unavailable."""
        self._code_context = None
        self._source_context = "None"
        self._code_context_labeled = ""
        self._call_line = ""
        self._num_context_lines = 0
        self._func_signature = None
        self._func_docstring = None

    def _ensure_source_loaded(self) -> None:
        """Load source metadata on first access when source loading is enabled."""
        if not self._source_loaded:
            self._load_source()

    def _load_source(self) -> None:
        """Load source context and function metadata from disk (one-shot).

        Uses ``linecache.getlines()`` which caches file contents per-path,
        so repeated calls for different lines in the same file are free.
        After loading, releases ``_frame_func_obj`` to avoid retaining
        a reference to the (potentially large) function object.
        """
        if self._source_loaded:
            return
        self._source_loaded = True

        # Load source lines via linecache (cached per file, no re-reads)
        all_lines = linecache.getlines(self.file)
        if all_lines:
            (
                self._code_context,
                self._source_context,
                self._code_context_labeled,
                self._call_line,
                self._num_context_lines,
            ) = _source_view(all_lines, self.line_number, self._num_context_lines_requested)
        else:
            self._initialize_no_source_state()

        # Release the func obj reference
        self._frame_func_obj = None

    def _source_text_state(self) -> tuple[tuple[Any, ...] | None, str]:
        """Return ``(text fields or None, status)`` for this frame.

        Snapshotted frames serve their stored text. An unsnapshotted frame
        with a capture fingerprint reads its file now (see
        :func:`_read_fingerprinted_lines`); the status names why no text is
        shown when it cannot.
        """

        self._ensure_source_loaded()
        stored = (
            self._code_context,
            self._source_context,
            self._code_context_labeled,
            self._call_line,
            self._num_context_lines,
        )
        if self.source_loading_enabled or self._code_context is not None:
            return (stored if self._code_context is not None else None), _SOURCE_TEXT_UNAVAILABLE
        # Runtime-only (never persisted); the capture path sets it on
        # unsnapshotted frames, and an unset cell means "no fingerprint".
        fingerprint = getattr(self, "_linecache_entry", None)
        if fingerprint is None or not self.file:
            return None, _SOURCE_TEXT_UNAVAILABLE
        lines = _read_fingerprinted_lines(self.file, fingerprint)
        if isinstance(lines, str):
            return None, lines
        return _source_view(lines, self.line_number, int(fingerprint[3])), ""

    def _text_field(self, index: int) -> Any:
        """Return one text field, the lazily read one when nothing is stored."""

        view, _status = self._source_text_state()
        if view is None:
            return (
                self._code_context,
                self._source_context,
                self._code_context_labeled,
                self._call_line,
                self._num_context_lines,
            )[index]
        return view[index]

    # --- Lazy properties ---
    # Each property calls _load_source() before returning its cached value.
    # The setters allow direct assignment (used by legacy path and tests).

    @property
    def code_context(self) -> list[str] | None:
        """Return source lines around this call site.

        Returns
        -------
        Optional[List[str]]
            Source context lines, or ``None`` when unavailable.
        """
        return self._text_field(0)

    @code_context.setter
    def code_context(self, value: list[str] | None) -> None:
        """Set source lines around this call site.

        Parameters
        ----------
        value:
            Source context lines, or ``None`` when unavailable.
        """
        self._code_context = value

    @property
    def source_context(self) -> str:
        """Return source context as a single string.

        Returns
        -------
        str
            Joined source context string.
        """
        return self._text_field(1)

    @source_context.setter
    def source_context(self, value: str) -> None:
        """Set source context as a single string.

        Parameters
        ----------
        value:
            Joined source context string.
        """
        self._source_context = value

    @property
    def code_context_labeled(self) -> str:
        """Return labeled source context with the call line marked.

        Returns
        -------
        str
            Source context annotated with a marker on the captured call line.
        """
        return self._text_field(2)

    @code_context_labeled.setter
    def code_context_labeled(self, value: str) -> None:
        """Set labeled source context with the call line marked.

        Parameters
        ----------
        value:
            Source context annotated with a marker on the captured call line.
        """
        self._code_context_labeled = value

    @property
    def call_line(self) -> str:
        """Return the exact source line that triggered this frame.

        Returns
        -------
        str
            Stripped source line for the captured call.
        """
        return self._text_field(3)

    @call_line.setter
    def call_line(self, value: str) -> None:
        """Set the exact source line that triggered this frame.

        Parameters
        ----------
        value:
            Stripped source line for the captured call.
        """
        self._call_line = value

    @property
    def num_context_lines(self) -> int:
        """Return the number of loaded source context lines.

        Returns
        -------
        int
            Count of source lines available for this call site.
        """
        return self._text_field(4)

    @num_context_lines.setter
    def num_context_lines(self, value: int) -> None:
        """Set the number of loaded source context lines.

        Parameters
        ----------
        value:
            Count of source lines available for this call site.
        """
        self._num_context_lines = value

    @property
    def func_signature(self) -> str | None:
        """Return the captured function signature string.

        Returns
        -------
        Optional[str]
            Stringified function signature, or ``None`` when unavailable.
        """
        self._ensure_source_loaded()
        return self._func_signature

    @func_signature.setter
    def func_signature(self, value: str | None) -> None:
        """Set the captured function signature string.

        Parameters
        ----------
        value:
            Stringified function signature, or ``None`` when unavailable.
        """
        self._func_signature = value

    @property
    def func_docstring(self) -> str | None:
        """Return the captured function docstring.

        Returns
        -------
        Optional[str]
            Function docstring, or ``None`` when unavailable.
        """
        self._ensure_source_loaded()
        return self._func_docstring

    @func_docstring.setter
    def func_docstring(self, value: str | None) -> None:
        """Set the captured function docstring.

        Parameters
        ----------
        value:
            Function docstring, or ``None`` when unavailable.
        """
        self._func_docstring = value

    __repr__ = frame_repr
    __str__ = frame_str

    def to_html_link(self) -> str:
        """Return a VS Code source-location anchor for HTML renderers.

        Returns
        -------
        str
            HTML anchor pointing at this frame's source location.
        """

        return vscode_file_line_link(self.file, self.line_number)

    def __getitem__(self, i: int | slice) -> str | list[str]:
        """Index into the source context lines."""
        if self.code_context is None:
            raise IndexError("code_context is None (source unavailable)")
        return self.code_context[i]

    def __len__(self) -> int:
        """Return the number of source context lines."""
        if self.code_context is None:
            return 0
        return len(self.code_context)

    def __tl_state_items__(self) -> Any:
        """Yield live state pairs from the backing row (M8 facade hook)."""

        from .._trace_core.record_rows import record_state_items

        return record_state_items(self)

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install a state mapping through the cell descriptors (M8 hook)."""

        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with live frame references stripped."""
        from ._state_adapter import state_items

        state = dict(state_items(self))
        state["_frame_func_obj"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state without reviving frame references."""
        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        default_fill_state(state, defaults={"_frame_func_obj": None})
        from .._io.state_keys import refuse_callable_shadowing_state_keys

        refuse_callable_shadowing_state_keys(type(self), state)
        from .._trace_core.record_rows import record_state_restore

        record_state_restore(self, state)


# The M8 facade: the declared stored fields (the literal _PORTABLE_STATE_SPEC
# keys) become row-cell descriptors; the instance dict keeps only the store
# binding + user extras. FuncCallLocation records are per-trace (cached in
# ``Trace._code_context_cache``), so cell storage is trace-scoped like every
# other record kind.
_FUNC_CALL_LOCATION_STORED_FIELDS: tuple[str, ...] = (
    "file",
    "line_number",
    "func_name",
    "code_firstlineno",
    "func_qualname",
    "col_offset",
    "source_loading_enabled",
    "_num_context_lines_requested",
    "_frame_func_obj",
    "_source_loaded",
    "_code_context",
    "_source_context",
    "_code_context_labeled",
    "_call_line",
    "_num_context_lines",
    "_func_signature",
    "_func_docstring",
    "_linecache_entry",
)


def _install_func_call_location_facade() -> None:
    """Install the row-cell descriptors (import-time, collision-safe)."""

    from .._trace_core.record_rows import install_record_facade

    install_record_facade(FuncCallLocation, _FUNC_CALL_LOCATION_STORED_FIELDS)


_install_func_call_location_facade()
