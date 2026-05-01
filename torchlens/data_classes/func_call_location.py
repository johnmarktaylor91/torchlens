"""FuncCallLocation: lightweight record for a single frame in a user-visible call stack.

Each FuncCallLocation captures one stack frame where a torch function was
invoked, including the source file, line number, function name, code object
identity fields, surrounding source context, and (optionally) the function's
signature and docstring.

**Dual construction paths**:

1. **New path** (from ``_get_func_call_stack`` in capture/call_stack.py):
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
from typing import Any, Dict, List, Optional, Union

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from .._source_links import terminal_file_line_link, vscode_file_line_link

# Sentinel object to distinguish "not yet loaded" from an actual None value.
# Used as the default for lazy-loading placeholders so we can tell the
# difference between "we haven't loaded this yet" and "we loaded it and
# the result was None".
_SENTINEL: Any = object()


def _snapshot_func_metadata(func_obj: Any) -> tuple[Optional[str], Optional[str]]:
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


class FuncCallLocation:
    """A location in source code where a function call occurred.

    Supports two construction paths:

    **New path** (from ``_get_func_call_stack``): pass ``file``, ``line_number``,
    ``func_name``, ``num_context_lines_requested``, and optionally
    ``_frame_func_obj``.  Source context and signature/docstring are loaded
    lazily on first access.

    **Legacy path** (direct construction, e.g. tests): pass the legacy
    source-context keyword arguments directly. Values are stored directly
    with no lazy loading.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "file": FieldPolicy.KEEP,
        "line_number": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "code_firstlineno": FieldPolicy.KEEP,
        "code_qualname": FieldPolicy.KEEP,
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

    def __init__(
        self,
        file: str,
        line_number: int,
        func_name: str,
        # New-path args
        num_context_lines_requested: int = _SENTINEL,
        _frame_func_obj: Any = _SENTINEL,
        # Legacy-path args (all default to sentinel)
        func_signature: Optional[str] = _SENTINEL,
        func_docstring: Optional[str] = _SENTINEL,
        call_line: str = _SENTINEL,
        code_context: Optional[List[str]] = _SENTINEL,
        source_context: str = _SENTINEL,
        code_context_labeled: str = _SENTINEL,
        num_context_lines: int = _SENTINEL,
        code_firstlineno: int = _SENTINEL,
        code_qualname: Optional[str] = None,
        col_offset: Optional[int] = None,
        source_loading_enabled: bool = True,
    ) -> None:
        self.file = file
        self.line_number = line_number
        self.func_name = func_name
        self.code_firstlineno = line_number if code_firstlineno is _SENTINEL else code_firstlineno
        self.code_qualname = code_qualname
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
            self._num_context_lines = num_context_lines if num_context_lines is not _SENTINEL else 0
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

        context_window = 2 * self._num_context_lines_requested + 1

        # Load source lines via linecache (cached per file, no re-reads)
        all_lines = linecache.getlines(self.file)
        if all_lines:
            # Extract context window around the call line
            start = max(0, self.line_number - self._num_context_lines_requested - 1)
            end = min(len(all_lines), start + context_window)
            code_lines = all_lines[start:end]

            self._code_context = code_lines
            self._source_context = "".join(code_lines)
            self._num_context_lines = len(code_lines)

            # The call line is at the position within the context window
            call_line_idx = self.line_number - 1 - start
            if 0 <= call_line_idx < len(code_lines):
                self._call_line = code_lines[call_line_idx].strip()
            else:
                self._call_line = ""

            # Build labeled context with arrow at the call line
            labeled_lines = []
            for i, line in enumerate(code_lines):
                if i == call_line_idx:
                    labeled_lines.append(f"  --->  {line.rstrip()}")
                else:
                    labeled_lines.append(f"        {line.rstrip()}")
            self._code_context_labeled = "\n".join(labeled_lines)
        else:
            self._initialize_no_source_state()

        # Release the func obj reference
        self._frame_func_obj = None

    # --- Lazy properties ---
    # Each property calls _load_source() before returning its cached value.
    # The setters allow direct assignment (used by legacy path and tests).

    @property
    def code_context(self) -> Optional[List[str]]:
        """Return source lines around this call site.

        Returns
        -------
        Optional[List[str]]
            Source context lines, or ``None`` when unavailable.
        """
        self._ensure_source_loaded()
        return self._code_context

    @code_context.setter
    def code_context(self, value: Optional[List[str]]) -> None:
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
        self._ensure_source_loaded()
        return self._source_context

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
        self._ensure_source_loaded()
        return self._code_context_labeled

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
        self._ensure_source_loaded()
        return self._call_line

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
        self._ensure_source_loaded()
        return self._num_context_lines

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
    def func_signature(self) -> Optional[str]:
        """Return the captured function signature string.

        Returns
        -------
        Optional[str]
            Stringified function signature, or ``None`` when unavailable.
        """
        self._ensure_source_loaded()
        return self._func_signature

    @func_signature.setter
    def func_signature(self, value: Optional[str]) -> None:
        """Set the captured function signature string.

        Parameters
        ----------
        value:
            Stringified function signature, or ``None`` when unavailable.
        """
        self._func_signature = value

    @property
    def func_docstring(self) -> Optional[str]:
        """Return the captured function docstring.

        Returns
        -------
        Optional[str]
            Function docstring, or ``None`` when unavailable.
        """
        self._ensure_source_loaded()
        return self._func_docstring

    @func_docstring.setter
    def func_docstring(self, value: Optional[str]) -> None:
        """Set the captured function docstring.

        Parameters
        ----------
        value:
            Function docstring, or ``None`` when unavailable.
        """
        self._func_docstring = value

    def __repr__(self) -> str:
        """Show file, line number, function name, and source context with arrow."""
        lines = [
            "FuncCallLocation:",
            f"  file: {terminal_file_line_link(self.file, self.line_number)}",
            f"  line: {self.line_number}",
            f"  function: {self.func_name}",
        ]
        if self.code_context is not None:
            lines.append("  code:")
            lines.append(self.code_context_labeled)
        else:
            lines.append("  code: source unavailable")
        return "\n".join(lines)

    def to_html_link(self) -> str:
        """Return a VS Code source-location anchor for HTML renderers.

        Returns
        -------
        str
            HTML anchor pointing at this frame's source location.
        """

        return vscode_file_line_link(self.file, self.line_number)

    def __getitem__(self, i: Union[int, slice]) -> Union[str, List[str]]:
        """Index into the source context lines."""
        if self.code_context is None:
            raise IndexError("code_context is None (source unavailable)")
        return self.code_context[i]

    def __len__(self) -> int:
        """Return the number of source context lines."""
        if self.code_context is None:
            return 0
        return len(self.code_context)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with live frame references stripped."""
        state = self.__dict__.copy()
        state["_frame_func_obj"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state without reviving frame references."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(state, defaults={"_frame_func_obj": None})
        self.__dict__.update(state)
