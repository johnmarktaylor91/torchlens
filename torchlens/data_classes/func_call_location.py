"""FuncCallLocation: lightweight record for a single frame in a user-visible call stack.

Each FuncCallLocation captures one stack frame where a torch function was
invoked, including the source file, line number, function name, surrounding
source context, and (optionally) the function's signature and docstring.

**Dual construction paths**:

1. **New path** (from ``_get_func_call_stack`` in capture/call_stack.py):
   Receives ``file``, ``line_number``, ``func_name``,
   ``num_context_lines_requested``, and ``_frame_func_obj``.  Source context
   and function metadata are loaded **lazily** on first access via
   ``_load_source()``, which uses ``linecache`` (cached per-file, no
   redundant disk reads).  After loading, ``_frame_func_obj`` is released
   to avoid retaining a reference to the frame's function object.

2. **Legacy path** (direct construction, e.g. from tests): All 10 keyword
   arguments are passed directly and stored immediately.  No lazy loading
   occurs (``_source_loaded`` is True from init).

The sentinel object ``_SENTINEL`` distinguishes "not yet loaded" from an
actual ``None`` value in the lazy-loading placeholders.
"""

import inspect
import linecache
from typing import Any, List, Optional, Union

# Sentinel object to distinguish "not yet loaded" from an actual None value.
# Used as the default for lazy-loading placeholders so we can tell the
# difference between "we haven't loaded this yet" and "we loaded it and
# the result was None".
_SENTINEL: Any = object()


class FuncCallLocation:
    """A location in source code where a function call occurred.

    Supports two construction paths:

    **New path** (from ``_get_func_call_stack``): pass ``file``, ``line_number``,
    ``func_name``, ``num_context_lines_requested``, and optionally
    ``_frame_func_obj``.  Source context and signature/docstring are loaded
    lazily on first access.

    **Legacy path** (direct construction, e.g. tests): pass all 10 original
    keyword arguments.  Values are stored directly with no lazy loading.
    """

    def __init__(
        self,
        file: str,
        line_number: int,
        func_name: str,
        # New-path args
        num_context_lines_requested: int = _SENTINEL,
        _frame_func_obj=_SENTINEL,
        # Legacy-path args (all default to sentinel)
        func_signature: Optional[str] = _SENTINEL,
        func_docstring: Optional[str] = _SENTINEL,
        call_line: str = _SENTINEL,
        code_context: Optional[List[str]] = _SENTINEL,
        source_context: str = _SENTINEL,
        code_context_labeled: str = _SENTINEL,
        num_context_lines: int = _SENTINEL,
    ):
        self.file = file
        self.line_number = line_number
        self.func_name = func_name

        # Detect which construction path: if func_signature was explicitly
        # passed, we're on the legacy path (direct construction from tests
        # or serialized data).  Otherwise, we're on the new path with lazy loading.
        legacy = func_signature is not _SENTINEL

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
            self._frame_func_obj = _frame_func_obj if _frame_func_obj is not _SENTINEL else None
            self._source_loaded = False
            # Placeholders (set by _load_source)
            self._code_context = _SENTINEL
            self._source_context = _SENTINEL
            self._code_context_labeled = _SENTINEL
            self._call_line = _SENTINEL
            self._num_context_lines = _SENTINEL
            self._func_signature = _SENTINEL
            self._func_docstring = _SENTINEL

    def _load_source(self):
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
            self._code_context = None
            self._source_context = "None"
            self._call_line = ""
            self._code_context_labeled = ""
            self._num_context_lines = 0

        # Resolve function signature and docstring from stored func object
        func_obj = self._frame_func_obj
        if func_obj is not None and callable(func_obj):
            try:
                self._func_signature = str(inspect.signature(func_obj))
            except (ValueError, TypeError):
                self._func_signature = None
            doc = getattr(func_obj, "__doc__", None)
            self._func_docstring = doc
        else:
            self._func_signature = None
            self._func_docstring = None

        # Release the func obj reference
        self._frame_func_obj = None

    # --- Lazy properties ---
    # Each property calls _load_source() before returning its cached value.
    # The setters allow direct assignment (used by legacy path and tests).

    @property
    def code_context(self) -> Optional[List[str]]:
        self._load_source()
        return self._code_context

    @code_context.setter
    def code_context(self, value: Optional[List[str]]) -> None:
        self._code_context = value

    @property
    def source_context(self) -> str:
        self._load_source()
        return self._source_context

    @source_context.setter
    def source_context(self, value: str) -> None:
        self._source_context = value

    @property
    def code_context_labeled(self) -> str:
        self._load_source()
        return self._code_context_labeled

    @code_context_labeled.setter
    def code_context_labeled(self, value: str) -> None:
        self._code_context_labeled = value

    @property
    def call_line(self) -> str:
        self._load_source()
        return self._call_line

    @call_line.setter
    def call_line(self, value: str) -> None:
        self._call_line = value

    @property
    def num_context_lines(self) -> int:
        self._load_source()
        return self._num_context_lines

    @num_context_lines.setter
    def num_context_lines(self, value: int) -> None:
        self._num_context_lines = value

    @property
    def func_signature(self) -> Optional[str]:
        self._load_source()
        return self._func_signature

    @func_signature.setter
    def func_signature(self, value: Optional[str]) -> None:
        self._func_signature = value

    @property
    def func_docstring(self) -> Optional[str]:
        self._load_source()
        return self._func_docstring

    @func_docstring.setter
    def func_docstring(self, value: Optional[str]) -> None:
        self._func_docstring = value

    def __repr__(self) -> str:
        """Show file, line number, function name, and source context with arrow."""
        lines = [
            "FuncCallLocation:",
            f"  file: {self.file}",
            f"  line: {self.line_number}",
            f"  function: {self.func_name}",
        ]
        if self.code_context is not None:
            lines.append("  code:")
            lines.append(self.code_context_labeled)
        else:
            lines.append("  code: source unavailable")
        return "\n".join(lines)

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
