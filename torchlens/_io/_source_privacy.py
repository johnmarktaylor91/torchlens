"""Source-embedding privacy policy for portable saves (split from scrub.py).

Every source-file reference in a bundle is relativized to a bare basename
(absolute paths embed the producer's ``$HOME``, OS username, and filesystem
layout -- host PII with no portable value), and ``include_source=False``
drops source text, docstrings, signature defaults, and source-file
references entirely. One policy function per record family: Trace/Module
source metadata, the Trace source-code blob, ``FuncCallLocation`` frames,
conditional records, and persisted capture advisories. ``scrub.py`` owns
the dispatch; this module owns the policy bodies.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scrub import _ScrubOptions


def _relativize_source_path(path: Any) -> Any:
    """Reduce a captured source path to its bare basename for portable save.

    Absolute source paths embed the producer's ``$HOME``, OS username, and
    site-packages / capturing-script filesystem layout -- host details with no
    portable value (a ``vscode://`` link built from them never resolves on
    another machine). Reducing every saved source path to its basename removes
    that host PII while keeping a human-useful filename hint. Handles both POSIX
    and Windows separators so a bundle produced on either host is scrubbed.

    Parameters
    ----------
    path:
        Candidate source path value.

    Returns
    -------
    Any
        Basename of ``path`` when it is a non-empty string, else ``path``.
    """

    if not isinstance(path, str) or not path:
        return path
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _relativize_source_text(value: Any) -> Any:
    """Return a source-blob value with its file path reduced to a basename.

    ``_source_code_blob`` values are ``SourceText`` (a ``str`` subclass carrying
    ``file_path``/``line_number``). A fresh copy is returned so the live Trace's
    ``SourceText`` objects are never mutated; values without a ``file_path`` are
    returned unchanged.

    Parameters
    ----------
    value:
        Source-blob value (a ``SourceText`` or plain string).

    Returns
    -------
    Any
        A path-relativized ``SourceText`` copy, or the original value.
    """

    file_path = getattr(value, "file_path", None)
    if file_path is None:
        return value
    # ``value`` is a ``SourceText`` (only that ``str`` subclass carries
    # ``file_path``). Reconstruct via ``type(value)`` rather than importing
    # ``SourceText`` so the save path never pulls in ``visualization.code_panel``
    # (and its ``graphviz`` import); a plain ``str`` never reaches here.
    return type(value)(
        str(value),
        file_path=_relativize_source_path(file_path),
        line_number=getattr(value, "line_number", None),
    )


# ``backward_*`` are only present on ``GradFn`` logs (for Python-inspectable custom
# autograd Functions); Trace/Module logs lack them, and each field is guarded by an
# ``in scrubbed_state`` check, so listing them here is inert where absent. Including
# them closes B8-21: the backward source path/docstring were outside the belt and
# persisted an absolute path unscrubbed.
_SOURCE_FILE_FIELDS = (
    "class_source_file",
    "init_source_file",
    "forward_source_file",
    "backward_source_file",
)
_DOCSTRING_FIELDS = (
    "class_docstring",
    "init_docstring",
    "forward_docstring",
    "backward_docstring",
)
# Signature strings are ``str(inspect.signature(...))`` snapshots. They are kept
# as structural interface metadata, but ``inspect.Signature.__str__`` renders
# every parameter default via ``repr(default)`` -- so a default like
# ``cfg='/home/user/x.yaml'`` embeds an absolute host path verbatim, and a
# ``token='SECRET'`` default ships source-derived VALUES, both surviving
# ``include_source=False`` (B8 R62). ``_func_signature`` is the FuncCallLocation
# owner; the rest ride Trace / Module / GradFn logs (guarded by presence).
_SIGNATURE_FIELDS = (
    "init_signature",
    "forward_signature",
    "backward_signature",
)
_FRAME_SIGNATURE_FIELD = "_func_signature"

_ABS_PATH_LITERAL = re.compile(r"^(?:/|~|[A-Za-z]:[\\/])")


def _relativize_path_literals(text: str) -> str:
    """Reduce absolute-path string literals inside a signature string to basenames.

    Scans ``text`` for quoted string literals and, for any whose content looks
    like an absolute host path (POSIX ``/...``, ``~...``, or a Windows drive
    ``X:\\...``), replaces it with its basename -- the same host-PII removal
    :func:`_relativize_source_path` applies to source-file fields, but applied to
    default-value reprs embedded in a signature. Non-path literals (plain
    strings, URLs) are left untouched.

    Parameters
    ----------
    text:
        Signature string (or one parameter of one).

    Returns
    -------
    str
        ``text`` with absolute-path literals relativized.
    """

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in "'\"":
            j = i + 1
            content: list[str] = []
            closed = False
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    content.append(text[j : j + 2])
                    j += 2
                    continue
                if text[j] == char:
                    closed = True
                    break
                content.append(text[j])
                j += 1
            literal = "".join(content)
            if closed:
                if _ABS_PATH_LITERAL.match(literal.replace("\\", "/")):
                    literal = _relativize_source_path(literal)
                out.append(char + literal + char)
                i = j + 1
                continue
            out.append(text[i:])
            break
        out.append(char)
        i += 1
    return "".join(out)


def _split_top_level_params(inner: str) -> list[str]:
    """Split a signature's inner text on top-level commas (bracket/quote aware)."""

    params: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    i = 0
    n = len(inner)
    while i < n:
        char = inner[i]
        if quote is not None:
            if char == "\\" and i + 1 < n:
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            params.append(inner[start:i])
            start = i + 1
        i += 1
    params.append(inner[start:])
    return [param.strip() for param in params if param.strip() != "" or inner == ""]


def _stub_param_default(param: str) -> str:
    """Replace a parameter's default value with ``...`` (structural stub)."""

    depth = 0
    quote: str | None = None
    i = 0
    n = len(param)
    while i < n:
        char = param[i]
        if quote is not None:
            if char == "\\" and i + 1 < n:
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            head = param[:i].rstrip()
            return f"{head} = ..." if ":" in head else f"{head}=..."
        i += 1
    return param


def _scrub_signature_string(signature: Any, *, include_source: bool) -> Any:
    """Scrub host paths (and, without source, default values) from a signature.

    Absolute-path literals are relativized in both modes (the ``no $HOME/username
    ever reaches the bundle`` guarantee). With ``include_source=False`` every
    parameter default is additionally stubbed to ``...`` because defaults are
    source-derived values the caller opted out of, keeping only the structural
    shape (names + annotations).

    Parameters
    ----------
    signature:
        Signature field value (``str`` or ``None``).
    include_source:
        Whether source-derived values may be embedded.

    Returns
    -------
    Any
        The scrubbed signature string, or the input unchanged when not a
        non-empty parenthesized signature.
    """

    if not isinstance(signature, str) or not signature.startswith("("):
        return signature
    # ``str(inspect.signature(...))`` is ``(params) -> return_annotation``; the
    # optional return-annotation suffix means the string does not end with ")".
    # Find the close paren balancing the leading "(" (bracket/quote aware), then
    # process the parameter group and preserve any suffix.
    depth = 0
    quote: str | None = None
    close: int | None = None
    i = 0
    n = len(signature)
    while i < n:
        char = signature[i]
        if quote is not None:
            if char == "\\" and i + 1 < n:
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                close = i
                break
        i += 1
    if close is None:
        return signature
    inner = signature[1:close]
    suffix = signature[close + 1 :]
    params = _split_top_level_params(inner)
    scrubbed: list[str] = []
    for param in params:
        cleaned = _relativize_path_literals(param)
        if not include_source:
            cleaned = _stub_param_default(cleaned)
        scrubbed.append(cleaned)
    # The return annotation is not a default value, so it is never stubbed; its
    # (unlikely) path literals are still relativized.
    return "(" + ", ".join(scrubbed) + ")" + _relativize_path_literals(suffix)


def _apply_source_metadata_policy(scrubbed_state: dict[str, Any], options: _ScrubOptions) -> None:
    """Apply the source-embedding privacy policy to scrubbed source metadata.

    Shared by the ``Trace`` and per-module ``Module`` logs, which both carry
    ``class``/``init``/``forward`` source-file paths and docstrings. Path
    relativization is unconditional (a pure privacy win: no host paths,
    ``$HOME``, or username ever reach the bundle). Docstrings are verbatim source
    text and are dropped when ``include_source=False``, along with the now-dangling
    source-file references. Function signatures are kept as structural interface
    metadata, but their default-value reprs are scrubbed: absolute-path literals
    are always relativized and, with ``include_source=False``, every default is
    stubbed to ``...`` (defaults are source-derived values). Source line numbers
    are structural and retained.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed field state for a Trace or Module log, mutated in place.
    options:
        Active scrub options carrying ``include_source``.
    """

    for field_name in _SIGNATURE_FIELDS:
        if field_name in scrubbed_state:
            scrubbed_state[field_name] = _scrub_signature_string(
                scrubbed_state[field_name], include_source=options.include_source
            )

    if options.include_source:
        for field_name in _SOURCE_FILE_FIELDS:
            if field_name in scrubbed_state:
                scrubbed_state[field_name] = _relativize_source_path(scrubbed_state[field_name])
        return

    for field_name in (*_SOURCE_FILE_FIELDS, *_DOCSTRING_FIELDS):
        if field_name in scrubbed_state:
            scrubbed_state[field_name] = None


def _apply_trace_blob_policy(scrubbed_state: dict[str, Any], options: _ScrubOptions) -> None:
    """Apply the source-embedding privacy policy to the Trace source-code blob.

    The ``_source_code_blob`` holds the model's verbatim class / ``__init__`` /
    ``forward`` source (with absolute ``file_path`` metadata). It is dropped
    entirely when ``include_source=False``; otherwise every entry is replaced with
    a fresh, path-relativized ``SourceText`` copy so the live Trace is never
    mutated and no host path is embedded.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed Trace field state, mutated in place.
    options:
        Active scrub options carrying ``include_source``.
    """

    if not options.include_source:
        scrubbed_state["_source_code_blob"] = {}
        return
    blob = scrubbed_state.get("_source_code_blob")
    if isinstance(blob, dict):
        scrubbed_state["_source_code_blob"] = {
            key: _relativize_source_text(value) for key, value in blob.items()
        }


def _apply_frame_source_policy(scrubbed_state: dict[str, Any], options: _ScrubOptions) -> None:
    """Apply the source-embedding privacy policy to a scrubbed call-stack frame.

    ``FuncCallLocation`` frames embed both the absolute source ``file`` and the
    surrounding source lines. The path is always relativized to a basename; with
    ``include_source=False`` the path and every source-text field are cleared to
    the canonical "source unavailable" frame state (mirroring
    ``FuncCallLocation._initialize_no_source_state``), keeping only structural
    location metadata (line numbers, function name/qualname, signature).

    Parameters
    ----------
    scrubbed_state:
        Scrubbed ``FuncCallLocation`` field state, mutated in place.
    options:
        Active scrub options carrying ``include_source``.
    """

    if _FRAME_SIGNATURE_FIELD in scrubbed_state:
        scrubbed_state[_FRAME_SIGNATURE_FIELD] = _scrub_signature_string(
            scrubbed_state[_FRAME_SIGNATURE_FIELD], include_source=options.include_source
        )

    if options.include_source:
        if "file" in scrubbed_state:
            scrubbed_state["file"] = _relativize_source_path(scrubbed_state["file"])
        return

    scrubbed_state["file"] = ""
    scrubbed_state["_source_loaded"] = True
    scrubbed_state["_code_context"] = None
    scrubbed_state["_source_context"] = "None"
    scrubbed_state["_code_context_labeled"] = ""
    scrubbed_state["_call_line"] = ""
    scrubbed_state["_num_context_lines"] = 0
    scrubbed_state["_num_context_lines_requested"] = 0
    scrubbed_state["_func_docstring"] = None
    scrubbed_state["_frame_func_obj"] = None


def _apply_conditional_source_policy(
    scrubbed_state: dict[str, Any], options: _ScrubOptions, *, drop_value: Any
) -> None:
    """Apply the source-embedding privacy policy to a scrubbed conditional record.

    Both ``ConditionalEvent.source_file`` (in ``Trace.conditional_records``) and
    ``Conditional.source_file`` (in the public ``Trace.conditionals`` accessor) hold
    the ABSOLUTE path of the user's forward-defining module. Like every other
    source-file reference in a bundle each is relativized to a bare basename
    (unconditional privacy win); with ``include_source=False`` it is cleared to
    ``drop_value``, matching the "no embedded source" contract honored for
    ``Trace``/``Module``/``FuncCallLocation``. The structural span/kind metadata is
    retained either way.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed conditional-record field state, mutated in place.
    options:
        Active scrub options carrying ``include_source``.
    drop_value:
        Value assigned to ``source_file`` when source is excluded (``""`` for the
        non-optional ``ConditionalEvent.source_file``; ``None`` for the optional
        ``Conditional.source_file``).
    """

    if "source_file" not in scrubbed_state:
        return
    if options.include_source:
        scrubbed_state["source_file"] = _relativize_source_path(scrubbed_state["source_file"])
    else:
        scrubbed_state["source_file"] = drop_value


def _apply_advisory_source_policy(scrubbed_state: dict[str, Any], options: _ScrubOptions) -> None:
    """Apply the source-embedding privacy policy to persisted capture advisories.

    ``Trace.annotations["capture_advisories"]`` rows (the SF5 scalar-escape
    disclosure) carry the ABSOLUTE path of the user's escape site in
    ``first_location`` (``"<file>:<line>"``). Like every other source-file
    reference in a bundle it is relativized to a bare basename
    (unconditional privacy win); with ``include_source=False`` it is dropped
    to ``None``, matching the "no embedded source" contract. The advisory row
    itself (kind, count, message) is an honesty fact and is retained either
    way. Fresh row containers are built so the live trace's session-time
    annotations are never mutated.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed Trace field state, mutated in place.
    options:
        Active scrub options carrying ``include_source``.
    """

    annotations_state = scrubbed_state.get("annotations")
    if not isinstance(annotations_state, dict):
        return
    advisories = annotations_state.get("capture_advisories")
    if not isinstance(advisories, list):
        return
    scrubbed_rows: list[Any] = []
    for row in advisories:
        if isinstance(row, dict) and row.get("first_location") is not None:
            row = dict(row)
            row["first_location"] = (
                _relativize_source_path(row["first_location"]) if options.include_source else None
            )
        scrubbed_rows.append(row)
    annotations_state["capture_advisories"] = scrubbed_rows
