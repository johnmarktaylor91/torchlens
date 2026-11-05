"""Capture-honesty comment blocks embedded in exported files."""

from __future__ import annotations

from typing import Any

from .._capture_honesty import capture_honesty_facts, honesty_preamble_lines

__tl_layer__ = "L7"


def _honesty_comment_block(log: Any, prefix: str) -> str:
    """Format the shared honesty preamble as comment lines.

    Parameters
    ----------
    log:
        Capture object being exported.
    prefix:
        Comment marker of the destination format (e.g. ``"# "``).

    Returns
    -------
    str
        Newline-terminated comment block.
    """

    return "".join(f"{prefix}{line}\n" for line in honesty_preamble_lines(log))


def _honesty_xml_comment(log: Any) -> str:
    """Format the shared honesty preamble as one XML/HTML comment.

    Returns
    -------
    str
        Single-line-per-fact comment; ``--`` is collapsed because XML
        comments must not contain double hyphens.
    """

    body = "\n".join(line.replace("--", "-") for line in honesty_preamble_lines(log))
    return f"<!-- {body} -->\n"


def _bundle_member_honesty(bundle: Any) -> dict[str, Any]:
    """Return per-member honesty facts for a Bundle export.

    Parameters
    ----------
    bundle:
        TorchLens ``Bundle``.

    Returns
    -------
    dict[str, Any]
        Member-name-keyed honesty fact blocks (best effort per member).
    """

    members: dict[str, Any] = {}
    for member_name in getattr(bundle, "names", ()) or ():
        # Per-member disclosure fallback: a member the bundle cannot serve (or
        # whose facts cannot be read) gets an explicit error row, never a
        # silent omission. The fact reader is getattr-defensive, so the
        # realistic raise surface is the member lookup itself.
        try:
            members[str(member_name)] = capture_honesty_facts(bundle[member_name])
        except (AttributeError, KeyError, TypeError):
            members[str(member_name)] = {"error": "member honesty facts unavailable"}
    return {"members": members}
