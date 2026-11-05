"""F12 roster pins: nine rows, tiers, subjects, compositions, memo settings.

The roster's exact settings are the memo's (trilabs themes memo section 3);
these pins hold the registry rows to the panel's ruling so a later edit
cannot silently move a settled cell (the dims collapse="none" 2-1 cell, the
three view pins, the transformer no-filter rule).
"""

from __future__ import annotations

from torchlens._visualization.theme_registry import describe_lens


def test_describe_teaches_exact_settings() -> None:
    """describe() prints real draw parameters with exact values (N12)."""

    text = describe_lens("dims")
    assert "size_by='dims'" in text
    assert "what shape is the data?" in text
