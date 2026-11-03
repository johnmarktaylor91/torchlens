"""Duplicate-key-safe reader for the keyed baseline TSVs that merge as a union.

``diagnostic_ratchets.tsv``, ``universe_cardinalities.tsv`` and
``harvest_counts.tsv`` are ``merge=union`` in ``.gitattributes`` so two
branches' appended rows both survive a merge. Their rows are KEYED (one count
per metric), so when two branches edit the same count the union keeps both
rows instead of conflicting, and a dict loader would silently keep whichever
row comes last. Every loader reads through :func:`parse_keyed_tsv`, which
refuses a repeated key so a bad union merge fails loudly.
"""

from __future__ import annotations


def parse_keyed_tsv(text: str, *, filename: str, header_key: str) -> dict[str, str]:
    """Parse ``key<TAB>value`` rows, refusing a key that appears twice.

    Parameters
    ----------
    text:
        File contents. Blank lines, ``#`` comments and the header row are skipped.
    filename:
        File name, for the error message.
    header_key:
        First column of the header row (``"metric"``, ``"universe_id"``).

    Returns
    -------
    dict[str, str]
        Key to stripped value, in file order.

    Raises
    ------
    AssertionError
        When a key appears on more than one row.
    """

    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith(f"{header_key}\t"):
            continue
        key, _, value = line.partition("\t")
        key, value = key.strip(), value.strip()
        if key in rows:
            raise AssertionError(
                f"{filename}: key {key!r} appears twice (values {rows[key]!r} and {value!r}). "
                "A union merge kept both branches' rows; keep one row and recompute it as "
                "the train value plus your delta."
            )
        rows[key] = value
    return rows
