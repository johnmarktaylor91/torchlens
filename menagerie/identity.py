"""Stable identity helpers for the TorchLens model menagerie catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from menagerie.catalog import CatalogRow


RECIPE_SCHEME_VERSION = 1
StableKey = tuple[str, str, str]
NAME_CHUNK_SIZE = 24
NAME_CHUNK_THRESHOLD = 60


def canonical_recipe_v1(row: CatalogRow) -> str:
    """Serialize the recipe identity for a catalog row.

    Parameters
    ----------
    row:
        Canonical catalog row.

    Returns
    -------
    str
        Frozen JSON serializer output for recipe scheme version 1.
    """

    if row.source == "classics":
        from menagerie.classics import CLASSICS

        entry = CLASSICS[row.name]
        recipe: dict[str, Any] = {
            "module_path": str(entry["module_path"]),
            "build_fn": getattr(entry["build"], "__name__", "build"),
        }
    else:
        recipe = {"constructor_call": row.constructor_call}
    payload = {
        "recipe_scheme_version": RECIPE_SCHEME_VERSION,
        "source": row.source,
        "recipe": recipe,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def recipe_revision_sha256(row: CatalogRow) -> str:
    """Return the SHA-256 fingerprint of a row's canonical recipe.

    Parameters
    ----------
    row:
        Canonical catalog row.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """

    return hashlib.sha256(canonical_recipe_v1(row).encode("utf-8")).hexdigest()


def _stable_id_ordinal(stable_id: str) -> int:
    """Return the numeric ordinal encoded in an opaque stable ID.

    Parameters
    ----------
    stable_id:
        Opaque stable ID.

    Returns
    -------
    int
        Parsed ordinal, or zero when the ID does not match the local ordinal format.
    """

    if stable_id.startswith("m") and stable_id[1:].isdigit():
        return int(stable_id[1:])
    return 0


def _encode_name(name: str) -> str | list[str]:
    """Encode a natural-key name for the stable-id table.

    Parameters
    ----------
    name:
        Natural-key name.

    Returns
    -------
    str | list[str]
        Raw name for short strings, or reversible chunks for long strings.
    """

    if len(name) <= NAME_CHUNK_THRESHOLD:
        return name
    return [name[index : index + NAME_CHUNK_SIZE] for index in range(0, len(name), NAME_CHUNK_SIZE)]


def _decode_name(value: object) -> str:
    """Decode a natural-key name from the stable-id table.

    Parameters
    ----------
    value:
        Encoded name.

    Returns
    -------
    str
        Decoded name.
    """

    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _load_stable_id_state(path: Path) -> tuple[dict[StableKey, str], dict[str, int]]:
    """Load stable IDs and zoo aliases from disk.

    Parameters
    ----------
    path:
        JSONL stable-id table.

    Returns
    -------
    tuple[dict[StableKey, str], dict[str, int]]
        Stable ID mapping and zoo-to-alias mapping.
    """

    mapping: dict[StableKey, str] = {}
    seen_ids: set[str] = set()
    zoo_by_index: dict[int, str] = {}
    zoo_indexes: dict[str, int] = {}
    if not path.exists():
        return mapping, zoo_indexes

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                key = (str(record["name"]), str(record["zoo"]), str(record.get("variant", "")))
                stable_id = str(record["stable_id"])
            elif isinstance(record, list) and len(record) == 2 and isinstance(record[0], int):
                index = int(record[0])
                zoo = str(record[1])
                if index in zoo_by_index:
                    raise ValueError(f"{path}:{line_number} duplicates zoo index {index}")
                zoo_by_index[index] = zoo
                zoo_indexes[zoo] = index
                continue
            elif isinstance(record, list) and len(record) in {3, 4}:
                name = _decode_name(record[0])
                zoo_index = int(record[1])
                zoo = zoo_by_index[zoo_index]
                if len(record) == 3:
                    variant = ""
                    stable_id = str(record[2])
                else:
                    variant = str(record[2])
                    stable_id = str(record[3])
                key = (name, zoo, variant)
            else:
                raise ValueError(f"{path}:{line_number} has unsupported stable-id record")
            if key in mapping:
                raise ValueError(f"{path}:{line_number} duplicates stable key {key!r}")
            if stable_id in seen_ids:
                raise ValueError(f"{path}:{line_number} duplicates stable_id {stable_id!r}")
            mapping[key] = stable_id
            seen_ids.add(stable_id)
    return mapping, zoo_indexes


def load_stable_ids(path: Path) -> dict[StableKey, str]:
    """Load stable IDs keyed by ``(name, zoo, variant)``.

    Parameters
    ----------
    path:
        JSONL stable-id table.

    Returns
    -------
    dict[StableKey, str]
        Stable ID mapping.
    """

    mapping, _zoo_indexes = _load_stable_id_state(path)
    return mapping


def ensure_stable_ids(
    keys: set[StableKey],
    path: Path,
) -> dict[StableKey, str]:
    """Ensure every natural key has an opaque stable ID.

    Parameters
    ----------
    keys:
        Natural keys that must exist in the ID table.
    path:
        JSONL stable-id table.

    Returns
    -------
    dict[StableKey, str]
        Complete stable ID mapping.
    """

    mapping, zoo_indexes = _load_stable_id_state(path)
    missing = sorted(keys - set(mapping))
    if not missing:
        return mapping

    max_ordinal = 0
    for stable_id in mapping.values():
        max_ordinal = max(max_ordinal, _stable_id_ordinal(stable_id))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for zoo in sorted({zoo for _name, zoo, _variant in missing} - set(zoo_indexes)):
            index = len(zoo_indexes)
            zoo_indexes[zoo] = index
            handle.write(json.dumps([index, zoo], separators=(",", ":")) + "\n")
        for offset, key in enumerate(missing, start=1):
            stable_id = f"m{max_ordinal + offset}"
            name, zoo, variant = key
            if variant:
                record = [_encode_name(name), zoo_indexes[zoo], variant, stable_id]
            else:
                record = [_encode_name(name), zoo_indexes[zoo], stable_id]
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            mapping[key] = stable_id
    return mapping
