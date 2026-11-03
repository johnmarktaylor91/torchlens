"""The 3.0 saved format: typed JSON records plus safetensors blobs, no pickle.

``contract`` holds the constants every builder imports against (format name and
major, the record index, the storage-kind to JSON type map, the typed-value tags,
the tensor reference fields and the refusal code table). ``writer``, ``reader``
and ``runnable`` are the three doors; until their lanes land, each public
function refuses typed and names the module that fills it. The contract text
is ``docs/maintainers/saved_format_3.md``.
"""

from __future__ import annotations

from . import contract as contract

__all__ = ["contract"]
