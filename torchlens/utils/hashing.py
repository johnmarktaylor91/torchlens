"""Barcode generation and argument hashing for tensor identity tracking.

Barcodes are short opaque identifiers attached to tensors during logging.
They serve two purposes:

1. **Random barcodes** (``make_random_barcode``): assigned to each tensor as
   it is created during the forward pass. These act as globally unique IDs
   so that the logging pipeline can track tensor identity across operations,
   even when the same ``torch.Tensor`` object is reused by in-place ops.

2. **Deterministic barcodes** (``make_short_barcode_from_input``): derived
   from the *content* of a tensor's creation arguments (param data pointers,
   buffer shapes, etc.). Two tensors with the same deterministic barcode
   originated from the same parameter/buffer and are candidates for
   *same-layer grouping* in loop detection — the barcode is the key signal
   that separate forward-pass operations actually reference the same weight.
"""

import base64
import random
import string
from typing import Any, List

_BARCODE_ALPHABET = string.ascii_letters + string.digits


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generate a random alphanumeric identifier for internal tensor tracking.

    These barcodes are invisible to the user and are used as unique
    internal keys for tensor entries in ``ModelLog``.

    Args:
        barcode_len: Length of the identifier string.

    Returns:
        Random alphanumeric string of the requested length.
    """
    return "".join(random.choices(_BARCODE_ALPHABET, k=barcode_len))


def make_short_barcode_from_input(things_to_hash: List[Any], barcode_len: int = 16) -> str:
    """Produce a deterministic short hash from a list of values.

    Used to create content-based barcodes for parameters and buffers so
    that loop detection can identify operations that share the same weights.
    The inputs are stringified, joined with a null-byte separator (to avoid
    accidental collisions from concatenation), hashed, and base64-encoded.

    Args:
        things_to_hash: Values to hash (must be stringifiable).
        barcode_len: Maximum length of the returned barcode.

    Returns:
        A URL-safe base64 string truncated to ``barcode_len`` characters.
    """
    # Null-byte separator prevents "ab" + "c" from colliding with "a" + "bc".
    joined = "\x00".join([str(x) for x in things_to_hash])
    hash_str = str(hash(joined))
    encoded: bytes = hash_str.encode("utf-8")
    b64: bytes = base64.urlsafe_b64encode(encoded)
    return b64.decode("utf-8")[:barcode_len]
