"""Barcode generation and argument hashing for tensor identity tracking."""

import base64
import random
import string
from typing import Any, List

_BARCODE_ALPHABET = string.ascii_letters + string.digits


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generates a random alphanumeric identifier string for a layer to use as internal label (invisible from user side).

    Args:
        barcode_len: Length of the desired barcode

    Returns:
        Random alphanumeric string.
    """
    return "".join(random.choices(_BARCODE_ALPHABET, k=barcode_len))


def make_short_barcode_from_input(things_to_hash: List[Any], barcode_len: int = 16) -> str:
    """Utility function that takes a list of anything and returns a short hash of it.

    Args:
        things_to_hash: List of things to hash; they must all be convertible to a string.
        barcode_len:

    Returns:
        Short hash of the input.
    """
    barcode = "\x00".join([str(x) for x in things_to_hash])
    barcode = str(hash(barcode))
    barcode = barcode.encode("utf-8")
    barcode = base64.urlsafe_b64encode(barcode)
    barcode = barcode.decode("utf-8")
    barcode = barcode[0:barcode_len]
    return barcode
