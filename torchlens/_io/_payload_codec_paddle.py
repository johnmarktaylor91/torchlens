"""Paddle place resolution for the payload codec."""

from __future__ import annotations

from typing import Any


def _resolve_paddle_place(paddle_module: Any, place_name: Any) -> Any | None:
    """Resolve a Paddle place object from manifest or map-location text."""

    if place_name is None:
        return None
    base_place = getattr(paddle_module, "Place", None)
    if base_place is not None and isinstance(place_name, base_place):
        return place_name
    if not isinstance(place_name, str):
        return None
    requested = place_name.lower()
    if requested in {"", "unknown"}:
        return None
    if requested in {"cpu", "place(cpu)"}:
        return paddle_module.CPUPlace()
    if requested in {"gpu", "cuda", "place(gpu)", "place(cuda)"}:
        cuda_place = getattr(paddle_module, "CUDAPlace", None)
        if cuda_place is not None:
            return cuda_place(0)
    for prefix in ("gpu:", "cuda:"):
        if requested.startswith(prefix):
            cuda_place = getattr(paddle_module, "CUDAPlace", None)
            if cuda_place is None:
                return None
            try:
                return cuda_place(int(requested.split(":", maxsplit=1)[1]))
            except ValueError:
                return None
    return None
