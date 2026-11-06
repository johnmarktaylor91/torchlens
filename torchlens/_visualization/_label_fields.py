"""The closed vocabulary of ``node_label_fields=`` and its refusal."""

from __future__ import annotations

from typing import Final

from .._errors import InvalidArgumentError

#: Field names ``node_label_fields=`` accepts, in the order the refusal lists them.
NODE_LABEL_FIELDS: Final = (
    "label",
    "name",
    "type",
    "op",
    "operation",
    "shape",
    "shape_summary",
    "dtype",
    "memory",
    "bytes",
    "module",
    "params",
    "pass",
    "flops",
    "time",
)


def require_node_label_field(field_name: str) -> None:
    """Refuse a ``node_label_fields=`` entry outside :data:`NODE_LABEL_FIELDS`."""

    if field_name not in NODE_LABEL_FIELDS:
        raise InvalidArgumentError(
            f"Unsupported node label field: {field_name!r}",
            code="node_label_field_invalid",
            remedy=f"pass node label field names from {', '.join(NODE_LABEL_FIELDS)}",
            field=str(field_name),
        )


__all__ = ("NODE_LABEL_FIELDS", "require_node_label_field")
