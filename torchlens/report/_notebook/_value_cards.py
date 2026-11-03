"""Notebook record cards reuse exactly the bounded text card rows."""

from __future__ import annotations

from html import escape
from typing import Any

from ..._data_classes._value_repr import layer_card, op_card, param_card
from .cardtree import Card, CardHtml


def _from_text(text: str, kind: str) -> Card:
    """Escape one bounded card into a two-column table, without tensor previews."""

    title, *lines = text.splitlines()
    rows = []
    for line in lines:
        label, value = line[2:11].strip(), line[12:]
        rows.append(f'<tr><th scope="row">{escape(label)}</th><td>{escape(value)}</td></tr>')
    table = '<table class="tl-record-facts"><tbody>' + "".join(rows) + "</tbody></table>"
    return Card(title=title, children=(CardHtml(table),), kind=kind)


def op_value_card(op: Any, *, kind: str = "op") -> Card:
    """The text Op card as safe notebook HTML with no automatic value grid."""

    return _from_text(op_card(op), kind)


def layer_value_card(layer: Any) -> Card:
    """The text Layer card, including its bounded per-pass rows."""

    return _from_text(layer_card(layer), "layer")


def param_value_card(param: Any) -> Card:
    """The text Param card; printing never triggers live statistics."""

    return _from_text(param_card(param), "param")
