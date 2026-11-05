"""Foreign graph-viewer export targets (bridge tier).

The Netron writer: a bridge-tier member of the export-target registry (its
shape comes from a foreign peer). The writer is the schema-v2 ``._netron``
module family; the package facade keeps importing ``NETRON_DISCLAIMER`` and
``netron`` from this module. The Model Explorer family lives in its own package
(``torchlens/export/_model_explorer/``) and writes the panel-converged schema v3.
"""

from __future__ import annotations

from ._netron import NETRON_DISCLAIMER, netron

__all__ = ["NETRON_DISCLAIMER", "netron"]

__tl_layer__ = "L8"
