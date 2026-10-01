"""Foreign graph-viewer export targets (bridge tier; C01 item 18).

The Netron writer: a bridge-tier member of the export-target registry (its
shape comes from a foreign peer). Lane F14 grew the writer into the schema-v2
``._netron`` module family without touching the package facade (which keeps
importing ``NETRON_DISCLAIMER`` and ``netron`` from this module); the Model
Explorer family moved to its own package
(``torchlens/export/_model_explorer/``, lane F15) when the v2 single-file
writer was superseded by the panel-converged schema v3.
"""

from __future__ import annotations

from ._netron import NETRON_DISCLAIMER, netron

__all__ = ["NETRON_DISCLAIMER", "netron"]

__tl_layer__ = "L8"
