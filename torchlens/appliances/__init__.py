"""The appliance door registry: ``tl.appliances.register`` and ``tl.appliances.names``."""

from ._appliances import names, register

__all__ = [
    "names",
    "register",
]
