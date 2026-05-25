"""HyperDEQ-PF pilot — minimal hypernetwork-conditioned PE_DEQ_PF for cross-voltage transfer.

Importing this package registers HyperDEQ_PF_Pilot in models.registry.
"""

from . import builder  # noqa: F401 -- registers the model
from .model import HyperDEQ_PF_Pilot  # noqa: F401

__all__ = ["HyperDEQ_PF_Pilot"]
