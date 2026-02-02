"""Compatibility shim for the refactored EdgeSelfAttn model.

The implementation lives in the package under `models/edge_selfattn/`.

Keeping this module (and the symbol name) avoids touching the rest of the codebase
that imports `GNSMsg_EdgeSelfAttn` from here.
"""

from __future__ import annotations

from .edge_selfattn import GNSMsg_EdgeSelfAttn

__all__ = ["GNSMsg_EdgeSelfAttn"]
