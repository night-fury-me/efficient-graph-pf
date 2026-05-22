"""PE_DEQ_PF model package.

Permutation-Equivariant Deep-Equilibrium AC-PF surrogate. Replaces the
K-step line-search post-correction in PIGNN-Attn-LS with a single
weight-tied operator whose fixed point is the AC-PF solution.

Public entrypoint:
    from models.pe_deq_pf import PE_DEQ_PF

Importing this package also imports `builder`, which self-registers
`PE_DEQ_PF` into `models.registry.MODEL_REGISTRY`.
"""

from .model import PE_DEQ_PF
from . import builder  # noqa: F401 -- side-effect: registers the model

__all__ = ["PE_DEQ_PF"]
