"""Model architectures for GNN load flow.

Each model lives in its own subpackage with the model class, message-passing
helpers, and a `builder.py` that registers the model in `models.registry`.

To add a new model:
  1. Create `models/<name>/` with at minimum `model.py` and `builder.py`
     (the builder must call `@register_model("<Name>")`).
  2. Import the subpackage here. Importing this manifest triggers builder
     registration for every model bundled with the project.
"""

from . import edge_selfattn  # noqa: F401  -- side-effect: registers GNSMsg_EdgeSelfAttn
