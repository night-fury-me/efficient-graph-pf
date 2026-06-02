"""Model architectures for GNN load flow.

Each model lives in its own subpackage with the model class, message-passing
helpers, and a `builder.py` that registers the model in `models.registry`.

To add a new model:
  1. Create `models/<name>/` with at minimum `model.py` and `builder.py`
     (the builder must call `@register_model("<Name>")`).
  2. Import the subpackage here. Importing this manifest triggers builder
     registration for every model bundled with the project.
"""

# Optional side-effect imports (model registration). Some bundled power-flow
# models depend on heavy/optional libs (e.g. torch_scatter); skip any that can't
# import so `import models` works in environments without them. The GNN
# experiments do not use these models.
import importlib as _il
for _m in ("edge_selfattn", "pe_deq_pf", "hyperdeq_pf_pilot"):
    try:
        _il.import_module(f".{_m}", __name__)  # noqa: F401 -- registers the model
    except Exception:
        pass
