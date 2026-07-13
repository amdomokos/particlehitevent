"""Model registry (Phase 3).

Phase 3 registers no models — an empty registry is the correct state until
Phase 4 adds one file per architecture, each calling ``register_model``.
The engine and ``Models/train.py`` look models up here by name so every
architecture runs through byte-identical training code.
"""

_MODEL_BUILDERS = {}


def register_model(name):
    """Decorator: register a callable(config: dict) -> nn.Module under ``name``."""
    def deco(fn):
        if name in _MODEL_BUILDERS:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_BUILDERS[name] = fn
        return fn
    return deco


def build_model(name, config):
    """Look up a registered builder and call it with the config dict."""
    if name not in _MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(_MODEL_BUILDERS)}"
        )
    return _MODEL_BUILDERS[name](config)


def list_models():
    """Sorted registered model names. Empty list (not an error) when empty,
    so argparse ``choices`` can be built lazily."""
    return sorted(_MODEL_BUILDERS)
