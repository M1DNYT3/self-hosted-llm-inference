# inference/backends/__init__.py
# Import all backends to trigger @register() side-effects.
from inference.backends import cpu, local, vastai  # noqa: F401
