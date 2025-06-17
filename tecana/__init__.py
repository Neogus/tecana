import inspect
import functools
from .core import Tecana

__version__ = '0.8.0'

_ta = Tecana()

# Expose public methods from Tecana with proper docstrings
__all__ = []

for name, method in inspect.getmembers(_ta, inspect.ismethod):
    if name.startswith('_'):
        continue
    # Wrap the bound method to preserve docstrings and metadata
    wrapped_method = functools.update_wrapper(method, getattr(Tecana, name))
    globals()[name] = wrapped_method
    __all__.append(name)