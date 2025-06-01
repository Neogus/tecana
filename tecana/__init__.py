import inspect
from .core import Tecana

_ta = Tecana()

for name, func in inspect.getmembers(_ta, inspect.ismethod):
    if name.startswith('_'):  # Skip private methods
        continue
    globals()[name] = func  # <-- fixed here

__version__ = '0.1.0'
