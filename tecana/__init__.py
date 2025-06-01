"""
Tecana - Technical Analysis Library

A Python library for technical analysis of financial markets.
Provides optimized implementations of common technical indicators and signals.
"""

import inspect
from .core import Tecana

# Create instance for direct function access
_ta = Tecana()

# Programmatically expose all functions based on their suffix pattern
for name, func in inspect.getmembers(_ta, inspect.ismethod):
    if name.startswith('_'):  # Skip private methods
        continue
    # Add to module's __dict__ instead of globals() for more reliable behavior
    __dict__[name] = func

# Version info
__version__ = '0.1.0'