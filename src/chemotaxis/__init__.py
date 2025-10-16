"""
Chemotaxis: A cell tracking program using ultrack and cellpose.

This package provides automated cell detection and tracking capabilities
for live-cell imaging analysis.
"""

__version__ = "0.1.0"
__author__ = "User"

from .tracker import CellTracker

__all__ = ["CellTracker", "__version__"]
