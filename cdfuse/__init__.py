"""CDFuse — compare two gridded NetCDF datasets.

The package is deliberately independent of Streamlit so the processing
pipeline can be imported from scripts, notebooks or tests. ``app.py``
provides the web interface on top of it.
"""

from __future__ import annotations

from .config import APP_NAME, APP_TAGLINE, VERSION

__all__ = ["APP_NAME", "APP_TAGLINE", "VERSION"]
__version__ = VERSION
