"""
AI-ON-FPGA package root
"""
from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # editable-install or source tree
    __version__ = "0.1.0"
