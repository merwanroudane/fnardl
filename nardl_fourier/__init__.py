"""
NARDL-Fourier: Nonlinear ARDL with Fourier Approximation and Bootstrap Tests
=============================================================================

A comprehensive Python library for Nonlinear Autoregressive Distributed Lag (NARDL) 
modeling with Fourier approximation and Bootstrap cointegration tests.

Three Core Models:
- NARDL: Standard Nonlinear ARDL (Shin, Yu & Greenwood-Nimmo, 2014)
- FourierNARDL: Fourier NARDL for smooth structural breaks (Zaghdoudi et al., 2023)
- BootstrapNARDL: Bootstrap cointegration tests (Bertelli, Vacca & Zoia, 2022)

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
GitHub: https://github.com/merwanroudane/fnardl
"""

__version__ = "1.0.1"
__author__ = "Dr. Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

from .core.nardl import NARDL
from .core.fnardl import FourierNARDL
from .core.fbnardl import BootstrapNARDL
from .bounds_test.pss_bounds import PSSBoundsTest
from .core.fbnardl import BootstrapCointegrationTest
from .output.tables import ResultsTable
from .output.plots import NARDLPlots

__all__ = [
    "NARDL",
    "FourierNARDL", 
    "BootstrapNARDL",
    "PSSBoundsTest",
    "BootstrapCointegrationTest",
    "ResultsTable",
    "NARDLPlots",
    "__version__",
    "__author__",
]
