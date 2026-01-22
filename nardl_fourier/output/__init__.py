"""
Output module for publication-ready tables and plots.
"""

from .tables import (
    ResultsTable,
    generate_nardl_summary,
    generate_fnardl_summary,
    generate_bootstrap_nardl_summary,
    generate_long_run_table,
    generate_short_run_table,
    generate_diagnostics_table,
)
from .plots import (
    NARDLPlots,
    plot_dynamic_multipliers,
    plot_cusum,
    plot_cusumsq,
)

__all__ = [
    "ResultsTable",
    "generate_nardl_summary",
    "generate_fnardl_summary", 
    "generate_bootstrap_nardl_summary",
    "generate_long_run_table",
    "generate_short_run_table",
    "generate_diagnostics_table",
    "NARDLPlots",
    "plot_dynamic_multipliers",
    "plot_cusum",
    "plot_cusumsq",
]
