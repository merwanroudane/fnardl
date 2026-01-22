"""
Simulation module for critical value computation.
"""

from .critical_values_sim import (
    simulate_pss_critical_values,
    simulate_bootstrap_critical_values,
)

__all__ = [
    "simulate_pss_critical_values",
    "simulate_bootstrap_critical_values",
]
