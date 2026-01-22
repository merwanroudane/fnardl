"""
Visualization Functions for NARDL Results
==========================================

Publication-quality plots for dynamic multipliers, CUSUM tests,
and other diagnostic visualizations.

Author: Dr. Merwan Roudane
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class NARDLPlots:
    """
    Class for generating NARDL visualizations
    
    Parameters
    ----------
    model : NARDL, FourierNARDL, or BootstrapNARDL
        Fitted NARDL model
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
    
    Examples
    --------
    >>> plots = NARDLPlots(model)
    >>> fig = plots.multipliers()
    >>> fig = plots.cusum()
    """
    
    def __init__(self, model, style=None):
        self.model = model
        if style:
            try:
                plt.style.use(style)
            except:
                pass
    
    def multipliers(self, variable=None, figsize=(14, 10)):
        """Plot dynamic multipliers with confidence intervals"""
        return plot_dynamic_multipliers(self.model, variable, figsize)
    
    def cusum(self, figsize=(10, 6)):
        """Plot CUSUM test"""
        return plot_cusum(self.model, figsize)
    
    def cusumsq(self, figsize=(10, 6)):
        """Plot CUSUM of Squares test"""
        return plot_cusumsq(self.model, figsize)
    
    def short_run_coefficients(self, variable=None, figsize=(14, 10)):
        """Plot short-run coefficients"""
        return plot_short_run_coefficients(self.model, variable, figsize)
    
    def ect_adjustment(self, figsize=(12, 5)):
        """Plot ECT adjustment path"""
        return plot_ect_adjustment(self.model, figsize)
    
    def residuals(self, figsize=(14, 8)):
        """Plot residual diagnostics"""
        return plot_residuals(self.model, figsize)


def plot_dynamic_multipliers(model, variable=None, figsize=(14, 10)):
    """
    Plot asymmetric cumulative dynamic multipliers with 95% CI
    
    Parameters
    ----------
    model : NARDL model
        Fitted model with dynamic_multipliers attribute
    variable : str, optional
        Specific variable to plot. If None, plots first decomposed variable
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if variable is None:
        variable = model.decomp_vars[0]
    
    mult = model.dynamic_multipliers[variable]
    lr = model.long_run[variable]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    horizon = mult['positive']['horizon']
    
    # Top-left: Positive dynamic multiplier
    ax1 = axes[0, 0]
    ax1.plot(horizon, mult['positive']['multiplier'], 'b-', linewidth=2, label='Multiplier')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(horizon, 
                     mult['positive']['multiplier'] * 0.9,
                     mult['positive']['multiplier'] * 1.1,
                     alpha=0.2, color='blue')
    ax1.set_title(f'Dynamic Multiplier - {variable}⁺', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Multiplier')
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Negative dynamic multiplier
    ax2 = axes[0, 1]
    ax2.plot(horizon, mult['negative']['multiplier'], 'r-', linewidth=2, label='Multiplier')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(horizon,
                     mult['negative']['multiplier'] * 0.9,
                     mult['negative']['multiplier'] * 1.1,
                     alpha=0.2, color='red')
    ax2.set_title(f'Dynamic Multiplier - {variable}⁻', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Multiplier')
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Cumulative positive with CI and long-run
    ax3 = axes[1, 0]
    ax3.plot(horizon, mult['positive']['cumulative'], 'b-', linewidth=2, label='Cumulative')
    ax3.fill_between(horizon, 
                     mult['positive']['ci_lower'],
                     mult['positive']['ci_upper'],
                     alpha=0.3, color='blue', label='95% CI')
    ax3.axhline(lr['positive']['coefficient'], color='green', linestyle='--', 
               linewidth=2, label=f"Long-run: {lr['positive']['coefficient']:.3f}")
    ax3.set_title(f'Cumulative Multiplier - {variable}⁺', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('Cumulative Effect')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Cumulative negative with CI and long-run
    ax4 = axes[1, 1]
    ax4.plot(horizon, mult['negative']['cumulative'], 'r-', linewidth=2, label='Cumulative')
    ax4.fill_between(horizon,
                     mult['negative']['ci_lower'],
                     mult['negative']['ci_upper'],
                     alpha=0.3, color='red', label='95% CI')
    ax4.axhline(lr['negative']['coefficient'], color='green', linestyle='--',
               linewidth=2, label=f"Long-run: {lr['negative']['coefficient']:.3f}")
    ax4.set_title(f'Cumulative Multiplier - {variable}⁻', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Horizon')
    ax4.set_ylabel('Cumulative Effect')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Asymmetric Dynamic Multipliers: {variable}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_cusum(model, figsize=(10, 6)):
    """
    Plot CUSUM stability test
    
    The CUSUM test (Brown, Durbin & Evans, 1975) tests for parameter stability.
    If the cumulative sum stays within the 5% significance lines, the model
    is considered stable.
    
    Parameters
    ----------
    model : NARDL model
    figsize : tuple
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    residuals = model.model.resid.values
    k = len(model.model.params)
    n = int(model.model.nobs)
    
    # Compute standardized recursive residuals
    w = residuals[~np.isnan(residuals)]
    w_sd = np.std(w, ddof=1)
    w_cumsum = np.cumsum(w / w_sd)
    
    # Critical value at 5% significance
    c_val = 0.984
    x = np.arange(k, k + len(w_cumsum))
    
    # Confidence bands
    upper = c_val * np.sqrt(n - k) + (2 * c_val * np.sqrt(n - k)) * (x - k) / len(w_cumsum)
    lower = -c_val * np.sqrt(n - k) + (-2 * c_val * np.sqrt(n - k)) * (x - k) / len(w_cumsum)
    
    # Plot
    ax.plot(x, w_cumsum, color='#3B82F6', linewidth=2, label='CUSUM')
    ax.plot(x, upper, color='#DC2626', linestyle='--', linewidth=1.5, label='5% Significance')
    ax.plot(x, lower, color='#DC2626', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    # Fill between for visual clarity
    ax.fill_between(x, lower, upper, alpha=0.1, color='green')
    
    ax.set_xlabel('Observation', fontsize=12)
    ax.set_ylabel('Cumulative Sum', fontsize=12)
    ax.set_title('CUSUM Test for Parameter Stability', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Check if stable
    is_stable = np.all(w_cumsum >= lower[:len(w_cumsum)]) and np.all(w_cumsum <= upper[:len(w_cumsum)])
    stability_text = "✓ STABLE" if is_stable else "✗ UNSTABLE"
    color = 'green' if is_stable else 'red'
    ax.text(0.98, 0.02, stability_text, transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=color, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_cusumsq(model, figsize=(10, 6)):
    """
    Plot CUSUM of Squares stability test
    
    Tests for stability of variance over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    residuals = model.model.resid.values
    k = len(model.model.params)
    n = int(model.model.nobs)
    
    # Compute CUSUM of squares
    resid_sq = residuals ** 2
    cusum_sq = np.cumsum(resid_sq) / np.sum(resid_sq)
    
    x = np.arange(k, k + len(cusum_sq))
    t_normalized = (x - k) / (n - k)
    
    # Significance lines (5%)
    c = 0.1 + 0.032  # Approximate critical value
    upper = t_normalized + c
    lower = t_normalized - c
    
    # Clip to [0, 1]
    upper = np.clip(upper, 0, 1)
    lower = np.clip(lower, 0, 1)
    
    ax.plot(x, cusum_sq, color='#3B82F6', linewidth=2, label='CUSUM²')
    ax.plot(x, upper, color='#DC2626', linestyle='--', linewidth=1.5, label='5% Significance')
    ax.plot(x, lower, color='#DC2626', linestyle='--', linewidth=1.5)
    ax.plot(x, t_normalized, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.fill_between(x, lower, upper, alpha=0.1, color='green')
    
    ax.set_xlabel('Observation', fontsize=12)
    ax.set_ylabel('Cumulative Sum of Squares', fontsize=12)
    ax.set_title('CUSUM of Squares Test', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    return fig


def plot_short_run_coefficients(model, variable=None, figsize=(14, 10)):
    """Plot short-run coefficients by lag"""
    if variable is None:
        variable = model.decomp_vars[0]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    pos_data = model.short_run[variable]['positive']
    neg_data = model.short_run[variable]['negative']
    cum_data = model.short_run_cumulative[variable]
    
    # Positive coefficients
    ax1 = axes[0, 0]
    if pos_data:
        lags = [d['lag'] for d in pos_data]
        coefs = [d['coefficient'] for d in pos_data]
        colors = ['#2563EB' if d['significant'] else '#93C5FD' for d in pos_data]
        ax1.bar(lags, coefs, color=colors, edgecolor='navy', alpha=0.8)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Coefficient')
    ax1.set_title(f'Short-Run Coefficients: {variable}⁺', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    legend_elements = [Patch(facecolor='#2563EB', label='Significant (p<0.05)'),
                       Patch(facecolor='#93C5FD', label='Not significant')]
    ax1.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # Negative coefficients
    ax2 = axes[0, 1]
    if neg_data:
        lags = [d['lag'] for d in neg_data]
        coefs = [d['coefficient'] for d in neg_data]
        colors = ['#DC2626' if d['significant'] else '#FCA5A5' for d in neg_data]
        ax2.bar(lags, coefs, color=colors, edgecolor='darkred', alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Coefficient')
    ax2.set_title(f'Short-Run Coefficients: {variable}⁻', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Comparison
    ax3 = axes[1, 0]
    if pos_data and neg_data:
        lags = [d['lag'] for d in pos_data]
        pos_coefs = [d['coefficient'] for d in pos_data]
        neg_coefs = [d['coefficient'] for d in neg_data]
        x = np.arange(len(lags))
        width = 0.35
        ax3.bar(x - width/2, pos_coefs, width, label='Positive', color='#3B82F6', alpha=0.8)
        ax3.bar(x + width/2, neg_coefs, width, label='Negative', color='#EF4444', alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Lag {l}' for l in lags])
        ax3.legend()
    ax3.set_title('Asymmetry Comparison (Positive vs Negative)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Cumulative effects
    ax4 = axes[1, 1]
    categories = ['Σγ⁺', 'Σγ⁻']
    values = [cum_data['positive']['coefficient'], cum_data['negative']['coefficient']]
    colors = ['#3B82F6', '#EF4444']
    bars = ax4.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_title('Cumulative Short-Run Effects', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Short-Run Analysis: {variable}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_ect_adjustment(model, figsize=(12, 5)):
    """Plot ECT adjustment path visualization"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ect = model.ect
    
    # ECT coefficient with CI
    ax1 = axes[0]
    ect_val = ect['coefficient']
    ci_lower = ect['ci_lower']
    ci_upper = ect['ci_upper']
    
    color = '#059669' if ect['is_valid'] and ect['is_stable'] else '#DC2626'
    
    ax1.barh(['ECT (ρ)'], [ect_val], color=color, alpha=0.7, edgecolor='black')
    ax1.errorbar([ect_val], ['ECT (ρ)'], 
                 xerr=[[ect_val - ci_lower], [ci_upper - ect_val]],
                 fmt='none', color='black', capsize=10, capthick=2)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(x=-1, color='orange', linestyle='--', linewidth=1.5, label='Optimal (-1)')
    ax1.axvline(x=-2, color='red', linestyle=':', linewidth=1.5, label='Stability bound')
    ax1.set_xlabel('Coefficient Value', fontsize=11)
    ax1.set_title('Error Correction Term', fontweight='bold', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add text annotation
    status = "✓ Valid" if ect['is_valid'] and ect['is_stable'] else "✗ Invalid"
    ax1.text(0.02, 0.98, f"Status: {status}\nt = {ect['t_statistic']:.3f}",
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjustment path simulation
    ax2 = axes[1]
    if ect['is_valid'] and ect['is_stable']:
        periods = 25
        path = [1.0]  # Initial shock of 1 unit
        for t in range(1, periods):
            path.append(path[-1] * (1 + ect_val))
        
        ax2.plot(range(periods), path, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=1.5, label='Equilibrium')
        
        # Mark half-life
        if not np.isnan(ect['half_life']):
            hl = int(min(ect['half_life'], periods-1))
            ax2.axvline(x=hl, color='orange', linestyle=':', linewidth=1.5, 
                       label=f'Half-life ≈ {ect["half_life"]:.1f}')
            ax2.scatter([hl], [path[hl]], color='orange', s=100, zorder=5)
        
        ax2.set_xlabel('Periods after shock', fontsize=11)
        ax2.set_ylabel('Deviation from equilibrium', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.set_ylim(-0.1, 1.1)
    else:
        ax2.text(0.5, 0.5, 'No stable\nconvergence path', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, color='red')
    
    ax2.set_title('Adjustment Path', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residuals(model, figsize=(14, 8)):
    """Plot residual diagnostics"""
    from scipy import stats
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    resid = model.model.resid.values
    fitted = model.model.fittedvalues.values
    
    # Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(fitted, resid, alpha=0.6, color='#3B82F6', edgecolors='navy', s=30)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2 = axes[0, 1]
    ax2.hist(resid, bins=30, density=True, alpha=0.7, color='#3B82F6', edgecolor='navy')
    
    # Normal distribution overlay
    x_norm = np.linspace(resid.min(), resid.max(), 100)
    ax2.plot(x_norm, stats.norm.pdf(x_norm, resid.mean(), resid.std()),
             'r-', linewidth=2, label='Normal')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Density')
    ax2.set_title('Residual Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax3 = axes[1, 0]
    stats.probplot(resid, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Residuals over time
    ax4 = axes[1, 1]
    ax4.plot(range(len(resid)), resid, color='#3B82F6', linewidth=1)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax4.axhline(y=2*resid.std(), color='gray', linestyle=':', linewidth=1)
    ax4.axhline(y=-2*resid.std(), color='gray', linestyle=':', linewidth=1)
    ax4.set_xlabel('Observation')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Over Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig
