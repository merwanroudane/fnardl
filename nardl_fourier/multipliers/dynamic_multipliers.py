"""
Dynamic Multipliers for NARDL Models
=====================================

Computation of asymmetric cumulative dynamic multipliers
with confidence intervals following Shin et al. (2014).

Author: Dr. Merwan Roudane
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_dynamic_multipliers(model, variable, horizon=None):
    """
    Compute asymmetric cumulative dynamic multipliers
    
    Following Shin, Yu & Greenwood-Nimmo (2014) Section 9.2.4
    
    The cumulative dynamic multiplier m_h measures the cumulative
    effect of a unit change in x on y from period 0 to h.
    
    Parameters
    ----------
    model : NARDL model object
        Fitted NARDL model
    variable : str
        Variable name to compute multipliers for
    horizon : int, optional
        Forecast horizon (default: q + 20)
    
    Returns
    -------
    dict
        Dictionary with positive and negative multipliers
    """
    coefs = model.model.params
    
    if horizon is None:
        horizon = max(model.best_lags['p'], model.best_lags['q']) + 20
    
    # Extract AR coefficients (φ)
    y_lag_names = [f'Y_L{j}' for j in range(1, model.best_lags['p'] + 1)]
    phi = np.array([coefs[name] if name in coefs else 0 for name in y_lag_names])
    
    # Extract DL coefficients (θ⁺ and θ⁻)
    q = model.best_lags['q']
    theta_pos = np.zeros(q + 1)
    theta_neg = np.zeros(q + 1)
    
    for j in range(q + 1):
        pos_name = f'{variable}_pos_L{j}'
        neg_name = f'{variable}_neg_L{j}'
        
        if pos_name in coefs:
            theta_pos[j] = coefs[pos_name]
        if neg_name in coefs:
            theta_neg[j] = coefs[neg_name]
    
    # Compute dynamic multipliers recursively
    # m_h = Σ(j=0 to min(h,p-1)) φ_j * m_{h-j-1} + θ_h (if h ≤ q)
    
    mult_pos = np.zeros(horizon)
    mult_neg = np.zeros(horizon)
    
    # h = 0
    mult_pos[0] = theta_pos[0]
    mult_neg[0] = theta_neg[0]
    
    # h = 1, 2, ..., horizon-1
    for h in range(1, horizon):
        # AR contribution
        for j in range(min(h, len(phi))):
            mult_pos[h] += phi[j] * mult_pos[h - j - 1]
            mult_neg[h] += phi[j] * mult_neg[h - j - 1]
        
        # DL contribution (only if h ≤ q)
        if h < len(theta_pos):
            mult_pos[h] += theta_pos[h]
            mult_neg[h] += theta_neg[h]
    
    # Cumulative multipliers
    cum_mult_pos = np.cumsum(mult_pos)
    cum_mult_neg = np.cumsum(mult_neg)
    
    # Long-run multipliers (as h → ∞)
    lr_pos = model.long_run[variable]['positive']['coefficient']
    lr_neg = model.long_run[variable]['negative']['coefficient']
    
    return {
        'positive': {
            'multiplier': mult_pos,
            'cumulative': cum_mult_pos,
            'long_run': lr_pos,
            'horizon': np.arange(horizon)
        },
        'negative': {
            'multiplier': mult_neg,
            'cumulative': cum_mult_neg,
            'long_run': lr_neg,
            'horizon': np.arange(horizon)
        },
        'asymmetry': cum_mult_pos - cum_mult_neg
    }


def bootstrap_multiplier_ci(model, variable, horizon=None, n_bootstrap=500, 
                            confidence=0.95, random_state=None):
    """
    Bootstrap confidence intervals for dynamic multipliers
    
    Uses residual resampling to generate bootstrap samples and
    compute confidence intervals for cumulative multipliers.
    
    Parameters
    ----------
    model : NARDL model object
    variable : str
        Variable name
    horizon : int, optional
        Forecast horizon
    n_bootstrap : int
        Number of bootstrap replications
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    random_state : int, optional
        Random seed
    
    Returns
    -------
    dict
        CI lower and upper bounds for positive and negative multipliers
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if horizon is None:
        horizon = max(model.best_lags['p'], model.best_lags['q']) + 20
    
    coefs = model.model.params
    vcov = model.model.cov_params()
    resid = model.model.resid.values
    
    boot_pos = []
    boot_neg = []
    
    for b in range(n_bootstrap):
        # Parametric bootstrap: sample coefficients from multivariate normal
        try:
            boot_coefs = np.random.multivariate_normal(coefs.values, vcov.values)
            boot_coefs_dict = dict(zip(coefs.index, boot_coefs))
            
            # Extract bootstrapped coefficients
            y_lag_names = [f'Y_L{j}' for j in range(1, model.best_lags['p'] + 1)]
            phi_boot = np.array([boot_coefs_dict.get(name, 0) for name in y_lag_names])
            
            q = model.best_lags['q']
            theta_pos_boot = np.zeros(q + 1)
            theta_neg_boot = np.zeros(q + 1)
            
            for j in range(q + 1):
                theta_pos_boot[j] = boot_coefs_dict.get(f'{variable}_pos_L{j}', 0)
                theta_neg_boot[j] = boot_coefs_dict.get(f'{variable}_neg_L{j}', 0)
            
            # Compute multipliers
            mult_pos_boot = np.zeros(horizon)
            mult_neg_boot = np.zeros(horizon)
            
            mult_pos_boot[0] = theta_pos_boot[0]
            mult_neg_boot[0] = theta_neg_boot[0]
            
            for h in range(1, horizon):
                for j in range(min(h, len(phi_boot))):
                    mult_pos_boot[h] += phi_boot[j] * mult_pos_boot[h - j - 1]
                    mult_neg_boot[h] += phi_boot[j] * mult_neg_boot[h - j - 1]
                if h < len(theta_pos_boot):
                    mult_pos_boot[h] += theta_pos_boot[h]
                    mult_neg_boot[h] += theta_neg_boot[h]
            
            boot_pos.append(np.cumsum(mult_pos_boot))
            boot_neg.append(np.cumsum(mult_neg_boot))
            
        except:
            continue
    
    boot_pos = np.array(boot_pos)
    boot_neg = np.array(boot_neg)
    
    alpha = (1 - confidence) / 2
    
    return {
        'positive': {
            'ci_lower': np.percentile(boot_pos, alpha * 100, axis=0),
            'ci_upper': np.percentile(boot_pos, (1 - alpha) * 100, axis=0),
            'mean': np.mean(boot_pos, axis=0),
            'std': np.std(boot_pos, axis=0)
        },
        'negative': {
            'ci_lower': np.percentile(boot_neg, alpha * 100, axis=0),
            'ci_upper': np.percentile(boot_neg, (1 - alpha) * 100, axis=0),
            'mean': np.mean(boot_neg, axis=0),
            'std': np.std(boot_neg, axis=0)
        }
    }


def plot_asymmetry(model, variable, figsize=(12, 6)):
    """
    Plot asymmetry between positive and negative multipliers
    
    Shows the difference m⁺_h - m⁻_h over the forecast horizon.
    
    Parameters
    ----------
    model : NARDL model
    variable : str
    figsize : tuple
    
    Returns
    -------
    matplotlib Figure
    """
    mult = model.dynamic_multipliers[variable]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    horizon = mult['positive']['horizon']
    
    # Left: Positive vs Negative cumulative multipliers
    ax1 = axes[0]
    ax1.plot(horizon, mult['positive']['cumulative'], 'b-', linewidth=2, label='m⁺ (Positive)')
    ax1.plot(horizon, mult['negative']['cumulative'], 'r-', linewidth=2, label='m⁻ (Negative)')
    ax1.axhline(mult['positive']['cumulative'][-1], color='blue', linestyle='--', 
               alpha=0.5, label=f"LR⁺: {mult['positive']['cumulative'][-1]:.3f}")
    ax1.axhline(mult['negative']['cumulative'][-1], color='red', linestyle='--',
               alpha=0.5, label=f"LR⁻: {mult['negative']['cumulative'][-1]:.3f}")
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Cumulative Multiplier')
    ax1.set_title(f'Cumulative Multipliers: {variable}', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right: Asymmetry (difference)
    ax2 = axes[1]
    asymmetry = mult['asymmetry']
    ax2.fill_between(horizon, 0, asymmetry, where=asymmetry >= 0,
                     color='#3B82F6', alpha=0.5, label='m⁺ > m⁻')
    ax2.fill_between(horizon, 0, asymmetry, where=asymmetry < 0,
                     color='#DC2626', alpha=0.5, label='m⁺ < m⁻')
    ax2.plot(horizon, asymmetry, 'k-', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Asymmetry (m⁺ - m⁻)')
    ax2.set_title('Dynamic Asymmetry', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Asymmetric Adjustment Analysis: {variable}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def long_run_convergence_check(model, variable, tolerance=0.01):
    """
    Check if cumulative multipliers converge to long-run values
    
    Parameters
    ----------
    model : NARDL model
    variable : str
    tolerance : float
        Convergence tolerance
    
    Returns
    -------
    dict
        Convergence analysis results
    """
    mult = model.dynamic_multipliers[variable]
    lr = model.long_run[variable]
    
    cum_pos = mult['positive']['cumulative']
    cum_neg = mult['negative']['cumulative']
    
    lr_pos = lr['positive']['coefficient']
    lr_neg = lr['negative']['coefficient']
    
    # Find convergence horizon
    pos_converged = np.where(np.abs(cum_pos - lr_pos) / (np.abs(lr_pos) + 1e-10) < tolerance)[0]
    neg_converged = np.where(np.abs(cum_neg - lr_neg) / (np.abs(lr_neg) + 1e-10) < tolerance)[0]
    
    pos_horizon = pos_converged[0] if len(pos_converged) > 0 else len(cum_pos)
    neg_horizon = neg_converged[0] if len(neg_converged) > 0 else len(cum_neg)
    
    return {
        'positive': {
            'converged': len(pos_converged) > 0,
            'convergence_horizon': int(pos_horizon),
            'final_value': cum_pos[-1],
            'long_run': lr_pos,
            'gap_percent': 100 * (cum_pos[-1] - lr_pos) / (np.abs(lr_pos) + 1e-10)
        },
        'negative': {
            'converged': len(neg_converged) > 0,
            'convergence_horizon': int(neg_horizon),
            'final_value': cum_neg[-1],
            'long_run': lr_neg,
            'gap_percent': 100 * (cum_neg[-1] - lr_neg) / (np.abs(lr_neg) + 1e-10)
        },
        'tolerance': tolerance
    }
