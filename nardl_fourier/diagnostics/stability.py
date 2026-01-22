"""
Stability Tests for NARDL Models
=================================

CUSUM and CUSUM of Squares tests for parameter stability.

Author: Dr. Merwan Roudane
"""

import numpy as np
from scipy import stats


def cusum_test(residuals, k, significance=0.05):
    """
    CUSUM test for parameter stability
    
    Tests whether model parameters are stable over time.
    Based on Brown, Durbin & Evans (1975).
    
    H0: Parameters are stable
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    k : int
        Number of parameters in the model
    significance : float
        Significance level (default 0.05)
    
    Returns
    -------
    dict
        Test results including:
        - cusum: Cumulative sum values
        - upper: Upper bound
        - lower: Lower bound  
        - is_stable: Whether model is stable
        - max_deviation: Maximum deviation from bounds
    """
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    n = len(residuals)
    
    # Standardize residuals
    sigma = np.std(residuals, ddof=1)
    w = residuals / sigma
    
    # Cumulative sum
    cusum = np.cumsum(w)
    
    # Critical value for 5% significance
    if significance == 0.05:
        c_val = 0.984
    elif significance == 0.01:
        c_val = 1.143
    elif significance == 0.10:
        c_val = 0.850
    else:
        c_val = 0.984
    
    # Confidence bands (expanding with observations)
    t = np.arange(k, k + len(cusum))
    a = c_val * np.sqrt(n - k)
    b = 2 * a / len(cusum)
    
    upper = a + b * (t - k)
    lower = -a - b * (t - k)
    
    # Check stability
    is_stable = np.all(cusum >= lower[:len(cusum)]) and np.all(cusum <= upper[:len(cusum)])
    
    # Maximum deviation
    deviation_upper = np.max(cusum - upper[:len(cusum)]) if len(cusum) > 0 else 0
    deviation_lower = np.max(lower[:len(cusum)] - cusum) if len(cusum) > 0 else 0
    max_deviation = max(deviation_upper, deviation_lower)
    
    return {
        'test': 'CUSUM',
        'cusum': cusum,
        'upper_bound': upper,
        'lower_bound': lower,
        'critical_value': c_val,
        'is_stable': is_stable,
        'max_deviation': max_deviation,
        'null_hypothesis': 'Parameters are stable',
        'conclusion': 'Stable' if is_stable else 'Unstable (structural break detected)'
    }


def cusumsq_test(residuals, k, significance=0.05):
    """
    CUSUM of Squares test for variance stability
    
    Tests whether the variance of residuals is stable over time.
    Based on Brown, Durbin & Evans (1975).
    
    H0: Variance is stable
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    k : int
        Number of parameters in the model
    significance : float
        Significance level
    
    Returns
    -------
    dict
        Test results
    """
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    n = len(residuals)
    
    # Squared residuals
    resid_sq = residuals ** 2
    
    # Cumulative sum of squares
    cusum_sq = np.cumsum(resid_sq) / np.sum(resid_sq)
    
    t = np.arange(k, k + len(cusum_sq))
    t_normalized = (t - k) / (n - k)
    
    # Critical values (approximate)
    if significance == 0.05:
        c = 0.1 + 0.032
    elif significance == 0.01:
        c = 0.1 + 0.05
    elif significance == 0.10:
        c = 0.1 + 0.020
    else:
        c = 0.132
    
    # Confidence bands
    upper = np.clip(t_normalized + c, 0, 1)
    lower = np.clip(t_normalized - c, 0, 1)
    
    # Check stability
    is_stable = np.all(cusum_sq >= lower[:len(cusum_sq)]) and np.all(cusum_sq <= upper[:len(cusum_sq)])
    
    # Maximum deviation
    deviation_upper = np.max(cusum_sq - upper[:len(cusum_sq)]) if len(cusum_sq) > 0 else 0
    deviation_lower = np.max(lower[:len(cusum_sq)] - cusum_sq) if len(cusum_sq) > 0 else 0
    max_deviation = max(deviation_upper, deviation_lower)
    
    return {
        'test': 'CUSUM of Squares',
        'cusum_sq': cusum_sq,
        'upper_bound': upper,
        'lower_bound': lower,
        'reference_line': t_normalized,
        'is_stable': is_stable,
        'max_deviation': max_deviation,
        'null_hypothesis': 'Variance is stable',
        'conclusion': 'Stable variance' if is_stable else 'Variance instability detected'
    }


def recursive_residuals(model):
    """
    Compute recursive residuals for stability testing
    
    Parameters
    ----------
    model : statsmodels OLS result
    
    Returns
    -------
    np.array
        Recursive residuals
    """
    y = model.model.endog
    X = model.model.exog
    n, k = X.shape
    
    recursive_resid = []
    
    for t in range(k + 1, n + 1):
        # Estimate with first t observations
        y_t = y[:t]
        X_t = X[:t]
        
        try:
            beta_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
            
            # One-step ahead forecast error
            if t < n:
                y_forecast = X[t] @ beta_t
                resid_t = y[t] - y_forecast
                
                # Standardize
                h_t = X[t] @ np.linalg.inv(X_t.T @ X_t) @ X[t]
                f_t = resid_t / np.sqrt(1 + h_t)
                recursive_resid.append(f_t)
        except:
            continue
    
    return np.array(recursive_resid)


def chow_test(model, break_point):
    """
    Chow test for structural break at known break point
    
    H0: No structural break at break_point
    
    Parameters
    ----------
    model : statsmodels OLS result
    break_point : int
        Index of suspected break point
    
    Returns
    -------
    dict
        Test results
    """
    from statsmodels.formula.api import ols
    
    y = model.model.endog
    X = model.model.exog
    n, k = X.shape
    
    if break_point <= k or break_point >= n - k:
        return {'test': 'Chow', 'error': 'Invalid break point'}
    
    try:
        # Full sample SSR
        ssr_full = model.ssr
        
        # Pre-break sample
        X1 = X[:break_point]
        y1 = y[:break_point]
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        ssr1 = np.sum((y1 - X1 @ beta1) ** 2)
        
        # Post-break sample
        X2 = X[break_point:]
        y2 = y[break_point:]
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        ssr2 = np.sum((y2 - X2 @ beta2) ** 2)
        
        # F-statistic
        f_stat = ((ssr_full - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n - 2 * k))
        f_pval = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
        
        return {
            'test': 'Chow',
            'f_statistic': f_stat,
            'p_value': f_pval,
            'df1': k,
            'df2': n - 2 * k,
            'break_point': break_point,
            'null_hypothesis': 'No structural break',
            'reject_h0': f_pval < 0.05,
            'conclusion': 'No structural break' if f_pval > 0.05 else 'Structural break detected'
        }
    except Exception as e:
        return {'test': 'Chow', 'error': str(e)}
