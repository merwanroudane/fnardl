"""
Helper Functions for NARDL Analysis
=====================================

Utility functions for data preprocessing, stationarity testing,
and output formatting.

Author: Dr. Merwan Roudane
"""

import numpy as np
import pandas as pd
from scipy import stats


def check_stationarity(series, method='adf', regression='c', maxlag=None):
    """
    Test for stationarity using ADF or KPSS test
    
    Parameters
    ----------
    series : pd.Series or np.array
        Time series to test
    method : str
        'adf' for Augmented Dickey-Fuller, 'kpss' for KPSS
    regression : str
        Type of regression ('c' for constant, 'ct' for constant + trend)
    maxlag : int, optional
        Maximum lag for ADF test
    
    Returns
    -------
    dict
        Test results
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    if method.lower() == 'adf':
        try:
            result = adfuller(series, regression=regression, maxlag=maxlag)
            
            return {
                'test': 'ADF',
                'statistic': result[0],
                'p_value': result[1],
                'lags_used': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'stationary': result[1] < 0.05,
                'conclusion': 'Stationary (I(0))' if result[1] < 0.05 else 'Non-stationary (I(1))'
            }
        except Exception as e:
            return {'test': 'ADF', 'error': str(e)}
    
    elif method.lower() == 'kpss':
        try:
            result = kpss(series, regression=regression, nlags='auto')
            
            # KPSS: H0 is stationarity
            return {
                'test': 'KPSS',
                'statistic': result[0],
                'p_value': result[1],
                'lags_used': result[2],
                'critical_values': result[3],
                'stationary': result[1] > 0.05,
                'conclusion': 'Stationary (I(0))' if result[1] > 0.05 else 'Non-stationary (I(1))'
            }
        except Exception as e:
            return {'test': 'KPSS', 'error': str(e)}
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'adf' or 'kpss'.")


def check_integration_order(series, max_diff=2):
    """
    Determine integration order of a series
    
    Parameters
    ----------
    series : array-like
        Time series
    max_diff : int
        Maximum differencing order to test
    
    Returns
    -------
    dict
        Integration order results
    """
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    results = {
        'level': check_stationarity(series, method='adf'),
        'integration_order': None
    }
    
    if results['level']['stationary']:
        results['integration_order'] = 0
        return results
    
    current_series = series
    for d in range(1, max_diff + 1):
        current_series = np.diff(current_series)
        results[f'd{d}'] = check_stationarity(current_series, method='adf')
        
        if results[f'd{d}']['stationary']:
            results['integration_order'] = d
            break
    
    if results['integration_order'] is None:
        results['integration_order'] = f'>{max_diff}'
    
    return results


def partial_sum_decomposition(series):
    """
    Decompose a series into positive and negative partial sums
    
    Following Shin et al. (2014):
    x_t^+ = sum(max(Δx_j, 0)) for j = 1 to t
    x_t^- = sum(min(Δx_j, 0)) for j = 1 to t
    
    Parameters
    ----------
    series : array-like
        Original series
    
    Returns
    -------
    tuple
        (positive_sum, negative_sum)
    """
    series = np.array(series)
    dx = np.diff(series)
    
    # Positive partial sum
    pos_changes = np.maximum(dx, 0)
    x_pos = np.concatenate([[0], np.cumsum(pos_changes)])
    
    # Negative partial sum
    neg_changes = np.minimum(dx, 0)
    x_neg = np.concatenate([[0], np.cumsum(neg_changes)])
    
    return x_pos, x_neg


def create_lagged_variables(df, var_name, max_lag):
    """
    Create lagged variables for a given column
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame
    var_name : str
        Variable name to lag
    max_lag : int
        Maximum lag order
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added lagged columns
    """
    df = df.copy()
    
    for lag in range(1, max_lag + 1):
        df[f'{var_name}_L{lag}'] = df[var_name].shift(lag)
    
    return df


def calculate_information_criteria(nobs, k, ssr, criterion='AIC'):
    """
    Calculate information criterion
    
    Parameters
    ----------
    nobs : int
        Number of observations
    k : int
        Number of parameters
    ssr : float
        Sum of squared residuals
    criterion : str
        'AIC' or 'BIC'
    
    Returns
    -------
    float
        Information criterion value
    """
    if criterion.upper() == 'AIC':
        return nobs * np.log(ssr / nobs) + 2 * k
    elif criterion.upper() == 'BIC':
        return nobs * np.log(ssr / nobs) + k * np.log(nobs)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def format_regression_table(model, precision=4):
    """
    Format regression results as a clean DataFrame
    
    Parameters
    ----------
    model : statsmodels OLS result
    precision : int
        Decimal precision
    
    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values.round(precision),
        'Std. Error': model.bse.values.round(precision),
        't-statistic': model.tvalues.values.round(precision - 1),
        'p-value': model.pvalues.values.round(precision)
    })
    
    # Add significance stars
    def add_stars(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.1:
            return '*'
        return ''
    
    df['Sig.'] = df['p-value'].apply(add_stars)
    
    return df


def half_life(rho):
    """
    Calculate half-life of adjustment from ECT coefficient
    
    Half-life = ln(0.5) / ln(1 + rho)
    
    Parameters
    ----------
    rho : float
        Error correction coefficient (should be negative)
    
    Returns
    -------
    float
        Half-life in periods
    """
    if rho >= 0:
        return np.inf
    if rho <= -2:
        return np.nan  # Unstable
    
    if (1 + rho) <= 0:
        return np.nan
    
    return np.log(0.5) / np.log(1 + rho)


def mean_adjustment_lag(rho):
    """
    Calculate mean adjustment lag from ECT coefficient
    
    MAL = 1 / |rho|
    
    Parameters
    ----------
    rho : float
        Error correction coefficient
    
    Returns
    -------
    float
        Mean adjustment lag in periods
    """
    if rho >= 0:
        return np.inf
    return -1 / rho


def validate_data(data, depvar, exog_vars, decomp_vars, min_obs=30):
    """
    Validate input data for NARDL estimation
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    depvar : str
        Dependent variable name
    exog_vars : list
        Control variable names
    decomp_vars : list
        Asymmetric variable names
    min_obs : int
        Minimum required observations
    
    Returns
    -------
    dict
        Validation results
    """
    issues = []
    
    # Check if DataFrame
    if not isinstance(data, pd.DataFrame):
        issues.append("Data must be a pandas DataFrame")
        return {'valid': False, 'issues': issues}
    
    # Check variables exist
    all_vars = [depvar] + exog_vars + decomp_vars
    for var in all_vars:
        if var not in data.columns:
            issues.append(f"Variable '{var}' not found in data")
    
    if issues:
        return {'valid': False, 'issues': issues}
    
    # Check for sufficient observations
    if len(data) < min_obs:
        issues.append(f"Insufficient observations: {len(data)} < {min_obs}")
    
    # Check for missing values
    missing = data[all_vars].isnull().sum()
    if missing.any():
        for var, count in missing[missing > 0].items():
            issues.append(f"Variable '{var}' has {count} missing values")
    
    # Check for constant variables
    for var in all_vars:
        if data[var].std() == 0:
            issues.append(f"Variable '{var}' is constant")
    
    # Check for I(2) variables
    for var in all_vars:
        order = check_integration_order(data[var].dropna())
        if order['integration_order'] == '>2' or order['integration_order'] == 2:
            issues.append(f"Variable '{var}' may be I(2) - not suitable for ARDL")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_obs': len(data),
        'n_vars': len(all_vars),
        'missing_total': data[all_vars].isnull().sum().sum()
    }
