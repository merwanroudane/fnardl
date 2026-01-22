"""
Diagnostic Tests for NARDL Models
==================================

Comprehensive suite of diagnostic tests for model validation.

Author: Dr. Merwan Roudane
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    acorr_breusch_godfrey,
    het_arch,
)
from statsmodels.stats.stattools import jarque_bera, durbin_watson


def jarque_bera_test(residuals):
    """
    Jarque-Bera test for normality
    
    H0: Residuals are normally distributed
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    
    Returns
    -------
    dict
        Test results
    """
    try:
        stat, pval, skew, kurt = jarque_bera(residuals)
        return {
            'test': 'Jarque-Bera',
            'statistic': stat,
            'p_value': pval,
            'skewness': skew,
            'kurtosis': kurt,
            'null_hypothesis': 'Residuals are normally distributed',
            'reject_h0': pval < 0.05,
            'conclusion': 'Normal' if pval > 0.05 else 'Non-normal'
        }
    except Exception as e:
        return {'test': 'Jarque-Bera', 'error': str(e)}


def breusch_godfrey_test(model, nlags=None):
    """
    Breusch-Godfrey LM test for serial correlation
    
    H0: No serial correlation up to lag order
    
    Parameters
    ----------
    model : statsmodels OLS result
    nlags : int, optional
        Number of lags to test
    
    Returns
    -------
    dict
        Test results
    """
    try:
        if nlags is None:
            nlags = min(4, int(model.nobs ** 0.25))
        
        lm_stat, lm_pval, f_stat, f_pval = acorr_breusch_godfrey(model, nlags=nlags)
        
        return {
            'test': 'Breusch-Godfrey',
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pval,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'lags': nlags,
            'null_hypothesis': 'No serial correlation',
            'reject_h0': lm_pval < 0.05,
            'conclusion': 'No autocorrelation' if lm_pval > 0.05 else 'Autocorrelation detected'
        }
    except Exception as e:
        return {'test': 'Breusch-Godfrey', 'error': str(e)}


def breusch_pagan_test(model):
    """
    Breusch-Pagan test for heteroskedasticity
    
    H0: Homoskedasticity (constant variance)
    
    Parameters
    ----------
    model : statsmodels OLS result
    
    Returns
    -------
    dict
        Test results
    """
    try:
        residuals = model.resid
        exog = model.model.exog
        
        lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, exog)
        
        return {
            'test': 'Breusch-Pagan',
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pval,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'null_hypothesis': 'Homoskedasticity',
            'reject_h0': lm_pval < 0.05,
            'conclusion': 'Homoskedastic' if lm_pval > 0.05 else 'Heteroskedastic'
        }
    except Exception as e:
        return {'test': 'Breusch-Pagan', 'error': str(e)}


def white_test(model):
    """
    White test for heteroskedasticity
    
    More general than Breusch-Pagan, includes cross-products
    
    Parameters
    ----------
    model : statsmodels OLS result
    
    Returns
    -------
    dict
        Test results
    """
    try:
        residuals = model.resid
        exog = model.model.exog
        
        lm_stat, lm_pval, f_stat, f_pval = het_white(residuals, exog)
        
        return {
            'test': 'White',
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pval,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'null_hypothesis': 'Homoskedasticity',
            'reject_h0': lm_pval < 0.05,
            'conclusion': 'Homoskedastic' if lm_pval > 0.05 else 'Heteroskedastic'
        }
    except Exception as e:
        return {'test': 'White', 'error': str(e)}


def arch_test(residuals, nlags=None):
    """
    ARCH test for autoregressive conditional heteroskedasticity
    
    H0: No ARCH effects
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    nlags : int, optional
        Number of lags
    
    Returns
    -------
    dict
        Test results
    """
    try:
        if nlags is None:
            nlags = min(4, len(residuals) // 10)
        
        lm_stat, lm_pval, f_stat, f_pval = het_arch(residuals, nlags=nlags)
        
        return {
            'test': 'ARCH',
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pval,
            'f_statistic': f_stat,
            'f_pvalue': f_pval,
            'lags': nlags,
            'null_hypothesis': 'No ARCH effects',
            'reject_h0': lm_pval < 0.05,
            'conclusion': 'No ARCH' if lm_pval > 0.05 else 'ARCH effects present'
        }
    except Exception as e:
        return {'test': 'ARCH', 'error': str(e)}


def ramsey_reset_test(model, power=3):
    """
    Ramsey RESET test for functional form misspecification
    
    H0: Model is correctly specified
    
    Parameters
    ----------
    model : statsmodels OLS result
    power : int
        Highest power of fitted values to include
    
    Returns
    -------
    dict
        Test results
    """
    try:
        from statsmodels.stats.diagnostic import linear_reset
        
        result = linear_reset(model, power=power, use_f=True)
        
        return {
            'test': 'Ramsey RESET',
            'f_statistic': result.fvalue,
            'f_pvalue': result.pvalue,
            'df': result.df_num,
            'df_denom': result.df_denom,
            'power': power,
            'null_hypothesis': 'Model is correctly specified',
            'reject_h0': result.pvalue < 0.05,
            'conclusion': 'Correct specification' if result.pvalue > 0.05 else 'Misspecification'
        }
    except Exception as e:
        return {'test': 'Ramsey RESET', 'error': str(e)}


def durbin_watson_test(residuals):
    """
    Durbin-Watson test for first-order autocorrelation
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    
    Returns
    -------
    dict
        Test results
    """
    try:
        dw_stat = durbin_watson(residuals)
        
        # Interpretation
        if dw_stat < 1.5:
            interpretation = 'Positive autocorrelation likely'
        elif dw_stat > 2.5:
            interpretation = 'Negative autocorrelation likely'
        else:
            interpretation = 'No significant autocorrelation'
        
        return {
            'test': 'Durbin-Watson',
            'statistic': dw_stat,
            'interpretation': interpretation,
            'note': 'DW ≈ 2 indicates no autocorrelation'
        }
    except Exception as e:
        return {'test': 'Durbin-Watson', 'error': str(e)}


def shapiro_wilk_test(residuals):
    """
    Shapiro-Wilk test for normality
    
    More powerful than Jarque-Bera for small samples
    
    Parameters
    ----------
    residuals : array-like
        Model residuals
    
    Returns
    -------
    dict
        Test results
    """
    try:
        stat, pval = stats.shapiro(residuals)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': pval,
            'null_hypothesis': 'Residuals are normally distributed',
            'reject_h0': pval < 0.05,
            'conclusion': 'Normal' if pval > 0.05 else 'Non-normal'
        }
    except Exception as e:
        return {'test': 'Shapiro-Wilk', 'error': str(e)}


def run_all_diagnostics(model):
    """
    Run all diagnostic tests on a fitted model
    
    Parameters
    ----------
    model : statsmodels OLS result or NARDL model
    
    Returns
    -------
    dict
        Dictionary of all test results
    """
    if hasattr(model, 'model'):
        # NARDL model object
        ols_model = model.model
        residuals = ols_model.resid.values
    else:
        # Direct OLS result
        ols_model = model
        residuals = model.resid.values if hasattr(model.resid, 'values') else model.resid
    
    results = {
        'normality': {
            'jarque_bera': jarque_bera_test(residuals),
            'shapiro_wilk': shapiro_wilk_test(residuals)
        },
        'serial_correlation': {
            'breusch_godfrey': breusch_godfrey_test(ols_model),
            'durbin_watson': durbin_watson_test(residuals)
        },
        'heteroskedasticity': {
            'breusch_pagan': breusch_pagan_test(ols_model),
            'white': white_test(ols_model),
            'arch': arch_test(residuals)
        },
        'specification': {
            'ramsey_reset': ramsey_reset_test(ols_model)
        }
    }
    
    # Summary
    issues = []
    if results['normality']['jarque_bera'].get('reject_h0'):
        issues.append('Non-normal residuals')
    if results['serial_correlation']['breusch_godfrey'].get('reject_h0'):
        issues.append('Serial correlation')
    if results['heteroskedasticity']['breusch_pagan'].get('reject_h0'):
        issues.append('Heteroskedasticity')
    if results['heteroskedasticity']['arch'].get('reject_h0'):
        issues.append('ARCH effects')
    if results['specification']['ramsey_reset'].get('reject_h0'):
        issues.append('Misspecification')
    
    results['summary'] = {
        'all_tests_passed': len(issues) == 0,
        'issues_detected': issues,
        'n_issues': len(issues)
    }
    
    return results


def diagnostics_summary_table(results):
    """
    Convert diagnostics results to summary DataFrame
    
    Parameters
    ----------
    results : dict
        Output from run_all_diagnostics
    
    Returns
    -------
    pd.DataFrame
    """
    rows = []
    
    # Normality
    jb = results['normality']['jarque_bera']
    if 'error' not in jb:
        rows.append({
            'Category': 'Normality',
            'Test': 'Jarque-Bera',
            'Statistic': f"{jb['statistic']:.4f}",
            'p-value': f"{jb['p_value']:.4f}",
            'Decision': jb['conclusion']
        })
    
    sw = results['normality']['shapiro_wilk']
    if 'error' not in sw:
        rows.append({
            'Category': 'Normality',
            'Test': 'Shapiro-Wilk',
            'Statistic': f"{sw['statistic']:.4f}",
            'p-value': f"{sw['p_value']:.4f}",
            'Decision': sw['conclusion']
        })
    
    # Serial correlation
    bg = results['serial_correlation']['breusch_godfrey']
    if 'error' not in bg:
        rows.append({
            'Category': 'Serial Correlation',
            'Test': f"Breusch-Godfrey ({bg['lags']} lags)",
            'Statistic': f"{bg['lm_statistic']:.4f}",
            'p-value': f"{bg['lm_pvalue']:.4f}",
            'Decision': bg['conclusion']
        })
    
    dw = results['serial_correlation']['durbin_watson']
    if 'error' not in dw:
        rows.append({
            'Category': 'Serial Correlation',
            'Test': 'Durbin-Watson',
            'Statistic': f"{dw['statistic']:.4f}",
            'p-value': '-',
            'Decision': dw['interpretation']
        })
    
    # Heteroskedasticity
    bp = results['heteroskedasticity']['breusch_pagan']
    if 'error' not in bp:
        rows.append({
            'Category': 'Heteroskedasticity',
            'Test': 'Breusch-Pagan',
            'Statistic': f"{bp['lm_statistic']:.4f}",
            'p-value': f"{bp['lm_pvalue']:.4f}",
            'Decision': bp['conclusion']
        })
    
    arch = results['heteroskedasticity']['arch']
    if 'error' not in arch:
        rows.append({
            'Category': 'Heteroskedasticity',
            'Test': f"ARCH ({arch['lags']} lags)",
            'Statistic': f"{arch['lm_statistic']:.4f}",
            'p-value': f"{arch['lm_pvalue']:.4f}",
            'Decision': arch['conclusion']
        })
    
    # Specification
    reset = results['specification']['ramsey_reset']
    if 'error' not in reset:
        rows.append({
            'Category': 'Specification',
            'Test': 'Ramsey RESET',
            'Statistic': f"{reset['f_statistic']:.4f}",
            'p-value': f"{reset['f_pvalue']:.4f}",
            'Decision': reset['conclusion']
        })
    
    return pd.DataFrame(rows)
