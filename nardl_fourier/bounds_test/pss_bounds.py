"""
PSS Bounds Test for Cointegration
==================================

Implementation of Pesaran, Shin & Smith (2001) bounds testing approach
for ARDL and NARDL models.

Author: Dr. Merwan Roudane
"""

import numpy as np
import pandas as pd
from scipy import stats

from .critical_values import get_critical_values, interpret_bounds_test, PSS_CRITICAL_VALUES


def compute_bounds_test(model, best_lags, decomp_vars, exog_vars, case=3):
    """
    Compute PSS bounds test for cointegration
    
    Parameters
    ----------
    model : statsmodels OLS result
        Fitted NARDL model
    best_lags : dict
        Lag structure {'p': int, 'q': int, 'r': dict}
    decomp_vars : list
        Decomposed variable names
    exog_vars : list
        Control variable names
    case : int
        PSS case (1-5)
    
    Returns
    -------
    dict
        Bounds test results
    """
    coefs = model.params
    
    # Identify level variables (lagged levels in ECM form)
    y_lag_names = [f'Y_L{j}' for j in range(1, best_lags['p'] + 1)]
    
    # X level variables (positive and negative)
    x_level_names = []
    for var in decomp_vars:
        x_level_names.append(f'{var}_pos_L0')
        x_level_names.append(f'{var}_neg_L0')
    
    # Z level variables
    z_level_names = [f'{var}_L0' for var in exog_vars]
    
    # All level variables
    all_level_vars = y_lag_names + x_level_names + z_level_names
    level_vars_in_model = [name for name in all_level_vars if name in coefs]
    
    # Count number of regressors k
    # For NARDL: each decomposed variable counts as 2 (pos and neg)
    k = len(decomp_vars) * 2 + len(exog_vars)
    
    # Compute F-statistic for bounds test
    # H0: All level coefficients = 0 (no cointegration)
    try:
        if len(level_vars_in_model) > 1:
            restriction = ' = '.join(level_vars_in_model) + ' = 0'
        else:
            restriction = f'{level_vars_in_model[0]} = 0'
        
        f_test = model.f_test(restriction)
        f_stat = float(f_test.fvalue)
        f_pvalue = float(f_test.pvalue)
    except:
        f_stat = np.nan
        f_pvalue = np.nan
    
    # Compute t-statistic (on first Y lag, proxy for ECT)
    try:
        if y_lag_names[0] in coefs:
            t_stat = float(model.tvalues[y_lag_names[0]])
        else:
            t_stat = np.nan
    except:
        t_stat = np.nan
    
    # Get critical values
    sample_size = int(model.nobs)
    
    # F-test critical values
    cv_f_10 = get_critical_values(k, case, 'F', sample_size, '10%')
    cv_f_5 = get_critical_values(k, case, 'F', sample_size, '5%')
    cv_f_1 = get_critical_values(k, case, 'F', sample_size, '1%')
    
    # T-test critical values
    cv_t_10 = get_critical_values(min(k, 10), case, 't', sample_size, '10%')
    cv_t_5 = get_critical_values(min(k, 10), case, 't', sample_size, '5%')
    cv_t_1 = get_critical_values(min(k, 10), case, 't', sample_size, '1%')
    
    # Interpret results
    f_decision_5 = interpret_bounds_test(f_stat, cv_f_5[0], cv_f_5[1])
    f_decision_1 = interpret_bounds_test(f_stat, cv_f_1[0], cv_f_1[1])
    
    # T-test interpretation (left-tailed)
    if not np.isnan(t_stat):
        if t_stat < cv_t_5[1]:  # Upper bound (more negative)
            t_decision_5 = 'Cointegration'
        elif t_stat > cv_t_5[0]:  # Lower bound
            t_decision_5 = 'No Cointegration'
        else:
            t_decision_5 = 'Inconclusive'
    else:
        t_decision_5 = 'N/A'
    
    return {
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        't_statistic': t_stat,
        'k': k,
        'n': sample_size,
        'case': case,
        'critical_values': {
            'F': {
                '10%': {'I(0)': cv_f_10[0], 'I(1)': cv_f_10[1]},
                '5%': {'I(0)': cv_f_5[0], 'I(1)': cv_f_5[1]},
                '1%': {'I(0)': cv_f_1[0], 'I(1)': cv_f_1[1]},
            },
            't': {
                '10%': {'I(0)': cv_t_10[0], 'I(1)': cv_t_10[1]},
                '5%': {'I(0)': cv_t_5[0], 'I(1)': cv_t_5[1]},
                '1%': {'I(0)': cv_t_1[0], 'I(1)': cv_t_1[1]},
            }
        },
        'decision': {
            'F_5%': f_decision_5,
            'F_1%': f_decision_1,
            't_5%': t_decision_5,
        },
        'cointegration': f_decision_5 == 'Cointegration' or f_decision_1 == 'Cointegration'
    }


class PSSBoundsTest:
    """
    PSS Bounds Test Class
    
    Performs Pesaran, Shin & Smith (2001) bounds testing for cointegration.
    
    Parameters
    ----------
    model : statsmodels OLS result or NARDL object
        Fitted model
    k : int
        Number of I(1) regressors
    case : int, default=3
        PSS case (1-5)
    sample_size : int, optional
        Sample size for small sample adjustments
    
    Attributes
    ----------
    f_statistic : float
        F-statistic for bounds test
    t_statistic : float
        t-statistic for ECT coefficient
    critical_values : dict
        Critical value bounds
    decision : str
        Test decision
    
    Examples
    --------
    >>> test = PSSBoundsTest(model, k=2)
    >>> print(test.summary())
    >>> print(test.decision)
    """
    
    def __init__(self, model, k, case=3, sample_size=None, restriction=None):
        self.model = model
        self.k = k
        self.case = case
        self.sample_size = sample_size or int(model.nobs)
        
        # Compute F-statistic
        if restriction is not None:
            try:
                f_test = model.f_test(restriction)
                self.f_statistic = float(f_test.fvalue)
                self.f_pvalue = float(f_test.pvalue)
            except:
                self.f_statistic = np.nan
                self.f_pvalue = np.nan
        else:
            self.f_statistic = np.nan
            self.f_pvalue = np.nan
        
        # Get critical values
        self._get_critical_values()
        
        # Make decision
        self._make_decision()
    
    def _get_critical_values(self):
        """Get critical values for the test"""
        self.critical_values = {
            'F': {},
            't': {}
        }
        
        for sig in ['10%', '5%', '1%']:
            cv_f = get_critical_values(self.k, self.case, 'F', self.sample_size, sig)
            cv_t = get_critical_values(min(self.k, 10), self.case, 't', self.sample_size, sig)
            
            self.critical_values['F'][sig] = {'I(0)': cv_f[0], 'I(1)': cv_f[1]}
            self.critical_values['t'][sig] = {'I(0)': cv_t[0], 'I(1)': cv_t[1]}
    
    def _make_decision(self):
        """Make test decision"""
        if np.isnan(self.f_statistic):
            self.decision = 'N/A'
            self.decision_5 = 'N/A'
            self.decision_1 = 'N/A'
            return
        
        cv_5 = self.critical_values['F']['5%']
        cv_1 = self.critical_values['F']['1%']
        
        self.decision_5 = interpret_bounds_test(self.f_statistic, cv_5['I(0)'], cv_5['I(1)'])
        self.decision_1 = interpret_bounds_test(self.f_statistic, cv_1['I(0)'], cv_1['I(1)'])
        
        # Overall decision
        if self.decision_5 == 'Cointegration' or self.decision_1 == 'Cointegration':
            self.decision = 'Cointegration'
        elif self.decision_5 == 'No Cointegration' and self.decision_1 == 'No Cointegration':
            self.decision = 'No Cointegration'
        else:
            self.decision = 'Inconclusive'
    
    def summary(self):
        """Generate summary of bounds test results"""
        lines = []
        lines.append("=" * 60)
        lines.append("PSS BOUNDS TEST FOR COINTEGRATION")
        lines.append("=" * 60)
        lines.append(f"Case: {self.case} | k = {self.k} | n = {self.sample_size}")
        lines.append("-" * 60)
        lines.append(f"F-statistic: {self.f_statistic:.4f}")
        lines.append("")
        lines.append("Critical Values (F-test):")
        lines.append("-" * 60)
        lines.append(f"{'Sig.':<10} {'I(0) Bound':<15} {'I(1) Bound':<15}")
        lines.append("-" * 60)
        for sig in ['10%', '5%', '1%']:
            cv = self.critical_values['F'][sig]
            lines.append(f"{sig:<10} {cv['I(0)']:<15.4f} {cv['I(1)']:<15.4f}")
        lines.append("-" * 60)
        lines.append(f"Decision at 5%: {self.decision_5}")
        lines.append(f"Decision at 1%: {self.decision_1}")
        lines.append("-" * 60)
        
        if self.decision == 'Cointegration':
            lines.append("CONCLUSION: Evidence of cointegration (long-run relationship)")
        elif self.decision == 'No Cointegration':
            lines.append("CONCLUSION: No evidence of cointegration")
        else:
            lines.append("CONCLUSION: Inconclusive - consider bootstrap test")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dataframe(self):
        """Convert results to DataFrame"""
        rows = []
        for sig in ['10%', '5%', '1%']:
            cv = self.critical_values['F'][sig]
            rows.append({
                'Significance': sig,
                'I(0) Bound': cv['I(0)'],
                'I(1) Bound': cv['I(1)'],
                'F-statistic': self.f_statistic,
                'Decision': interpret_bounds_test(self.f_statistic, cv['I(0)'], cv['I(1)'])
            })
        return pd.DataFrame(rows)
