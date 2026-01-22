"""
Fourier Nonlinear ARDL (Fourier-NARDL) Model
============================================

Implementation based on Zaghdoudi et al. (2023):
"Asymmetric connectedness between oil price, coal and renewable energy consumption 
in China: Evidence from Fourier NARDL approach"
Energy, 285, 129416.

The Fourier NARDL extends the standard NARDL by incorporating Fourier trigonometric
terms to capture smooth structural breaks without estimating break dates.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from itertools import product
import warnings

from .nardl import NARDL

warnings.filterwarnings('ignore')


class FourierNARDL(NARDL):
    """
    Fourier Nonlinear ARDL Model with Smooth Structural Breaks
    
    Extends the standard NARDL model by incorporating Fourier trigonometric terms
    to capture gradual structural changes in the data without requiring knowledge
    of break dates.
    
    The model includes:
    - sin(2πkt/T) and cos(2πkt/T) terms
    - Optimal frequency k selection via AIC/BIC
    - All features of standard NARDL
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the time series data
    depvar : str
        Name of the dependent variable
    exog_vars : list
        List of exogenous/control variable names (enter symmetrically)
    decomp_vars : list
        List of variable names to decompose into positive/negative components
    maxlag : int, default=4
        Maximum lag order to consider for model selection
    max_freq : int, default=3
        Maximum Fourier frequency to consider
    ic : str, default='AIC'
        Information criterion for model selection ('AIC' or 'BIC')
    case : int, default=3
        PSS bounds test case (1-5)
    freq_step : float, default=0.1
        Step size for frequency grid search (smaller = finer grid)
    
    Attributes
    ----------
    best_freq : float
        Optimal Fourier frequency
    fourier_test : dict
        F-test for significance of Fourier terms
    
    Examples
    --------
    >>> from nardl_fourier import FourierNARDL
    >>> model = FourierNARDL(
    ...     data=df,
    ...     depvar='coal',
    ...     exog_vars=['gdp', 'gdp2'],
    ...     decomp_vars=['oil_price'],
    ...     maxlag=4,
    ...     max_freq=3,
    ...     ic='AIC'
    ... )
    >>> print(model.summary())
    >>> print(f"Optimal frequency: {model.best_freq}")
    
    References
    ----------
    Zaghdoudi, T., Tissaoui, K., Maaloul, M. H., Bahou, Y., & Kammoun, N. (2023).
    Asymmetric connectedness between oil price, coal and renewable energy consumption
    in China: Evidence from Fourier NARDL approach. Energy, 285, 129416.
    
    Enders, W., & Lee, J. (2012). The flexible Fourier form and Dickey–Fuller type
    unit root tests. Economics Letters, 117(1), 196-199.
    """
    
    def __init__(self, data, depvar, exog_vars, decomp_vars, maxlag=4, max_freq=3,
                 ic='AIC', case=3, freq_step=0.1):
        # Store Fourier-specific parameters (before calling parent __init__)
        self.max_freq = max_freq
        self.freq_step = freq_step
        self._is_fourier = True
        
        # Call parent constructor
        # Note: We override _fit_model so it will use Fourier terms
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if depvar not in data.columns:
            raise ValueError(f"Dependent variable '{depvar}' not found in data")
        for var in exog_vars + decomp_vars:
            if var not in data.columns:
                raise ValueError(f"Variable '{var}' not found in data")
        
        self.data = data.copy()
        self.depvar = depvar
        self.exog_vars = exog_vars
        self.decomp_vars = decomp_vars
        self.maxlag = maxlag
        self.ic = ic.upper()
        self.case = case
        
        # Fit the model with Fourier terms
        self._decompose_variables()
        self._fit_model_fourier()
        self._compute_long_run()
        self._compute_short_run_ecm()
        self._wald_tests()
        self._short_run_wald_tests()
        self._diagnostic_tests()
        self._fourier_significance_test()
        self._bounds_test()
        self._compute_dynamic_multipliers()
    
    def _fit_model_fourier(self):
        """Fit Fourier NARDL model with optimal lag and frequency selection"""
        n = len(self.data)
        
        # Frequency grid: from 0.1 to max_freq
        freq_grid = np.arange(self.freq_step, self.max_freq + self.freq_step, self.freq_step)
        
        best_ic = np.inf
        best_model = None
        best_lags = None
        best_freq = 0
        
        r_combinations = self._get_r_combinations()
        
        for kstar in freq_grid:
            # Add Fourier terms
            t = np.arange(n)
            self.data['sin_t'] = np.sin(2 * np.pi * kstar * t / n)
            self.data['cos_t'] = np.cos(2 * np.pi * kstar * t / n)
            
            for p in range(1, self.maxlag + 1):
                for q in range(0, self.maxlag + 1):
                    for r_vals in r_combinations:
                        model_data = self._build_model_data(p, q, r_vals)
                        
                        if model_data is None:
                            continue
                        
                        try:
                            formula_str = self._build_formula_fourier(p, q, r_vals)
                            model = ols(formula_str, data=model_data).fit()
                            
                            nobs = model.nobs
                            k_params = len(model.params)
                            ssr = model.ssr
                            
                            if self.ic == 'AIC':
                                ic_val = nobs * np.log(ssr / nobs) + 2 * k_params
                            else:  # BIC
                                ic_val = nobs * np.log(ssr / nobs) + k_params * np.log(nobs)
                            
                            if ic_val < best_ic:
                                best_ic = ic_val
                                best_model = model
                                best_lags = {'p': p, 'q': q, 'r': dict(zip(self.exog_vars, r_vals))}
                                best_freq = kstar
                        except:
                            continue
        
        if best_model is None:
            raise ValueError("No valid models found. Check your data for missing values.")
        
        # Store final Fourier terms with optimal frequency
        t = np.arange(n)
        self.data['sin_t'] = np.sin(2 * np.pi * best_freq * t / n)
        self.data['cos_t'] = np.cos(2 * np.pi * best_freq * t / n)
        
        self.model = best_model
        self.best_lags = best_lags
        self.best_freq = best_freq
        self.best_ic = best_ic
        self._model_data = self._build_model_data(best_lags['p'], best_lags['q'],
                                                   tuple(best_lags['r'].values()))
    
    def _build_formula_fourier(self, p, q, r_vals):
        """Build OLS formula string with Fourier terms"""
        rhs = []
        
        # Y lags
        for j in range(1, p + 1):
            rhs.append(f'Y_L{j}')
        
        # Decomposed variables
        for var in self.decomp_vars:
            rhs.append(f'{var}_pos_L0')
            rhs.append(f'{var}_neg_L0')
            
            for j in range(1, q + 1):
                rhs.append(f'{var}_pos_L{j}')
                rhs.append(f'{var}_neg_L{j}')
        
        # Control variables
        for idx, var in enumerate(self.exog_vars):
            r_x = r_vals[idx]
            rhs.append(f'{var}_L0')
            
            for j in range(1, r_x + 1):
                rhs.append(f'{var}_L{j}')
        
        # Fourier terms
        rhs.extend(['sin_t', 'cos_t'])
        
        return f"Y ~ {' + '.join(rhs)}"
    
    def _fourier_significance_test(self):
        """Test significance of Fourier terms (F-test)"""
        try:
            # Test H0: coefficients on sin_t and cos_t are jointly zero
            restriction = 'sin_t = cos_t = 0'
            f_test = self.model.f_test(restriction)
            
            self.fourier_test = {
                'f_statistic': float(f_test.fvalue),
                'p_value': float(f_test.pvalue),
                'significant': float(f_test.pvalue) < 0.05,
                'optimal_frequency': self.best_freq
            }
        except:
            self.fourier_test = {
                'f_statistic': np.nan,
                'p_value': np.nan,
                'significant': None,
                'optimal_frequency': self.best_freq
            }
    
    def _fourier_adf_test(self, series, regression='c'):
        """
        Fourier ADF unit root test (Enders & Lee, 2012)
        
        Parameters
        ----------
        series : pd.Series or np.array
            Time series to test
        regression : str
            Type of regression ('c' for constant, 'ct' for constant and trend)
        
        Returns
        -------
        dict
            Test results including statistic, p-value, and optimal frequency
        """
        from statsmodels.tsa.stattools import adfuller
        
        n = len(series)
        t = np.arange(n)
        
        best_ssr = np.inf
        best_k = 1
        best_resid = None
        
        # Find optimal frequency
        for k in np.arange(0.1, 5.1, 0.1):
            sin_t = np.sin(2 * np.pi * k * t / n)
            cos_t = np.cos(2 * np.pi * k * t / n)
            
            if regression == 'ct':
                X = np.column_stack([np.ones(n), t, sin_t, cos_t])
            else:
                X = np.column_stack([np.ones(n), sin_t, cos_t])
            
            try:
                beta = np.linalg.lstsq(X, series, rcond=None)[0]
                resid = series - X @ beta
                ssr = np.sum(resid**2)
                
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_k = k
                    best_resid = resid
            except:
                continue
        
        # Perform ADF test on residuals
        if best_resid is not None:
            adf_result = adfuller(best_resid, regression='n')
            
            return {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'optimal_frequency': best_k,
                'lags_used': adf_result[2],
                'nobs': adf_result[3],
                'critical_values': adf_result[4]
            }
        else:
            return None
    
    def summary(self):
        """Generate comprehensive model summary including Fourier terms"""
        from ..output.tables import generate_fnardl_summary
        return generate_fnardl_summary(self)
    
    def plot_fourier_terms(self, figsize=(12, 4)):
        """Plot the Fourier approximation terms"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        n = len(self.data)
        t = np.arange(n)
        
        # Sin term
        axes[0].plot(t, np.sin(2 * np.pi * self.best_freq * t / n), 
                     color='#3B82F6', linewidth=2)
        axes[0].set_title(f'sin(2π × {self.best_freq:.2f} × t/T)', fontweight='bold')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Cos term
        axes[1].plot(t, np.cos(2 * np.pi * self.best_freq * t / n),
                     color='#DC2626', linewidth=2)
        axes[1].set_title(f'cos(2π × {self.best_freq:.2f} × t/T)', fontweight='bold')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
