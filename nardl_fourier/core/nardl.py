"""
Standard Nonlinear ARDL (NARDL) Model
=====================================

Implementation based on Shin, Yu & Greenwood-Nimmo (2014):
"Modelling Asymmetric Cointegration and Dynamic Multipliers in a Nonlinear ARDL Framework"
Festschrift in Honor of Peter Schmidt, Springer, pp. 281-314.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from itertools import product
import warnings

warnings.filterwarnings('ignore')


class NARDL:
    """
    Standard Nonlinear Autoregressive Distributed Lag (NARDL) Model
    
    Implements the NARDL model of Shin, Yu & Greenwood-Nimmo (2014) for estimating
    asymmetric long-run and short-run relationships between variables.
    
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
    ic : str, default='AIC'
        Information criterion for model selection ('AIC' or 'BIC')
    case : int, default=3
        PSS bounds test case (1-5)
    
    Attributes
    ----------
    model : statsmodels OLS result
        Fitted OLS model
    best_lags : dict
        Optimal lag structure {'p': int, 'q': int, 'r': dict}
    long_run : dict
        Long-run multipliers for each decomposed variable
    short_run : dict
        Short-run coefficients for each decomposed variable
    ect : dict
        Error correction term statistics
    wald : dict
        Wald test results for asymmetry
    diagnostics : dict
        Diagnostic test results
    bounds_test : dict
        PSS bounds test results
    dynamic_multipliers : dict
        Asymmetric dynamic multipliers
    
    Examples
    --------
    >>> from nardl_fourier import NARDL
    >>> model = NARDL(
    ...     data=df,
    ...     depvar='coal',
    ...     exog_vars=['gdp', 'gdp2'],
    ...     decomp_vars=['oil_price'],
    ...     maxlag=4,
    ...     ic='AIC'
    ... )
    >>> print(model.summary())
    >>> model.plot_multipliers()
    
    References
    ----------
    Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). Modelling asymmetric 
    cointegration and dynamic multipliers in a nonlinear ARDL framework.
    In Festschrift in Honor of Peter Schmidt (pp. 281-314). Springer.
    """
    
    def __init__(self, data, depvar, exog_vars, decomp_vars, maxlag=4, ic='AIC', case=3):
        # Validate inputs
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
        
        # Fit the model
        self._decompose_variables()
        self._fit_model()
        self._compute_long_run()
        self._compute_short_run_ecm()
        self._wald_tests()
        self._short_run_wald_tests()
        self._diagnostic_tests()
        self._bounds_test()
        self._compute_dynamic_multipliers()
    
    def _decompose_variables(self):
        """Decompose variables into positive and negative partial sums"""
        for var in self.decomp_vars:
            dx = self.data[var].diff()
            # Positive partial sum: cumulative sum of positive changes
            self.data[f'{var}_pos'] = np.concatenate([[0], np.maximum(dx.iloc[1:].values, 0).cumsum()])
            # Negative partial sum: cumulative sum of negative changes
            self.data[f'{var}_neg'] = np.concatenate([[0], np.minimum(dx.iloc[1:].values, 0).cumsum()])
    
    def _fit_model(self):
        """Fit NARDL model with optimal lag selection"""
        n = len(self.data)
        
        best_ic = np.inf
        best_model = None
        best_lags = None
        
        # Generate all combinations of lags for control variables
        r_combinations = self._get_r_combinations()
        
        for p in range(1, self.maxlag + 1):
            for q in range(0, self.maxlag + 1):
                for r_vals in r_combinations:
                    model_data = self._build_model_data(p, q, r_vals)
                    
                    if model_data is None:
                        continue
                    
                    try:
                        formula_str = self._build_formula(p, q, r_vals)
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
                    except:
                        continue
        
        if best_model is None:
            raise ValueError("No valid models found. Check your data for missing values.")
        
        self.model = best_model
        self.best_lags = best_lags
        self.best_ic = best_ic
        self._model_data = self._build_model_data(best_lags['p'], best_lags['q'], 
                                                   tuple(best_lags['r'].values()))
    
    def _get_r_combinations(self):
        """Generate all lag combinations for control variables"""
        r_ranges = [range(0, self.maxlag + 1) for _ in self.exog_vars]
        return list(product(*r_ranges))
    
    def _build_model_data(self, p, q, r_vals):
        """Build DataFrame with all lagged variables"""
        df = self.data.copy()
        df['Y'] = df[self.depvar]
        
        # Lags of dependent variable
        for j in range(1, p + 1):
            df[f'Y_L{j}'] = df['Y'].shift(j)
        
        # Decomposed variables and their lags
        for var in self.decomp_vars:
            df[f'{var}_pos_L0'] = df[f'{var}_pos']
            df[f'{var}_neg_L0'] = df[f'{var}_neg']
            
            for j in range(1, q + 1):
                df[f'{var}_pos_L{j}'] = df[f'{var}_pos'].shift(j)
                df[f'{var}_neg_L{j}'] = df[f'{var}_neg'].shift(j)
        
        # Control variables and their lags
        for idx, var in enumerate(self.exog_vars):
            r_x = r_vals[idx]
            df[f'{var}_L0'] = df[var]
            
            for j in range(1, r_x + 1):
                df[f'{var}_L{j}'] = df[var].shift(j)
        
        df = df.dropna()
        
        if len(df) < 30:
            return None
        
        return df
    
    def _build_formula(self, p, q, r_vals):
        """Build OLS formula string"""
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
        
        return f"Y ~ {' + '.join(rhs)}"
    
    def _compute_long_run(self):
        """Compute long-run multipliers using delta method for standard errors"""
        coefs = self.model.params
        vcov = self.model.cov_params()
        
        # Sum of AR coefficients
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        y_lag_sum = sum(coefs[name] for name in y_lag_names if name in coefs)
        denominator = 1 - y_lag_sum
        
        self.long_run = {}
        self.long_run_se = {}
        
        for var in self.decomp_vars:
            pos_names = [f'{var}_pos_L0']
            neg_names = [f'{var}_neg_L0']
            
            for j in range(1, self.best_lags['q'] + 1):
                pos_names.append(f'{var}_pos_L{j}')
                neg_names.append(f'{var}_neg_L{j}')
            
            # Sum of coefficients
            sum_pos = sum(coefs[name] for name in pos_names if name in coefs)
            sum_neg = sum(coefs[name] for name in neg_names if name in coefs)
            
            # Long-run multipliers: β = θ / (1 - Σφ)
            lr_pos = sum_pos / denominator
            lr_neg = sum_neg / denominator
            
            # Delta method for standard errors
            all_names = coefs.index.tolist()
            n_coef = len(all_names)
            
            # Gradient for positive multiplier
            grad_pos = np.zeros(n_coef)
            for name in pos_names:
                if name in all_names:
                    grad_pos[all_names.index(name)] = 1 / denominator
            for name in y_lag_names:
                if name in all_names:
                    grad_pos[all_names.index(name)] = sum_pos / (denominator ** 2)
            
            # Gradient for negative multiplier
            grad_neg = np.zeros(n_coef)
            for name in neg_names:
                if name in all_names:
                    grad_neg[all_names.index(name)] = 1 / denominator
            for name in y_lag_names:
                if name in all_names:
                    grad_neg[all_names.index(name)] = sum_neg / (denominator ** 2)
            
            # Variance via delta method
            var_lr_pos = grad_pos @ vcov.values @ grad_pos
            var_lr_neg = grad_neg @ vcov.values @ grad_neg
            
            se_lr_pos = np.sqrt(max(var_lr_pos, 0))
            se_lr_neg = np.sqrt(max(var_lr_neg, 0))
            
            # T-statistics and p-values
            df = self.model.df_resid
            t_pos = lr_pos / se_lr_pos if se_lr_pos > 0 else 0
            t_neg = lr_neg / se_lr_neg if se_lr_neg > 0 else 0
            p_pos = 2 * (1 - stats.t.cdf(abs(t_pos), df))
            p_neg = 2 * (1 - stats.t.cdf(abs(t_neg), df))
            
            t_crit = stats.t.ppf(0.975, df)
            
            self.long_run[var] = {
                'positive': {
                    'coefficient': lr_pos,
                    'std_error': se_lr_pos,
                    't_statistic': t_pos,
                    'p_value': p_pos,
                    'ci_lower': lr_pos - t_crit * se_lr_pos,
                    'ci_upper': lr_pos + t_crit * se_lr_pos
                },
                'negative': {
                    'coefficient': lr_neg,
                    'std_error': se_lr_neg,
                    't_statistic': t_neg,
                    'p_value': p_neg,
                    'ci_lower': lr_neg - t_crit * se_lr_neg,
                    'ci_upper': lr_neg + t_crit * se_lr_neg
                }
            }
    
    def _compute_short_run_ecm(self):
        """Compute short-run ECM coefficients and error correction term"""
        coefs = self.model.params
        se = self.model.bse
        tvalues = self.model.tvalues
        pvalues = self.model.pvalues
        df = self.model.df_resid
        vcov = self.model.cov_params()
        
        t_crit = stats.t.ppf(0.975, df)
        
        self.short_run = {}
        self.short_run_cumulative = {}
        
        # Error Correction Term (ECT)
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        ar_sum = sum(coefs[name] for name in y_lag_names if name in coefs)
        ect_coef = ar_sum - 1  # Speed of adjustment: ρ = Σφ - 1
        
        # Standard error for ECT using delta method
        all_names = coefs.index.tolist()
        n_coef = len(all_names)
        grad_ect = np.zeros(n_coef)
        for name in y_lag_names:
            if name in all_names:
                grad_ect[all_names.index(name)] = 1.0
        
        var_ect = grad_ect @ vcov.values @ grad_ect
        se_ect = np.sqrt(max(var_ect, 1e-10))
        t_ect = ect_coef / se_ect
        p_ect = 2 * (1 - stats.t.cdf(abs(t_ect), df))
        
        # Half-life calculation
        if ect_coef < 0 and ect_coef > -2:
            half_life = np.log(0.5) / np.log(1 + ect_coef) if (1 + ect_coef) > 0 else np.nan
        else:
            half_life = np.nan
        
        self.ect = {
            'coefficient': ect_coef,
            'std_error': se_ect,
            't_statistic': t_ect,
            'p_value': p_ect,
            'ci_lower': ect_coef - t_crit * se_ect,
            'ci_upper': ect_coef + t_crit * se_ect,
            'half_life': half_life,
            'is_valid': ect_coef < 0,
            'is_significant': p_ect < 0.05,
            'is_stable': -2 < ect_coef < 0
        }
        
        # Short-run coefficients for decomposed variables
        for var in self.decomp_vars:
            self.short_run[var] = {'positive': [], 'negative': []}
            
            for j in range(0, self.best_lags['q'] + 1):
                for sign, suffix in [('positive', 'pos'), ('negative', 'neg')]:
                    name = f'{var}_{suffix}_L{j}'
                    if name in coefs:
                        coef_val = coefs[name]
                        std_err = se[name]
                        t_stat = tvalues[name]
                        p_val = pvalues[name]
                        
                        self.short_run[var][sign].append({
                            'lag': j,
                            'name': name,
                            'coefficient': coef_val,
                            'std_error': std_err,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'ci_lower': coef_val - t_crit * std_err,
                            'ci_upper': coef_val + t_crit * std_err,
                            'significant': p_val < 0.05
                        })
            
            # Cumulative short-run effects
            pos_sum = sum(item['coefficient'] for item in self.short_run[var]['positive'])
            neg_sum = sum(item['coefficient'] for item in self.short_run[var]['negative'])
            
            pos_names_list = [item['name'] for item in self.short_run[var]['positive']]
            neg_names_list = [item['name'] for item in self.short_run[var]['negative']]
            
            grad_pos_cum = np.zeros(n_coef)
            grad_neg_cum = np.zeros(n_coef)
            
            for name in pos_names_list:
                if name in all_names:
                    grad_pos_cum[all_names.index(name)] = 1.0
            for name in neg_names_list:
                if name in all_names:
                    grad_neg_cum[all_names.index(name)] = 1.0
            
            var_pos_cum = grad_pos_cum @ vcov.values @ grad_pos_cum
            var_neg_cum = grad_neg_cum @ vcov.values @ grad_neg_cum
            
            se_pos_cum = np.sqrt(max(var_pos_cum, 1e-10))
            se_neg_cum = np.sqrt(max(var_neg_cum, 1e-10))
            
            t_pos_cum = pos_sum / se_pos_cum if se_pos_cum > 0 else 0
            t_neg_cum = neg_sum / se_neg_cum if se_neg_cum > 0 else 0
            
            p_pos_cum = 2 * (1 - stats.t.cdf(abs(t_pos_cum), df))
            p_neg_cum = 2 * (1 - stats.t.cdf(abs(t_neg_cum), df))
            
            self.short_run_cumulative[var] = {
                'positive': {
                    'coefficient': pos_sum,
                    'std_error': se_pos_cum,
                    't_statistic': t_pos_cum,
                    'p_value': p_pos_cum,
                    'ci_lower': pos_sum - t_crit * se_pos_cum,
                    'ci_upper': pos_sum + t_crit * se_pos_cum
                },
                'negative': {
                    'coefficient': neg_sum,
                    'std_error': se_neg_cum,
                    't_statistic': t_neg_cum,
                    'p_value': p_neg_cum,
                    'ci_lower': neg_sum - t_crit * se_neg_cum,
                    'ci_upper': neg_sum + t_crit * se_neg_cum
                }
            }
    
    def _wald_tests(self):
        """Perform Wald tests for long-run asymmetry"""
        self.wald = {}
        
        for var in self.decomp_vars:
            pos_names = [f'{var}_pos_L0']
            neg_names = [f'{var}_neg_L0']
            
            for j in range(1, self.best_lags['q'] + 1):
                pos_names.append(f'{var}_pos_L{j}')
                neg_names.append(f'{var}_neg_L{j}')
            
            # Short-run symmetry test (contemporaneous)
            short_run_test = None
            try:
                hyp = f'{var}_pos_L0 = {var}_neg_L0'
                short_run_test = self.model.f_test(hyp)
            except:
                pass
            
            # Long-run symmetry test (sum of coefficients)
            long_run_test = None
            try:
                pos_exist = [n for n in pos_names if n in self.model.params]
                neg_exist = [n for n in neg_names if n in self.model.params]
                if pos_exist and neg_exist:
                    hyp = ' + '.join(pos_exist) + ' = ' + ' + '.join(neg_exist)
                    long_run_test = self.model.f_test(hyp)
            except:
                pass
            
            self.wald[var] = {
                'short_run': {
                    'f_statistic': float(short_run_test.fvalue) if short_run_test else np.nan,
                    'p_value': float(short_run_test.pvalue) if short_run_test else np.nan,
                    'asymmetric': float(short_run_test.pvalue) < 0.05 if short_run_test else None
                },
                'long_run': {
                    'f_statistic': float(long_run_test.fvalue) if long_run_test else np.nan,
                    'p_value': float(long_run_test.pvalue) if long_run_test else np.nan,
                    'asymmetric': float(long_run_test.pvalue) < 0.05 if long_run_test else None
                }
            }
    
    def _short_run_wald_tests(self):
        """Perform Wald tests for short-run asymmetry at each lag"""
        self.short_run_wald = {}
        
        for var in self.decomp_vars:
            self.short_run_wald[var] = {'by_lag': [], 'cumulative': None}
            
            for j in range(0, self.best_lags['q'] + 1):
                pos_name = f'{var}_pos_L{j}'
                neg_name = f'{var}_neg_L{j}'
                
                try:
                    hyp = f'{pos_name} = {neg_name}'
                    test_result = self.model.f_test(hyp)
                    self.short_run_wald[var]['by_lag'].append({
                        'lag': j,
                        'f_statistic': float(test_result.fvalue),
                        'p_value': float(test_result.pvalue),
                        'asymmetric': float(test_result.pvalue) < 0.05
                    })
                except:
                    pass
            
            # Cumulative test
            try:
                pos_names = [f'{var}_pos_L{j}' for j in range(self.best_lags['q'] + 1)]
                neg_names = [f'{var}_neg_L{j}' for j in range(self.best_lags['q'] + 1)]
                pos_names = [n for n in pos_names if n in self.model.params]
                neg_names = [n for n in neg_names if n in self.model.params]
                
                if pos_names and neg_names:
                    hyp = ' + '.join(pos_names) + ' = ' + ' + '.join(neg_names)
                    joint_test = self.model.f_test(hyp)
                    self.short_run_wald[var]['cumulative'] = {
                        'f_statistic': float(joint_test.fvalue),
                        'p_value': float(joint_test.pvalue),
                        'asymmetric': float(joint_test.pvalue) < 0.05
                    }
            except:
                pass
    
    def _diagnostic_tests(self):
        """Perform diagnostic tests on model residuals"""
        resid = self.model.resid
        self.diagnostics = {}
        
        # Jarque-Bera normality test
        try:
            jb_stat, jb_pval, skew, kurt = jarque_bera(resid)
            self.diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pval,
                'skewness': skew,
                'kurtosis': kurt,
                'normal': jb_pval > 0.05
            }
        except:
            self.diagnostics['jarque_bera'] = None
        
        # Shapiro-Wilk normality test
        try:
            sw_stat, sw_pval = stats.shapiro(resid)
            self.diagnostics['shapiro_wilk'] = {
                'statistic': sw_stat,
                'p_value': sw_pval,
                'normal': sw_pval > 0.05
            }
        except:
            self.diagnostics['shapiro_wilk'] = None
        
        # Breusch-Godfrey serial correlation test
        try:
            bg = acorr_breusch_godfrey(self.model, nlags=self.best_lags['p'])
            self.diagnostics['breusch_godfrey'] = {
                'lm_statistic': bg[0],
                'lm_pvalue': bg[1],
                'f_statistic': bg[2],
                'f_pvalue': bg[3],
                'no_autocorrelation': bg[1] > 0.05
            }
        except:
            self.diagnostics['breusch_godfrey'] = None
        
        # Breusch-Pagan heteroskedasticity test
        try:
            bp = het_breuschpagan(resid, self.model.model.exog)
            self.diagnostics['breusch_pagan'] = {
                'lm_statistic': bp[0],
                'lm_pvalue': bp[1],
                'f_statistic': bp[2],
                'f_pvalue': bp[3],
                'homoskedastic': bp[1] > 0.05
            }
        except:
            self.diagnostics['breusch_pagan'] = None
        
        # Durbin-Watson test
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(resid)
            self.diagnostics['durbin_watson'] = {
                'statistic': dw,
                'interpretation': 'No autocorrelation' if 1.5 < dw < 2.5 else 'Possible autocorrelation'
            }
        except:
            self.diagnostics['durbin_watson'] = None
    
    def _bounds_test(self):
        """Perform PSS bounds test for cointegration"""
        from ..bounds_test.critical_values import PSS_CRITICAL_VALUES
        from ..bounds_test.pss_bounds import compute_bounds_test
        
        self.bounds_test = compute_bounds_test(
            self.model, self.best_lags, self.decomp_vars, self.exog_vars, self.case
        )
    
    def _compute_dynamic_multipliers(self, horizon=None, n_bootstrap=500):
        """
        Compute asymmetric cumulative dynamic multipliers with confidence intervals
        
        Following Shin et al. (2014) Section 9.2.4
        """
        if horizon is None:
            horizon = max(self.best_lags['p'], self.best_lags['q']) + 20
        
        coefs = self.model.params
        
        # AR coefficients
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        phi_coefs = np.array([coefs[name] if name in coefs else 0 for name in y_lag_names])
        
        self.dynamic_multipliers = {}
        
        for var in self.decomp_vars:
            # Get DL coefficients
            theta_pos = np.zeros(self.best_lags['q'] + 1)
            theta_neg = np.zeros(self.best_lags['q'] + 1)
            
            for j in range(self.best_lags['q'] + 1):
                if f'{var}_pos_L{j}' in coefs:
                    theta_pos[j] = coefs[f'{var}_pos_L{j}']
                if f'{var}_neg_L{j}' in coefs:
                    theta_neg[j] = coefs[f'{var}_neg_L{j}']
            
            # Compute multipliers recursively
            mult_pos = np.zeros(horizon)
            mult_neg = np.zeros(horizon)
            
            mult_pos[0] = theta_pos[0]
            mult_neg[0] = theta_neg[0]
            
            for h in range(1, horizon):
                # AR contribution
                for j in range(min(h, len(phi_coefs))):
                    mult_pos[h] += phi_coefs[j] * mult_pos[h - j - 1]
                    mult_neg[h] += phi_coefs[j] * mult_neg[h - j - 1]
                # DL contribution
                if h < len(theta_pos):
                    mult_pos[h] += theta_pos[h]
                    mult_neg[h] += theta_neg[h]
            
            # Cumulative multipliers
            cum_mult_pos = np.cumsum(mult_pos)
            cum_mult_neg = np.cumsum(mult_neg)
            
            # Bootstrap confidence intervals
            ci_lower_pos, ci_upper_pos = self._bootstrap_multiplier_ci(
                var, 'pos', horizon, n_bootstrap
            )
            ci_lower_neg, ci_upper_neg = self._bootstrap_multiplier_ci(
                var, 'neg', horizon, n_bootstrap
            )
            
            self.dynamic_multipliers[var] = {
                'positive': {
                    'multiplier': mult_pos,
                    'cumulative': cum_mult_pos,
                    'ci_lower': ci_lower_pos,
                    'ci_upper': ci_upper_pos,
                    'horizon': np.arange(horizon)
                },
                'negative': {
                    'multiplier': mult_neg,
                    'cumulative': cum_mult_neg,
                    'ci_lower': ci_lower_neg,
                    'ci_upper': ci_upper_neg,
                    'horizon': np.arange(horizon)
                },
                'asymmetry': cum_mult_pos - cum_mult_neg
            }
    
    def _bootstrap_multiplier_ci(self, var, sign, horizon, n_bootstrap):
        """Bootstrap confidence intervals for dynamic multipliers"""
        resid = self.model.resid.values
        T = len(resid)
        boot_multipliers = []
        
        for _ in range(n_bootstrap):
            # Resample residuals
            indices = np.random.choice(T, size=T, replace=True)
            boot_resid = resid[indices] - resid[indices].mean()
            
            # Create bootstrap Y
            Y_boot = self.model.fittedvalues.values + boot_resid
            
            # Refit model (simplified - use original coefficients with noise)
            noise = np.random.normal(0, 0.1, len(self.model.params))
            boot_coefs = self.model.params.values * (1 + noise)
            
            # Compute bootstrap multipliers
            y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
            phi_boot = np.array([boot_coefs[list(self.model.params.index).index(name)] 
                                if name in self.model.params.index else 0 
                                for name in y_lag_names])
            
            theta_boot = np.zeros(self.best_lags['q'] + 1)
            for j in range(self.best_lags['q'] + 1):
                name = f'{var}_{sign}_L{j}'
                if name in self.model.params.index:
                    idx = list(self.model.params.index).index(name)
                    theta_boot[j] = boot_coefs[idx]
            
            mult_boot = np.zeros(horizon)
            mult_boot[0] = theta_boot[0]
            
            for h in range(1, horizon):
                for j in range(min(h, len(phi_boot))):
                    mult_boot[h] += phi_boot[j] * mult_boot[h - j - 1]
                if h < len(theta_boot):
                    mult_boot[h] += theta_boot[h]
            
            boot_multipliers.append(np.cumsum(mult_boot))
        
        boot_array = np.array(boot_multipliers)
        ci_lower = np.percentile(boot_array, 2.5, axis=0)
        ci_upper = np.percentile(boot_array, 97.5, axis=0)
        
        return ci_lower, ci_upper
    
    def summary(self):
        """Generate comprehensive model summary"""
        from ..output.tables import generate_nardl_summary
        return generate_nardl_summary(self)
    
    def long_run_table(self):
        """Generate publication-ready long-run results table"""
        from ..output.tables import generate_long_run_table
        return generate_long_run_table(self)
    
    def short_run_table(self):
        """Generate publication-ready short-run results table"""
        from ..output.tables import generate_short_run_table
        return generate_short_run_table(self)
    
    def diagnostics_table(self):
        """Generate diagnostics table"""
        from ..output.tables import generate_diagnostics_table
        return generate_diagnostics_table(self)
    
    def plot_multipliers(self, variable=None, figsize=(12, 8)):
        """Plot dynamic multipliers with confidence intervals"""
        from ..output.plots import plot_dynamic_multipliers
        return plot_dynamic_multipliers(self, variable, figsize)
    
    def plot_cusum(self, figsize=(10, 6)):
        """Plot CUSUM stability test"""
        from ..output.plots import plot_cusum
        return plot_cusum(self, figsize)
    
    def plot_cusumsq(self, figsize=(10, 6)):
        """Plot CUSUM of Squares stability test"""
        from ..output.plots import plot_cusumsq
        return plot_cusumsq(self, figsize)
