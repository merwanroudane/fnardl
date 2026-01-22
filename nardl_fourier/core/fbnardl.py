"""
Bootstrap Fourier NARDL Model with Cointegration Tests
=======================================================

Implementation based on Bertelli, Vacca & Zoia (2022):
"Bootstrap cointegration tests in ARDL models"
Economic Modelling, 116, 105987.

This model extends the Fourier-NARDL with bootstrap-based cointegration tests
that eliminate the inconclusive zone problem of PSS bounds tests.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings

from .fnardl import FourierNARDL

warnings.filterwarnings('ignore')


class BootstrapCointegrationTest:
    """
    Bootstrap Cointegration Tests for ARDL Models
    Based on Bertelli, Vacca & Zoia (2022) - Economic Modelling
    
    This class implements bootstrap versions of:
    - F_ov: Overall F-test for cointegration
    - t-test: Test for degenerate case of first type (a_yy = 0)
    - F_ind: F-test on independent variables (degenerate case of second type)
    
    The bootstrap procedure eliminates the inconclusive zone problem
    and provides data-specific critical values.
    
    Parameters
    ----------
    model : statsmodels OLS result
        Fitted ARDL model
    data : pd.DataFrame
        Model data
    depvar : str
        Dependent variable name
    decomp_vars : list
        Decomposed variable names
    exog_vars : list
        Control variable names
    best_lags : dict
        Optimal lag structure {'p': int, 'q': int, 'r': dict}
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    case : str, default='III'
        Case for bounds test ('II' or 'III')
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, model, data, depvar, decomp_vars, exog_vars, best_lags,
                 n_bootstrap=1000, case='III', random_state=None):
        self.model = model
        self.data = data.copy()
        self.depvar = depvar
        self.decomp_vars = decomp_vars
        self.exog_vars = exog_vars
        self.best_lags = best_lags
        self.n_bootstrap = n_bootstrap
        self.case = case
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Compute original statistics
        self.original_stats = self._compute_test_statistics(model)
        
        # Run bootstrap procedure
        self._run_bootstrap()
    
    def _compute_test_statistics(self, model):
        """Compute F_ov, t, and F_ind statistics from fitted model"""
        coefs = model.params
        
        # Identify level variables
        y_lag_names = [f'Y_L{j}' for j in range(1, self.best_lags['p'] + 1)]
        
        # Level variables for X (positive and negative at lag 0)
        x_level_names = []
        for var in self.decomp_vars:
            x_level_names.append(f'{var}_pos_L0')
            x_level_names.append(f'{var}_neg_L0')
        
        # Z level variables
        z_level_names = [f'{var}_L0' for var in self.exog_vars]
        
        # All level variables for F_ov test
        all_level_vars = y_lag_names + x_level_names + z_level_names
        
        # F_ov test: H0: a_yy = 0 AND a_yx = 0 (all level coefficients = 0)
        try:
            level_vars_in_model = [name for name in all_level_vars if name in coefs]
            if len(level_vars_in_model) > 1:
                restriction = ' = '.join(level_vars_in_model) + ' = 0'
            else:
                restriction = f'{level_vars_in_model[0]} = 0'
            f_ov_test = model.f_test(restriction)
            f_ov = float(f_ov_test.fvalue)
        except:
            f_ov = np.nan
        
        # t-test: H0: a_yy = 0 (coefficient on first Y lag as proxy)
        try:
            if y_lag_names[0] in coefs:
                t_stat = float(model.tvalues[y_lag_names[0]])
            else:
                t_stat = np.nan
        except:
            t_stat = np.nan
        
        # F_ind test: H0: a_yx = 0 (coefficients on X level variables = 0)
        try:
            x_vars_in_model = [name for name in x_level_names if name in coefs]
            if len(x_vars_in_model) > 1:
                x_restriction = ' = '.join(x_vars_in_model) + ' = 0'
            else:
                x_restriction = f'{x_vars_in_model[0]} = 0'
            f_ind_test = model.f_test(x_restriction)
            f_ind = float(f_ind_test.fvalue)
        except:
            f_ind = np.nan
        
        return {
            'F_ov': f_ov,
            't': t_stat,
            'F_ind': f_ind
        }
    
    def _run_bootstrap(self):
        """
        Run the complete bootstrap procedure following Bertelli et al. (2022)
        
        Algorithm:
        1. Obtain residuals from the unrestricted model
        2. For each bootstrap replication:
           a. Resample residuals with replacement
           b. Recenter resampled residuals
           c. Generate bootstrap Y values under H0 (no cointegration)
           d. Refit model and compute test statistics
        3. Compute bootstrap critical values and p-values
        """
        original_resid = self.model.resid.values
        T = len(original_resid)
        p = self.best_lags['p']
        
        # Storage for bootstrap statistics
        boot_F_ov = []
        boot_t = []
        boot_F_ind = []
        
        # Get model data and formula
        model_data = self.model.model.data.frame.copy()
        formula = self.model.model.formula
        
        for b in range(self.n_bootstrap):
            try:
                # Step 1: Resample residuals with replacement
                indices = np.random.choice(T, size=T, replace=True)
                boot_resid = original_resid[indices]
                
                # Step 2: Recenter (Davidson & MacKinnon, 2005)
                boot_resid = boot_resid - np.mean(boot_resid)
                
                # Step 3: Generate bootstrap Y values
                boot_data = model_data.copy()
                Y_original = boot_data['Y'].values
                Y_boot = Y_original + boot_resid
                boot_data['Y'] = Y_boot
                
                # Update lagged Y values consistently
                for j in range(1, p + 1):
                    col_name = f'Y_L{j}'
                    if col_name in boot_data.columns:
                        boot_data[col_name] = np.roll(Y_boot, j)
                
                # Step 4: Estimate model on bootstrap data
                boot_model = ols(formula, data=boot_data).fit()
                boot_stats = self._compute_test_statistics(boot_model)
                
                # Store valid statistics
                if not np.isnan(boot_stats['F_ov']):
                    boot_F_ov.append(boot_stats['F_ov'])
                if not np.isnan(boot_stats['t']):
                    boot_t.append(boot_stats['t'])
                if not np.isnan(boot_stats['F_ind']):
                    boot_F_ind.append(boot_stats['F_ind'])
                    
            except:
                continue
        
        # Store bootstrap distributions
        self.boot_F_ov = np.array(boot_F_ov) if boot_F_ov else np.array([np.nan])
        self.boot_t = np.array(boot_t) if boot_t else np.array([np.nan])
        self.boot_F_ind = np.array(boot_F_ind) if boot_F_ind else np.array([np.nan])
        
        # Compute critical values and p-values
        self._compute_critical_values()
    
    def _compute_critical_values(self):
        """Compute bootstrap critical values at various significance levels"""
        self.critical_values = {
            'F_ov': {},
            't': {},
            'F_ind': {}
        }
        self.p_values = {}
        
        # For F tests (right-tailed)
        for test_name, boot_dist in [('F_ov', self.boot_F_ov), ('F_ind', self.boot_F_ind)]:
            if len(boot_dist) > 10 and not np.all(np.isnan(boot_dist)):
                boot_clean = boot_dist[~np.isnan(boot_dist)]
                self.critical_values[test_name] = {
                    '10%': np.percentile(boot_clean, 90),
                    '5%': np.percentile(boot_clean, 95),
                    '1%': np.percentile(boot_clean, 99)
                }
                # P-value: proportion of bootstrap stats >= observed
                orig_stat = self.original_stats[test_name]
                if not np.isnan(orig_stat):
                    self.p_values[test_name] = np.mean(boot_clean >= orig_stat)
                else:
                    self.p_values[test_name] = np.nan
            else:
                self.critical_values[test_name] = {'10%': np.nan, '5%': np.nan, '1%': np.nan}
                self.p_values[test_name] = np.nan
        
        # For t test (left-tailed)
        if len(self.boot_t) > 10 and not np.all(np.isnan(self.boot_t)):
            boot_t_clean = self.boot_t[~np.isnan(self.boot_t)]
            self.critical_values['t'] = {
                '10%': np.percentile(boot_t_clean, 10),
                '5%': np.percentile(boot_t_clean, 5),
                '1%': np.percentile(boot_t_clean, 1)
            }
            # P-value: proportion of bootstrap stats <= observed
            orig_stat = self.original_stats['t']
            if not np.isnan(orig_stat):
                self.p_values['t'] = np.mean(boot_t_clean <= orig_stat)
            else:
                self.p_values['t'] = np.nan
        else:
            self.critical_values['t'] = {'10%': np.nan, '5%': np.nan, '1%': np.nan}
            self.p_values['t'] = np.nan
    
    def get_results(self):
        """Return comprehensive test results"""
        results = {
            'statistics': self.original_stats,
            'critical_values': self.critical_values,
            'p_values': self.p_values,
            'n_bootstrap': self.n_bootstrap,
            'case': self.case,
            'conclusions': {}
        }
        
        alpha = 0.05
        
        # F_ov conclusion
        if not np.isnan(self.original_stats['F_ov']) and not np.isnan(self.critical_values['F_ov']['5%']):
            results['conclusions']['F_ov'] = self.original_stats['F_ov'] > self.critical_values['F_ov']['5%']
        else:
            results['conclusions']['F_ov'] = None
        
        # t conclusion
        if not np.isnan(self.original_stats['t']) and not np.isnan(self.critical_values['t']['5%']):
            results['conclusions']['t'] = self.original_stats['t'] < self.critical_values['t']['5%']
        else:
            results['conclusions']['t'] = None
        
        # F_ind conclusion
        if not np.isnan(self.original_stats['F_ind']) and not np.isnan(self.critical_values['F_ind']['5%']):
            results['conclusions']['F_ind'] = self.original_stats['F_ind'] > self.critical_values['F_ind']['5%']
        else:
            results['conclusions']['F_ind'] = None
        
        # Overall cointegration conclusion following the flowchart in paper
        if results['conclusions']['F_ov']:
            if results['conclusions']['t']:
                if results['conclusions']['F_ind']:
                    results['overall'] = 'COINTEGRATION'
                    results['interpretation'] = 'Strong evidence of cointegrating relationship'
                else:
                    results['overall'] = 'DEGENERATE_TYPE2'
                    results['interpretation'] = 'a_yy ≠ 0, but a_yx = 0: X does not appear in long-run'
            else:
                if results['conclusions']['F_ind']:
                    results['overall'] = 'DEGENERATE_TYPE1'
                    results['interpretation'] = 'a_yy = 0, but a_yx ≠ 0: Y does not adjust to equilibrium'
                else:
                    results['overall'] = 'NO_COINTEGRATION'
                    results['interpretation'] = 'No long-run equilibrium relationship'
        else:
            results['overall'] = 'NO_COINTEGRATION'
            results['interpretation'] = 'No long-run equilibrium relationship'
        
        return results
    
    def summary_table(self):
        """Generate summary table for bootstrap test results"""
        results = self.get_results()
        
        rows = []
        for test in ['F_ov', 't', 'F_ind']:
            stat = results['statistics'][test]
            cv_1 = results['critical_values'][test]['1%']
            cv_5 = results['critical_values'][test]['5%']
            cv_10 = results['critical_values'][test]['10%']
            pval = results['p_values'][test]
            
            if test == 't':
                decision = '✓ Reject H₀' if results['conclusions'][test] else '✗ Fail to reject'
            else:
                decision = '✓ Reject H₀' if results['conclusions'][test] else '✗ Fail to reject'
            
            rows.append({
                'Test': test,
                'Statistic': f'{stat:.4f}' if not np.isnan(stat) else 'N/A',
                'CV (1%)': f'{cv_1:.4f}' if not np.isnan(cv_1) else 'N/A',
                'CV (5%)': f'{cv_5:.4f}' if not np.isnan(cv_5) else 'N/A',
                'CV (10%)': f'{cv_10:.4f}' if not np.isnan(cv_10) else 'N/A',
                'p-value': f'{pval:.4f}' if not np.isnan(pval) else 'N/A',
                'Decision': decision
            })
        
        return pd.DataFrame(rows)


class BootstrapNARDL(FourierNARDL):
    """
    Bootstrap Fourier NARDL Model
    
    Combines Fourier-NARDL with bootstrap cointegration tests from 
    Bertelli, Vacca & Zoia (2022) for robust inference.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the time series data
    depvar : str
        Name of the dependent variable
    exog_vars : list
        List of exogenous/control variable names
    decomp_vars : list
        List of variable names to decompose
    maxlag : int, default=4
        Maximum lag order
    max_freq : int, default=3
        Maximum Fourier frequency
    ic : str, default='AIC'
        Information criterion
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    bootstrap_case : str, default='III'
        Case for bootstrap test
    random_state : int, optional
        Random seed
    
    Attributes
    ----------
    bootstrap_test : BootstrapCointegrationTest
        Bootstrap cointegration test object
    bootstrap_results : dict
        Bootstrap test results
    
    Examples
    --------
    >>> from nardl_fourier import BootstrapNARDL
    >>> model = BootstrapNARDL(
    ...     data=df,
    ...     depvar='coal',
    ...     exog_vars=['gdp', 'gdp2'],
    ...     decomp_vars=['oil_price'],
    ...     maxlag=4,
    ...     n_bootstrap=1000
    ... )
    >>> print(model.bootstrap_results['overall'])
    >>> model.plot_bootstrap_distributions()
    
    References
    ----------
    Bertelli, S., Vacca, G., & Zoia, M. (2022). Bootstrap cointegration tests 
    in ARDL models. Economic Modelling, 116, 105987.
    """
    
    def __init__(self, data, depvar, exog_vars, decomp_vars, maxlag=4, max_freq=3,
                 ic='AIC', case=3, n_bootstrap=1000, bootstrap_case='III', 
                 random_state=42):
        
        self.n_bootstrap = n_bootstrap
        self.bootstrap_case = bootstrap_case
        self.random_state = random_state
        
        # Call parent constructor
        super().__init__(data, depvar, exog_vars, decomp_vars, maxlag, max_freq, ic, case)
        
        # Run bootstrap cointegration tests
        self._run_bootstrap_tests()
    
    def _run_bootstrap_tests(self):
        """Run bootstrap cointegration tests"""
        try:
            self.bootstrap_test = BootstrapCointegrationTest(
                model=self.model,
                data=self._model_data,
                depvar=self.depvar,
                decomp_vars=self.decomp_vars,
                exog_vars=self.exog_vars,
                best_lags=self.best_lags,
                n_bootstrap=self.n_bootstrap,
                case=self.bootstrap_case,
                random_state=self.random_state
            )
            self.bootstrap_results = self.bootstrap_test.get_results()
        except Exception as e:
            self.bootstrap_test = None
            self.bootstrap_results = None
            warnings.warn(f"Bootstrap test failed: {e}")
    
    def summary(self):
        """Generate comprehensive summary including bootstrap tests"""
        from ..output.tables import generate_bootstrap_nardl_summary
        return generate_bootstrap_nardl_summary(self)
    
    def plot_bootstrap_distributions(self, figsize=(15, 5)):
        """Plot bootstrap distributions for all three test statistics"""
        import matplotlib.pyplot as plt
        
        if self.bootstrap_test is None:
            raise ValueError("Bootstrap test not available")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # F_ov distribution
        ax1 = axes[0]
        if len(self.bootstrap_test.boot_F_ov) > 10:
            boot_clean = self.bootstrap_test.boot_F_ov[~np.isnan(self.bootstrap_test.boot_F_ov)]
            ax1.hist(boot_clean, bins=50, density=True, alpha=0.7, 
                     color='#3B82F6', edgecolor='navy')
            ax1.axvline(self.bootstrap_test.original_stats['F_ov'], color='red', 
                       linewidth=2, label=f"Observed: {self.bootstrap_test.original_stats['F_ov']:.3f}")
            ax1.axvline(self.bootstrap_test.critical_values['F_ov']['5%'], color='green',
                       linewidth=2, linestyle='--', 
                       label=f"5% CV: {self.bootstrap_test.critical_values['F_ov']['5%']:.3f}")
            ax1.legend(fontsize=9)
        ax1.set_title('Bootstrap Distribution: F_ov', fontweight='bold')
        ax1.set_xlabel('F-statistic')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # t distribution
        ax2 = axes[1]
        if len(self.bootstrap_test.boot_t) > 10:
            boot_clean = self.bootstrap_test.boot_t[~np.isnan(self.bootstrap_test.boot_t)]
            ax2.hist(boot_clean, bins=50, density=True, alpha=0.7,
                     color='#059669', edgecolor='darkgreen')
            ax2.axvline(self.bootstrap_test.original_stats['t'], color='red',
                       linewidth=2, label=f"Observed: {self.bootstrap_test.original_stats['t']:.3f}")
            ax2.axvline(self.bootstrap_test.critical_values['t']['5%'], color='orange',
                       linewidth=2, linestyle='--',
                       label=f"5% CV: {self.bootstrap_test.critical_values['t']['5%']:.3f}")
            ax2.legend(fontsize=9)
        ax2.set_title('Bootstrap Distribution: t-test', fontweight='bold')
        ax2.set_xlabel('t-statistic')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # F_ind distribution
        ax3 = axes[2]
        if len(self.bootstrap_test.boot_F_ind) > 10:
            boot_clean = self.bootstrap_test.boot_F_ind[~np.isnan(self.bootstrap_test.boot_F_ind)]
            ax3.hist(boot_clean, bins=50, density=True, alpha=0.7,
                     color='#DC2626', edgecolor='darkred')
            ax3.axvline(self.bootstrap_test.original_stats['F_ind'], color='blue',
                       linewidth=2, label=f"Observed: {self.bootstrap_test.original_stats['F_ind']:.3f}")
            ax3.axvline(self.bootstrap_test.critical_values['F_ind']['5%'], color='green',
                       linewidth=2, linestyle='--',
                       label=f"5% CV: {self.bootstrap_test.critical_values['F_ind']['5%']:.3f}")
            ax3.legend(fontsize=9)
        ax3.set_title('Bootstrap Distribution: F_ind', fontweight='bold')
        ax3.set_xlabel('F-statistic')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def cointegration_decision(self):
        """Get cointegration test decision with interpretation"""
        if self.bootstrap_results is None:
            return "Bootstrap test not available"
        
        overall = self.bootstrap_results['overall']
        interpretation = self.bootstrap_results['interpretation']
        
        decision_map = {
            'COINTEGRATION': '✅ COINTEGRATION DETECTED',
            'DEGENERATE_TYPE1': '⚠️ DEGENERATE CASE TYPE 1',
            'DEGENERATE_TYPE2': '⚠️ DEGENERATE CASE TYPE 2',
            'NO_COINTEGRATION': '❌ NO COINTEGRATION'
        }
        
        return f"{decision_map.get(overall, overall)}\n{interpretation}"
