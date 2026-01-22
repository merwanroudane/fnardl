"""
Publication-Ready Tables for NARDL Results
===========================================

Generate formatted tables for academic papers in LaTeX, HTML, and markdown.

Author: Dr. Merwan Roudane
"""

import numpy as np
import pandas as pd
from scipy import stats


def format_pvalue(p):
    """Format p-value for display"""
    if np.isnan(p):
        return "N/A"
    elif p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


def get_significance_stars(p):
    """Get significance stars"""
    if np.isnan(p):
        return ""
    elif p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


class ResultsTable:
    """
    Class for generating publication-ready tables
    
    Parameters
    ----------
    model : NARDL, FourierNARDL, or BootstrapNARDL
        Fitted NARDL model
    
    Examples
    --------
    >>> table = ResultsTable(model)
    >>> print(table.regression_table())
    >>> table.to_latex('results.tex')
    """
    
    def __init__(self, model):
        self.model = model
    
    def regression_table(self, output='dataframe'):
        """
        Generate regression results table
        
        Parameters
        ----------
        output : str
            Output format ('dataframe', 'latex', 'html', 'markdown')
        """
        m = self.model.model
        
        df = pd.DataFrame({
            'Variable': m.params.index,
            'Coefficient': m.params.values,
            'Std. Error': m.bse.values,
            't-statistic': m.tvalues.values,
            'p-value': m.pvalues.values
        })
        df['Sig.'] = df['p-value'].apply(get_significance_stars)
        df = df.round({'Coefficient': 6, 'Std. Error': 6, 't-statistic': 4, 'p-value': 4})
        
        if output == 'latex':
            return self._to_latex(df, 'Regression Results')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def long_run_table(self, output='dataframe'):
        """Generate long-run multipliers table"""
        rows = []
        for var in self.model.decomp_vars:
            lr = self.model.long_run[var]
            for sign in ['positive', 'negative']:
                data = lr[sign]
                rows.append({
                    'Variable': f"{var}{'⁺' if sign == 'positive' else '⁻'}",
                    'Coefficient': data['coefficient'],
                    'Std. Error': data['std_error'],
                    't-statistic': data['t_statistic'],
                    'p-value': data['p_value'],
                    '95% CI Lower': data['ci_lower'],
                    '95% CI Upper': data['ci_upper'],
                })
        
        df = pd.DataFrame(rows)
        df['Sig.'] = df['p-value'].apply(get_significance_stars)
        df = df.round(4)
        
        if output == 'latex':
            return self._to_latex(df, 'Long-Run Multipliers')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def short_run_table(self, output='dataframe'):
        """Generate short-run coefficients table"""
        rows = []
        
        # ECT
        ect = self.model.ect
        rows.append({
            'Variable': 'ECT (ρ)',
            'Coefficient': ect['coefficient'],
            'Std. Error': ect['std_error'],
            't-statistic': ect['t_statistic'],
            'p-value': ect['p_value'],
        })
        
        # Short-run coefficients
        for var in self.model.decomp_vars:
            for sign in ['positive', 'negative']:
                for item in self.model.short_run[var][sign]:
                    rows.append({
                        'Variable': item['name'],
                        'Coefficient': item['coefficient'],
                        'Std. Error': item['std_error'],
                        't-statistic': item['t_statistic'],
                        'p-value': item['p_value'],
                    })
        
        df = pd.DataFrame(rows)
        df['Sig.'] = df['p-value'].apply(get_significance_stars)
        df = df.round(4)
        
        if output == 'latex':
            return self._to_latex(df, 'Short-Run Coefficients')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def diagnostics_table(self, output='dataframe'):
        """Generate diagnostics table"""
        diag = self.model.diagnostics
        rows = []
        
        if diag.get('jarque_bera'):
            jb = diag['jarque_bera']
            rows.append({
                'Test': 'Jarque-Bera (Normality)',
                'Statistic': jb['statistic'],
                'p-value': jb['p_value'],
                'Decision': 'Normal' if jb['normal'] else 'Non-Normal'
            })
        
        if diag.get('breusch_godfrey'):
            bg = diag['breusch_godfrey']
            rows.append({
                'Test': 'Breusch-Godfrey (Serial Corr.)',
                'Statistic': bg['lm_statistic'],
                'p-value': bg['lm_pvalue'],
                'Decision': 'No Autocorr.' if bg['no_autocorrelation'] else 'Autocorr.'
            })
        
        if diag.get('breusch_pagan'):
            bp = diag['breusch_pagan']
            rows.append({
                'Test': 'Breusch-Pagan (Heterosk.)',
                'Statistic': bp['lm_statistic'],
                'p-value': bp['lm_pvalue'],
                'Decision': 'Homosked.' if bp['homoskedastic'] else 'Heterosked.'
            })
        
        if diag.get('durbin_watson'):
            dw = diag['durbin_watson']
            rows.append({
                'Test': 'Durbin-Watson',
                'Statistic': dw['statistic'],
                'p-value': np.nan,
                'Decision': dw['interpretation']
            })
        
        df = pd.DataFrame(rows)
        df = df.round(4)
        
        if output == 'latex':
            return self._to_latex(df, 'Diagnostic Tests')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def bounds_test_table(self, output='dataframe'):
        """Generate bounds test table"""
        bt = self.model.bounds_test
        
        rows = []
        for sig in ['10%', '5%', '1%']:
            cv = bt['critical_values']['F'][sig]
            rows.append({
                'Significance': sig,
                'I(0) Bound': cv['I(0)'],
                'I(1) Bound': cv['I(1)'],
                'F-statistic': bt['f_statistic'],
                'Decision': bt['decision'].get(f'F_{sig}', 'N/A')
            })
        
        df = pd.DataFrame(rows)
        df = df.round(4)
        
        if output == 'latex':
            return self._to_latex(df, 'PSS Bounds Test')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def wald_test_table(self, output='dataframe'):
        """Generate Wald test for asymmetry table"""
        rows = []
        
        for var in self.model.decomp_vars:
            wald = self.model.wald[var]
            
            rows.append({
                'Variable': var,
                'Test': 'Short-Run (contemporaneous)',
                'F-statistic': wald['short_run']['f_statistic'],
                'p-value': wald['short_run']['p_value'],
                'Asymmetric': 'Yes' if wald['short_run']['asymmetric'] else 'No'
            })
            
            rows.append({
                'Variable': var,
                'Test': 'Long-Run',
                'F-statistic': wald['long_run']['f_statistic'],
                'p-value': wald['long_run']['p_value'],
                'Asymmetric': 'Yes' if wald['long_run']['asymmetric'] else 'No'
            })
        
        df = pd.DataFrame(rows)
        df = df.round(4)
        
        if output == 'latex':
            return self._to_latex(df, 'Wald Tests for Asymmetry')
        elif output == 'html':
            return df.to_html(index=False)
        elif output == 'markdown':
            return df.to_markdown(index=False)
        return df
    
    def _to_latex(self, df, caption):
        """Convert DataFrame to LaTeX table"""
        latex = df.to_latex(index=False, escape=False, 
                           caption=caption, label=f"tab:{caption.lower().replace(' ', '_')}")
        return latex
    
    def to_latex(self, filename, tables='all'):
        """Export all tables to LaTeX file"""
        with open(filename, 'w') as f:
            f.write("% NARDL Results Tables\n")
            f.write("% Generated by nardl-fourier library\n\n")
            
            if tables == 'all' or 'regression' in tables:
                f.write(self.regression_table('latex'))
                f.write("\n\n")
            
            if tables == 'all' or 'longrun' in tables:
                f.write(self.long_run_table('latex'))
                f.write("\n\n")
            
            if tables == 'all' or 'shortrun' in tables:
                f.write(self.short_run_table('latex'))
                f.write("\n\n")
            
            if tables == 'all' or 'diagnostics' in tables:
                f.write(self.diagnostics_table('latex'))
                f.write("\n\n")
            
            if tables == 'all' or 'bounds' in tables:
                f.write(self.bounds_test_table('latex'))
                f.write("\n\n")


def generate_nardl_summary(model):
    """Generate comprehensive NARDL summary"""
    lines = []
    lines.append("=" * 80)
    lines.append("NARDL MODEL ESTIMATION RESULTS")
    lines.append("=" * 80)
    lines.append(f"Dependent Variable: {model.depvar}")
    lines.append(f"Decomposed Variables: {', '.join(model.decomp_vars)}")
    lines.append(f"Control Variables: {', '.join(model.exog_vars)}")
    lines.append("-" * 80)
    lines.append(f"Observations: {int(model.model.nobs)}")
    lines.append(f"R-squared: {model.model.rsquared:.4f}")
    lines.append(f"Adj. R-squared: {model.model.rsquared_adj:.4f}")
    lines.append(f"Information Criterion ({model.ic}): {model.best_ic:.4f}")
    lines.append("-" * 80)
    lines.append("Optimal Lag Structure:")
    lines.append(f"  p (Y lags): {model.best_lags['p']}")
    lines.append(f"  q (X lags): {model.best_lags['q']}")
    r_str = ", ".join([f"{k}={v}" for k, v in model.best_lags['r'].items()])
    lines.append(f"  r (Z lags): {r_str}")
    lines.append("=" * 80)
    
    # ECT
    lines.append("\nERROR CORRECTION TERM")
    lines.append("-" * 40)
    ect = model.ect
    lines.append(f"ECT (ρ): {ect['coefficient']:.6f}")
    lines.append(f"Std. Error: {ect['std_error']:.6f}")
    lines.append(f"t-statistic: {ect['t_statistic']:.4f}")
    lines.append(f"p-value: {format_pvalue(ect['p_value'])}")
    if ect['is_valid'] and ect['is_stable']:
        lines.append(f"Half-life: {ect['half_life']:.2f} periods")
        lines.append("Status: ✓ Valid (negative and stable)")
    else:
        lines.append("Status: ✗ Invalid")
    
    # Bounds Test
    lines.append("\n" + "=" * 80)
    lines.append("PSS BOUNDS TEST FOR COINTEGRATION")
    lines.append("-" * 80)
    bt = model.bounds_test
    lines.append(f"F-statistic: {bt['f_statistic']:.4f}")
    lines.append(f"Critical Values (5%): I(0)={bt['critical_values']['F']['5%']['I(0)']:.4f}, "
                f"I(1)={bt['critical_values']['F']['5%']['I(1)']:.4f}")
    lines.append(f"Decision: {bt['decision']['F_5%']}")
    
    # Long-run
    lines.append("\n" + "=" * 80)
    lines.append("LONG-RUN MULTIPLIERS")
    lines.append("-" * 80)
    for var in model.decomp_vars:
        lr = model.long_run[var]
        lines.append(f"\n{var}:")
        lines.append(f"  L⁺: {lr['positive']['coefficient']:.6f} "
                    f"(SE={lr['positive']['std_error']:.4f}, p={format_pvalue(lr['positive']['p_value'])})"
                    f" {get_significance_stars(lr['positive']['p_value'])}")
        lines.append(f"  L⁻: {lr['negative']['coefficient']:.6f} "
                    f"(SE={lr['negative']['std_error']:.4f}, p={format_pvalue(lr['negative']['p_value'])})"
                    f" {get_significance_stars(lr['negative']['p_value'])}")
        
        # Wald test
        wald = model.wald[var]['long_run']
        lines.append(f"  Asymmetry Test: F={wald['f_statistic']:.3f}, p={format_pvalue(wald['p_value'])}")
        lines.append(f"  → {'Asymmetric' if wald['asymmetric'] else 'Symmetric'}")
    
    lines.append("\n" + "=" * 80)
    lines.append("Note: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_fnardl_summary(model):
    """Generate Fourier NARDL summary"""
    lines = []
    lines.append("=" * 80)
    lines.append("FOURIER NARDL MODEL ESTIMATION RESULTS")
    lines.append("=" * 80)
    lines.append(f"Dependent Variable: {model.depvar}")
    lines.append(f"Decomposed Variables: {', '.join(model.decomp_vars)}")
    lines.append(f"Control Variables: {', '.join(model.exog_vars)}")
    lines.append("-" * 80)
    lines.append(f"Observations: {int(model.model.nobs)}")
    lines.append(f"R-squared: {model.model.rsquared:.4f}")
    lines.append(f"Adj. R-squared: {model.model.rsquared_adj:.4f}")
    lines.append(f"Information Criterion ({model.ic}): {model.best_ic:.4f}")
    lines.append("-" * 80)
    lines.append("Optimal Lag Structure:")
    lines.append(f"  p (Y lags): {model.best_lags['p']}")
    lines.append(f"  q (X lags): {model.best_lags['q']}")
    r_str = ", ".join([f"{k}={v}" for k, v in model.best_lags['r'].items()])
    lines.append(f"  r (Z lags): {r_str}")
    lines.append("-" * 80)
    lines.append("FOURIER APPROXIMATION:")
    lines.append(f"  Optimal Frequency (k*): {model.best_freq:.2f}")
    
    if hasattr(model, 'fourier_test') and model.fourier_test:
        ft = model.fourier_test
        lines.append(f"  F-test (sin, cos): F={ft['f_statistic']:.3f}, p={format_pvalue(ft['p_value'])}")
        lines.append(f"  → {'Significant' if ft['significant'] else 'Not significant'}")
    
    lines.append("=" * 80)
    
    # Rest is similar to standard NARDL
    # ECT
    lines.append("\nERROR CORRECTION TERM")
    lines.append("-" * 40)
    ect = model.ect
    lines.append(f"ECT (ρ): {ect['coefficient']:.6f}")
    lines.append(f"t-statistic: {ect['t_statistic']:.4f}, p-value: {format_pvalue(ect['p_value'])}")
    if ect['is_valid'] and ect['is_stable']:
        lines.append(f"Half-life: {ect['half_life']:.2f} periods")
        lines.append("Status: ✓ Valid (negative and stable)")
    
    # Long-run multipliers
    lines.append("\n" + "=" * 80)
    lines.append("LONG-RUN MULTIPLIERS")
    lines.append("-" * 80)
    for var in model.decomp_vars:
        lr = model.long_run[var]
        lines.append(f"\n{var}:")
        lines.append(f"  L⁺: {lr['positive']['coefficient']:.6f} {get_significance_stars(lr['positive']['p_value'])}")
        lines.append(f"  L⁻: {lr['negative']['coefficient']:.6f} {get_significance_stars(lr['negative']['p_value'])}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def generate_bootstrap_nardl_summary(model):
    """Generate Bootstrap NARDL summary"""
    lines = []
    lines.append("=" * 80)
    lines.append("BOOTSTRAP FOURIER NARDL MODEL RESULTS")
    lines.append("(Bertelli, Vacca & Zoia, 2022)")
    lines.append("=" * 80)
    lines.append(f"Dependent Variable: {model.depvar}")
    lines.append(f"Observations: {int(model.model.nobs)}")
    lines.append(f"R-squared: {model.model.rsquared:.4f}")
    lines.append(f"Optimal Frequency: {model.best_freq:.2f}")
    lines.append(f"Bootstrap Replications: {model.n_bootstrap}")
    lines.append("=" * 80)
    
    # Bootstrap cointegration results
    if model.bootstrap_results:
        results = model.bootstrap_results
        
        lines.append("\nBOOTSTRAP COINTEGRATION TESTS")
        lines.append("-" * 80)
        
        for test in ['F_ov', 't', 'F_ind']:
            stat = results['statistics'][test]
            cv = results['critical_values'][test]['5%']
            pval = results['p_values'][test]
            decision = '✓ Reject H₀' if results['conclusions'][test] else '✗ Fail to reject'
            
            lines.append(f"\n{test}:")
            lines.append(f"  Statistic: {stat:.4f}")
            lines.append(f"  Bootstrap CV (5%): {cv:.4f}")
            lines.append(f"  p-value: {format_pvalue(pval)}")
            lines.append(f"  Decision: {decision}")
        
        lines.append("\n" + "-" * 80)
        overall = results['overall']
        interp = results['interpretation']
        
        if overall == 'COINTEGRATION':
            lines.append("✅ CONCLUSION: COINTEGRATION DETECTED")
        elif overall == 'DEGENERATE_TYPE1':
            lines.append("⚠️ CONCLUSION: DEGENERATE CASE TYPE 1")
        elif overall == 'DEGENERATE_TYPE2':
            lines.append("⚠️ CONCLUSION: DEGENERATE CASE TYPE 2")
        else:
            lines.append("❌ CONCLUSION: NO COINTEGRATION")
        
        lines.append(f"   {interp}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def generate_long_run_table(model):
    """Generate long-run results DataFrame"""
    table = ResultsTable(model)
    return table.long_run_table()


def generate_short_run_table(model):
    """Generate short-run results DataFrame"""
    table = ResultsTable(model)
    return table.short_run_table()


def generate_diagnostics_table(model):
    """Generate diagnostics DataFrame"""
    table = ResultsTable(model)
    return table.diagnostics_table()
