<p align="center">
  <img src="https://img.shields.io/badge/NARDL--Fourier-v1.0.1-blue?style=for-the-badge&logo=python&logoColor=white" alt="Version"/>
</p>

<h1 align="center">🌊 NARDL-Fourier</h1>

<p align="center">
  <b>Nonlinear Autoregressive Distributed Lag with Fourier Approximation & Bootstrap Cointegration</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT"/></a>
  <a href="https://pypi.org/project/nardl-fourier/"><img src="https://img.shields.io/badge/PyPI-nardl--fourier-brightgreen?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://merwanroudane.github.io/fnardl/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-blue?style=flat-square&logo=github" alt="Documentation"/></a>
</p>

<p align="center">
  <i>A comprehensive Python library for asymmetric cointegration analysis with structural breaks detection and robust bootstrap inference.</i>
</p>

---

## 🎯 Overview

**NARDL-Fourier** is a state-of-the-art econometric library that implements three advanced models for analyzing asymmetric long-run relationships between time series variables:

| Model | Description | Key Feature | Reference |
|:------|:------------|:------------|:----------|
| **NARDL** | Standard Nonlinear ARDL | Asymmetric decomposition | Shin, Yu & Greenwood-Nimmo (2014) |
| **FourierNARDL** | Fourier-augmented NARDL | Smooth structural breaks | Zaghdoudi et al. (2023) |
| **BootstrapNARDL** | Bootstrap NARDL | Robust inference | Bertelli, Vacca & Zoia (2022) |

---

## 📚 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [📐 Theoretical Background](#-theoretical-background)
- [🚀 Installation](#-installation)
- [📖 Quick Start](#-quick-start)
- [🔬 Model Specifications](#-model-specifications)
- [📊 Complete Examples](#-complete-examples)
- [📈 Visualizations](#-visualizations)
- [🧪 Diagnostic Tests](#-diagnostic-tests)
- [📋 Export Options](#-export-options)
- [🔧 API Reference](#-api-reference)
- [📚 References](#-references)
- [👨‍🔬 Author](#-author)
- [📄 License](#-license)

---

## ✨ Features

### 🔷 Asymmetric Analysis
- **Partial sum decomposition**: Separate positive (x⁺) and negative (x⁻) changes
- **Long-run multipliers**: L⁺ and L⁻ with delta-method standard errors
- **Short-run dynamics**: Asymmetric ECM with lag-specific coefficients
- **Wald tests**: LR and SR asymmetry testing with p-values

### 🔷 Structural Breaks
- **Fourier approximation**: Captures unknown smooth breaks
- **Optimal frequency selection**: AIC/BIC-based k* determination
- **No break date required**: Data-driven structural change detection

### 🔷 Robust Inference
- **PSS bounds test**: Pesaran et al. (2001) critical values for I(0)/I(1)
- **Bootstrap cointegration**: McNown et al. (2018) F and t tests
- **No inconclusive zone**: Bootstrap eliminates bounds uncertainty
- **Dynamic multipliers**: 95% bootstrap confidence intervals

### 🔷 Automatic Optimization
- **Lag selection**: AIC/BIC/HQ optimization for p, q, r
- **General-to-specific**: Automatic insignificant lag removal
- **Variance-weighted**: Partial sum normalization options

### 🔷 Diagnostic Suite
- **Normality**: Jarque-Bera, Shapiro-Wilk tests
- **Serial correlation**: Breusch-Godfrey, Durbin-Watson, Ljung-Box
- **Heteroskedasticity**: Breusch-Pagan, White, ARCH-LM tests
- **Stability**: CUSUM and CUSUM of Squares plots

### 🔷 Publication-Ready Output
- **LaTeX tables**: Journal-quality regression tables
- **HTML/Markdown**: Web-ready formatted output
- **High-resolution plots**: Publication-quality figures
- **Comprehensive summaries**: All results in one view

---

## 📐 Theoretical Background

### 1️⃣ Standard NARDL Model

The Nonlinear ARDL framework by **Shin, Yu & Greenwood-Nimmo (2014)** captures asymmetric long-run relationships through partial sum decomposition:

**Partial Sum Decomposition:**
```
x⁺ₜ = Σⱼ₌₁ᵗ Δxⱼ⁺ = Σⱼ₌₁ᵗ max(Δxⱼ, 0)
x⁻ₜ = Σⱼ₌₁ᵗ Δxⱼ⁻ = Σⱼ₌₁ᵗ min(Δxⱼ, 0)
```

**Error Correction Model:**
```
Δyₜ = α + ρyₜ₋₁ + θ⁺x⁺ₜ₋₁ + θ⁻x⁻ₜ₋₁ + Σᵢ₌₁ᵖ γᵢΔyₜ₋ᵢ + Σⱼ₌₀ᵍ (π⁺ⱼΔx⁺ₜ₋ⱼ + π⁻ⱼΔx⁻ₜ₋ⱼ) + εₜ
```

**Long-Run Multipliers:**
```
L⁺ = -θ⁺/ρ    (Effect of positive changes)
L⁻ = -θ⁻/ρ    (Effect of negative changes)
```

**Asymmetry Testing:**
- **LR Asymmetry**: H₀: L⁺ = L⁻ (Wald test)
- **SR Asymmetry**: H₀: Σπ⁺ᵢ = Σπ⁻ᵢ (Wald test)

### 2️⃣ Fourier NARDL Model

**Zaghdoudi et al. (2023)** extend NARDL with Fourier terms to capture smooth structural breaks without requiring prior knowledge of break dates:

**Fourier Terms:**
```
sin(2πkt/T) and cos(2πkt/T)    where k = 1, 2, ..., max_freq
```

**Extended Model:**
```
Δyₜ = α + ρyₜ₋₁ + θ⁺x⁺ₜ₋₁ + θ⁻x⁻ₜ₋₁ + β₁sin(2πkt/T) + β₂cos(2πkt/T) 
      + Σᵢ₌₁ᵖ γᵢΔyₜ₋ᵢ + Σⱼ₌₀ᵍ (π⁺ⱼΔx⁺ₜ₋ⱼ + π⁻ⱼΔx⁻ₜ₋ⱼ) + εₜ
```

**Optimal Frequency Selection:**
The optimal k* minimizes the information criterion (AIC/BIC) across all candidate frequencies.

### 3️⃣ Bootstrap NARDL Model

**Bertelli, Vacca & Zoia (2022)** propose bootstrap-based cointegration tests that eliminate the "inconclusive zone" problem of PSS bounds tests:

**Three Test Statistics:**
1. **F_overall (Fₒᵥ)**: Joint test on all lagged levels
2. **t-statistic (t)**: Test on the error correction term
3. **F_independent (Fᵢₙd)**: Test on lagged X levels only

**Bootstrap Procedure:**
1. Estimate the restricted (no cointegration) model under H₀
2. Generate B bootstrap samples using residual resampling
3. Compute test statistics for each bootstrap sample
4. Derive empirical critical values from bootstrap distribution

**Advantages:**
- ✅ Correct size in small samples
- ✅ No inconclusive zone
- ✅ Valid inference regardless of I(0)/I(1) uncertainty

---

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install nardl-fourier
```

### From Source

```bash
git clone https://github.com/merwanroudane/fnardl.git
cd fnardl
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/merwanroudane/fnardl.git
cd fnardl
pip install -e ".[dev,notebook]"
```

### Dependencies

| Package | Minimum Version | Purpose |
|:--------|:---------------|:--------|
| numpy | ≥1.20.0 | Array operations |
| pandas | ≥1.3.0 | Data handling |
| scipy | ≥1.7.0 | Statistical functions |
| statsmodels | ≥0.13.0 | Econometric models |
| matplotlib | ≥3.4.0 | Visualizations |
| tabulate | ≥0.8.9 | Table formatting |

---

## 📖 Quick Start

### Basic Usage

```python
from nardl_fourier import NARDL, FourierNARDL, BootstrapNARDL
import pandas as pd

# Load your time series data
data = pd.read_excel('your_data.xlsx')

# Option 1: Standard NARDL
model = NARDL(
    data=data,
    depvar='y',              # Dependent variable column
    exog_vars=['z1', 'z2'],  # Control variables (symmetric)
    decomp_vars=['x'],       # Variable to decompose asymmetrically
    maxlag=4,
    ic='AIC'
)

# Option 2: Fourier NARDL (with structural breaks)
model = FourierNARDL(
    data=data,
    depvar='y',
    exog_vars=['z1', 'z2'],
    decomp_vars=['x'],
    maxlag=4,
    max_freq=3,              # Maximum Fourier frequency to test
    ic='AIC'
)

# Option 3: Bootstrap NARDL (robust inference)
model = BootstrapNARDL(
    data=data,
    depvar='y',
    exog_vars=['z1', 'z2'],
    decomp_vars=['x'],
    maxlag=4,
    n_bootstrap=1000
)

# View results
print(model.summary())
```

---

## 🔬 Model Specifications

### NARDL Class

```python
NARDL(
    data: pd.DataFrame,           # Time series data
    depvar: str,                  # Dependent variable name
    exog_vars: List[str] = None,  # Symmetric control variables
    decomp_vars: List[str] = None,# Variables to decompose (asymmetric)
    maxlag: int = 4,              # Maximum lag order
    ic: str = 'AIC',              # Information criterion ('AIC', 'BIC', 'HQ')
    case: int = 3,                # Deterministic case (1-5)
    threshold: float = 0.0,       # Threshold for decomposition
)
```

### FourierNARDL Class

```python
FourierNARDL(
    data: pd.DataFrame,
    depvar: str,
    exog_vars: List[str] = None,
    decomp_vars: List[str] = None,
    maxlag: int = 4,
    max_freq: int = 3,            # Maximum Fourier frequency
    ic: str = 'AIC',
    case: int = 3,
)
```

### BootstrapNARDL Class

```python
BootstrapNARDL(
    data: pd.DataFrame,
    depvar: str,
    exog_vars: List[str] = None,
    decomp_vars: List[str] = None,
    maxlag: int = 4,
    n_bootstrap: int = 1000,      # Bootstrap replications
    confidence_level: float = 0.95,
    seed: int = None,             # Random seed for reproducibility
)
```

---

## 📊 Complete Examples

### Example 1: Standard NARDL Analysis

```python
from nardl_fourier import NARDL
import pandas as pd

# Load data
data = pd.read_excel('energy_data.xlsx')

# Estimate NARDL model
model = NARDL(
    data=data,
    depvar='coal',                    # Coal consumption
    exog_vars=['gdp', 'gdp2'],        # GDP and GDP squared
    decomp_vars=['oil'],              # Oil price (asymmetric)
    maxlag=4,
    ic='AIC'
)

# Display full summary
print(model.summary())

# Long-run coefficients table
print("\n=== LONG-RUN COEFFICIENTS ===")
print(model.long_run_table())

# Short-run coefficients table  
print("\n=== SHORT-RUN COEFFICIENTS ===")
print(model.short_run_table())

# Asymmetry tests
print("\n=== ASYMMETRY TESTS ===")
lr_asymmetry = model.wald_lr_asymmetry()
sr_asymmetry = model.wald_sr_asymmetry()
print(f"Long-run asymmetry: Wald={lr_asymmetry['statistic']:.4f}, p={lr_asymmetry['pvalue']:.4f}")
print(f"Short-run asymmetry: Wald={sr_asymmetry['statistic']:.4f}, p={sr_asymmetry['pvalue']:.4f}")

# PSS Bounds Test
print("\n=== COINTEGRATION TEST ===")
bounds = model.bounds_test()
print(f"F-statistic: {bounds['f_statistic']:.4f}")
print(f"Critical Values (5%): I(0)={bounds['cv_I0_5pct']:.3f}, I(1)={bounds['cv_I1_5pct']:.3f}")
print(f"Decision: {bounds['decision']}")

# Plot dynamic multipliers
fig = model.plot_multipliers(horizon=20)
fig.savefig('multipliers.png', dpi=300)
```

### Example 2: Fourier NARDL with Structural Breaks

```python
from nardl_fourier import FourierNARDL
import pandas as pd

# Load data with potential structural breaks
data = pd.read_excel('macro_data.xlsx')

# Estimate Fourier NARDL
model = FourierNARDL(
    data=data,
    depvar='inflation',
    exog_vars=['unemployment'],
    decomp_vars=['oil_price'],
    maxlag=4,
    max_freq=3,                       # Test frequencies k=1,2,3
    ic='AIC'
)

# Optimal frequency
print(f"Optimal Fourier frequency: k* = {model.best_freq}")
print(f"Fourier coefficients significant: {model.fourier_significant}")

# Summary with Fourier terms
print(model.summary())

# Compare IC across frequencies
print("\n=== FREQUENCY SELECTION ===")
for k, ic_val in model.frequency_comparison.items():
    print(f"k={k}: AIC={ic_val:.4f}")

# Plot Fourier component
fig = model.plot_fourier_component()
fig.savefig('fourier_breaks.png', dpi=300)
```

### Example 3: Bootstrap Cointegration Tests

```python
from nardl_fourier import BootstrapNARDL
import pandas as pd

# Load data
data = pd.read_excel('finance_data.xlsx')

# Estimate with bootstrap inference
model = BootstrapNARDL(
    data=data,
    depvar='stock_returns',
    exog_vars=['interest_rate'],
    decomp_vars=['exchange_rate'],
    maxlag=4,
    n_bootstrap=2000,                 # More replications for precision
    seed=42                           # For reproducibility
)

# Bootstrap cointegration tests
print("\n=== BOOTSTRAP COINTEGRATION TESTS ===")
tests = model.cointegration_tests()

for test_name, results in tests.items():
    print(f"\n{test_name}:")
    print(f"  Statistic: {results['statistic']:.4f}")
    print(f"  Bootstrap CV (5%): {results['critical_value']:.4f}")
    print(f"  Bootstrap p-value: {results['pvalue']:.4f}")
    print(f"  Decision: {'Reject H₀' if results['reject'] else 'Fail to Reject H₀'}")

# Overall conclusion
print(f"\n🔍 CONCLUSION: {model.cointegration_decision()}")

# Plot bootstrap distributions
fig = model.plot_bootstrap_distributions()
fig.savefig('bootstrap_distributions.png', dpi=300)

# Long-run multipliers with bootstrap CIs
print("\n=== LONG-RUN MULTIPLIERS (Bootstrap 95% CI) ===")
multipliers = model.long_run_multipliers_bootstrap()
for var, stats in multipliers.items():
    print(f"{var}: {stats['estimate']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
```

### Example 4: Complete Workflow

```python
from nardl_fourier import NARDL, FourierNARDL, BootstrapNARDL, ResultsTable, NARDLPlots
from nardl_fourier.diagnostics import run_all_diagnostics
import pandas as pd

# ============================================
# STEP 1: Data Preparation
# ============================================
data = pd.read_excel('dataset.xlsx')

# Check for stationarity (external test recommended)
print("Sample size:", len(data))
print("Variables:", data.columns.tolist())

# ============================================
# STEP 2: Model Comparison
# ============================================
# Fit all three models
nardl = NARDL(data=data, depvar='y', decomp_vars=['x'], maxlag=4)
fnardl = FourierNARDL(data=data, depvar='y', decomp_vars=['x'], maxlag=4, max_freq=3)
bnardl = BootstrapNARDL(data=data, depvar='y', decomp_vars=['x'], maxlag=4, n_bootstrap=1000)

# Compare model fit
print("\n=== MODEL COMPARISON ===")
print(f"NARDL:        AIC={nardl.aic:.4f}, Adj-R²={nardl.adj_rsquared:.4f}")
print(f"FourierNARDL: AIC={fnardl.aic:.4f}, Adj-R²={fnardl.adj_rsquared:.4f}, k*={fnardl.best_freq}")
print(f"BootstrapNARDL: Cointegration={bnardl.cointegration_decision()}")

# ============================================
# STEP 3: Diagnostic Tests
# ============================================
best_model = fnardl  # Choose based on comparison
diagnostics = run_all_diagnostics(best_model)

print("\n=== DIAGNOSTIC TESTS ===")
print(f"Normality (JB): p-value = {diagnostics['normality']['jarque_bera']['pvalue']:.4f}")
print(f"Serial Corr (BG): p-value = {diagnostics['serial_correlation']['breusch_godfrey']['pvalue']:.4f}")
print(f"Heterosk (BP): p-value = {diagnostics['heteroskedasticity']['breusch_pagan']['pvalue']:.4f}")
print(f"Issues detected: {diagnostics['summary']['issues_detected']}")

# ============================================
# STEP 4: Results Output
# ============================================
table = ResultsTable(best_model)

# Export to LaTeX
table.to_latex('results.tex')

# Individual tables
print(table.long_run_table('markdown'))
print(table.short_run_table('markdown'))
print(table.diagnostics_table('markdown'))

# ============================================
# STEP 5: Visualizations
# ============================================
plots = NARDLPlots(best_model)

# Dynamic multipliers
plots.multipliers(save_path='multipliers.png')

# Stability tests
plots.cusum(save_path='cusum.png')
plots.cusumsq(save_path='cusumsq.png')

# Residual diagnostics
plots.residuals(save_path='residuals.png')
```

---

## 📈 Visualizations

### Available Plots

| Plot | Method | Description |
|:-----|:-------|:------------|
| Dynamic Multipliers | `model.plot_multipliers()` | Cumulative asymmetric effects with 95% CI |
| CUSUM | `model.plot_cusum()` | Recursive residuals stability |
| CUSUM of Squares | `model.plot_cusumsq()` | Variance stability |
| Bootstrap Distributions | `model.plot_bootstrap_distributions()` | Bootstrap test distributions |
| Fourier Component | `model.plot_fourier_component()` | Estimated structural breaks |

### Using NARDLPlots Class

```python
from nardl_fourier import NARDLPlots

# Initialize with fitted model
plots = NARDLPlots(model)

# Generate all plots
fig1 = plots.multipliers(horizon=20, save_path='fig1.png')
fig2 = plots.cusum(save_path='fig2.png')
fig3 = plots.cusumsq(save_path='fig3.png')
fig4 = plots.residuals(save_path='fig4.png')
fig5 = plots.short_run_coefficients(save_path='fig5.png')
fig6 = plots.ect_path(save_path='fig6.png')

# Customize appearance
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

fig = plots.multipliers(
    horizon=30,
    figsize=(12, 6),
    colors={'positive': '#2ecc71', 'negative': '#e74c3c'},
    title='Dynamic Multiplier Effects'
)
```

---

## 🧪 Diagnostic Tests

### Comprehensive Diagnostics

```python
from nardl_fourier.diagnostics import run_all_diagnostics

# Run all tests
results = run_all_diagnostics(model)

# Access individual results
print(results['normality']['jarque_bera'])
print(results['normality']['shapiro_wilk'])
print(results['serial_correlation']['breusch_godfrey'])
print(results['serial_correlation']['durbin_watson'])
print(results['serial_correlation']['ljung_box'])
print(results['heteroskedasticity']['breusch_pagan'])
print(results['heteroskedasticity']['white'])
print(results['heteroskedasticity']['arch_lm'])
print(results['stability']['cusum'])
print(results['stability']['cusumsq'])

# Summary
print(results['summary']['overall_assessment'])
print(results['summary']['issues_detected'])
```

### Individual Tests

```python
from nardl_fourier.diagnostics import (
    jarque_bera_test,
    shapiro_wilk_test,
    breusch_godfrey_test,
    durbin_watson_test,
    ljung_box_test,
    breusch_pagan_test,
    white_test,
    arch_lm_test,
    cusum_test,
    cusumsq_test
)

# Normality
jb = jarque_bera_test(model.resid)
print(f"Jarque-Bera: stat={jb['statistic']:.4f}, p={jb['pvalue']:.4f}")

# Serial correlation
bg = breusch_godfrey_test(model, nlags=4)
print(f"Breusch-Godfrey: stat={bg['statistic']:.4f}, p={bg['pvalue']:.4f}")

# Heteroskedasticity
bp = breusch_pagan_test(model)
print(f"Breusch-Pagan: stat={bp['statistic']:.4f}, p={bp['pvalue']:.4f}")
```

---

## 📋 Export Options

### ResultsTable Class

```python
from nardl_fourier import ResultsTable

table = ResultsTable(model)

# Full regression table
print(table.regression_table('markdown'))
print(table.regression_table('latex'))
print(table.regression_table('html'))

# Long-run coefficients
print(table.long_run_table('latex'))

# Short-run coefficients
print(table.short_run_table('latex'))

# Diagnostics summary
print(table.diagnostics_table('latex'))

# Bounds test results
print(table.bounds_test_table('latex'))

# Export all to file
table.to_latex('all_results.tex')
table.to_html('all_results.html')
table.to_markdown('all_results.md')
```

### LaTeX Table Example

```latex
\begin{table}[htbp]
\centering
\caption{Long-Run Coefficients}
\begin{tabular}{lcccc}
\hline
Variable & Coefficient & Std. Error & t-stat & p-value \\
\hline
oil$^+$ & 0.4521 & 0.0892 & 5.068 & 0.000*** \\
oil$^-$ & -0.2134 & 0.0756 & -2.823 & 0.006** \\
gdp & 0.8765 & 0.1234 & 7.103 & 0.000*** \\
\hline
\end{tabular}
\end{table}
```

---

## 🔧 API Reference

### Core Classes

| Class | Description | Module |
|:------|:------------|:-------|
| `NARDL` | Standard Nonlinear ARDL | `nardl_fourier.core.nardl` |
| `FourierNARDL` | Fourier NARDL with breaks | `nardl_fourier.core.fnardl` |
| `BootstrapNARDL` | Bootstrap NARDL | `nardl_fourier.core.fbnardl` |
| `PSSBoundsTest` | PSS bounds test | `nardl_fourier.bounds_test.pss_bounds` |
| `BootstrapCointegrationTest` | Bootstrap bounds | `nardl_fourier.core.fbnardl` |
| `ResultsTable` | Table formatting | `nardl_fourier.output.tables` |
| `NARDLPlots` | Visualization | `nardl_fourier.output.plots` |

### Common Methods

| Method | Returns | Description |
|:-------|:--------|:------------|
| `.summary()` | str | Full model summary |
| `.long_run_table()` | DataFrame | Long-run coefficients |
| `.short_run_table()` | DataFrame | Short-run coefficients |
| `.bounds_test()` | dict | PSS bounds test results |
| `.wald_lr_asymmetry()` | dict | Long-run asymmetry Wald test |
| `.wald_sr_asymmetry()` | dict | Short-run asymmetry Wald test |
| `.plot_multipliers()` | Figure | Dynamic multiplier plot |
| `.plot_cusum()` | Figure | CUSUM stability plot |
| `.diagnostics()` | dict | All diagnostic tests |

### Key Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `.params` | Series | Estimated coefficients |
| `.bse` | Series | Standard errors |
| `.tvalues` | Series | t-statistics |
| `.pvalues` | Series | p-values |
| `.resid` | Series | Residuals |
| `.fittedvalues` | Series | Fitted values |
| `.rsquared` | float | R-squared |
| `.adj_rsquared` | float | Adjusted R-squared |
| `.aic` | float | Akaike Information Criterion |
| `.bic` | float | Bayesian Information Criterion |
| `.nobs` | int | Number of observations |
| `.ect` | float | Error correction term coefficient |
| `.half_life` | float | Half-life of adjustment |

---

## 📚 References

### Core Methodology

1. **Shin, Y., Yu, B., & Greenwood-Nimmo, M.** (2014). Modelling asymmetric cointegration and dynamic multipliers in a nonlinear ARDL framework. In *Festschrift in Honor of Peter Schmidt* (pp. 281-314). Springer.
   > *Introduces the NARDL framework with partial sum decomposition*

2. **Pesaran, M. H., Shin, Y., & Smith, R. J.** (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289-326.
   > *Original ARDL bounds testing approach*

### Extensions

3. **Zaghdoudi, T., Tissaoui, K., Maaloul, M. H., Bahou, Y., & Kammoun, N.** (2023). Asymmetric connectedness between oil price, coal and renewable energy consumption in China: Evidence from Fourier NARDL approach. *Energy*, 285, 129416.
   > *Fourier NARDL for smooth structural breaks*

4. **Bertelli, S., Vacca, G., & Zoia, M.** (2022). Bootstrap cointegration tests in ARDL models. *Economic Modelling*, 116, 105987.
   > *Bootstrap bounds test methodology*

5. **McNown, R., Sam, C. Y., & Goh, S. K.** (2018). Bootstrapping the autoregressive distributed lag test for cointegration. *Applied Economics*, 50(13), 1509-1521.
   > *Bootstrap implementation details*

### Critical Values

6. **Narayan, P. K.** (2005). The saving and investment nexus for China: Evidence from cointegration tests. *Applied Economics*, 37(17), 1979-1990.
   > *Small sample critical values*

---

## 👨‍🔬 Author

<p align="center">
  <b>Dr. Merwan Roudane</b><br>
  <i>Econometrician & Data Scientist</i>
</p>

<p align="center">
  📧 <a href="mailto:merwanroudane920@gmail.com">merwanroudane920@gmail.com</a><br>
  🌐 <a href="https://github.com/merwanroudane/fnardl">GitHub Repository</a><br>
  📖 <a href="https://merwanroudane.github.io/fnardl/">Documentation</a>
</p>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Dr. Merwan Roudane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a **Pull Request**.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📊 Citation

If you use **NARDL-Fourier** in your research, please cite:

```bibtex
@software{roudane2024nardlfourier,
  author = {Roudane, Merwan},
  title = {NARDL-Fourier: Nonlinear ARDL with Fourier Approximation and Bootstrap Tests},
  version = {1.0.1},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/merwanroudane/fnardl}
}
```

---

<p align="center">
  <b>⭐ If you find this library useful, please give it a star on GitHub! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by Dr. Merwan Roudane
</p>
