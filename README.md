# American Options

`American_Options` is an educational quantitative-finance library for pricing American options and related structured products with numerical methods. The installable package is `comp_methods`.

The project started from course/research notebooks and now exposes the reusable pieces as a Python package. It is intended for learning, experimentation, and method comparison rather than trading or production risk systems.

## Highlights

- American and European option pricing with binomial and trinomial trees.
- Black-Scholes closed-form pricing and payoff helpers.
- Continuous-dividend yield support for the main vanilla pricers.
- Flat and interpolated zero-rate curves for discounting and curve-aware pricing inputs.
- Closed-form and bump-based Greeks.
- Finite-difference solvers on stock-price and log-price grids.
- Longstaff-Schwartz Monte Carlo for American puts.
- Monte Carlo utilities for GBM, antithetic variates, control variates, moment matching, Sobol normals, Euler/Milstein discretization, and two-factor stochastic-volatility examples.
- Monte Carlo examples for Asian, barrier, lookback, and digital options.
- Implied volatility and simple calibration helpers.
- Random-number generation exercises including LCG, Bernoulli/binomial/exponential draws, Box-Muller, and Polar-Marsaglia normals.
- Interest-rate examples for CIR and G2++ zero-coupon bonds and bond options.
- Structured-product examples including mortgage default simulation, MBS pricing, OAS, IO/PO pricing, and Heston down-and-out puts.
- Notebook source material and exported scripts under `notebooks/`.
- Regression tests for key numerical fixes and smoke tests for core pricing functions.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Kizo07/American_Options.git
cd American_Options
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

For notebooks or optional examples:

```bash
python -m pip install -e ".[notebooks,examples]"
```

If you use Anaconda, run validation through the Anaconda environment:

```bash
conda run -n base python -m pytest
conda run -n base python -m compileall -q comp_methods tests examples notebooks/exports
```

## Quick Start

Price an American put with the binomial tree:

```python
from comp_methods import binomial_tree

price = binomial_tree(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.20,
    T=1.0,
    n=100,
    option_type="put",
)

print(f"American put: {price:.4f}")
```

Compare a no-dividend American call tree price to Black-Scholes:

```python
from comp_methods import black_scholes, trinomial_tree

tree_price = trinomial_tree(100, 100, 0.05, 0.20, 1.0, 200, option_type="call")
bs_price = black_scholes(100, 100, 0.05, 0.20, 1.0, option_type="call")

print(tree_price, bs_price)
```

Use a dividend yield and curve object:

```python
from comp_methods import FlatCurve, black_scholes

curve = FlatCurve(0.05)
price = black_scholes(100, 100, curve, 0.20, 1.0, option_type="call", q=0.02)

print(price)
```

Run Longstaff-Schwartz Monte Carlo with a seeded NumPy generator:

```python
import numpy as np
from comp_methods import lsmc

rng = np.random.default_rng(42)
price = lsmc(100, 100, 0.05, 0.20, 1.0, N=20_000, num_steps=50, k=3, rng=rng)

print(f"LSMC American put: {price:.4f}")
```

Compute Greeks, implied volatility, and an exotic price:

```python
from comp_methods import asian_option_mc, black_scholes, black_scholes_greeks, implied_vol

greeks = black_scholes_greeks(100, 100, 0.05, 0.20, 1.0, option_type="call")
price = black_scholes(100, 100, 0.05, 0.25, 1.0, option_type="put")
iv = implied_vol(price, 100, 100, 0.05, 1.0, option_type="put")
asian_price, asian_err = asian_option_mc(100, 100, 0.05, 0.20, 1.0, n_paths=20_000)

print(greeks["delta"], iv, asian_price, asian_err)
```

## Feature Map

### Analytic Models

Module: `comp_methods.analytic_models`

- `black_scholes(S0, K, r, sigma, T, option_type="call", q=0.0)`
- `black_scholes_approx(S0, K, r, sigma, T, q=0.0)`
- `call_payoff(S_T, K)`
- `put_payoff(S_T, K)`
- `norm_cdf_approx(x)`

Use these functions for closed-form European option references and payoff construction. They are also useful for validating tree, finite-difference, and Monte Carlo outputs. `r` may be a scalar or a supported curve object, and `q` is a continuous dividend yield.

### Curves

Module: `comp_methods.curves`

- `FlatCurve(rate)`
- `ZeroCurve(times, rates, compounding="continuous")`
- `discount_factor(rate_or_curve, t)`
- `as_flat_rate(rate_or_curve, t)`

Curve objects expose `discount(t)`, `zero_rate(t)`, and `forward_rate(t1, t2)`. The main vanilla pricers accept either scalar rates or curve objects for `r`.

Example:

```python
from comp_methods import FlatCurve, ZeroCurve, black_scholes

flat = FlatCurve(0.05)
curve = ZeroCurve([0.0, 1.0, 2.0], [0.03, 0.04, 0.05])

print(flat.discount(1.0))
print(curve.forward_rate(1.0, 2.0))
print(black_scholes(100, 100, flat, 0.20, 1.0))
```

### Tree Methods

Module: `comp_methods.trees`

- `binomial_tree(...)`: binomial American/European-style early-exercise tree for calls and puts.
- `crr_american_put(...)`: Cox-Ross-Rubinstein American put with a simple delta estimate.
- `trinomial_tree(...)`: recombining log-price trinomial tree for American calls and puts.

The tree methods are the most direct tools in the library for American exercise. The trinomial implementation uses a recombining log-price grid and validates method/option inputs. Tree methods support continuous dividend yield through `q`.

### Finite Difference Methods

Module: `comp_methods.finite_difference`

- `fd_bs(K, sigma, T, r, dt, dS, S0_range, method="explicit", option_type="put", q=0.0)`
- `fd_log(K, sigma, T, r, dt, dx, S0_range, method="explicit", option_type="put", q=0.0)`

Supported methods:

- `explicit`
- `implicit`
- `crank_nicolson`

The solvers support American-style early exercise through projection onto intrinsic value. Explicit methods validate stability by default and raise `ValueError` on unstable grids unless `allow_unstable=True` is passed.

Example:

```python
from comp_methods import fd_bs

price = fd_bs(
    K=100,
    sigma=0.20,
    T=1.0,
    r=0.05,
    dt=0.01,
    dS=1.0,
    S0_range=[100],
    method="crank_nicolson",
    option_type="put",
)

print(price[0])
```

### Longstaff-Schwartz Monte Carlo

Module: `comp_methods.lsmc`

- `lsmc(S0, K, r, sigma, T, N, num_steps, k, poly_type="laguerre", rng=None)`
- Basis helpers: `laguerre`, `hermite`, `monomial`

Supported basis types:

- `laguerre`
- `hermite`
- `monomial`

The LSMC implementation supports seeded NumPy generators through `rng`, handles odd path counts, supports continuous dividend yield through `q`, and applies an immediate-exercise lower bound for deep in-the-money puts.

### Greeks

Module: `comp_methods.greeks`

- `black_scholes_greeks(...)`
- `bump_greeks(price_func, params, bumps=None)`
- `tree_greeks(...)`
- `fd_greeks(...)`

The Black-Scholes Greek function returns a dictionary with `price`, `delta`, `gamma`, `vega`, `theta`, and `rho`. Bump-based helpers are useful for educational comparisons across pricing engines.

Example:

```python
from comp_methods import black_scholes_greeks

greeks = black_scholes_greeks(100, 100, 0.05, 0.20, 1.0, option_type="call", q=0.01)
print(greeks)
```

### Monte Carlo and Stochastic Processes

Modules: `comp_methods.monte_carlo`, `comp_methods.stochastic_processes`

Path and vanilla-option utilities:

- `simulate_path_S(...)`
- `mc_call_option(...)`
- `mc_call_option_antithetic(...)`
- `sobol_normals(...)`
- `euler_discretization(...)`
- `milstein_discretization(...)`
- `brownian_motion(...)`
- `correlated_brownian_motion(...)`
- `gbm(...)`

Two-factor and Heston-style examples:

- `two_factor_gbm(...)`
- `two_factor_mc_call(...)`
- `heston_down_out_put(...)`

Several stochastic functions accept `rng=np.random.default_rng(seed)` for reproducible experiments. Correlation parameters are validated so invalid `rho` values fail explicitly instead of producing NaNs. Vanilla Monte Carlo calls support `variance_reduction="antithetic"`, `"control_variate"`, or `"moment_matching"`.

Example:

```python
import numpy as np
from comp_methods import simulate_path_S

rng = np.random.default_rng(7)
path = simulate_path_S(100, 0.05, 0.20, 1.0, num_steps=252, rng=rng)

print(path[0], path[-1])
```

Variance-reduction example:

```python
import numpy as np
from comp_methods import mc_call_option, sobol_normals

rng = np.random.default_rng(123)
price, err = mc_call_option(
    100,
    100,
    0.05,
    0.20,
    1.0,
    50_000,
    rng=rng,
    variance_reduction="control_variate",
)
normals = sobol_normals(1024, 12, seed=7)

print(price, err, normals.shape)
```

### Exotics

Module: `comp_methods.exotics`

- `asian_option_mc(...)`
- `barrier_option_mc(...)`
- `digital_option_bs(...)`
- `lookback_option_mc(...)`
- `vanilla_reference_price(...)`

These functions provide compact MC/closed-form examples for common exotic payoff shapes. They are intended as educational references and benchmarks for future method extensions.

Example:

```python
from comp_methods import asian_option_mc, barrier_option_mc, digital_option_bs, lookback_option_mc

asian, asian_err = asian_option_mc(100, 100, 0.05, 0.20, 1.0)
barrier, barrier_err = barrier_option_mc(100, 100, 0.05, 0.20, 1.0, barrier=90)
digital = digital_option_bs(100, 100, 0.05, 0.20, 1.0)
lookback, lookback_err = lookback_option_mc(100, 0.05, 0.20, 1.0)

print(asian, barrier, digital, lookback)
```

### Random Number Generators

Module: `comp_methods.random_numbers`

- `lcg_uniform(...)`
- `bernoulli(...)`
- `binomial(...)`
- `exponential(...)`
- `box_muller(...)`
- `polar_marsaglia(...)`

These are primarily educational implementations for understanding simulation inputs. For serious Monte Carlo experiments, prefer NumPy generators unless you are specifically studying the custom RNG routines.

### Interest-Rate Models

Module: `comp_methods.interest_rates`

CIR examples:

- `cir_zero_bond_price(...)`
- `cir_bond_price_mc(...)`
- `cir_call_option_mc(...)`
- `cir_call_option_pde(...)`

G2++ examples:

- `g2pp_zero_bond_price(...)`
- `g2pp_put_option_mc(...)`
- `g2pp_put_option_explicit(...)`

These functions provide compact examples of affine-rate model pricing, Monte Carlo pricing, and PDE-style bond-option valuation.

### Calibration

Module: `comp_methods.calibration`

- `implied_vol(...)`
- `calibrate_vol_surface(...)`
- `bootstrap_zero_curve(...)`

The calibration helpers are intentionally small: implied volatility uses scalar root finding, the surface helper fits a single flat volatility to synthetic or educational quote sets, and `bootstrap_zero_curve` wraps the curve object used elsewhere.

Example:

```python
from comp_methods import black_scholes, calibrate_vol_surface, implied_vol

target = black_scholes(100, 100, 0.05, 0.30, 1.0)
iv = implied_vol(target, 100, 100, 0.05, 1.0)

quotes = [
    {"S0": 100, "K": 90, "r": 0.05, "T": 1.0, "price": black_scholes(100, 90, 0.05, 0.30, 1.0)},
    {"S0": 100, "K": 100, "r": 0.05, "T": 1.0, "price": black_scholes(100, 100, 0.05, 0.30, 1.0)},
    {"S0": 100, "K": 110, "r": 0.05, "T": 1.0, "price": black_scholes(100, 110, 0.05, 0.30, 1.0)},
]
fit = calibrate_vol_surface(quotes)

print(iv, fit)
```

### Credit, Mortgage, and Structured Products

Modules: `comp_methods.credit_risk`, `comp_methods.mbs`, `comp_methods.monte_carlo`

Credit-risk simulation:

- `SimulationParams`
- `simulate_mortgage_default(...)`
- `calculate_outstanding_loan(...)`
- `calculate_trigger_ratio(...)`
- `evolve_asset_values(...)`

Mortgage-backed security examples:

- `numerix_prepayment_rate(...)`
- `mbs_pricing(...)`
- `compute_oas(...)`
- `io_po_pricing(...)`
- `MBSConfig`
- `PrepaymentConfig`

Barrier/stochastic-volatility example:

- `heston_down_out_put(...)`
- `HestonBarrierConfig`

These are research/example implementations with embedded modeling assumptions. They are useful as starting points for experiments, not as calibrated production valuation models.

Config example:

```python
from comp_methods import HestonBarrierConfig, MBSConfig, heston_down_out_put, mbs_pricing

mbs_price = mbs_pricing(config=MBSConfig(notional=250_000, N_sims=5_000))
heston_prices = heston_down_out_put(config=HestonBarrierConfig(N_sims=20_000))

print(mbs_price, heston_prices)
```

### Greeks and Utility Functions

Module: `comp_methods.utils`

- `delta_vs_stock(...)`
- `greeks_vs_time(...)`
- `vega_vs_stock(...)`
- `display_results(...)`

The Greek helpers are simple finite-difference/binomial utilities intended for exploration and plotting workflows.

## Project Structure

```text
comp_methods/          Installable package
tests/                 Pytest smoke and numerical regression tests
notebooks/             Jupyter notebooks used as source/research material
notebooks/exports/     Python exports of notebook work
examples/              Optional examples and demos
pyproject.toml         Package metadata and optional dependencies
```

## Testing

Run tests with Anaconda:

```bash
conda run -n base python -m pytest
conda run -n base python -m compileall -q comp_methods tests examples notebooks/exports
```

Expected current result:

```text
22 passed
```

The numerical test suite covers:

- Trinomial tree convergence and comparison against binomial/Black-Scholes references.
- LSMC odd path counts, deep in-the-money exercise, and seeded Monte Carlo behavior.
- Stochastic path initialization and seeded reproducibility.
- Correlation validation and truncation-method behavior.
- Finite-difference method validation and explicit stability guardrails.
- Dividend-yield, curve, Greek, calibration, variance-reduction, exotic-option, and config-dataclass features.

## Notes and Limitations

- This repository is educational/research code, not production valuation infrastructure.
- Most models assume constant volatilities unless a specific model function says otherwise.
- Continuous dividend yields, simple curves, implied volatility, and flat-vol calibration are implemented; discrete dividends and full market data integrations are not.
- Some structured-product routines intentionally encode assignment-style assumptions and should be reviewed before reuse.
- Monte Carlo outputs depend on path count, seed, basis choice, and discretization settings.

## License

This project is licensed under the MIT License.
