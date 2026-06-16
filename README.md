# American Options

`American_Options` is an educational quantitative-finance library for pricing American options and related structured products with numerical methods. The installable package is `comp_methods`.

The project started from course/research notebooks and now exposes the reusable pieces as a Python package. It is intended for learning, experimentation, and method comparison rather than trading or production risk systems.

## Highlights

- American and European option pricing with binomial and trinomial trees.
- Black-Scholes closed-form pricing and payoff helpers.
- Finite-difference solvers on stock-price and log-price grids.
- Longstaff-Schwartz Monte Carlo for American puts.
- Monte Carlo utilities for GBM, antithetic variates, Euler/Milstein discretization, and two-factor stochastic-volatility examples.
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

Run Longstaff-Schwartz Monte Carlo with a seeded NumPy generator:

```python
import numpy as np
from comp_methods import lsmc

rng = np.random.default_rng(42)
price = lsmc(100, 100, 0.05, 0.20, 1.0, N=20_000, num_steps=50, k=3, rng=rng)

print(f"LSMC American put: {price:.4f}")
```

## Feature Map

### Analytic Models

Module: `comp_methods.analytic_models`

- `black_scholes(S0, K, r, sigma, T, option_type="call")`
- `black_scholes_approx(S0, K, r, sigma, T)`
- `call_payoff(S_T, K)`
- `put_payoff(S_T, K)`
- `norm_cdf_approx(x)`

Use these functions for closed-form European option references and payoff construction. They are also useful for validating tree, finite-difference, and Monte Carlo outputs.

### Tree Methods

Module: `comp_methods.trees`

- `binomial_tree(...)`: binomial American/European-style early-exercise tree for calls and puts.
- `crr_american_put(...)`: Cox-Ross-Rubinstein American put with a simple delta estimate.
- `trinomial_tree(...)`: recombining log-price trinomial tree for American calls and puts.

The tree methods are the most direct tools in the library for American exercise. The trinomial implementation uses a recombining log-price grid and validates method/option inputs.

### Finite Difference Methods

Module: `comp_methods.finite_difference`

- `fd_bs(K, sigma, T, r, dt, dS, S0_range, method="explicit", option_type="put")`
- `fd_log(K, sigma, T, r, dt, dx, S0_range, method="explicit", option_type="put")`

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

The LSMC implementation supports seeded NumPy generators through `rng`, handles odd path counts, and applies an immediate-exercise lower bound for deep in-the-money puts.

### Monte Carlo and Stochastic Processes

Modules: `comp_methods.monte_carlo`, `comp_methods.stochastic_processes`

Path and vanilla-option utilities:

- `simulate_path_S(...)`
- `mc_call_option(...)`
- `mc_call_option_antithetic(...)`
- `euler_discretization(...)`
- `milstein_discretization(...)`
- `brownian_motion(...)`
- `correlated_brownian_motion(...)`
- `gbm(...)`

Two-factor and Heston-style examples:

- `two_factor_gbm(...)`
- `two_factor_mc_call(...)`
- `heston_down_out_put(...)`

Several stochastic functions accept `rng=np.random.default_rng(seed)` for reproducible experiments. Correlation parameters are validated so invalid `rho` values fail explicitly instead of producing NaNs.

Example:

```python
import numpy as np
from comp_methods import simulate_path_S

rng = np.random.default_rng(7)
path = simulate_path_S(100, 0.05, 0.20, 1.0, num_steps=252, rng=rng)

print(path[0], path[-1])
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

Barrier/stochastic-volatility example:

- `heston_down_out_put(...)`

These are research/example implementations with embedded modeling assumptions. They are useful as starting points for experiments, not as calibrated production valuation models.

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
15 passed
```

The numerical test suite covers:

- Trinomial tree convergence and comparison against binomial/Black-Scholes references.
- LSMC odd path counts, deep in-the-money exercise, and seeded Monte Carlo behavior.
- Stochastic path initialization and seeded reproducibility.
- Correlation validation and truncation-method behavior.
- Finite-difference method validation and explicit stability guardrails.

## Notes and Limitations

- This repository is educational/research code, not production valuation infrastructure.
- Most models assume constant rates and volatilities unless a specific model function says otherwise.
- Dividend yields, discrete dividends, calibration routines, and market data integrations are not implemented.
- Some structured-product routines intentionally encode assignment-style assumptions and should be reviewed before reuse.
- Monte Carlo outputs depend on path count, seed, basis choice, and discretization settings.

## License

This project is licensed under the MIT License.
