# American Options

`comp_methods` is a Python library for numerical derivative-pricing methods, with a focus on American options and selected structured-product examples.

## Features

- Binomial-tree pricing for American and European calls/puts.
- Finite-difference solvers for Black-Scholes-style grids.
- Monte Carlo utilities for vanilla options and selected stochastic-process examples.
- Longstaff-Schwartz Monte Carlo for American puts.
- Educational notebooks and exported scripts under `notebooks/`.

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

## Usage

Here is a simple example of how to price an American Put option using the Binomial Tree method:

```python
from comp_methods import binomial_tree

S0 = 100  # Initial stock price
K = 100   # Strike price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 1.0   # Time to maturity, in years
n = 100   # Number of time steps

price = binomial_tree(S0, K, r, sigma, T, n, option_type="put")

print(f"American Put Option Price: {price:.4f}")
```

## Structure

- `comp_methods/`: Main package containing the implementations.
- `tests/`: Automated smoke and regression tests.
- `notebooks/`: Jupyter notebooks and exported notebook source scripts.
- `examples/`: Optional demos that may require extra dependencies.

## Testing

```bash
python -m pytest
python -m compileall -q comp_methods tests examples notebooks/exports
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
