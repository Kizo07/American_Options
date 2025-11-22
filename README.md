# comp_methods

`comp_methods` is a Python library dedicated to computational methods for pricing financial derivatives, specifically focusing on American Options. It implements various numerical techniques such as Binomial Trees and Finite Difference Methods.

## Features

- **Binomial Tree Model**: Cox-Ross-Rubinstein (CRR) implementation for American and European options.
- **Finite Difference Methods**:
  - Explicit Method
  - Implicit Method
  - Crank-Nicolson Method
- **Greeks Calculation**: Tools to compute Delta, Gamma, Theta, etc.
- **Visualization**: Utilities for plotting convergence and option price surfaces.

## Installation

To use this library, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/American_Options.git
cd American_Options
pip install -r requirements.txt
```

## Usage

Here is a simple example of how to price an American Put option using the Binomial Tree method:

```python
from comp_methods import BinomialTree

# Parameters: S0, K, T, r, sigma, N
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1.0   # Time to maturity (1 year)
r = 0.05  # Risk-free rate
sigma = 0.2 # Volatility
N = 100   # Number of time steps

model = BinomialTree(S0, K, T, r, sigma, N, option_type='put', style='american')
price = model.price()

print(f"American Put Option Price: {price:.4f}")
```

## Structure

- `comp_methods/`: Main package containing the implementations.
- `tests/`: Unit tests for the algorithms.
- `notebooks/`: Jupyter notebooks with examples and analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
