import numpy as np
from scipy.optimize import brentq, least_squares

from .analytic_models import black_scholes
from .curves import ZeroCurve


def implied_vol(price, S0, K, r, T, q=0.0, option_type="call", bracket=(1e-6, 5.0)):
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if option_type == "put":
        intrinsic = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    upper = S0 * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)
    if price < intrinsic - 1e-12 or price > upper + 1e-12:
        raise ValueError("price is outside no-arbitrage bounds")

    def objective(sigma):
        return black_scholes(S0, K, r, sigma, T, option_type=option_type, q=q) - price

    return brentq(objective, *bracket)


def calibrate_vol_surface(quotes, initial_sigma=0.2):
    def residual(x):
        sigma = float(x[0])
        return [
            black_scholes(
                quote["S0"],
                quote["K"],
                quote["r"],
                sigma,
                quote["T"],
                option_type=quote.get("option_type", "call"),
                q=quote.get("q", 0.0),
            ) - quote["price"]
            for quote in quotes
        ]

    result = least_squares(residual, x0=np.array([initial_sigma]), bounds=(1e-6, 5.0))
    return {"sigma": float(result.x[0]), "cost": float(result.cost), "success": bool(result.success)}


def bootstrap_zero_curve(times, zero_rates, compounding="continuous"):
    return ZeroCurve(times, zero_rates, compounding=compounding)
