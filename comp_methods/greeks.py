import numpy as np
from scipy.stats import norm

from .analytic_models import black_scholes


def black_scholes_greeks(S0, K, r, sigma, T, q=0.0, option_type="call"):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    pdf = norm.pdf(d1)

    if option_type == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            -S0 * df_q * pdf * sigma / (2 * np.sqrt(T))
            - r * K * df_r * norm.cdf(d2)
            + q * S0 * df_q * norm.cdf(d1)
        )
        rho = K * T * df_r * norm.cdf(d2)
    elif option_type == "put":
        delta = df_q * (norm.cdf(d1) - 1)
        theta = (
            -S0 * df_q * pdf * sigma / (2 * np.sqrt(T))
            + r * K * df_r * norm.cdf(-d2)
            - q * S0 * df_q * norm.cdf(-d1)
        )
        rho = -K * T * df_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "price": black_scholes(S0, K, r, sigma, T, option_type=option_type, q=q),
        "delta": delta,
        "gamma": df_q * pdf / (S0 * sigma * np.sqrt(T)),
        "vega": S0 * df_q * pdf * np.sqrt(T),
        "theta": theta,
        "rho": rho,
    }


def bump_greeks(price_func, params, bumps=None):
    bumps = bumps or {"S0": 1e-2, "sigma": 1e-4, "T": 1 / 365, "r": 1e-4}
    base = dict(params)
    price = price_func(**base)

    up = dict(base)
    down = dict(base)
    up["S0"] += bumps["S0"]
    down["S0"] -= bumps["S0"]
    p_up = price_func(**up)
    p_down = price_func(**down)
    delta = (p_up - p_down) / (2 * bumps["S0"])
    gamma = (p_up - 2 * price + p_down) / bumps["S0"] ** 2

    up = dict(base)
    down = dict(base)
    up["sigma"] += bumps["sigma"]
    down["sigma"] -= bumps["sigma"]
    vega = (price_func(**up) - price_func(**down)) / (2 * bumps["sigma"])

    up = dict(base)
    down = dict(base)
    up["r"] += bumps["r"]
    down["r"] -= bumps["r"]
    rho = (price_func(**up) - price_func(**down)) / (2 * bumps["r"])

    down = dict(base)
    down["T"] = max(base["T"] - bumps["T"], 1e-8)
    theta = (price_func(**down) - price) / bumps["T"]

    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def tree_greeks(S0, K, r, sigma, T, n, q=0.0, option_type="put", bump=1e-2, pricer=None):
    from .trees import binomial_tree

    pricer = pricer or binomial_tree
    params = {"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T, "n": n, "q": q, "option_type": option_type}
    return bump_greeks(lambda **kwargs: pricer(**kwargs), params, bumps={"S0": bump, "sigma": 1e-4, "T": 1 / 365, "r": 1e-4})


def fd_greeks(S0, K, r, sigma, T, dt, dS, q=0.0, option_type="put", method="crank_nicolson"):
    from .finite_difference import fd_bs

    def price_func(S0, K, r, sigma, T, q, option_type):
        return float(fd_bs(K, sigma, T, r, dt, dS, [S0], method=method, option_type=option_type, q=q)[0])

    return bump_greeks(price_func, {"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T, "q": q, "option_type": option_type})
