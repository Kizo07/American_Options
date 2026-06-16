import numpy as np
from scipy.stats import norm

from .analytic_models import black_scholes


def _rng_normal(rng, size):
    if rng is None:
        return np.random.normal(size=size)
    return rng.normal(size=size)


def _gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0, rng=None):
    dt = T / n_steps
    z = _rng_normal(rng, (n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
    return paths


def asian_option_mc(S0, K, r, sigma, T, n_paths=50000, n_steps=252, q=0.0, option_type="call", average_type="arithmetic", rng=None):
    paths = _gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, rng=rng)
    if average_type == "arithmetic":
        avg = paths[:, 1:].mean(axis=1)
    elif average_type == "geometric":
        avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric'")
    if option_type == "call":
        payoff = np.maximum(avg - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - avg, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    disc_payoff = np.exp(-r * T) * payoff
    return float(np.mean(disc_payoff)), float(np.std(disc_payoff, ddof=1) / np.sqrt(n_paths))


def barrier_option_mc(S0, K, r, sigma, T, barrier, barrier_type="down_and_out", n_paths=50000, n_steps=252, q=0.0, option_type="call", rng=None):
    paths = _gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, rng=rng)
    if barrier_type == "down_and_out":
        alive = paths.min(axis=1) > barrier
    elif barrier_type == "up_and_out":
        alive = paths.max(axis=1) < barrier
    else:
        raise ValueError("barrier_type must be 'down_and_out' or 'up_and_out'")
    terminal = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(terminal - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - terminal, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    disc_payoff = np.exp(-r * T) * payoff * alive
    return float(np.mean(disc_payoff)), float(np.std(disc_payoff, ddof=1) / np.sqrt(n_paths))


def digital_option_bs(S0, K, r, sigma, T, q=0.0, option_type="call", cash=1.0):
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return cash * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return cash * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError("option_type must be 'call' or 'put'")


def lookback_option_mc(S0, r, sigma, T, n_paths=50000, n_steps=252, q=0.0, option_type="call", rng=None):
    paths = _gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=q, rng=rng)
    terminal = paths[:, -1]
    if option_type == "call":
        payoff = terminal - paths.min(axis=1)
    elif option_type == "put":
        payoff = paths.max(axis=1) - terminal
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    disc_payoff = np.exp(-r * T) * payoff
    return float(np.mean(disc_payoff)), float(np.std(disc_payoff, ddof=1) / np.sqrt(n_paths))


def vanilla_reference_price(S0, K, r, sigma, T, q=0.0, option_type="call"):
    return black_scholes(S0, K, r, sigma, T, option_type=option_type, q=q)
