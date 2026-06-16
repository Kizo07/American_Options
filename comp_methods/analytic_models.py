import numpy as np
from scipy.stats import norm
from .curves import as_flat_rate

def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0)

def put_payoff(S_T, K):
    return np.maximum(K - S_T, 0)

def black_scholes(S0, K, r, sigma, T, option_type='call', q=0.0):
    r = as_flat_rate(r, T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def norm_cdf_approx(x):
    a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    L = np.abs(x)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-L**2 / 2.0) * (a1 * K + a2 * K**2 + a3 * K**3 + a4 * K**4 + a5 * K**5)
    return np.where(x >= 0, w, 1.0 - w)

def black_scholes_approx(S0, K, r, sigma, T, q=0.0):
    r = as_flat_rate(r, T)
    S0 = np.asarray(S0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm_cdf_approx(d1) - K * np.exp(-r * T) * norm_cdf_approx(d2)
