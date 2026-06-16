from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc
from .stochastic_processes import brownian_motion, gbm, two_factor_gbm
from .analytic_models import call_payoff
from ._validation import validate_positive, validate_rho
from .curves import as_flat_rate


@dataclass(frozen=True)
class HestonBarrierConfig:
    K: float = 100
    T: float = 1
    gamma: float = 0.25
    v0: float = 0.1
    alpha: float = 0.45
    beta: float = -5.105
    S0: float = 100
    r: float = 0.05
    rho: float = -0.75
    N_sims: int = 100000

def _rng_normal(rng, loc, scale, size):
    if rng is None:
        return np.random.normal(loc, scale, size)
    return rng.normal(loc, scale, size)

def simulate_path_S(S0, r, sigma, T, num_steps, rng=None, q=0.0):
    r = as_flat_rate(r, T)
    for name, value in {"S0": S0, "sigma": sigma, "T": T, "num_steps": num_steps}.items():
        validate_positive(name, value)
    dt = T / num_steps
    dW = brownian_motion(dt, num_steps, 1, rng=rng).flatten()
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * dW
    path = np.empty(num_steps + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(increments))
    return path

def euler_discretization(S0, r, sigma, T, steps, N, rng=None, q=0.0):
    r = as_flat_rate(r, T)
    dt = T / steps
    Wt = brownian_motion(dt, steps, N, rng=rng)
    increments = 1 + (r - q) * dt + sigma * Wt
    return S0 * np.prod(increments, axis=1)

def milstein_discretization(S0, r, sigma, T, steps, N, rng=None, q=0.0):
    r = as_flat_rate(r, T)
    dt = T / steps
    Wt = brownian_motion(dt, steps, N, rng=rng)
    increments = 1 + (r - q) * dt + sigma * Wt + 0.5 * sigma**2 * (Wt**2 - dt)
    return S0 * np.prod(increments, axis=1)

def mc_call_option(S0, K, r, sigma, T, N, steps=None, discretization_func=None, rng=None, q=0.0, variance_reduction=None):
    r = as_flat_rate(r, T)
    if variance_reduction == "antithetic":
        return mc_call_option_antithetic(S0, K, r, sigma, T, N, rng=rng, q=q)
    if steps and discretization_func:
        S_T = discretization_func(S0, r, sigma, T, steps, N, rng=rng, q=q)
    else:
        Wt = _rng_normal(rng, 0.0, np.sqrt(T), N)
        S_T = gbm(S0, r - q, sigma, T, Wt)
        
    payoffs = call_payoff(S_T, K)
    if variance_reduction == "control_variate":
        expected_ST = S0 * np.exp((r - q) * T)
        cov = np.cov(payoffs, S_T, ddof=1)[0, 1]
        var = np.var(S_T, ddof=1)
        beta = cov / var if var > 0 else 0.0
        adjusted = payoffs - beta * (S_T - expected_ST)
        price = np.exp(-r * T) * np.mean(adjusted)
        std_err = np.exp(-r * T) * np.std(adjusted, ddof=1) / np.sqrt(N)
        return price, std_err
    if variance_reduction == "moment_matching":
        z = (Wt / np.sqrt(T) - np.mean(Wt / np.sqrt(T))) / np.std(Wt / np.sqrt(T), ddof=1)
        S_T = gbm(S0, r - q, sigma, T, np.sqrt(T) * z)
        payoffs = call_payoff(S_T, K)
    elif variance_reduction not in (None, "control_variate"):
        raise ValueError("variance_reduction must be None, 'antithetic', 'control_variate', or 'moment_matching'")
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(N)
    return price, std_err

def sobol_normals(n_paths, n_steps, scramble=True, seed=None):
    sampler = qmc.Sobol(d=n_steps, scramble=scramble, seed=seed)
    u = sampler.random(n_paths)
    eps = np.finfo(float).eps
    return norm.ppf(np.clip(u, eps, 1 - eps))

def mc_call_option_antithetic(S0, K, r, sigma, T, N, rng=None, q=0.0):
    r = as_flat_rate(r, T)
    Wt = _rng_normal(rng, 0.0, np.sqrt(T), N)
    S_T = gbm(S0, r - q, sigma, T, Wt)
    S_T_anti = gbm(S0, r - q, sigma, T, -Wt)
    
    payoffs = (call_payoff(S_T, K) + call_payoff(S_T_anti, K)) / 2
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N)
    return price, std_err

def two_factor_mc_call(S0, V0, r, alpha, beta, sigma, rho, K, T, N, n, method="partial_truncation", rng=None):
    validate_rho(rho)
    S_T = two_factor_gbm(S0, V0, r, alpha, beta, sigma, rho, T, N, n, method, rng=rng)
    payoffs = call_payoff(S_T, K)
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n)
    return price, std_err

def heston_down_out_put(K=100, T=1, gamma=0.25, v0=0.1, alpha=0.45, beta=-5.105, S0=100, r=0.05, rho=-0.75, N_sims=100000, config=None):
    if config is not None:
        K = config.K
        T = config.T
        gamma = config.gamma
        v0 = config.v0
        alpha = config.alpha
        beta = config.beta
        S0 = config.S0
        r = config.r
        rho = config.rho
        N_sims = config.N_sims
    dt = 1/252
    N_steps = int(T / dt)
    
    t_grid = np.arange(N_steps) * dt
    barrier_1 = np.full(N_steps, 94.0)
    barrier_2 = 6 * t_grid / T + 91
    barrier_3 = -6 * t_grid / T + 97
    
    S_paths = np.full(N_sims, float(S0))
    v_paths = np.full(N_sims, float(v0))
    
    hit_1 = np.zeros(N_sims, dtype=bool)
    hit_2 = np.zeros(N_sims, dtype=bool)
    hit_3 = np.zeros(N_sims, dtype=bool)
    
    sqrt_dt = np.sqrt(dt)
    sqrt_rho = np.sqrt(1 - rho**2)
    
    for i in range(N_steps):
        dW1 = np.random.normal(0, sqrt_dt, N_sims)
        dW2 = rho * dW1 + sqrt_rho * np.random.normal(0, sqrt_dt, N_sims)
        
        v_pos = np.maximum(v_paths, 0)
        sqrt_v = np.sqrt(v_pos)
        
        S_paths += r * S_paths * dt + S_paths * sqrt_v * dW1
        v_paths += (alpha + beta * v_pos) * dt + gamma * sqrt_v * dW2
        
        hit_1 |= (S_paths <= barrier_1[i])
        hit_2 |= (S_paths <= barrier_2[i])
        hit_3 |= (S_paths <= barrier_3[i])
    
    payoffs = np.maximum(K - S_paths, 0) * np.exp(-r * T)
    
    return (np.mean(payoffs * (~hit_1)), 
            np.mean(payoffs * (~hit_2)), 
            np.mean(payoffs * (~hit_3)))
