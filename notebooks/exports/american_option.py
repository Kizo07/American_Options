import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from scipy import optimize
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags


from scipy.optimize import brentq
from numba import jit
import timeit
from tqdm import tqdm
from joblib import Parallel, delayed


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)

def lcg_uniform_rn(x_0, n):
    a = 7**5
    b = 0
    m = 2**31 - 1
    x = np.empty(n)
    x[0] = (a * x_0 + b) % m
    for i in range(1, n):
        x[i] = (a * x[i - 1] + b) % m
    return x / m

def bernoulli(x_0, p, n):
    x = lcg_uniform_rn(x_0, n)
    return (x < p).astype(int)

def binomial(x_0, n, p, N):
    b = bernoulli(x_0, p, n * N)
    return b.reshape(N, n).sum(axis=1)

def exp_rn(x_0, l, n):
    x = lcg_uniform_rn(x_0, n)
    return -(1/l)*np.log(x)

def bm_normal_rn(x_0, n):
    x = lcg_uniform_rn(x_0, n)
    x = x.reshape(-1, 2)
    z1 = np.sqrt(-2 * np.log(x[:, 0])) * np.cos(2 * np.pi * x[:, 1])
    z2 = np.sqrt(-2 * np.log(x[:, 0])) * np.sin(2 * np.pi * x[:, 1])
    return np.concatenate((z1.reshape(-1,1), z2.reshape(-1,1)), axis=1).reshape(-1)

def pm_normal_rn(x_0, n):
    x = lcg_uniform_rn(x_0, int(n*1.3))
    x = x.reshape(-1, 2)
    s = x[:, 0]**2 + x[:, 1]**2
    s = np.concatenate((x, s.reshape(-1,1)), axis=1)
    y = x*(np.where(s[:,2] < 1, np.sqrt(-2 * np.log(s[:,2]) / s[:,2]), np.nan).reshape(-1, 1))
    y = y[~np.isnan(y).any(axis=1)].reshape(-1)
    return y[:n]

def simulate_Wt(t, N):
  Z = np.random.randn(N)
  Wt = np.sqrt(t) * Z
  return Wt

def black_scholes(S0, K, r, sigma, T):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d1 = d2 + sigma * np.sqrt(T)
    C_0 = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_0

def call_payoff(S_T, K):
    C_T = np.maximum(S_T - K, 0)
    return C_T

def gbm(S0, r, sigma, T, Wt):
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * Wt)
    return S_T

def call_option_price(S0, K, r, sigma, T, N):
    Wt = simulate_Wt(T, N)
    S_T = gbm(S0, r, sigma, T, Wt)
    C_T = call_payoff(S_T, K)
    C_0 = np.exp(-r * T) * np.mean(C_T)
    std_error = np.exp(-r * T) * np.std(C_T) / np.sqrt(N)
    return C_0, std_error

def call_option_price_antithetic(S0, K, r, sigma, T, N):
    Wt = simulate_Wt(T, N)
    S_T = gbm(S0, r, sigma, T, Wt)
    S_T_antithetic = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * Wt)
    C_T = call_payoff(S_T, K)
    C_T_antithetic = call_payoff(S_T_antithetic, K)
    C_0 = np.exp(-r * T) * np.mean((C_T + C_T_antithetic) / 2)
    std_error = np.exp(-r * T) * np.std((C_T + C_T_antithetic) / 2) / np.sqrt(N)
    return C_0, std_error

def display_results(C_0, std_error, method):
    print(f"{method} Estimate: {C_0:.6f} (Standard Error: {std_error:.6f})")

def simulate_path_S(S0, r, sigma, T, num_steps):
    dt = T / num_steps
    ln_S_t = np.cumsum((r - 0.5 * sigma**2) * dt + sigma * simulate_Wt(dt, num_steps+1))
    S_t = S0 * np.exp(ln_S_t)
    return S_t

def simulate_Wt(dt, steps, N):
    return np.sqrt(dt) * np.random.randn(N, steps)

def euler_discretization_vec(S0, r, sigma, T, steps, N):
    dt = T / steps
    Wt = simulate_Wt(dt, steps, N)
    increments = 1 + r * dt + sigma * Wt
    S_T = S0 * np.prod(increments, axis=1)
    return S_T

def milstein_discretization_vec(S0, r, sigma, T, steps, N):
    dt = T / steps
    Wt = simulate_Wt(dt, steps, N)
    increments = 1 + r * dt + sigma * Wt + 0.5 * sigma**2 * (Wt**2 - dt)
    S_T = S0 * np.prod(increments, axis=1)
    return S_T

def monte_carlo_call_option(S0, K, r, sigma, T, N, steps, discretization_func):
    S_T = discretization_func(S0, r, sigma, T, steps, N)
    payoffs = call_payoff(S_T, K)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N)
    return option_price, std_error

def norm_cdf_approx_vec(x):
    a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    L = np.abs(x)
    K_inv = 1.0 + 0.2316419 * L
    K = 1.0 / K_inv
    w = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-L**2 / 2.0) * (a1 * K + a2 * K**2 + a3 * K**3 + a4 * K**4 + a5 * K**5)
    return np.where(x >= 0, w, 1.0 - w)

def black_scholes_approx_vec(S0, K, r, sigma, T):
    S0 = np.asarray(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm_cdf_approx_vec(d1) - K * np.exp(-r * T) * norm_cdf_approx_vec(d2)
    return price

def simulate_Wt_correlated(T, N, n, rho):
    dt = T / N
    dW1 = np.random.normal(0.0, np.sqrt(dt), (n, N))
    dW2 = np.random.normal(0.0, np.sqrt(dt), (n, N))
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2
    return dW1, dW2

def two_factor_gbm_vec(S0, V0, r, alpha, beta, sigma, rho, T, N, n, method="partial_truncation"):
    dt = T / N
    S_paths = np.zeros((n, N + 1))
    V_paths = np.zeros((n, N + 1))
    S_paths[:, 0] = S0
    V_paths[:, 0] = V0

    dW1, dW2 = simulate_Wt_correlated(T, N, n, rho)

    for t in range(N):
        S = S_paths[:, t]
        V = V_paths[:, t]
        sqrt_V = np.sqrt(np.maximum(V, 0))
        S_paths[:, t + 1] = S * np.exp((r - 0.5 * V) * dt + sqrt_V * dW1[:, t])

        if method == "full_truncation":
            V_next = V + alpha * (beta - V) * dt + sigma * sqrt_V * dW2[:, t]
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "partial_truncation":
            V_next = V + alpha * (beta - V) * dt + sigma * sqrt_V * dW2[:, t]
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "reflection":
            V_next = V + alpha * (beta - V) * dt + sigma * sqrt_V * dW2[:, t]
            V_paths[:, t + 1] = np.abs(V_next)
        else:
             raise ValueError("Unsupported method. Choose 'full_truncation', 'partial_truncation', or 'reflection'.")


    return S_paths[:, -1]

def european_call_option_price_two_factor_vec(S0, V0, r, alpha, beta, sigma, rho, K, T, N, n, method="partial_truncation"):
    S_T = two_factor_gbm_vec(S0, V0, r, alpha, beta, sigma, rho, T, N, n, method)
    C_T = call_payoff(S_T, K)
    C_0 = np.exp(-r * T) * np.mean(C_T)
    std_error = np.exp(-r * T) * np.std(C_T) / np.sqrt(n)
    return C_0, std_error

def binomial_tree_a(S0, K, r, sigma, T, n):
    dt = T / n
    c = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
    d = c - np.sqrt(c**2 - 1)
    u = 1 / d
    p = (np.exp(r * dt) - d) / (u - d)
    
    stock = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**(i-j)) * (d**j)
    
    option = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        option[i, n] = max(K - stock[i, n], 0)
    
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            expected = p * option[j, i+1] + (1-p) * option[j+1, i+1]
            expected *= np.exp(-r * dt)
            option[j, i] = max(expected, K - stock[j, i])
    
    return option[0, 0]

def binomial_tree_b(S0, K, r, sigma, T, n):
    dt = T / n
    u = np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - sigma**2/2) * dt - sigma * np.sqrt(dt))
    p = 0.5
    
    stock = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**(i-j)) * (d**j)
    
    option = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        option[i, n] = max(K - stock[i, n], 0)
    
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            expected = p * option[j, i+1] + (1-p) * option[j+1, i+1]
            expected *= np.exp(-r * dt)
            option[j, i] = max(expected, K - stock[j, i])
    
    return option[0, 0]

def crr_binomial_american_put(S0, K, r, sigma, T, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    stock = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**(i-j)) * (d**j)
    
    option = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        option[i, n] = max(K - stock[i, n], 0)
    
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            expected = p * option[j, i+1] + (1-p) * option[j+1, i+1]
            expected *= np.exp(-r * dt)
            option[j, i] = max(expected, K - stock[j, i])
    
    if n > 1:
        delta = (option[0, 1] - option[1, 1]) / (stock[0, 1] - stock[1, 1])
    else:
        delta = (option[0, 0] - K + S0) / S0 if S0 < K else -1
    
    return option[0, 0], delta

def calculate_delta_vs_stock(K, r, sigma, T, n, stock_range):
    deltas = []
    prices = []
    for S0 in stock_range:
        price, delta = crr_binomial_american_put(S0, K, r, sigma, T, n)
        prices.append(price)
        deltas.append(delta)
    return prices, deltas

def calculate_greeks_vs_time(S0, K, r, sigma, time_range, n):
    deltas = []
    thetas = []
    base_price = None
    
    for T in time_range:
        price, delta = crr_binomial_american_put(S0, K, r, sigma, T, n)
        deltas.append(delta)
        
        if T + 1/365 in time_range:
            next_idx = np.where(np.isclose(time_range, T + 1/365))[0][0]
            next_price = crr_binomial_american_put(S0, K, r, sigma, time_range[next_idx], n)[0]
            theta = (next_price - price) / (1/365)
        else:
            if T > 1/365:
                prev_T = max(0, T - 1/365)
                prev_price = crr_binomial_american_put(S0, K, r, sigma, prev_T, n)[0]
                theta = (price - prev_price) / (1/365)
            else:
                theta = 0
        
        thetas.append(theta)
    
    return deltas, thetas

def calculate_vega_vs_stock(K, r, sigma, T, n, stock_range, d_sigma=0.01):
    vegas = []
    for S0 in stock_range:
        price1 = crr_binomial_american_put(S0, K, r, sigma, T, n)[0]
        price2 = crr_binomial_american_put(S0, K, r, sigma + d_sigma, T, n)[0]
        vega = (price2 - price1) / d_sigma
        vegas.append(vega)
    return vegas

def trinomial_tree_a(S0, K, r, sigma, T, n):
    dt = T / n
    d = np.exp(-sigma * np.sqrt(3 * dt))
    u = 1 / d
    
    p_d = (r * dt * (1 - u) + (r * dt)**2 + sigma**2 * dt) / ((u - d) * (1 - d))
    p_u = (r * dt * (1 - d) + (r * dt)**2 + sigma**2 * dt) / ((u - d) * (u - 1))
    p_m = 1 - p_u - p_d
    
    stock = np.zeros((2*n+1, n+1))
    stock[n, 0] = S0
    
    for j in range(1, n+1):
        for i in range(2*n+1):
            up_power = max(0, j - i + n)
            down_power = max(0, i - n)
            
            if up_power + down_power <= j:
                stock[i, j] = S0 * (u ** up_power) * (d ** down_power)
    
    option = np.zeros((2*n+1, n+1))
    
    for i in range(2*n+1):
        if stock[i, n] > 0:
            option[i, n] = max(K - stock[i, n], 0)
    
    for j in range(n-1, -1, -1):
        for i in range(2*n+1):
            if stock[i, j] > 0:
                up_idx = max(0, min(2*n, i-1))
                mid_idx = i
                down_idx = min(2*n, i+1)
                
                expected = p_u * option[up_idx, j+1] + p_m * option[mid_idx, j+1] + p_d * option[down_idx, j+1]
                expected *= np.exp(-r * dt)
                
                option[i, j] = max(expected, K - stock[i, j])
    
    return option[n, 0]

def trinomial_tree_b(S0, K, r, sigma, T, n):
    dt = T / n
    dX_u = sigma * np.sqrt(3 * dt)
    dX_d = -dX_u
    
    drift = r - sigma**2/2
    p_d = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dX_u**2 - drift * dt / dX_u)
    p_u = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dX_u**2 + drift * dt / dX_u)
    p_m = 1 - p_u - p_d
    
    logS = np.zeros((2*n+1, n+1))
    logS[n, 0] = np.log(S0)
    
    for j in range(1, n+1):
        for i in range(2*n+1):
            net_moves = j - i + n
            
            if 0 <= net_moves <= 2*j:
                logS[i, j] = np.log(S0) + (j - i + n) * dX_u + (i - n) * dX_d
    
    stock = np.exp(logS)
    
    option = np.zeros((2*n+1, n+1))
    
    for i in range(2*n+1):
        if stock[i, n] > 0:
            option[i, n] = max(K - stock[i, n], 0)
    
    for j in range(n-1, -1, -1):
        for i in range(2*n+1):
            if stock[i, j] > 0:
                up_idx = max(0, min(2*n, i-1))
                mid_idx = i
                down_idx = min(2*n, i+1)
                
                expected = p_u * option[up_idx, j+1] + p_m * option[mid_idx, j+1] + p_d * option[down_idx, j+1]
                expected *= np.exp(-r * dt)
                
                option[i, j] = max(expected, K - stock[i, j])
    
    return option[n, 0]

def laguerre(x, k):
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return 1 - x
    elif k == 2:
        return 0.5 * (x**2 - 4*x + 2)
    elif k == 3:
        return (1/6) * (-x**3 + 9*x**2 - 18*x + 6)
    elif k == 4:
        return (1/24) * (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)
    elif k == 5:
        return (1/120) * (-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)

def hermite(x, k):
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    elif k == 2:
        return x**2 - 1
    elif k == 3:
        return x**3 - 3*x
    elif k == 4:
        return x**4 - 6*x**2 + 3
    elif k == 5:
        return x**5 - 10*x**3 + 15*x

def monomial(x, k):
    if k == 0:
        return np.ones_like(x)
    else:
        return x**k

def lsmc(S0, K, r, sigma, T, N, num_steps, k, poly_type='laguerre'):
    dt = T / num_steps
    df = np.exp(-r * dt)
    
    N_half = N // 2
    np.random.seed(42)
    Z = np.random.normal(0, 1, (N_half, num_steps))
    
    Z_anti = -Z
    Z_all = np.vstack((Z, Z_anti))
    
    S = np.zeros((N, num_steps + 1))
    S[:, 0] = S0
    
    for i in range(num_steps):
        S[:, i+1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_all[:, i])
    
    V = np.maximum(K - S[:, -1], 0)
    
    if poly_type == 'laguerre':
        poly_func = laguerre
    elif poly_type == 'hermite':
        poly_func = hermite
    else:
        poly_func = monomial
    
    for i in range(num_steps - 1, 0, -1):
        itm = K - S[:, i] > 0
        if sum(itm) > 0:
            S_itm = S[itm, i]
            
            V_next = V * df
            
            X = np.ones((sum(itm), k))
            for j in range(k):
                if j > 0:
                    X[:, j] = poly_func(S_itm / K, j)
            
            beta, _, _, _ = np.linalg.lstsq(X, V_next[itm], rcond=None)
            
            C = np.dot(X, beta)
            
            exercise = np.zeros(N, dtype=bool)
            exercise[itm] = (K - S_itm) > C
            
            V_temp = V.copy()
            V[exercise] = K - S[exercise, i]
            V[~exercise] = V_temp[~exercise] * df
        else:
            V = V * df
    
    return np.mean(V) * df

def explicit_fd_log(K, sigma, T, r, dt, dx, S0_range):
    x_min = np.log(min(S0_range) * 0.5)
    x_max = np.log(max(S0_range) * 2.0)
    
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    
    nt = int(T / dt) + 1
    t = np.linspace(0, T, nt)
    
    v = np.zeros((nx, nt))
    for i in range(nx):
        S = np.exp(x[i])
        v[i, -1] = max(K - S, 0)
    
    alpha = dt / (dx**2)
    beta = dt / (2 * dx)
    gamma = r - sigma**2/2
    
    if alpha > 0.5:
        print(f"Warning: Explicit method may be unstable. alpha = {alpha} > 0.5")
    
    for j in range(nt-2, -1, -1):
        for i in range(1, nx-1):
            a = 0.5 * sigma**2 * alpha
            b = gamma * beta
            v[i, j] = a * v[i+1, j+1] + (1 - 2*a - r*dt) * v[i, j+1] + (a - b) * v[i-1, j+1]
            
            S = np.exp(x[i])
            v[i, j] = max(v[i, j], K - S)
        
        v[0, j] = K * np.exp(-r * (T - t[j]))
        v[nx-1, j] = 0
    
    prices = np.zeros(len(S0_range))
    for i, S0 in enumerate(S0_range):
        x0 = np.log(S0)
        idx = np.abs(x - x0).argmin()
        if x[idx] == x0:
            prices[i] = v[idx, 0]
        else:
            if x[idx] < x0 and idx < nx - 1:
                t = (x0 - x[idx]) / (x[idx+1] - x[idx])
                prices[i] = v[idx, 0] * (1 - t) + v[idx+1, 0] * t
            elif x[idx] > x0 and idx > 0:
                t = (x0 - x[idx-1]) / (x[idx] - x[idx-1])
                prices[i] = v[idx-1, 0] * (1 - t) + v[idx, 0] * t
            else:
                prices[i] = v[idx, 0]
    
    return prices

def implicit_fd_log(K, sigma, T, r, dt, dx, S0_range):
    x_min = np.log(min(S0_range) * 0.5)
    x_max = np.log(max(S0_range) * 2.0)
    
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    
    nt = int(T / dt) + 1
    t = np.linspace(0, T, nt)
    
    v = np.zeros((nx, nt))
    for i in range(nx):
        S = np.exp(x[i])
        v[i, -1] = max(K - S, 0)
    
    alpha = dt / (dx**2)
    beta = dt / (2 * dx)
    gamma = r - sigma**2/2
    
    for j in range(nt-2, -1, -1):
        a = 0.5 * sigma**2 * alpha
        b = gamma * beta
        
        lower_diag = np.ones(nx-2) * (a - b)
        main_diag = np.ones(nx-1) * (1 + 2*a + r*dt)
        upper_diag = np.ones(nx-2) * (a + b)
        
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        rhs = v[1:nx, j+1].copy()
        
        boundary_val = K * np.exp(-r * (T - t[j]))
        rhs[0] -= (a - b) * boundary_val
        
        v_new = np.linalg.solve(A, rhs)
        
        v[1:nx, j] = v_new
        v[0, j] = boundary_val
        
        for i in range(nx):
            S = np.exp(x[i])
            v[i, j] = max(v[i, j], K - S)
    
    prices = np.zeros(len(S0_range))
    for i, S0 in enumerate(S0_range):
        x0 = np.log(S0)
        idx = np.abs(x - x0).argmin()
        if x[idx] == x0:
            prices[i] = v[idx, 0]
        else:
            if x[idx] < x0 and idx < nx - 1:
                t = (x0 - x[idx]) / (x[idx+1] - x[idx])
                prices[i] = v[idx, 0] * (1 - t) + v[idx+1, 0] * t
            elif x[idx] > x0 and idx > 0:
                t = (x0 - x[idx-1]) / (x[idx] - x[idx-1])
                prices[i] = v[idx-1, 0] * (1 - t) + v[idx, 0] * t
            else:
                prices[i] = v[idx, 0]
    
    return prices

def crank_nicolson_fd_log(K, sigma, T, r, dt, dx, S0_range):
    x_min = np.log(min(S0_range) * 0.5)
    x_max = np.log(max(S0_range) * 2.0)
    
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    
    nt = int(T / dt) + 1
    t = np.linspace(0, T, nt)
    
    v = np.zeros((nx, nt))
    for i in range(nx):
        S = np.exp(x[i])
        v[i, -1] = max(K - S, 0)
    
    alpha = dt / (2 * dx**2)
    beta = dt / (4 * dx)
    gamma = r - sigma**2/2
    
    for j in range(nt-2, -1, -1):
        a = 0.5 * sigma**2 * alpha
        b = gamma * beta
        
        lower_diag_A = np.ones(nx-2) * (-a + b)
        main_diag_A = np.ones(nx-1) * (1 + 2*a + 0.5*r*dt)
        upper_diag_A = np.ones(nx-2) * (-a - b)
        
        A = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        lower_diag_B = np.ones(nx-2) * (a - b)
        main_diag_B = np.ones(nx-1) * (1 - 2*a - 0.5*r*dt)
        upper_diag_B = np.ones(nx-2) * (a + b)
        
        B = diags([lower_diag_B, main_diag_B, upper_diag_B], [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        rhs = B @ v[1:nx, j+1]
        
        boundary_val = K * np.exp(-r * (T - t[j]))
        rhs[0] += (-a + b) * boundary_val
        
        v_new = np.linalg.solve(A, rhs)
        
        v[1:nx, j] = v_new
        v[0, j] = boundary_val
        v[nx-1, j] = 0
        
        for i in range(nx):
            S = np.exp(x[i])
            v[i, j] = max(v[i, j], K - S)
    
    prices = np.zeros(len(S0_range))
    for i, S0 in enumerate(S0_range):
        x0 = np.log(S0)
        idx = np.abs(x - x0).argmin()
        if x[idx] == x0:
            prices[i] = v[idx, 0]
        else:
            if x[idx] < x0 and idx < nx - 1:
                t = (x0 - x[idx]) / (x[idx+1] - x[idx])
                prices[i] = v[idx, 0] * (1 - t) + v[idx+1, 0] * t
            elif x[idx] > x0 and idx > 0:
                t = (x0 - x[idx-1]) / (x[idx] - x[idx-1])
                prices[i] = v[idx-1, 0] * (1 - t) + v[idx, 0] * t
            else:
                prices[i] = v[idx, 0]
    
    return prices

def explicit_fd_bs(K, sigma, T, r, dt, dS, S0_range):
    S_min = max(min(S0_range) - 50, 1.0)
    S_max = max(S0_range) + 100.0
    
    nS = int(round((S_max - S_min) / dS)) + 1
    S = np.linspace(S_min, S_max, nS)
    
    nt = int(round(T / dt)) + 1
    time_grid = np.linspace(0, T, nt) 
    
    v = np.zeros((nS, nt))
    
    payoff = np.maximum(K - S, 0)
    v[:, -1] = payoff
    
    for j in range(nt - 2, -1, -1):
        v[0, j] = K - S[0] 
        v[nS - 1, j] = 0.0
        
        for i in range(1, nS - 1):
            Si = S[i]
            
            coeff_Si2_term = sigma**2 * Si**2 * dt / dS**2
            coeff_Si_term = r * Si * dt / dS
            
            v[i, j] = (0.5 * coeff_Si2_term - 0.5 * coeff_Si_term) * v[i-1, j+1] + \
                        (1.0 - coeff_Si2_term - r * dt) * v[i, j+1] + \
                        (0.5 * coeff_Si2_term + 0.5 * coeff_Si_term) * v[i+1, j+1]
        
        v[:, j] = np.maximum(v[:, j], payoff)

    prices = np.zeros(len(S0_range))
    for i_S0, S0_val in enumerate(S0_range):
        idx = np.abs(S - S0_val).argmin()
        if np.isclose(S[idx], S0_val):
            prices[i_S0] = v[idx, 0]
        else:
            if S[idx] < S0_val and idx + 1 < nS:
                interp_ratio = (S0_val - S[idx]) / (S[idx+1] - S[idx])
                prices[i_S0] = v[idx, 0] * (1 - interp_ratio) + v[idx+1, 0] * interp_ratio
            elif S[idx] > S0_val and idx > 0:
                interp_ratio = (S0_val - S[idx-1]) / (S[idx] - S[idx-1])
                prices[i_S0] = v[idx-1, 0] * (1 - interp_ratio) + v[idx, 0] * interp_ratio
            else:
                prices[i_S0] = v[idx, 0]
    
    return prices

def implicit_fd_bs(K, sigma, T, r, dt, dS, S0_range):
    S_min = max(min(S0_range) - 50, 1.0)
    S_max = max(S0_range) + 100.0
    
    nS = int(round((S_max - S_min) / dS)) + 1
    S = np.linspace(S_min, S_max, nS)
    
    nt = int(round(T / dt)) + 1
    time_grid = np.linspace(0, T, nt)
    
    v = np.zeros((nS, nt))
    
    payoff = np.maximum(K - S, 0)
    v[:, -1] = payoff
    
    alpha = 0.5 * sigma**2 * S**2 * dt / dS**2
    beta = 0.5 * r * S * dt / dS
    
    
    lower_diag_coeffs = beta[1:nS-1] - alpha[1:nS-1]
    main_diag_coeffs = 1.0 + 2 * alpha[1:nS-1] + r * dt
    upper_diag_coeffs = -beta[1:nS-1] - alpha[1:nS-1]
    
    if nS <= 2:
        if nS == 1:
             v[0, :] = np.maximum(K - S[0],0)
        elif nS == 2:
             v[0, :] = np.maximum(K - S[0],0)
             v[1, :] = np.maximum(K - S[1],0)

        prices = np.zeros(len(S0_range))
        for i_S0, S0_val in enumerate(S0_range):
            idx = np.abs(S - S0_val).argmin()
            prices[i_S0] = v[idx, 0]
        return prices

    A_matrix = diags([lower_diag_coeffs[1:], main_diag_coeffs, upper_diag_coeffs[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
    
    for j in range(nt - 2, -1, -1):
        rhs_b_vector = v[1:nS-1, j+1].copy()
        
        boundary_val_S_min_j = K - S[0]
        boundary_val_S_max_j = 0.0

        rhs_b_vector[0] -= lower_diag_coeffs[0] * boundary_val_S_min_j
        rhs_b_vector[-1] -= upper_diag_coeffs[-1] * boundary_val_S_max_j
        
        v_interior = np.linalg.solve(A_matrix, rhs_b_vector)
        
        v[1:nS-1, j] = v_interior
        v[0, j] = boundary_val_S_min_j
        v[nS-1, j] = boundary_val_S_max_j
        
        v[:, j] = np.maximum(v[:, j], payoff)
    
    prices = np.zeros(len(S0_range))
    for i_S0, S0_val in enumerate(S0_range):
        idx = np.abs(S - S0_val).argmin()
        if np.isclose(S[idx], S0_val):
            prices[i_S0] = v[idx, 0]
        else:
            if S[idx] < S0_val and idx + 1 < nS:
                interp_ratio = (S0_val - S[idx]) / (S[idx+1] - S[idx])
                prices[i_S0] = v[idx, 0] * (1 - interp_ratio) + v[idx+1, 0] * interp_ratio
            elif S[idx] > S0_val and idx > 0:
                interp_ratio = (S0_val - S[idx-1]) / (S[idx] - S[idx-1])
                prices[i_S0] = v[idx-1, 0] * (1 - interp_ratio) + v[idx, 0] * interp_ratio
            else:
                prices[i_S0] = v[idx, 0]
                
    return prices

def crank_nicolson_fd_bs(K, sigma, T, r, dt, dS, S0_range):
    S_min = max(min(S0_range) - 50, 1.0)
    S_max = max(S0_range) + 100.0
    
    nS = int(round((S_max - S_min) / dS)) + 1
    S = np.linspace(S_min, S_max, nS)
    
    nt = int(round(T / dt)) + 1
    time_grid = np.linspace(0, T, nt)
        
    v = np.zeros((nS, nt))
    
    payoff = np.maximum(K - S, 0)
    v[:, -1] = payoff
    
    alpha_cn_half = 0.25 * sigma**2 * S**2 * dt / dS**2 
    beta_cn_half = 0.25 * r * S * dt / dS           

    if nS <= 2:
        if nS == 1:
             v[0, :] = np.maximum(K - S[0],0)
        elif nS == 2:
             v[0, :] = np.maximum(K - S[0],0)
             v[1, :] = np.maximum(K - S[1],0)

        prices = np.zeros(len(S0_range))
        for i_S0, S0_val in enumerate(S0_range):
            idx = np.abs(S - S0_val).argmin()
            prices[i_S0] = v[idx, 0]
        return prices

    m1_lower_coeffs = beta_cn_half[1:nS-1] - alpha_cn_half[1:nS-1]
    m1_main_coeffs = 1.0 + 2.0 * alpha_cn_half[1:nS-1] + 0.5 * r * dt
    m1_upper_coeffs = -beta_cn_half[1:nS-1] - alpha_cn_half[1:nS-1]
    M1_matrix = diags([m1_lower_coeffs[1:], m1_main_coeffs, m1_upper_coeffs[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
    
    m2_lower_coeffs = -(beta_cn_half[1:nS-1] - alpha_cn_half[1:nS-1])
    m2_main_coeffs = 1.0 - 2.0 * alpha_cn_half[1:nS-1] - 0.5 * r * dt
    m2_upper_coeffs = -(-beta_cn_half[1:nS-1] - alpha_cn_half[1:nS-1])
    M2_matrix = diags([m2_lower_coeffs[1:], m2_main_coeffs, m2_upper_coeffs[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
        
    for j in range(nt - 2, -1, -1):
        rhs_cn_vector = M2_matrix @ v[1:nS-1, j+1]
        
        boundary_val_S_min_j = K - S[0]
        boundary_val_S_max_j = 0.0
        
        val_S_min_jplus1 = v[0, j+1] 
        val_S_max_jplus1 = v[nS-1, j+1]

        rhs_cn_vector[0] -= m1_lower_coeffs[0] * boundary_val_S_min_j
        rhs_cn_vector[0] -= m2_lower_coeffs[0] * val_S_min_jplus1 
        
        rhs_cn_vector[-1] -= m1_upper_coeffs[-1] * boundary_val_S_max_j
        rhs_cn_vector[-1] -= m2_upper_coeffs[-1] * val_S_max_jplus1
                
        v_interior = np.linalg.solve(M1_matrix, rhs_cn_vector)
        
        v[1:nS-1, j] = v_interior
        v[0, j] = boundary_val_S_min_j
        v[nS-1, j] = boundary_val_S_max_j
        
        v[:, j] = np.maximum(v[:, j], payoff)
        
    prices = np.zeros(len(S0_range))
    for i_S0, S0_val in enumerate(S0_range):
        idx = np.abs(S - S0_val).argmin()
        if np.isclose(S[idx], S0_val):
            prices[i_S0] = v[idx, 0]
        else:
            if S[idx] < S0_val and idx + 1 < nS:
                interp_ratio = (S0_val - S[idx]) / (S[idx+1] - S[idx])
                prices[i_S0] = v[idx, 0] * (1 - interp_ratio) + v[idx+1, 0] * interp_ratio
            elif S[idx] > S0_val and idx > 0:
                interp_ratio = (S0_val - S[idx-1]) / (S[idx] - S[idx-1])
                prices[i_S0] = v[idx-1, 0] * (1 - interp_ratio) + v[idx, 0] * interp_ratio
            else:
                prices[i_S0] = v[idx, 0]
                
    return prices

import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationParams:
    # Asset Params
    V0: float = 20000.0      # Initial Asset Value
    mu: float = -0.1         # Drift (expected asset depreciation)
    sigma: float = 0.2       # Volatility
    lambda1: float = 0.2     # Jump frequency (per year)
    gamma: float = -0.4      # Jump size (percentage drop)
    
    # Loan Params
    L0: float = 22000.0      # Initial Loan Amount
    T: int = 5               # Term in years
    r0: float = 0.055        # Risk-free rate (for discounting)
    
    # Loan Interest Rate Logic
    delta: float = 0.25
    lambda2: float = 0.4
    
    # Default Trigger Logic
    alpha: float = 0.7       # Start trigger ratio
    epsilon: float = 0.95    # End trigger / Recovery rate
    
    # Sim Settings
    N_sims: int = 100000
    dt: float = 1/252
    seed: int = 42

    @property
    def mortgage_rate(self):
        return (self.r0 + self.delta * self.lambda2) / 12

    @property
    def N_steps(self):
        return int(self.T / self.dt)

def get_loan_schedule_constants(p: SimulationParams):
    """Pre-calculates constants a, b, c for the amortization formula."""
    r = p.mortgage_rate
    n_months = int(p.T * 12)
    PMT = p.L0 * r / (1 - (1 + r)**(-n_months))
    
    a = PMT / r
    b = PMT / (r * (1 + r)**n_months)
    c = 1 + r
    beta = (p.epsilon - p.alpha) / p.T
    return a, b, c, beta

def calculate_outstanding_loan(t, a, b, c):
    """Calculates Loan Balance (L_t) at time t."""
    return a - b * c**(t * 12)

def calculate_trigger_ratio(t, alpha, beta):
    """Calculates the default trigger ratio (q_t) at time t."""
    return alpha + beta * t

def evolve_asset_values(V, p: SimulationParams, active_mask):
    """Updates asset values V based on Jump-Diffusion process."""
    num_active = np.sum(active_mask)
    
    # 1. Poisson Jumps (Market Crashes)
    jumps = np.random.poisson(p.lambda1 * p.dt, size=num_active) > 0
    
    # Create a temporary mask relative to the whole array
    # (Only update specific active indices)
    jump_indices = np.where(active_mask)[0][jumps]
    V[jump_indices] *= (1 + p.gamma)

    # 2. Diffusion (Standard Volatility)
    dW = np.random.normal(0, np.sqrt(p.dt), size=num_active)
    drift_diffusion = np.exp((p.mu - 0.5 * p.sigma**2) * p.dt + p.sigma * dW)
    V[active_mask] *= drift_diffusion
    
    return V

def simulate_mortgage_default_option(params: SimulationParams):
    np.random.seed(params.seed)
    
    # --- Initialization ---
    V = np.full(params.N_sims, params.V0, dtype=float)
    default_payoffs = np.zeros(params.N_sims)
    default_times = np.zeros(params.N_sims)
    active_sims = np.ones(params.N_sims, dtype=bool)
    
    # Pre-calculate constants to speed up the loop
    a, b, c, beta = get_loan_schedule_constants(params)

    # --- Time Loop ---
    for step in range(params.N_steps):
        if not np.any(active_sims): 
            break
            
        t = step * params.dt
        
        # 1. Update Deterministic Variables (Loan & Trigger)
        L_t = calculate_outstanding_loan(t, a, b, c)
        q_t = calculate_trigger_ratio(t, params.alpha, beta)
        
        # 2. Update Stochastic Variable (Asset Value)
        V = evolve_asset_values(V, params, active_sims)
        
        # 3. Check for Defaults
        default_threshold = q_t * L_t
        # Identify newly defaulted simulations
        new_defaults_mask = (V <= default_threshold) & active_sims
        
        # 4. Process Payoffs for Defaults
        if np.any(new_defaults_mask):
            # Payoff = Loan Balance - Recovered Asset Value
            loss = params.L0 # safeguard if L_t logic fails, though L_t is accurate
            recovered_value = params.epsilon * V[new_defaults_mask]
            payoff = np.maximum(L_t - recovered_value, 0)
            
            # Discount to Present Value
            default_payoffs[new_defaults_mask] = payoff * np.exp(-params.r0 * t)
            default_times[new_defaults_mask] = t
            
            # Remove defaulted loans from active pool
            active_sims[new_defaults_mask] = False

    # --- Aggregation ---
    default_count = np.sum(default_times > 0)
    
    option_price = np.mean(default_payoffs)
    prob_default = default_count / params.N_sims
    expected_time = np.sum(default_times) / default_count if default_count > 0 else 0
    
    return option_price, prob_default, expected_time

def heston_down_out_put(K=100, T=1, gamma=0.25):
    np.random.seed(42)
    
    v0, alpha, beta = 0.1, 0.45, -5.105
    S0, r, rho = 100, 0.05, -0.75
    
    N_sims = 100000
    dt = 1/252
    N_steps = int(T / dt)
    
    # Pre-compute time grid and barriers
    t_grid = np.arange(N_steps) * dt
    barrier_1 = np.full(N_steps, 94.0)
    barrier_2 = 6 * t_grid / T + 91
    barrier_3 = -6 * t_grid / T + 97
    
    # Initialize arrays for vectorized computation
    S_paths = np.full(N_sims, float(S0))
    v_paths = np.full(N_sims, float(v0))
    barrier_hit_1 = np.zeros(N_sims, dtype=bool)
    barrier_hit_2 = np.zeros(N_sims, dtype=bool)
    barrier_hit_3 = np.zeros(N_sims, dtype=bool)
    
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1 - rho**2)
    
    # Vectorized simulation
    for step in range(N_steps):
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, sqrt_dt, N_sims)
        dW2 = rho * dW1 + sqrt_one_minus_rho2 * np.random.normal(0, sqrt_dt, N_sims)
        
        # Full truncation
        v_pos = np.maximum(v_paths, 0)
        sqrt_v_pos = np.sqrt(v_pos)
        
        # Update paths
        S_paths += r * S_paths * dt + S_paths * sqrt_v_pos * dW1
        v_paths += (alpha + beta * v_pos) * dt + gamma * sqrt_v_pos * dW2
        
        # Check barriers (vectorized)
        barrier_hit_1 |= (S_paths <= barrier_1[step])
        barrier_hit_2 |= (S_paths <= barrier_2[step])
        barrier_hit_3 |= (S_paths <= barrier_3[step])
    
    # Calculate payoffs
    final_payoffs = np.maximum(K - S_paths, 0) * np.exp(-r * T)
    
    P1 = np.mean(final_payoffs * (~barrier_hit_1))
    P2 = np.mean(final_payoffs * (~barrier_hit_2))
    P3 = np.mean(final_payoffs * (~barrier_hit_3))
    
    return P1, P2, P3

from scipy.linalg import solve_banded
def cir_bond_price(r0=0.05, sigma=0.12, kappa=0.92, r_bar=0.055, T=4):
    np.random.seed(42)
    
    N_sims = 50000  # Reduced for better performance
    dt = 1/252
    N_steps = int(T / dt)
    
    coupon_times = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    coupons = np.array([30, 30, 30, 30, 30, 30, 30, 1030])
    
    # Vectorized simulation
    r_paths = np.full(N_sims, r0)
    integral_r = np.zeros(N_sims)
    bond_values = np.zeros(N_sims)
    
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        t = step * dt
        
        if t >= T:
            break
        
        # Vectorized CIR updates
        dW = np.random.normal(0, sqrt_dt, N_sims)
        r_pos = np.maximum(r_paths, 0)
        sqrt_r = np.sqrt(r_pos)
        
        r_paths += kappa * (r_bar - r_paths) * dt + sigma * sqrt_r * dW
        r_paths = np.maximum(r_paths, 0)
        
        integral_r += r_paths * dt
        
        # Check for coupon payments (vectorized)
        for i, coupon_time in enumerate(coupon_times):
            if abs(t - coupon_time) < dt/2:
                bond_values += coupons[i] * np.exp(-integral_r)
    
    return np.mean(bond_values)

def cir_zero_bond_price(r, T, kappa=0.92, r_bar=0.055, sigma=0.12):
    if T <= 0:
        return 1.0
    
    gamma = np.sqrt(kappa**2 + 2*sigma**2)
    
    B = (2 * (np.exp(gamma * T) - 1)) / ((kappa + gamma) * (np.exp(gamma * T) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((kappa + gamma) * T / 2)) / ((kappa + gamma) * (np.exp(gamma * T) - 1) + 2 * gamma))**(2 * kappa * r_bar / sigma**2)
    
    return A * np.exp(-B * r)

def cir_call_option_mc(r0=0.05, T=0.5, S=1, K=980, sigma=0.12, kappa=0.92, r_bar=0.055):
    np.random.seed(42)
    
    N_sims = 50000  # Reduced for better performance
    dt = 1/252
    N_steps = int(T / dt)
    
    # Vectorized simulation
    r_paths = np.full(N_sims, r0)
    integral_r = np.zeros(N_sims)
    
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        # Vectorized CIR updates
        dW = np.random.normal(0, sqrt_dt, N_sims)
        r_pos = np.maximum(r_paths, 0)
        sqrt_r = np.sqrt(r_pos)
        
        r_paths += kappa * (r_bar - r_paths) * dt + sigma * sqrt_r * dW
        r_paths = np.maximum(r_paths, 0)
        
        integral_r += r_paths * dt
    
    # Vectorized bond pricing and option payoffs
    bond_prices_at_T = np.array([cir_zero_bond_price(r, S - T, kappa, r_bar, sigma) * 1000 for r in r_paths])
    payoffs = np.maximum(bond_prices_at_T - K, 0)
    option_values = payoffs * np.exp(-integral_r)
    
    return np.mean(option_values)

def cir_call_option_pde(r0=0.05, T=0.5, S=1, K=980, sigma=0.12, kappa=0.92, r_bar=0.055):
    r_max = 0.5
    r_min = 0.0
    N_r = 200
    N_t = 1000
    
    dr = (r_max - r_min) / N_r
    dt = T / N_t
    
    r_grid = np.linspace(r_min, r_max, N_r + 1)
    interior_r_grid = r_grid[1:-1]
    
    option_values = np.zeros((N_t + 1, N_r + 1))
    
    bond_maturity_at_T = S - T 
    bond_prices_at_T = cir_zero_bond_price(r_grid, bond_maturity_at_T, kappa, r_bar, sigma) * 1000
    option_values[N_t, :] = np.maximum(bond_prices_at_T - K, 0)
    
    aj = (dt / 2) * (kappa * (r_bar - interior_r_grid) / dr - (sigma**2 * interior_r_grid) / dr**2)
    bj = 1 + dt * ( (sigma**2 * interior_r_grid) / dr**2 + interior_r_grid )
    cj = (dt / 2) * (-kappa * (r_bar - interior_r_grid) / dr - (sigma**2 * interior_r_grid) / dr**2)

    for i in range(N_t - 1, -1, -1):
        known_b = option_values[i + 1, 1:-1]
        
        known_b[0] -= aj[0] * option_values[i + 1, 0]
        known_b[-1] -= cj[-1] * option_values[i + 1, -1]

        banded_M = np.vstack([np.append(0, cj[:-1]), bj, np.append(aj[1:], 0)])
        
        x = solve_banded((1, 1), banded_M, known_b)
        option_values[i, 1:-1] = x
        
        option_values[i, 0] = option_values[i, 1]
        option_values[i, -1] = option_values[i, -2]

    final_price = np.interp(r0, r_grid, option_values[0, :])
    
    return final_price

def g2pp_zero_bond_price(t, T, x, y, phi, a=0.1, b=0.3, sigma=0.05, eta=0.09, rho=0.7):
    """
    Calculates the G2++ zero-coupon bond price P(t, T).
    't' is the current time, 'T' is the bond's maturity.
    """
    tau = T - t
    if tau <= 1e-10:
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0

    # B(t,T) terms are correct
    B_x = (1 - np.exp(-a * tau)) / a
    B_y = (1 - np.exp(-b * tau)) / b
    
    # A(t,T) term components - CORRECTED to include rho and use proper integral for covariance
    v_x = (sigma**2 / (2 * a**2)) * (tau - 2 * B_x + (1 - np.exp(-2 * a * tau)) / (2 * a))
    v_y = (eta**2 / (2 * b**2)) * (tau - 2 * B_y + (1 - np.exp(-2 * b * tau)) / (2 * b))
    v_xy = (rho * sigma * eta / (a * b)) * (tau - B_x - B_y + (1 - np.exp(-(a + b) * tau)) / (a + b))
    
    log_A_t = v_x + v_y + v_xy
    
    return np.exp(log_A_t - B_x * x - B_y * y - phi * tau)


# --- Corrected and Vectorized Monte Carlo Simulation ---
def g2pp_put_option_mc(T=0.5, S=1, K=950, rho=0.7):
    np.random.seed(42)
    
    x0, y0, phi0 = 0.0, 0.0, 0.055
    a, b, sigma, eta = 0.1, 0.3, 0.05, 0.09
    
    N_sims = 100000
    dt = 1/252
    N_steps = int(T / dt)
    
    x_paths = np.full(N_sims, x0, dtype=float)
    y_paths = np.full(N_sims, y0, dtype=float)
    integral_r = np.zeros(N_sims, dtype=float)
    
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho2 = np.sqrt(1 - rho**2)
    
    for _ in range(N_steps):
        dW1 = np.random.normal(0, sqrt_dt, N_sims)
        dW2 = rho * dW1 + sqrt_one_minus_rho2 * np.random.normal(0, sqrt_dt, N_sims)
        
        # Euler discretization for x and y
        x_paths = x_paths - a * x_paths * dt + sigma * dW1
        y_paths = y_paths - b * y_paths * dt + eta * dW2
        
        # Accumulate the integral for the discount factor
        integral_r += (x_paths + y_paths + phi0) * dt
    
    # Price the underlying bond P(T, S) at option expiration T
    # This is now fully vectorized, removing the slow Python loop.
    bond_prices_at_T = g2pp_zero_bond_price(T, S, x_paths, y_paths, phi0, a, b, sigma, eta, rho) * 1000
    
    # Calculate payoffs and discount back to t=0
    payoffs = np.maximum(K - bond_prices_at_T, 0)
    put_values = payoffs * np.exp(-integral_r)
    
    return np.mean(put_values)


# --- Corrected Explicit Formula ---
def g2pp_put_option_explicit(T=0.5, S=1, K=950, rho=0.7):
    a, b, sigma, eta = 0.1, 0.3, 0.05, 0.09
    phi0 = 0.055
    
    # Calculate zero-coupon bond prices P(0,T) and P(0,S)
    # Note: x0=0 and y0=0
    P_0T = g2pp_zero_bond_price(0, T, 0, 0, phi0, a, b, sigma, eta, rho)
    P_0S = g2pp_zero_bond_price(0, S, 0, 0, phi0, a, b, sigma, eta, rho)
    
    # Calculate the volatility term Σ(T) for the forward bond P(T,S) - CORRECTED
    B_x_TS = (1 - np.exp(-a * (S - T))) / a
    B_y_TS = (1 - np.exp(-b * (S - T))) / b
    
    var_x = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * T)) * (B_x_TS**2)
    var_y = (eta**2 / (2 * b)) * (1 - np.exp(-2 * b * T)) * (B_y_TS**2)
    cov_xy = (rho * sigma * eta / (a + b)) * (1 - np.exp(-(a + b) * T)) * B_x_TS * B_y_TS
    
    total_variance = var_x + var_y + 2 * cov_xy
    v = np.sqrt(total_variance)
    
    # Standard Black's model formula for a put
    d1 = (np.log(P_0S / ( (K/1000) * P_0T)) / v) + 0.5 * v
    d2 = d1 - v
    
    # Final put price formula - CORRECTED scaling
    put_price = (K * P_0T * norm.cdf(-d2)) - (1000 * P_0S * norm.cdf(-d1))
    
    return put_price

def numerix_prepayment_rate(age, r_market, WAC=0.08):
    """Calculates the Numerix prepayment rate in a vectorized manner."""
    r_incentive = (WAC - r_market) * 100
    
    RI = np.maximum(r_incentive, 0)
    BU = 0.3 + 0.7 * np.minimum(age / 30, 1)
    SG = np.minimum(1, (age / 6) * np.exp(-age / 40))
    SY = 0.94 - 0.1 * np.minimum(age / 25, 1)
    
    return (RI * BU * SG * SY) / 100

def _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional):
    """Runs the core MBS Monte Carlo simulation and returns key metrics."""
    np.random.seed(42)
    
    N_sims = 25000
    dt = 1/12
    N_months = 360
    
    
    monthly_wac = WAC / 12
    if monthly_wac == 0:
        monthly_payment = notional / N_months
    else:
        monthly_payment = notional * (monthly_wac * (1 + monthly_wac)**N_months) / ((1 + monthly_wac)**N_months - 1)

    r_paths = np.full(N_sims, r0, dtype=float)
    balance_paths = np.full(N_sims, float(notional), dtype=float)
    
    integral_r_paths = np.zeros(N_sims, dtype=float)
    mbs_values = np.zeros(N_sims, dtype=float)
    io_values = np.zeros(N_sims, dtype=float)
    po_values = np.zeros(N_sims, dtype=float)
    
    sqrt_dt = np.sqrt(dt)
    
    for month in range(1, N_months + 1):
        active_mask = balance_paths > 1e-6
        n_active = np.sum(active_mask)
        
        if n_active == 0:
            break
        
        current_balances = balance_paths[active_mask]
        current_rates = r_paths[active_mask]
        
        integral_r_paths[active_mask] += current_rates * dt
        discount_factors = np.exp(-integral_r_paths[active_mask])
        
        dW = np.random.normal(0, 1.0, n_active)
        new_rates = current_rates + kappa * (r_bar - current_rates) * dt + sigma * np.sqrt(current_rates) * sqrt_dt * dW
        r_paths[active_mask] = np.maximum(new_rates, 0)

        interest_payments = current_balances * monthly_wac
        scheduled_principal = monthly_payment - interest_payments
        
        prepayment_rates = numerix_prepayment_rate(month, r_paths[active_mask], WAC)
        prepayments = current_balances * prepayment_rates
        
        total_principal = np.minimum(scheduled_principal + prepayments, current_balances)
        total_cashflow = interest_payments + total_principal
        
        mbs_values[active_mask] += total_cashflow * discount_factors
        io_values[active_mask] += interest_payments * discount_factors
        po_values[active_mask] += total_principal * discount_factors
        
        balance_paths[active_mask] -= total_principal

    return np.mean(mbs_values), np.mean(io_values), np.mean(po_values)

def mbs_pricing(r_bar=0.08, kappa=0.6, sigma=0.12, r0=0.078, WAC=0.08, notional=100000):
    """Optimized: Prices the full pass-through MBS."""
    mbs_val, _, _ = _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional)
    return mbs_val

def compute_oas(market_price, r_bar, kappa, sigma, r0=0.078, WAC=0.08, notional=100000):
    """Optimized: Computes the Option-Adjusted Spread (OAS)."""
    def objective(spread):
        return mbs_pricing(r_bar + spread, kappa, sigma, r0, WAC, notional) - market_price
    
    try:
        oas = brentq(objective, -0.05, 0.05, xtol=1e-6, rtol=1e-6)
        return oas
    except ValueError:
        print("OAS not found in the initial search interval.")
        return np.nan

def io_po_pricing(r_bar=0.08, kappa=0.6, sigma=0.12, r0=0.078, WAC=0.08, notional=100000):
    """Optimized: Prices the IO and PO strips of the MBS."""
    _, io_val, po_val = _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional)
    return io_val, po_val