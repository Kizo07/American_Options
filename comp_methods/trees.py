import numpy as np
from ._validation import validate_choice, validate_positive
from .curves import as_flat_rate

def binomial_tree(S0, K, r, sigma, T, n, method='a', option_type='put', q=0.0):
    r = as_flat_rate(r, T)
    validate_choice("option_type", option_type, {"call", "put"})
    validate_choice("method", method, {"a", "b"})
    for name, value in {"S0": S0, "K": K, "sigma": sigma, "T": T, "n": n}.items():
        validate_positive(name, value)
    dt = T / n
    if method == 'a':
        c = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
        d = c - np.sqrt(c**2 - 1)
        u = 1 / d
        p = (np.exp((r - q) * dt) - d) / (u - d)
    else:
        u = np.exp((r - q - sigma**2/2) * dt + sigma * np.sqrt(dt))
        d = np.exp((r - q - sigma**2/2) * dt - sigma * np.sqrt(dt))
        p = 0.5
    
    stock = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**(i-j)) * (d**j)
    
    option = np.zeros((n+1, n+1))
    for i in range(n+1):
        if option_type == 'call':
            option[i, n] = max(stock[i, n] - K, 0)
        else:
            option[i, n] = max(K - stock[i, n], 0)
    
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            expected = p * option[j, i+1] + (1-p) * option[j+1, i+1]
            if option_type == 'call':
                payoff = stock[j, i] - K
            else:
                payoff = K - stock[j, i]
            option[j, i] = max(expected * np.exp(-r * dt), payoff)
            
    return option[0, 0]

def crr_american_put(S0, K, r, sigma, T, n, q=0.0):
    r = as_flat_rate(r, T)
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
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
            option[j, i] = max(expected * np.exp(-r * dt), K - stock[j, i])
            
    if n > 1:
        delta = (option[0, 1] - option[1, 1]) / (stock[0, 1] - stock[1, 1])
    else:
        delta = (option[0, 0] - K + S0) / S0 if S0 < K else -1
        
    return option[0, 0], delta

def trinomial_tree(S0, K, r, sigma, T, n, method='a', option_type='put', q=0.0):
    r = as_flat_rate(r, T)
    validate_choice("method", method, {"a", "b"})
    validate_choice("option_type", option_type, {"call", "put"})
    for name, value in {"S0": S0, "K": K, "sigma": sigma, "T": T, "n": n}.items():
        validate_positive(name, value)

    dt = T / n
    dx = sigma * np.sqrt(3 * dt)
    drift = r - q - 0.5 * sigma**2
    p_u = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dx**2 + drift * dt / dx)
    p_d = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dx**2 - drift * dt / dx)
    p_m = 1.0 - p_u - p_d

    if min(p_u, p_m, p_d) < -1e-14:
        raise ValueError("trinomial probabilities are negative; increase n or adjust parameters")

    node_indices = np.arange(-n, n + 1)
    stock = S0 * np.exp(node_indices * dx)
    if option_type == 'call':
        option = np.maximum(stock - K, 0.0)
    else:
        option = np.maximum(K - stock, 0.0)

    disc = np.exp(-r * dt)
    for step in range(n - 1, -1, -1):
        continuation = disc * (
            p_d * option[:-2]
            + p_m * option[1:-1]
            + p_u * option[2:]
        )
        active_indices = np.arange(-step, step + 1)
        active_stock = S0 * np.exp(active_indices * dx)
        if option_type == 'call':
            intrinsic = np.maximum(active_stock - K, 0.0)
        else:
            intrinsic = np.maximum(K - active_stock, 0.0)
        option = np.maximum(continuation, intrinsic)

    return option[0]
