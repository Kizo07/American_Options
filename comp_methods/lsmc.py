import numpy as np
from ._validation import validate_choice, validate_positive

def laguerre(x, k):
    if k == 0: return np.ones_like(x)
    if k == 1: return 1 - x
    if k == 2: return 0.5 * (x**2 - 4*x + 2)
    if k == 3: return (1/6) * (-x**3 + 9*x**2 - 18*x + 6)
    if k == 4: return (1/24) * (x**4 - 16*x**3 + 72*x**2 - 96*x + 24)
    if k == 5: return (1/120) * (-x**5 + 25*x**4 - 200*x**3 + 600*x**2 - 600*x + 120)
    return np.zeros_like(x)

def hermite(x, k):
    if k == 0: return np.ones_like(x)
    if k == 1: return x
    if k == 2: return x**2 - 1
    if k == 3: return x**3 - 3*x
    if k == 4: return x**4 - 6*x**2 + 3
    if k == 5: return x**5 - 10*x**3 + 15*x
    return np.zeros_like(x)

def monomial(x, k):
    return x**k

def _rng_normal(rng, loc, scale, size):
    if rng is None:
        return np.random.normal(loc, scale, size)
    return rng.normal(loc, scale, size)

def lsmc(S0, K, r, sigma, T, N, num_steps, k, poly_type='laguerre', rng=None):
    validate_choice("poly_type", poly_type, {"laguerre", "hermite", "monomial"})
    for name, value in {"S0": S0, "K": K, "sigma": sigma, "T": T, "N": N, "num_steps": num_steps, "k": k}.items():
        validate_positive(name, value)
    dt = T / num_steps
    df = np.exp(-r * dt)
    
    N_half = int(np.ceil(N / 2))
    Z_half = _rng_normal(rng, 0.0, 1.0, (N_half, num_steps))
    Z = np.vstack((Z_half, -Z_half))[:N]
    
    S = np.zeros((N, num_steps + 1))
    S[:, 0] = S0
    for i in range(num_steps):
        S[:, i+1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    V = np.maximum(K - S[:, -1], 0)
    
    poly_func = {'laguerre': laguerre, 'hermite': hermite}.get(poly_type, monomial)
    
    for i in range(num_steps - 1, 0, -1):
        itm = K - S[:, i] > 0
        if np.any(itm):
            S_itm = S[itm, i]
            V_next = V * df
            
            basis_count = min(k, np.sum(itm))
            X = np.ones((np.sum(itm), basis_count))
            for j in range(1, basis_count):
                X[:, j] = poly_func(S_itm / K, j)
            
            beta = np.linalg.lstsq(X, V_next[itm], rcond=None)[0]
            C = np.dot(X, beta)
            
            exercise = np.zeros(N, dtype=bool)
            exercise[itm] = (K - S_itm) > C
            
            V[exercise] = K - S[exercise, i]
            V[~exercise] *= df
        else:
            V *= df
            
    return max(float(np.mean(V) * df), max(K - S0, 0.0))
