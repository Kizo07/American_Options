import numpy as np
from ._validation import validate_choice, validate_positive, validate_rho

def _rng_normal(rng, loc, scale, size):
    if rng is None:
        return np.random.normal(loc, scale, size)
    return rng.normal(loc, scale, size)

def brownian_motion(dt, steps, N, rng=None):
    validate_positive("dt", dt)
    validate_positive("steps", steps)
    validate_positive("N", N)
    return _rng_normal(rng, 0.0, np.sqrt(dt), (N, steps))

def correlated_brownian_motion(T, N, n, rho, rng=None):
    validate_positive("T", T)
    validate_positive("N", N)
    validate_positive("n", n)
    validate_rho(rho)
    dt = T / N
    dW1 = _rng_normal(rng, 0.0, np.sqrt(dt), (n, N))
    dW2 = _rng_normal(rng, 0.0, np.sqrt(dt), (n, N))
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2
    return dW1, dW2

def gbm(S0, r, sigma, T, Wt):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * Wt)

def two_factor_gbm(S0, V0, r, alpha, beta, sigma, rho, T, N, n, method="partial_truncation", rng=None):
    validate_choice("method", method, {"full_truncation", "partial_truncation", "reflection"})
    validate_rho(rho)
    for name, value in {"S0": S0, "T": T, "N": N, "n": n}.items():
        validate_positive(name, value)
    dt = T / N
    S_paths = np.zeros((n, N + 1))
    V_paths = np.zeros((n, N + 1))
    S_paths[:, 0] = S0
    V_paths[:, 0] = V0

    dW1, dW2 = correlated_brownian_motion(T, N, n, rho, rng=rng)

    for t in range(N):
        S = S_paths[:, t]
        V = V_paths[:, t]
        V_pos = np.maximum(V, 0)
        sqrt_V = np.sqrt(V_pos)
        variance_for_drift = V_pos if method == "full_truncation" else V
        S_paths[:, t + 1] = S * np.exp((r - 0.5 * variance_for_drift) * dt + sqrt_V * dW1[:, t])

        drift_variance = V_pos if method == "full_truncation" else V
        V_next = V + alpha * (beta - drift_variance) * dt + sigma * sqrt_V * dW2[:, t]
        
        if method == "full_truncation":
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "partial_truncation":
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "reflection":
            V_paths[:, t + 1] = np.abs(V_next)

    return S_paths[:, -1]
