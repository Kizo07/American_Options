import numpy as np

def brownian_motion(dt, steps, N):
    return np.sqrt(dt) * np.random.randn(N, steps)

def correlated_brownian_motion(T, N, n, rho):
    dt = T / N
    dW1 = np.random.normal(0.0, np.sqrt(dt), (n, N))
    dW2 = np.random.normal(0.0, np.sqrt(dt), (n, N))
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2
    return dW1, dW2

def gbm(S0, r, sigma, T, Wt):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * Wt)

def two_factor_gbm(S0, V0, r, alpha, beta, sigma, rho, T, N, n, method="partial_truncation"):
    dt = T / N
    S_paths = np.zeros((n, N + 1))
    V_paths = np.zeros((n, N + 1))
    S_paths[:, 0] = S0
    V_paths[:, 0] = V0

    dW1, dW2 = correlated_brownian_motion(T, N, n, rho)

    for t in range(N):
        S = S_paths[:, t]
        V = V_paths[:, t]
        sqrt_V = np.sqrt(np.maximum(V, 0))
        S_paths[:, t + 1] = S * np.exp((r - 0.5 * V) * dt + sqrt_V * dW1[:, t])

        V_next = V + alpha * (beta - V) * dt + sigma * sqrt_V * dW2[:, t]
        
        if method == "full_truncation":
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "partial_truncation":
            V_paths[:, t + 1] = np.maximum(V_next, 0)
        elif method == "reflection":
            V_paths[:, t + 1] = np.abs(V_next)
        else:
             raise ValueError("Invalid method. Use 'full_truncation', 'partial_truncation', or 'reflection'.")

    return S_paths[:, -1]
