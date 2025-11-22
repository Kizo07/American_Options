import numpy as np

def lcg_uniform(x0, n, a=7**5, b=0, m=2**31 - 1):
    x = np.empty(n)
    x[0] = (a * x0 + b) % m
    for i in range(1, n):
        x[i] = (a * x[i - 1] + b) % m
    return x / m

def bernoulli(x0, p, n):
    x = lcg_uniform(x0, n)
    return (x < p).astype(int)

def binomial(x0, n, p, N):
    b = bernoulli(x0, p, n * N)
    return b.reshape(N, n).sum(axis=1)

def exponential(x0, lam, n):
    x = lcg_uniform(x0, n)
    return -(1/lam) * np.log(x)

def box_muller(x0, n):
    x = lcg_uniform(x0, n)
    if n % 2 != 0:
        x = np.append(x, lcg_uniform(x[-1], 1)) # Ensure even number for pairs
    
    x = x.reshape(-1, 2)
    z1 = np.sqrt(-2 * np.log(x[:, 0])) * np.cos(2 * np.pi * x[:, 1])
    z2 = np.sqrt(-2 * np.log(x[:, 0])) * np.sin(2 * np.pi * x[:, 1])
    
    res = np.concatenate((z1, z2))
    return res[:n]

def polar_marsaglia(x0, n):
    # Generate more than needed to account for rejection
    count = int(n * 1.3)
    x = lcg_uniform(x0, count)
    x = x.reshape(-1, 2) * 2 - 1 # Map to [-1, 1]
    
    s = x[:, 0]**2 + x[:, 1]**2
    mask = (s < 1) & (s > 0)
    x_valid = x[mask]
    s_valid = s[mask]
    
    factor = np.sqrt(-2 * np.log(s_valid) / s_valid)
    z1 = x_valid[:, 0] * factor
    z2 = x_valid[:, 1] * factor
    
    res = np.concatenate((z1, z2))
    if len(res) < n:
        # Recursive call if not enough samples
        return np.concatenate((res, polar_marsaglia(x0 + 123, n - len(res))))
    return res[:n]
