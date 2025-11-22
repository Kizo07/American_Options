import numpy as np

def binomial_tree(S0, K, r, sigma, T, n, method='a', option_type='put'):
    dt = T / n
    if method == 'a':
        c = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
        d = c - np.sqrt(c**2 - 1)
        u = 1 / d
        p = (np.exp(r * dt) - d) / (u - d)
    else:
        u = np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt))
        d = np.exp((r - sigma**2/2) * dt - sigma * np.sqrt(dt))
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

def crr_american_put(S0, K, r, sigma, T, n):
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
            option[j, i] = max(expected * np.exp(-r * dt), K - stock[j, i])
            
    if n > 1:
        delta = (option[0, 1] - option[1, 1]) / (stock[0, 1] - stock[1, 1])
    else:
        delta = (option[0, 0] - K + S0) / S0 if S0 < K else -1
        
    return option[0, 0], delta

def trinomial_tree(S0, K, r, sigma, T, n, method='a'):
    dt = T / n
    if method == 'a':
        d = np.exp(-sigma * np.sqrt(3 * dt))
        u = 1 / d
        p_d = (r * dt * (1 - u) + (r * dt)**2 + sigma**2 * dt) / ((u - d) * (1 - d))
        p_u = (r * dt * (1 - d) + (r * dt)**2 + sigma**2 * dt) / ((u - d) * (u - 1))
        p_m = 1 - p_u - p_d
        
        stock = np.zeros((2*n+1, n+1))
        stock[n, 0] = S0
        for j in range(1, n+1):
            for i in range(2*n+1):
                up = max(0, j - i + n)
                down = max(0, i - n)
                if up + down <= j:
                    stock[i, j] = S0 * (u ** up) * (d ** down)
    else:
        dX = sigma * np.sqrt(3 * dt)
        drift = r - sigma**2/2
        p_d = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dX**2 - drift * dt / dX)
        p_u = 0.5 * ((sigma**2 * dt + drift**2 * dt**2) / dX**2 + drift * dt / dX)
        p_m = 1 - p_u - p_d
        
        logS = np.zeros((2*n+1, n+1))
        logS[n, 0] = np.log(S0)
        for j in range(1, n+1):
            for i in range(2*n+1):
                if 0 <= j - i + n <= 2*j:
                    logS[i, j] = np.log(S0) + (j - i + n) * dX + (i - n) * (-dX)
        stock = np.exp(logS)

    option = np.zeros((2*n+1, n+1))
    for i in range(2*n+1):
        if stock[i, n] > 0:
            option[i, n] = max(K - stock[i, n], 0)
            
    for j in range(n-1, -1, -1):
        for i in range(2*n+1):
            if stock[i, j] > 0:
                up = max(0, min(2*n, i-1))
                mid = i
                down = min(2*n, i+1)
                
                expected = p_u * option[up, j+1] + p_m * option[mid, j+1] + p_d * option[down, j+1]
                option[i, j] = max(expected * np.exp(-r * dt), K - stock[i, j])
                
    return option[n, 0]
