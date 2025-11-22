import numpy as np
from scipy.sparse import diags

def _fd_solver_log(K, sigma, T, r, dt, dx, S0_range, method='explicit', option_type='put'):
    x_min = np.log(min(S0_range) * 0.5)
    x_max = np.log(max(S0_range) * 2.0)
    nx = int((x_max - x_min) / dx) + 1
    x = np.linspace(x_min, x_max, nx)
    nt = int(T / dt) + 1
    t = np.linspace(0, T, nt)
    
    v = np.zeros((nx, nt))
    for i in range(nx):
        if option_type == 'call':
            v[i, -1] = max(np.exp(x[i]) - K, 0)
        else:
            v[i, -1] = max(K - np.exp(x[i]), 0)
        
    alpha = dt / (dx**2)
    beta = dt / (2 * dx)
    gamma = r - sigma**2/2
    
    if method == 'explicit':
        # Stability check
        if alpha * sigma**2 > 1.0:
             print(f"Warning: Explicit method might be unstable. alpha*sigma^2 = {alpha*sigma**2:.2f} > 1.0. Consider reducing dt or increasing dx.")

        for j in range(nt-2, -1, -1):
            for i in range(1, nx-1):
                a = 0.5 * sigma**2 * alpha
                b = gamma * beta
                v[i, j] = a * v[i+1, j+1] + (1 - 2*a - r*dt) * v[i, j+1] + (a - b) * v[i-1, j+1]
                if option_type == 'call':
                    v[i, j] = max(v[i, j], np.exp(x[i]) - K)
                else:
                    v[i, j] = max(v[i, j], K - np.exp(x[i]))
            
            if option_type == 'call':
                v[0, j] = 0
                v[nx-1, j] = (np.exp(x_max) - K) * np.exp(-r * (T - t[j])) # Approx
            else:
                v[0, j] = K * np.exp(-r * (T - t[j]))
                v[nx-1, j] = 0
            
    elif method == 'implicit':
        a = 0.5 * sigma**2 * alpha
        b = gamma * beta
        diagonals = [np.ones(nx-2)*(a-b), np.ones(nx-1)*(1+2*a+r*dt), np.ones(nx-2)*(a+b)]
        A = diags(diagonals, [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        for j in range(nt-2, -1, -1):
            rhs = v[1:nx, j+1].copy()
            if option_type == 'call':
                boundary_low = 0
                boundary_high = (np.exp(x_max) - K) * np.exp(-r * (T - t[j]))
            else:
                boundary_low = K * np.exp(-r * (T - t[j]))
                boundary_high = 0
                
            rhs[0] -= (a - b) * boundary_low
            rhs[-1] -= (a + b) * boundary_high # Need to handle upper boundary for implicit too if non-zero
            
            v[1:nx, j] = np.linalg.solve(A, rhs)
            v[0, j] = boundary_low
            v[nx-1, j] = boundary_high
            
            for i in range(nx):
                if option_type == 'call':
                    v[i, j] = max(v[i, j], np.exp(x[i]) - K)
                else:
                    v[i, j] = max(v[i, j], K - np.exp(x[i]))
                
    elif method == 'crank_nicolson':
        alpha /= 2
        beta /= 2
        a = 0.5 * sigma**2 * alpha
        b = gamma * beta
        
        A_diags = [np.ones(nx-2)*(-a+b), np.ones(nx-1)*(1+2*a+0.5*r*dt), np.ones(nx-2)*(-a-b)]
        A = diags(A_diags, [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        B_diags = [np.ones(nx-2)*(a-b), np.ones(nx-1)*(1-2*a-0.5*r*dt), np.ones(nx-2)*(a+b)]
        B = diags(B_diags, [-1, 0, 1], shape=(nx-1, nx-1)).toarray()
        
        for j in range(nt-2, -1, -1):
            rhs = B @ v[1:nx, j+1]
            
            if option_type == 'call':
                boundary_low = 0
                boundary_high = (np.exp(x_max) - K) * np.exp(-r * (T - t[j]))
            else:
                boundary_low = K * np.exp(-r * (T - t[j]))
                boundary_high = 0
            
            rhs[0] += (-a + b) * boundary_low
            rhs[-1] += (-a - b) * boundary_high
            
            v[1:nx, j] = np.linalg.solve(A, rhs)
            v[0, j] = boundary_low
            v[nx-1, j] = boundary_high
            for i in range(nx):
                if option_type == 'call':
                    v[i, j] = max(v[i, j], np.exp(x[i]) - K)
                else:
                    v[i, j] = max(v[i, j], K - np.exp(x[i]))

    prices = np.zeros(len(S0_range))
    for i, S0 in enumerate(S0_range):
        prices[i] = np.interp(np.log(S0), x, v[:, 0])
    return prices

def fd_log(K, sigma, T, r, dt, dx, S0_range, method='explicit', option_type='put'):
    return _fd_solver_log(K, sigma, T, r, dt, dx, S0_range, method, option_type)

def _fd_solver_bs(K, sigma, T, r, dt, dS, S0_range, method='explicit', option_type='put'):
    S_min = max(min(S0_range) - 50, 1.0)
    S_max = max(S0_range) + 100.0
    nS = int(round((S_max - S_min) / dS)) + 1
    S = np.linspace(S_min, S_max, nS)
    nt = int(round(T / dt)) + 1
    
    v = np.zeros((nS, nt))
    if option_type == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    v[:, -1] = payoff
    
    if method == 'explicit':
        # Stability check
        alpha_max = 0.5 * sigma**2 * S_max**2 * dt / dS**2
        if alpha_max > 0.5:
            print(f"Warning: Explicit method might be unstable. Max alpha = {alpha_max:.2f} > 0.5. Consider reducing dt or increasing dS.")

        for j in range(nt - 2, -1, -1):
            if option_type == 'call':
                v[0, j] = 0
                v[nS - 1, j] = S_max - K * np.exp(-r * (T - j*dt)) # Approx
            else:
                v[0, j] = K - S[0]
                v[nS - 1, j] = 0.0
                
            for i in range(1, nS - 1):
                Si = S[i]
                c1 = sigma**2 * Si**2 * dt / dS**2
                c2 = r * Si * dt / dS
                v[i, j] = (0.5*c1 - 0.5*c2)*v[i-1, j+1] + (1 - c1 - r*dt)*v[i, j+1] + (0.5*c1 + 0.5*c2)*v[i+1, j+1]
            v[:, j] = np.maximum(v[:, j], payoff)
            
    elif method == 'implicit':
        alpha = 0.5 * sigma**2 * S**2 * dt / dS**2
        beta = 0.5 * r * S * dt / dS
        
        lower = beta[1:nS-1] - alpha[1:nS-1]
        main = 1.0 + 2 * alpha[1:nS-1] + r * dt
        upper = -beta[1:nS-1] - alpha[1:nS-1]
        
        A = diags([lower[1:], main, upper[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
        
        for j in range(nt - 2, -1, -1):
            rhs = v[1:nS-1, j+1].copy()
            
            if option_type == 'call':
                boundary_low = 0
                boundary_high = S_max - K * np.exp(-r * (T - j*dt))
            else:
                boundary_low = K - S[0]
                boundary_high = 0
                
            rhs[0] -= lower[0] * boundary_low
            rhs[-1] -= upper[-1] * boundary_high
            
            v[1:nS-1, j] = np.linalg.solve(A, rhs)
            v[0, j] = boundary_low
            v[nS-1, j] = boundary_high
            v[:, j] = np.maximum(v[:, j], payoff)
            
    elif method == 'crank_nicolson':
        alpha = 0.25 * sigma**2 * S**2 * dt / dS**2
        beta = 0.25 * r * S * dt / dS
        
        m1_lower = beta[1:nS-1] - alpha[1:nS-1]
        m1_main = 1.0 + 2.0 * alpha[1:nS-1] + 0.5 * r * dt
        m1_upper = -beta[1:nS-1] - alpha[1:nS-1]
        M1 = diags([m1_lower[1:], m1_main, m1_upper[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
        
        m2_lower = -(beta[1:nS-1] - alpha[1:nS-1])
        m2_main = 1.0 - 2.0 * alpha[1:nS-1] - 0.5 * r * dt
        m2_upper = -(-beta[1:nS-1] - alpha[1:nS-1])
        M2 = diags([m2_lower[1:], m2_main, m2_upper[:-1]], [-1, 0, 1], shape=(nS-2, nS-2)).toarray()
        
        for j in range(nt - 2, -1, -1):
            rhs = M2 @ v[1:nS-1, j+1]
            
            if option_type == 'call':
                boundary_low = 0
                boundary_high = S_max - K * np.exp(-r * (T - j*dt))
            else:
                boundary_low = K - S[0]
                boundary_high = 0
            
            rhs[0] -= m1_lower[0] * boundary_low + m2_lower[0] * v[0, j+1]
            rhs[-1] -= m1_upper[-1] * boundary_high + m2_upper[-1] * v[nS-1, j+1]
            
            v[1:nS-1, j] = np.linalg.solve(M1, rhs)
            v[0, j] = boundary_low
            v[nS-1, j] = boundary_high
            v[:, j] = np.maximum(v[:, j], payoff)

    prices = np.zeros(len(S0_range))
    for i, S0 in enumerate(S0_range):
        prices[i] = np.interp(S0, S, v[:, 0])
    return prices

def fd_bs(K, sigma, T, r, dt, dS, S0_range, method='explicit', option_type='put'):
    return _fd_solver_bs(K, sigma, T, r, dt, dS, S0_range, method, option_type)
