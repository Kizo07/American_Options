import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm

def cir_zero_bond_price(r, T, kappa=0.92, r_bar=0.055, sigma=0.12):
    if T <= 0: return 1.0
    gamma = np.sqrt(kappa**2 + 2*sigma**2)
    B = (2 * (np.exp(gamma * T) - 1)) / ((kappa + gamma) * (np.exp(gamma * T) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((kappa + gamma) * T / 2)) / ((kappa + gamma) * (np.exp(gamma * T) - 1) + 2 * gamma))**(2 * kappa * r_bar / sigma**2)
    return A * np.exp(-B * r)

def cir_bond_price_mc(r0=0.05, sigma=0.12, kappa=0.92, r_bar=0.055, T=4, N_sims=50000):
    dt = 1/252
    N_steps = int(T / dt)
    coupon_times = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    coupons = np.array([30, 30, 30, 30, 30, 30, 30, 1030])
    
    r_paths = np.full(N_sims, r0)
    integral_r = np.zeros(N_sims)
    bond_values = np.zeros(N_sims)
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        t = step * dt
        if t >= T: break
        
        dW = np.random.normal(0, sqrt_dt, N_sims)
        r_pos = np.maximum(r_paths, 0)
        r_paths += kappa * (r_bar - r_paths) * dt + sigma * np.sqrt(r_pos) * dW
        r_paths = np.maximum(r_paths, 0)
        integral_r += r_paths * dt
        
        for i, ct in enumerate(coupon_times):
            if abs(t - ct) < dt/2:
                bond_values += coupons[i] * np.exp(-integral_r)
    
    return np.mean(bond_values)

def cir_call_option_mc(r0=0.05, T=0.5, S=1, K=980, sigma=0.12, kappa=0.92, r_bar=0.055, N_sims=50000):
    dt = 1/252
    N_steps = int(T / dt)
    r_paths = np.full(N_sims, r0)
    integral_r = np.zeros(N_sims)
    sqrt_dt = np.sqrt(dt)
    
    for step in range(N_steps):
        dW = np.random.normal(0, sqrt_dt, N_sims)
        r_pos = np.maximum(r_paths, 0)
        r_paths += kappa * (r_bar - r_paths) * dt + sigma * np.sqrt(r_pos) * dW
        r_paths = np.maximum(r_paths, 0)
        integral_r += r_paths * dt
    
    bond_prices = np.array([cir_zero_bond_price(r, S - T, kappa, r_bar, sigma) * 1000 for r in r_paths])
    return np.mean(np.maximum(bond_prices - K, 0) * np.exp(-integral_r))

def cir_call_option_pde(r0=0.05, T=0.5, S=1, K=980, sigma=0.12, kappa=0.92, r_bar=0.055):
    r_max, r_min, N_r, N_t = 0.5, 0.0, 200, 1000
    dr = (r_max - r_min) / N_r
    dt = T / N_t
    r_grid = np.linspace(r_min, r_max, N_r + 1)
    
    option_values = np.zeros((N_t + 1, N_r + 1))
    bond_prices = cir_zero_bond_price(r_grid, S - T, kappa, r_bar, sigma) * 1000
    option_values[N_t, :] = np.maximum(bond_prices - K, 0)
    
    ir = r_grid[1:-1]
    aj = (dt / 2) * (kappa * (r_bar - ir) / dr - (sigma**2 * ir) / dr**2)
    bj = 1 + dt * ((sigma**2 * ir) / dr**2 + ir)
    cj = (dt / 2) * (-kappa * (r_bar - ir) / dr - (sigma**2 * ir) / dr**2)

    for i in range(N_t - 1, -1, -1):
        known_b = option_values[i + 1, 1:-1]
        known_b[0] -= aj[0] * option_values[i + 1, 0]
        known_b[-1] -= cj[-1] * option_values[i + 1, -1]
        
        banded_M = np.vstack([np.append(0, cj[:-1]), bj, np.append(aj[1:], 0)])
        option_values[i, 1:-1] = solve_banded((1, 1), banded_M, known_b)
        option_values[i, 0] = option_values[i, 1]
        option_values[i, -1] = option_values[i, -2]

    return np.interp(r0, r_grid, option_values[0, :])

def g2pp_zero_bond_price(t, T, x, y, phi, a=0.1, b=0.3, sigma=0.05, eta=0.09, rho=0.7):
    tau = T - t
    if tau <= 1e-10: return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
    
    B_x = (1 - np.exp(-a * tau)) / a
    B_y = (1 - np.exp(-b * tau)) / b
    
    v_x = (sigma**2 / (2 * a**2)) * (tau - 2 * B_x + (1 - np.exp(-2 * a * tau)) / (2 * a))
    v_y = (eta**2 / (2 * b**2)) * (tau - 2 * B_y + (1 - np.exp(-2 * b * tau)) / (2 * b))
    v_xy = (rho * sigma * eta / (a * b)) * (tau - B_x - B_y + (1 - np.exp(-(a + b) * tau)) / (a + b))
    
    return np.exp(v_x + v_y + v_xy - B_x * x - B_y * y - phi * tau)

def g2pp_put_option_mc(T=0.5, S=1, K=950, rho=0.7, N_sims=100000):
    x0, y0, phi0 = 0.0, 0.0, 0.055
    a, b, sigma, eta = 0.1, 0.3, 0.05, 0.09
    dt = 1/252
    N_steps = int(T / dt)
    
    x_paths = np.full(N_sims, x0)
    y_paths = np.full(N_sims, y0)
    integral_r = np.zeros(N_sims)
    sqrt_dt = np.sqrt(dt)
    sqrt_rho = np.sqrt(1 - rho**2)
    
    for _ in range(N_steps):
        dW1 = np.random.normal(0, sqrt_dt, N_sims)
        dW2 = rho * dW1 + sqrt_rho * np.random.normal(0, sqrt_dt, N_sims)
        
        x_paths += -a * x_paths * dt + sigma * dW1
        y_paths += -b * y_paths * dt + eta * dW2
        integral_r += (x_paths + y_paths + phi0) * dt
    
    bond_prices = g2pp_zero_bond_price(T, S, x_paths, y_paths, phi0, a, b, sigma, eta, rho) * 1000
    return np.mean(np.maximum(K - bond_prices, 0) * np.exp(-integral_r))

def g2pp_put_option_explicit(T=0.5, S=1, K=950, rho=0.7):
    a, b, sigma, eta = 0.1, 0.3, 0.05, 0.09
    phi0 = 0.055
    
    P_0T = g2pp_zero_bond_price(0, T, 0, 0, phi0, a, b, sigma, eta, rho)
    P_0S = g2pp_zero_bond_price(0, S, 0, 0, phi0, a, b, sigma, eta, rho)
    
    B_x = (1 - np.exp(-a * (S - T))) / a
    B_y = (1 - np.exp(-b * (S - T))) / b
    
    var = (sigma**2/(2*a))*(1-np.exp(-2*a*T))*B_x**2 + \
          (eta**2/(2*b))*(1-np.exp(-2*b*T))*B_y**2 + \
          2*(rho*sigma*eta/(a+b))*(1-np.exp(-(a+b)*T))*B_x*B_y
          
    v = np.sqrt(var)
    d1 = (np.log(P_0S / ((K/1000) * P_0T)) / v) + 0.5 * v
    d2 = d1 - v
    
    return K * P_0T * norm.cdf(-d2) - 1000 * P_0S * norm.cdf(-d1)
