import numpy as np
from .trees import crr_american_put

def display_results(price, std_err, method):
    print(f"{method} Estimate: {price:.6f} (Standard Error: {std_err:.6f})")

def delta_vs_stock(K, r, sigma, T, n, stock_range):
    deltas = []
    prices = []
    for S0 in stock_range:
        price, delta = crr_american_put(S0, K, r, sigma, T, n)
        prices.append(price)
        deltas.append(delta)
    return prices, deltas

def greeks_vs_time(S0, K, r, sigma, time_range, n):
    deltas = []
    thetas = []
    
    for T in time_range:
        price, delta = crr_american_put(S0, K, r, sigma, T, n)
        deltas.append(delta)
        
        if T + 1/365 in time_range:
            next_idx = np.where(np.isclose(time_range, T + 1/365))[0][0]
            next_price = crr_american_put(S0, K, r, sigma, time_range[next_idx], n)[0]
            theta = (next_price - price) * 365
        else:
            if T > 1/365:
                prev_T = max(0, T - 1/365)
                prev_price = crr_american_put(S0, K, r, sigma, prev_T, n)[0]
                theta = (price - prev_price) * 365
            else:
                theta = 0
        thetas.append(theta)
    
    return deltas, thetas

def vega_vs_stock(K, r, sigma, T, n, stock_range, d_sigma=0.01):
    vegas = []
    for S0 in stock_range:
        p1 = crr_american_put(S0, K, r, sigma, T, n)[0]
        p2 = crr_american_put(S0, K, r, sigma + d_sigma, T, n)[0]
        vegas.append((p2 - p1) / d_sigma)
    return vegas
