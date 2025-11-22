import numpy as np
from scipy.optimize import brentq

def numerix_prepayment_rate(age, r_market, WAC=0.08):
    r_incentive = np.maximum((WAC - r_market) * 100, 0)
    BU = 0.3 + 0.7 * np.minimum(age / 30, 1)
    SG = np.minimum(1, (age / 6) * np.exp(-age / 40))
    SY = 0.94 - 0.1 * np.minimum(age / 25, 1)
    return (r_incentive * BU * SG * SY) / 100

def _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional, N_sims=25000):
    np.random.seed(42)
    dt = 1/12
    N_months = 360
    
    monthly_wac = WAC / 12
    if monthly_wac == 0:
        monthly_payment = notional / N_months
    else:
        monthly_payment = notional * (monthly_wac * (1 + monthly_wac)**N_months) / ((1 + monthly_wac)**N_months - 1)

    r_paths = np.full(N_sims, r0)
    balance_paths = np.full(N_sims, float(notional))
    integral_r = np.zeros(N_sims)
    mbs_val = np.zeros(N_sims)
    io_val = np.zeros(N_sims)
    po_val = np.zeros(N_sims)
    sqrt_dt = np.sqrt(dt)
    
    for month in range(1, N_months + 1):
        active = balance_paths > 1e-6
        if not np.any(active): break
        
        curr_bal = balance_paths[active]
        curr_r = r_paths[active]
        
        integral_r[active] += curr_r * dt
        df = np.exp(-integral_r[active])
        
        dW = np.random.normal(0, 1.0, np.sum(active))
        r_paths[active] = np.maximum(curr_r + kappa * (r_bar - curr_r) * dt + sigma * np.sqrt(curr_r) * sqrt_dt * dW, 0)

        interest = curr_bal * monthly_wac
        principal = monthly_payment - interest
        prepay = curr_bal * numerix_prepayment_rate(month, r_paths[active], WAC)
        
        total_principal = np.minimum(principal + prepay, curr_bal)
        total_cf = interest + total_principal
        
        mbs_val[active] += total_cf * df
        io_val[active] += interest * df
        po_val[active] += total_principal * df
        
        balance_paths[active] -= total_principal

    return np.mean(mbs_val), np.mean(io_val), np.mean(po_val)

def mbs_pricing(r_bar=0.08, kappa=0.6, sigma=0.12, r0=0.078, WAC=0.08, notional=100000):
    return _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional)[0]

def compute_oas(market_price, r_bar, kappa, sigma, r0=0.078, WAC=0.08, notional=100000):
    def objective(spread):
        return mbs_pricing(r_bar + spread, kappa, sigma, r0, WAC, notional) - market_price
    try:
        return brentq(objective, -0.05, 0.05, xtol=1e-6, rtol=1e-6)
    except ValueError:
        return np.nan

def io_po_pricing(r_bar=0.08, kappa=0.6, sigma=0.12, r0=0.078, WAC=0.08, notional=100000):
    _, io, po = _run_mbs_simulation(r_bar, kappa, sigma, r0, WAC, notional)
    return io, po
