import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationParams:
    V0: float = 20000.0
    mu: float = -0.1
    sigma: float = 0.2
    lambda1: float = 0.2
    gamma: float = -0.4
    L0: float = 22000.0
    T: int = 5
    r0: float = 0.055
    delta: float = 0.25
    lambda2: float = 0.4
    alpha: float = 0.7
    epsilon: float = 0.95
    N_sims: int = 100000
    dt: float = 1/252
    seed: int = 42

    @property
    def mortgage_rate(self):
        return (self.r0 + self.delta * self.lambda2) / 12

    @property
    def N_steps(self):
        return int(self.T / self.dt)

def get_loan_schedule_constants(p: SimulationParams):
    r = p.mortgage_rate
    n_months = int(p.T * 12)
    PMT = p.L0 * r / (1 - (1 + r)**(-n_months))
    return PMT / r, PMT / (r * (1 + r)**n_months), 1 + r, (p.epsilon - p.alpha) / p.T

def calculate_outstanding_loan(t, a, b, c):
    return a - b * c**(t * 12)

def calculate_trigger_ratio(t, alpha, beta):
    return alpha + beta * t

def evolve_asset_values(V, p: SimulationParams, active_mask):
    num_active = np.sum(active_mask)
    jumps = np.random.poisson(p.lambda1 * p.dt, size=num_active) > 0
    
    jump_indices = np.where(active_mask)[0][jumps]
    V[jump_indices] *= (1 + p.gamma)

    dW = np.random.normal(0, np.sqrt(p.dt), size=num_active)
    V[active_mask] *= np.exp((p.mu - 0.5 * p.sigma**2) * p.dt + p.sigma * dW)
    return V

def simulate_mortgage_default(params: SimulationParams):
    np.random.seed(params.seed)
    V = np.full(params.N_sims, params.V0, dtype=float)
    default_payoffs = np.zeros(params.N_sims)
    default_times = np.zeros(params.N_sims)
    active_sims = np.ones(params.N_sims, dtype=bool)
    
    a, b, c, beta = get_loan_schedule_constants(params)

    for step in range(params.N_steps):
        if not np.any(active_sims): break
        t = step * params.dt
        
        L_t = calculate_outstanding_loan(t, a, b, c)
        q_t = calculate_trigger_ratio(t, params.alpha, beta)
        V = evolve_asset_values(V, params, active_sims)
        
        new_defaults = (V <= q_t * L_t) & active_sims
        if np.any(new_defaults):
            payoff = np.maximum(L_t - params.epsilon * V[new_defaults], 0)
            default_payoffs[new_defaults] = payoff * np.exp(-params.r0 * t)
            default_times[new_defaults] = t
            active_sims[new_defaults] = False

    default_count = np.sum(default_times > 0)
    return (np.mean(default_payoffs), 
            default_count / params.N_sims, 
            np.sum(default_times) / default_count if default_count > 0 else 0)
