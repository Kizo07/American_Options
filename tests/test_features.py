import numpy as np
import pytest
from scipy.stats import norm

import comp_methods as cm


def test_continuous_dividend_black_scholes_matches_merton_formula():
    price = cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="call", q=0.03)
    d1 = (np.log(1.0) + (0.05 - 0.03 + 0.5 * 0.2**2)) / 0.2
    d2 = d1 - 0.2
    expected = 100 * np.exp(-0.03) * norm.cdf(d1) - 100 * np.exp(-0.05) * norm.cdf(d2)

    assert price == pytest.approx(expected)
    assert cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="call", q=0.0) == pytest.approx(
        10.450583572185565
    )


def test_tree_and_lsmc_accept_dividend_yield():
    no_div_call = cm.trinomial_tree(100, 100, 0.05, 0.2, 1, 150, option_type="call", q=0.0)
    div_call = cm.trinomial_tree(100, 100, 0.05, 0.2, 1, 150, option_type="call", q=0.03)

    assert div_call < no_div_call

    rng = np.random.default_rng(42)
    assert cm.lsmc(100, 100, 0.05, 0.2, 1, 1001, 50, 3, q=0.02, rng=rng) > 0


def test_flat_and_zero_curves_discount_and_forward_rates():
    flat = cm.FlatCurve(0.05)
    assert flat.discount(2.0) == pytest.approx(np.exp(-0.10))
    assert flat.forward_rate(1.0, 3.0) == pytest.approx(0.05)
    assert cm.black_scholes(100, 100, flat, 0.2, 1) == pytest.approx(
        cm.black_scholes(100, 100, 0.05, 0.2, 1)
    )

    curve = cm.ZeroCurve([0, 1, 2], [0.02, 0.03, 0.04])
    assert curve.zero_rate(1.5) == pytest.approx(0.035)
    assert curve.discount(2.0) == pytest.approx(np.exp(-0.08))


def test_black_scholes_greeks_and_implied_vol():
    greeks = cm.black_scholes_greeks(100, 100, 0.05, 0.2, 1, option_type="call", q=0.0)

    assert greeks["price"] == pytest.approx(cm.black_scholes(100, 100, 0.05, 0.2, 1))
    assert greeks["delta"] == pytest.approx(0.6368306511756191)
    assert greeks["gamma"] == pytest.approx(0.018762017345846895)

    price = cm.black_scholes(100, 100, 0.05, 0.27, 1, option_type="put", q=0.01)
    assert cm.implied_vol(price, 100, 100, 0.05, 1, q=0.01, option_type="put") == pytest.approx(0.27)


def test_monte_carlo_variance_reduction_and_sobol_normals():
    rng_plain = np.random.default_rng(7)
    rng_cv = np.random.default_rng(7)
    plain_price, plain_err = cm.mc_call_option(100, 100, 0.05, 0.2, 1, 20_000, rng=rng_plain)
    cv_price, cv_err = cm.mc_call_option(
        100, 100, 0.05, 0.2, 1, 20_000, rng=rng_cv, variance_reduction="control_variate"
    )

    bs = cm.black_scholes(100, 100, 0.05, 0.2, 1)
    assert abs(cv_price - bs) < abs(plain_price - bs)
    assert cv_err < plain_err
    assert cm.sobol_normals(16, 3, seed=1).shape == (16, 3)


def test_exotics_basic_relationships_and_digital_closed_form():
    rng_vanilla = np.random.default_rng(4)
    rng_barrier = np.random.default_rng(4)
    vanilla, _ = cm.mc_call_option(100, 100, 0.05, 0.2, 1, 10_000, rng=rng_vanilla)
    barrier, _ = cm.barrier_option_mc(100, 100, 0.05, 0.2, 1, barrier=90, n_paths=10_000, n_steps=50, rng=rng_barrier)

    assert barrier < vanilla

    asian, _ = cm.asian_option_mc(100, 100, 0.05, 0.2, 1, n_paths=10_000, n_steps=50, rng=np.random.default_rng(5))
    assert asian < vanilla

    digital = cm.digital_option_bs(100, 100, 0.05, 0.2, 1)
    d2 = (0.05 - 0.5 * 0.2**2) / 0.2
    assert digital == pytest.approx(np.exp(-0.05) * norm.cdf(d2))


def test_calibration_and_config_defaults():
    quotes = [
        {"S0": 100, "K": 90, "r": 0.05, "T": 1, "price": cm.black_scholes(100, 90, 0.05, 0.23, 1)},
        {"S0": 100, "K": 100, "r": 0.05, "T": 1, "price": cm.black_scholes(100, 100, 0.05, 0.23, 1)},
        {"S0": 100, "K": 110, "r": 0.05, "T": 1, "price": cm.black_scholes(100, 110, 0.05, 0.23, 1)},
    ]
    fit = cm.calibrate_vol_surface(quotes, initial_sigma=0.2)

    assert fit["success"]
    assert fit["sigma"] == pytest.approx(0.23)

    config = cm.MBSConfig(N_sims=100)
    assert cm.mbs_pricing(config=config) > 0
    assert all(value >= 0 for value in cm.heston_down_out_put(config=cm.HestonBarrierConfig(N_sims=100)))
