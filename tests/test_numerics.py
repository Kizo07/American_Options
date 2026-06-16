import numpy as np
import pytest

import comp_methods as cm


def test_trinomial_tree_matches_binomial_put_and_black_scholes_call():
    tri_put = cm.trinomial_tree(100, 100, 0.05, 0.2, 1, 100, option_type="put")
    binomial_put = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 500, option_type="put")

    assert tri_put == pytest.approx(binomial_put, abs=0.03)

    tri_call = cm.trinomial_tree(100, 100, 0.05, 0.2, 1, 200, option_type="call")
    bs_call = cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="call")

    assert tri_call == pytest.approx(bs_call, abs=0.03)


def test_trinomial_tree_converges_instead_of_collapsing_to_zero():
    values = [
        cm.trinomial_tree(100, 100, 0.05, 0.2, 1, n, option_type="put")
        for n in (50, 100, 200)
    ]

    assert all(value > 5.9 for value in values)
    assert abs(values[-1] - values[-2]) < abs(values[1] - values[0])


def test_lsmc_handles_odd_paths_and_immediate_exercise():
    odd_price = cm.lsmc(100, 100, 0.05, 0.2, 1, 1001, 50, 3, rng=np.random.default_rng(42))
    assert odd_price > 0

    deep_itm = cm.lsmc(20, 100, 0.05, 0.2, 1, 1000, 50, 3, rng=np.random.default_rng(42))
    assert deep_itm >= 80


def test_lsmc_atm_price_is_close_to_binomial_with_fixed_rng():
    lsmc_price = cm.lsmc(100, 100, 0.05, 0.2, 1, 20_000, 50, 3, rng=np.random.default_rng(7))
    binomial_put = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 500, option_type="put")

    assert lsmc_price == pytest.approx(binomial_put, abs=0.35)


def test_lsmc_rejects_unknown_basis():
    with pytest.raises(ValueError):
        cm.lsmc(100, 100, 0.05, 0.2, 1, 1000, 50, 3, poly_type="chebyshev")


def test_simulate_path_starts_at_initial_stock_and_seed_is_reproducible():
    path_a = cm.simulate_path_S(100, 0.05, 0.2, 1, 5, rng=np.random.default_rng(1))
    path_b = cm.simulate_path_S(100, 0.05, 0.2, 1, 5, rng=np.random.default_rng(1))

    assert len(path_a) == 6
    assert path_a[0] == 100
    assert np.allclose(path_a, path_b)


def test_correlated_brownian_motion_rejects_invalid_rho():
    with pytest.raises(ValueError):
        cm.correlated_brownian_motion(1, 10, 3, 1.5)


def test_two_factor_truncation_methods_are_distinguishable():
    args = (100, -0.1, 0.05, 0.45, 0.1, 0.25, -0.5, 1, 5, 4)
    full = cm.two_factor_gbm(*args, method="full_truncation", rng=np.random.default_rng(5))
    partial = cm.two_factor_gbm(*args, method="partial_truncation", rng=np.random.default_rng(5))

    assert not np.allclose(full, partial)


def test_fd_log_matches_black_scholes_call_on_stable_grid():
    price = cm.fd_log(100, 0.2, 1, 0.05, 0.0001, 0.02, [100], method="crank_nicolson", option_type="call")
    bs_call = cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="call")

    assert price[0] == pytest.approx(bs_call, abs=0.08)


def test_finite_difference_rejects_invalid_method_and_unstable_explicit_grid():
    with pytest.raises(ValueError):
        cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], method="bad")

    with pytest.raises(ValueError):
        cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], method="explicit", option_type="put")
