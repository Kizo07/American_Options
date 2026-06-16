import numpy as np
import pytest

import comp_methods as cm


def test_black_scholes_reference_values():
    assert cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="call") == pytest.approx(
        10.450583572185565
    )
    assert cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type="put") == pytest.approx(
        5.573526022256971
    )


def test_binomial_tree_smoke_values():
    put = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 100, option_type="put")
    call = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 100, option_type="call")

    assert put == pytest.approx(6.08718954693357)
    assert call == pytest.approx(10.435445512498049)


def test_finite_difference_crank_nicolson_smoke_values():
    put = cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], option_type="put", method="crank_nicolson")
    call = cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], option_type="call", method="crank_nicolson")

    assert put == pytest.approx(np.array([6.08214561]))
    assert call == pytest.approx(np.array([10.44604968]))


def test_lsmc_smoke_value_with_fixed_seed():
    price = cm.lsmc(100, 100, 0.05, 0.2, 1, 1000, 50, 3, rng=np.random.default_rng(42))

    assert price == pytest.approx(6.238340937876154)


def test_core_package_import_does_not_require_example_dependencies():
    assert hasattr(cm, "black_scholes")
    assert hasattr(cm, "binomial_tree")
    assert not hasattr(cm, "BinomialTree")
