import comp_methods as cm
import numpy as np

def test_library():
    print("Testing Random Numbers...")
    rn = cm.lcg_uniform(123, 10)
    print(f"LCG: {rn}")
    
    print("\nTesting Black-Scholes...")
    bs_call = cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type='call')
    bs_put = cm.black_scholes(100, 100, 0.05, 0.2, 1, option_type='put')
    print(f"BS Call: {bs_call}")
    print(f"BS Put: {bs_put}")
    
    print("\nTesting Monte Carlo...")
    mc_price, mc_err = cm.mc_call_option(100, 100, 0.05, 0.2, 1, 10000)
    print(f"MC Call Price: {mc_price} +/- {mc_err}")
    
    print("\nTesting Binomial Tree...")
    bt_put = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 100, option_type='put')
    bt_call = cm.binomial_tree(100, 100, 0.05, 0.2, 1, 100, option_type='call')
    print(f"Binomial Tree Put: {bt_put}")
    print(f"Binomial Tree Call: {bt_call}")
    
    print("\nTesting Finite Difference...")
    fd_put = cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], option_type='put', method='crank_nicolson')
    fd_call = cm.fd_bs(100, 0.2, 1, 0.05, 0.01, 1, [100], option_type='call', method='crank_nicolson')
    print(f"FD Put: {fd_put}")
    print(f"FD Call: {fd_call}")
    
    print("\nTesting LSMC...")
    lsmc_price = cm.lsmc(100, 100, 0.05, 0.2, 1, 1000, 50, 3)
    print(f"LSMC Put Price: {lsmc_price}")

if __name__ == "__main__":
    test_library()
