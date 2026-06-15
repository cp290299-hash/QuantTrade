import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_black_scholes_gamma():
    from app import black_scholes_gamma
    
    # 一般情況：gamma 應為正數且不太大
    gamma = black_scholes_gamma(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    assert gamma > 0
    assert gamma < 1
    
    # 到期日為 0 時 gamma 應為 0
    gamma_zero = black_scholes_gamma(S=100, K=100, T=0, r=0.05, sigma=0.2)
    assert gamma_zero == 0
    
    # 股價為 0 時 gamma 應為 0
    gamma_zero_price = black_scholes_gamma(S=0, K=100, T=0.5, r=0.05, sigma=0.2)
    assert gamma_zero_price == 0