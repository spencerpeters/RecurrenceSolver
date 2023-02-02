import numpy as np
from scipy.special import binom


def log_base_k(a, k):
    return np.log(a) / np.log(k)

def C_of(k, n, tau):
    assert k <= n
    C = 0 if tau == 0 else binom(n - k + tau - 1, tau - 1)
    return int(C)