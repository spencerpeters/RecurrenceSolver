import numpy as np
from StepType import StepType
from scipy.special import gamma as gamma_fn

def log_base_k(a, k):
    return np.log(a) / np.log(k)

def common_base_cases(recurrence):
    # Base cases common to all oracles.
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.C.stop_index
    base_cases = np.full((maxN, maxN, maxCIndex), np.inf)
    base_case_types = np.full((maxN, maxN, maxCIndex), np.nan, dtype='U16')

    # N, L, C = np.ogrid[:maxN, :maxN, :maxCIndex]
    N, L = np.ogrid[:maxN, :maxN]

    # base_cases[:, :, 0] = (N * np.minimum(L, N - L)).squeeze()
    # base_case_types[:, :, 0] = StepType.LLL.name
    a = np.log(4/3) / np.log(2)

    # base_cases[:, :, :] = N * np.minimum(L, N - L)
    base_cases[:, :, 0] = (a/4) * np.squeeze((L * (N - L)))
    base_case_types[:, :, 0] = StepType.LLL.name

    L, C = np.ogrid[:maxN, :maxCIndex]
    if recurrence.k is not None:
        base_cases[recurrence.k, :, :] = (a/4) * (L * (recurrence.k - L))
    # 0 oracle queries -> LLL -> log_approximation = n * min(l, n - l)

    # for i in range(maxN):
    #     base_cases[i, i, :] = 0
    #     base_case_types[i, i, :] = StepType.TRIVIAL.name

    # # l = n is free
    # # the "dual" l = 0 "dunnaevenmakeanysense" as Noah would say.
    # # So, we leave the corresponding entries as np.nan values.

    return base_cases, base_case_types

def blichfeldt_bound_on_lambda1(n):
  blichfeldt = (2 / np.pi) * gamma_fn(2 + n/2)**(2/n)
  return np.sqrt(blichfeldt)

def best_bound_on_lambda1(n):
  known_values_of_gamma_n_to_n = {
      1: 1,
      2: 4/3,
      3: 2,
      4: 4,
      5: 8,
      6: 64/3,
      7: 64,
      8: 256,
  }

  if n in known_values_of_gamma_n_to_n:
    return np.sqrt(known_values_of_gamma_n_to_n[n]**(1/n))
  else:
    return blichfeldt_bound_on_lambda1(n)