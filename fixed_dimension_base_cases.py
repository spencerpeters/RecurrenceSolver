import numpy as np
from utilities import log_base_k, common_base_cases, best_bound_on_lambda1
from StepType import StepType

# Everything should be in log_2 terms. Sheesh.

# def common_base_cases(k, maxN, max_runtime_parameter):
#     # Base cases common to all oracles.
#     base_cases = np.full((maxN, maxN, max_runtime_parameter), np.nan)
#     base_case_types = np.full((maxN, maxN, max_runtime_parameter), np.nan, dtype='U16')
#
#     N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]
#     # Numpy notation for working with indices.
#
#     # Spencer: let's make the LLL base case more realistic.
#     # base_cases[:, :, 0] = (N * np.minimum(L, N - L)).squeeze()
#     a = np.log(4/3) / np.log(2)
#
#     base_cases[:, :, 0] = (a/2) * (L * (N - L)).squeeze()
#     base_case_types[:, :, 0] = StepType.LLL.name
#
#     # print(log_approximation[:, :, 0])
#     # 0 oracle queries -> LLL -> log_approximation = n * l. Replaced l with min(l, n - l)
#
#     for i in range(maxN):
#         base_cases[i, i, :] = 0
#         base_case_types[i, i, :] = StepType.TRIVIAL.name
#
#     # l = n is free
#     # the "dual" l = 0 "dunnaevenmakeanysense" as Noah would say.
#     # So, we leave the corresponding entries as np.nan values.
#
#     return base_cases, base_case_types


def hkz_base_cases(k, maxN, max_runtime_parameter):
    assert False
    base_cases, base_case_types = common_base_cases(k, maxN, max_runtime_parameter)
    N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]

    for n in range(1, k + 1):
        base_cases[n, :, 1:] = np.minimum(L, n - L) * log_base_k(n, k) / 2
        base_case_types[n, :, 1:] = StepType.HKZ.name
        # Dimension n <= k, l != 1, C > 0 -> HKZ -> log_approximation = log_k(sqrt(n)) * l. (Might even be slightly better.)
        # HKZ includes SVP and "duality then SVP" as special cases.

    return base_cases, base_case_types

def svp_in_dim_k_only_base_cases(recurrence):
    # base_cases, base_case_types = common_base_cases(recurrence)
    maxN = recurrence.N.stop_index
    max_runtime_parameter = recurrence.C.stop_index
    k = recurrence.k
    base_cases, base_case_types = common_base_cases(recurrence)

    base_cases[k, :, :] = np.inf
    base_case_types[k, :, :] = StepType.STUCK.name
    # Dimension k, l != 1 -> infinity, out of luck! Not allowed to do anything.
    base_cases[k, 1, 1:] = np.log2(best_bound_on_lambda1(k))
    base_case_types[k, 1, 1:] = StepType.SVP.name
    # Dimension k, l = 1, C > 0 -> SVP oracle -> log_approximation = 1/2
    base_cases[k, k - 1, 1:] = np.log2(best_bound_on_lambda1(k))
    base_case_types[k, k - 1, 1:] = StepType.SVP.name
    # The dual

    return base_cases, base_case_types


def svp_and_lll_base_cases(k, maxN, max_runtime_parameter):
    assert False
    base_cases, base_case_types = common_base_cases(k, maxN, max_runtime_parameter)

    N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]

    base_cases[k, :, :] = k * np.minimum(L, k - L)
    base_case_types[k, :, :] = StepType.LLL.name 
    # Dimension k, l != 1, C = 0 -> LLL -> log_approximation = k * l.

    for n in range(1, k + 1):
        base_cases[n, 1, 1:] = log_base_k(n, k) / 2
        base_case_types[n, 1, 1:] = StepType.SVP.name
        # Dimension n <= k, l = 1, C > 0 -> SVP oracle -> log_approximation = log_k(n)/2
        base_cases[n, n - 1, 1:] = log_base_k(n, k) / 2
        base_case_types[n, n - 1, 1:] = StepType.SVP.name
        # The dual

    return base_cases, base_case_types

def svp_only_base_cases(recurrence):
    base_cases, base_case_types = common_base_cases(recurrence)
    maxN = recurrence.N.stop_index
    max_runtime_parameter = recurrence.C.stop_index
    k = recurrence.k
    N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]

    base_cases[k, :, :] = np.inf
    base_case_types[k, :, :] = StepType.STUCK.name
    # Dimension k, l != 1 -> infinity, out of luck! Not allowed to do anything. Not even LLL.

    for n in range(1, k + 1):
        # base_cases[n, 1, 1:] = (1 / 2) * np.log(k) / np.log(2)
        base_cases[n, 1, 1:] = np.log2(best_bound_on_lambda1(n))
        base_case_types[n, 1, 1:] = StepType.SVP.name
        # base_cases[n, n - 1, 1:] = (1 / 2) * np.log(k) / np.log(2)
        base_cases[n, n - 1, 1:] = np.log2(best_bound_on_lambda1(n))
        base_case_types[n, n - 1, 1:] = StepType.SVP.name

    return base_cases, base_case_types

def dsp_base_cases(k, maxN, max_runtime_parameter):
    assert False
    base_cases, base_case_types = common_base_cases(k, maxN, max_runtime_parameter)
    N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]

    base_cases[k, :, :] = L * (k - L) / (2 * (k - 1))
    base_case_types[k, :, :] = StepType.DSP.name
    # Dimension k, l -> DSP oracle -> log_approximation = l * (k - l) / (2 (k - 1))
    # Note: this is only our GUESS for the appropriate Rankin's constant!

    return base_cases, base_case_types