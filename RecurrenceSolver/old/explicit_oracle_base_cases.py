import numpy as np
from utilities import log_base_k
from StepType import StepType

# costs are expressed in log_2 terms (weird I know)
# LLL costs 0
# Also fuck setting base cases up quickly

def common_base_cases(recurrence):
    # Base cases common to all oracles.
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.max_runtime_parameter()
    base_cases = np.full((maxN, maxN, maxCIndex), np.inf)
    base_case_types = np.full((maxN, maxN, maxCIndex), np.nan, dtype='U16')

    N, L, C = np.ogrid[:maxN, :maxN, :maxCIndex]
    # Numpy notation for working with indices.


    base_cases[:, :, 0] = (N * np.minimum(L, N - L)).squeeze()
    base_case_types[:, :, 0] = StepType.LLL.name

    # print(log_approximation[:, :, 0])
    # 0 oracle queries -> LLL -> log_approximation = n * l. Replaced l with min(l, n - l)

    for i in range(maxN):
        base_cases[i, i, :] = 0
        base_case_types[i, i, :] = StepType.TRIVIAL.name
    #
    # # l = n is free
    # # the "dual" l = 0 "dunnaevenmakeanysense" as Noah would say.
    # # So, we leave the corresponding entries as np.nan values.

    return base_cases, base_case_types


def hkz_base_cases(recurrence):
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.max_runtime_parameter()
    base_cases, base_case_types = common_base_cases(recurrence)
    # N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]
    #
    # for n in range(1, k + 1):
    #     base_cases[n, :, 1:] = np.minimum(L, n - L) * log_base_k(n, k) / 2
    #     base_case_types[n, :, 1:] = StepType.HKZ.name
        # Dimension n <= k, l != 1, C > 0 -> HKZ -> log_approximation = log_k(sqrt(n)) * l. (Might even be slightly better.)
        # HKZ includes SVP and "duality then SVP" as special cases.

    for n in range(1,maxN):
        for l in range(1, n):
            for C_index in range(maxCIndex):
                logC = recurrence.get_log_value_of(C_index)
                if logC >= n:
                    # base_cases[n, l, C_index] = min(l,n-l) * log_base_k(n,k) / 2
                    base_cases[n, l, C_index] = np.log2(n) * min(l, n-l) / 2

                    base_case_types[n, l, C_index] = StepType.HKZ.name

    return base_cases, base_case_types

# def svp_in_dim_k_only_base_cases(recurrence):
#     base_cases, base_case_types = common_base_cases(k, maxN, max_runtime_parameter)
#
#     base_cases[k, :, :] = np.inf
#     base_case_types[k, :, :] = StepType.STUCK.name
#     # Dimension k, l != 1 -> infinity, out of luck! Not allowed to do anything.
#     base_cases[k, 1, 1:] = 1 / 2
#     base_case_types[k, 1, 1:] = StepType.SVP.name
#     # Dimension k, l = 1, C > 0 -> SVP oracle -> log_approximation = 1/2
#     base_cases[k, k - 1, 1:] = 1 / 2
#     base_case_types[k, k - 1, 1:] = StepType.SVP.name
#     # The dual
#
#     return base_cases, base_case_types


# def svp_and_lll_base_cases(recurrence):
#     base_cases, base_case_types = common_base_cases(k, maxN, max_runtime_parameter)
#
#     N, L, C = np.ogrid[:maxN, :maxN, :max_runtime_parameter]
#
#     base_cases[k, :, :] = k * np.minimum(L, k - L)
#     base_case_types[k, :, :] = StepType.LLL.name
#     # Dimension k, l != 1, C = 0 -> LLL -> log_approximation = k * l.
#
#     for n in range(1, k + 1):
#         base_cases[n, 1, 1:] = log_base_k(n, k) / 2
#         base_case_types[n, 1, 1:] = StepType.SVP.name
#         # Dimension n <= k, l = 1, C > 0 -> SVP oracle -> log_approximation = log_k(n)/2
#         base_cases[n, n - 1, 1:] = log_base_k(n, k) / 2
#         base_case_types[n, n - 1, 1:] = StepType.SVP.name
#         # The dual
#
#     return base_cases, base_case_types

def svp_only_base_cases(recurrence):
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.max_runtime_parameter()
    base_cases, base_case_types = common_base_cases(recurrence)

    for n in range(1,maxN):
        for C_index in range(maxCIndex):
            logC = recurrence.get_log_value_of(C_index)
            # TODO hack
            if logC >= n:
            # if logC >= n:
                base_cases[n, 1, C_index] = np.log2(n) / 2
                base_case_types[n, 1, C_index] = StepType.SVP.name

    # base_cases[k, :, :] = np.inf
    # base_case_types[k, :, :] = StepType.STUCK.name
    # # Dimension k, l != 1 -> infinity, out of luck! Not allowed to do anything. Not even LLL.
    #
    # for n in range(1, k + 1):
    #     base_cases[n, 1, 1:] = log_base_k(n, k) / 2
    #     base_case_types[n, 1, 1:] = StepType.SVP.name
    #     base_cases[n, n - 1, 1:] = log_base_k(n, k) / 2
    #     base_case_types[n, n - 1, 1:] = StepType.SVP.name

    return base_cases, base_case_types

def dsp_base_cases(recurrence):
    # let's assume for now that this costs you 2^(nl) running time
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.max_runtime_parameter()
    base_cases, base_case_types = common_base_cases(recurrence)

    for n in range(1,maxN):
        for l in range(1, n):
            for C_index in range(maxCIndex):
                logC = recurrence.get_log_value_of(C_index)
                # TODO hack
                if logC >= n * l:
                # if logC >= n:
                    base_cases[n, l, C_index] = np.log2(n) * l * (n - l) / (2 * (n - 1))
                    base_case_types[n, l, C_index] = StepType.DSP.name

    return base_cases, base_case_types