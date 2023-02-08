import numpy as np
from StepType import StepType

def log_base_k(a, k):
    return np.log(a) / np.log(k)

def common_base_cases(recurrence):
    # Base cases common to all oracles.
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.C.stop_index
    base_cases = np.full((maxN, maxN, maxCIndex), np.inf)
    base_case_types = np.full((maxN, maxN, maxCIndex), np.nan, dtype='U16')

    N, L, C = np.ogrid[:maxN, :maxN, :maxCIndex]

    # base_cases[:, :, 0] = (N * np.minimum(L, N - L)).squeeze()
    # base_case_types[:, :, 0] = StepType.LLL.name
    base_cases[:, :, :] = N * np.minimum(L, N - L)
    base_case_types[:, :, :] = StepType.LLL.name
    # 0 oracle queries -> LLL -> log_approximation = n * min(l, n - l)

    for i in range(maxN):
        base_cases[i, i, :] = 0
        base_case_types[i, i, :] = StepType.TRIVIAL.name

    # # l = n is free
    # # the "dual" l = 0 "dunnaevenmakeanysense" as Noah would say.
    # # So, we leave the corresponding entries as np.nan values.

    return base_cases, base_case_types