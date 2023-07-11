import numpy as np
from StepType import StepType
from utilities import common_base_cases, best_bound_on_lambda1

# costs are expressed in log_2 terms
# LLL costs 0


def hkz_base_cases(recurrence):
    assert False
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.C.stop_index
    base_cases, base_case_types = common_base_cases(recurrence)

    for n in range(1,maxN):
        for l in range(1, n):
            for C_index in range(maxCIndex):
                logC = recurrence.get_log_value_of(C_index)
                if logC >= n:
                    base_cases[n, l, C_index] = np.log2(n) * min(l, n-l) / 2

                    base_case_types[n, l, C_index] = StepType.HKZ.name

    return base_cases, base_case_types


def svp_only_base_cases(recurrence):
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.C.stop_index
    base_cases, base_case_types = common_base_cases(recurrence)

    for n in range(1,maxN):
        for C_index in range(maxCIndex):
            logC = recurrence.get_log_value_of(C_index)
            if logC >= n:
                # base_cases[n, 1, C_index] = np.log2(n) / 2
                # base_cases[n, 1, C_index] = (1 / 2) * np.log(n) / np.log(2)
                base_cases[n, 1, C_index] = best_bound_on_lambda1(n)
                # base_cases[n, 1, C_index] = np.log2(n) / 2 + 1e-10 * (logC - n)

                base_case_types[n, 1, C_index] = StepType.SVP.name

    return base_cases, base_case_types

def dsp_base_cases_cost(recurrence, cost):
    # let's assume for now that this costs you 2^(nl) running time
    maxN = recurrence.N.stop_index
    maxCIndex = recurrence.C.stop_index
    base_cases, base_case_types = common_base_cases(recurrence)

    for n in range(1,maxN):
        for l in range(1, n):
            for C_index in range(maxCIndex):
                logC = recurrence.get_log_value_of(C_index)
                if logC >= cost(n, l):
                # if logC >= n:
                    base_cases[n, l, C_index] = np.log2(n) * l * (n - l) / (2 * (n - 1))
                    base_case_types[n, l, C_index] = StepType.DSP.name

    return base_cases, base_case_types

def dsp_base_cases_nl(recurrence):
    assert False
    return dsp_base_cases_cost(recurrence, cost=lambda n, l: n * l)

def dsp_base_cases_n(recurrence):
    assert False
    return dsp_base_cases_cost(recurrence, cost=lambda n, l: n)