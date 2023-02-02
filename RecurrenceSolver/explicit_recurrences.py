from CoarseGrainedBasisReductionRecurrence import CoarseGrainedBasisReductionRecurrence
import explicit_oracle_base_cases
from BaseCaseTypes import BaseCaseTypes
from StepType import StepType

import numpy as np

class ExplicitReduction(CoarseGrainedBasisReductionRecurrence):

    def indices_in_solve_order(self):
        return [((n, l, C), (n, n - l, C))
                for n in range(1, self.N.stop_index)
                for C in range(1, self.C.stop_index)
                for l in range(1, n // 2 + 1)]

    # There is no k now
    def recurrence_function(self, current_indices):
        n, l, C_index = current_indices
        # print(n, l, C_index)
        # min_n = max(l, self.k)
        min_n = l
        low = self.lowest_index_reduces_to(C_index)
        assert low < C_index or C_index == 1
        left_term = self.objective_values[n - 1:min_n - 1:-1, l, C_index:low-1:-1]

        # print(left_term.shape)
        max_l_star = left_term.shape[0]
        l_stars = np.ogrid[:max_l_star] + 1

        zero_part = np.expand_dims(self.objective_values[n, 1:n - min_n + 1, 0], 1)
        rest = self.objective_values[n, 1:n - min_n+1, low:C_index]
        right_term = (l * np.concatenate([zero_part, rest], axis=1).T / (n - l_stars)).T

        objective = left_term + right_term

        best_l_index, objective_index = np.unravel_index(np.argmin(objective), objective.shape)
        best_l_star = best_l_index + 1

        # one_same_exponent = self.lowest_index_of_same_exponent(C_index)

        right_C_index = 0 if objective_index == 0 else \
                        objective_index + low - 1
        assert right_C_index < C_index
        left_C_index = C_index - objective_index
        assert left_C_index <= C_index
        assert self.get_value_of(left_C_index) + self.get_value_of(right_C_index) == self.get_value_of(C_index), f"""
        Sum of C indices left + right {self.get_value_of(left_C_index)} + {self.get_value_of(right_C_index)}
        is not equal to parent {self.get_value_of(C_index)}.
        C_indices are: {left_C_index} {right_C_index} {C_index}
        """

        optimal_left_parameters = (n - best_l_star, l, left_C_index)
        optimal_right_parameters = (n, best_l_star, right_C_index)

        minimizer = self.objective_values[optimal_left_parameters] + \
                    (l / (n - best_l_star)) * self.objective_values[optimal_right_parameters]



        assert np.isclose(minimizer, np.min(objective)), f"""
        Claimed minimizer is {minimizer} but
        min of objective is {np.min(objective)}.
        Objective is {objective}.
        Optimal parameters are {optimal_left_parameters}, {optimal_right_parameters}.
        {self.objective_values[optimal_left_parameters]}
        {self.objective_values[optimal_right_parameters]}
        """

        # TODO big hack
        if self.objective_values[current_indices] < minimizer:
            minimizer = self.objective_values[current_indices]

        best_C_star = self.get_value_of(right_C_index)
        return minimizer, [best_l_star, best_C_star], StepType.RECURSIVE, \
            [optimal_left_parameters, optimal_right_parameters]

class ExplicitReductionToHKZ(ExplicitReduction):

    def __init__(self, k, maxN, base, max_exponent):
        self.base_case_type = BaseCaseTypes.HKZ
        super().__init__(k, maxN, base, max_exponent)

    def base_cases_and_types(self):
        return explicit_oracle_base_cases.hkz_base_cases(self)

class ExplicitReductionToDSP(ExplicitReduction):
    def __init__(self, k, maxN, base, max_exponent):
        self.base_case_type = BaseCaseTypes.DSP
        super().__init__(k, maxN, base, max_exponent)

    # Base cases for DSP in dimension k, using our optimistic guess k^(l * (k - l) / (2 * (k - 1))) for the approximation factor.
    def base_cases_and_types(self):
        return explicit_oracle_base_cases.dsp_base_cases(self)