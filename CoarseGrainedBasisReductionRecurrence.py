import numpy as np

from BasisReductionRecurrence import BasisReductionRecurrence
from Parameter import Parameter
from Parameters import Parameters
from StepType import StepType
from abc import ABC

class CoarseGrainedBasisReductionRecurrence(BasisReductionRecurrence, ABC):

    def __init__(self, k, max_N, base, max_exponent, debug=False):
        self.base = base
        self.max_exponent = max_exponent
        max_C = (max_exponent + 1) * (base - 1) + 1
        super().__init__(k, max_N, max_C, debug)

    def recurrence_function(self, current_indices):
        n, l, C_index = current_indices
        # print(n, l, C_index)
        min_n = max(l, self.k)
        low = self.lowest_index_reduces_to(C_index)
        assert low < C_index or C_index == 1
        left_term = self.objective_values[n - 1:min_n - 1:-1, l, C_index:low - 1:-1]

        # print(left_term.shape)
        max_l_star = left_term.shape[0]
        l_stars = np.ogrid[:max_l_star] + 1

        zero_part = np.expand_dims(self.objective_values[n, 1:n - min_n + 1, 0], 1)
        rest = self.objective_values[n, 1:n - min_n + 1, low:C_index]
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

        minimizer = np.min(objective)

        ### SANITY CHECK ###

        optimal_left_parameters = (n - best_l_star, l, left_C_index)
        optimal_right_parameters = (n, best_l_star, right_C_index)
        recomputed_minimizer = self.objective_values[optimal_left_parameters] + \
                    (l / (n - best_l_star)) * self.objective_values[optimal_right_parameters]

        assert np.isclose(minimizer, np.min(objective)), f"""
         Claimed minimizer is {minimizer} but
         min of objective is {np.min(objective)}.
         Objective is {objective}.
         Optimal parameters are {optimal_left_parameters}, {optimal_right_parameters}.
         {self.objective_values[optimal_left_parameters]}
         {self.objective_values[optimal_right_parameters]}
         """

        # best_C_star = self.get_value_of(right_C_index)
        return minimizer, [best_l_star, right_C_index], StepType.RECURSIVE
            # , \
            # [optimal_left_parameters, optimal_right_parameters]

    def get_C_index(self, exponent, digit):
        return digit + exponent * (self.base - 1)

    def get_exponent(self, C_index):
        if C_index == 0:
            return 0
        exponent = (C_index - 1) // (self.base - 1)
        return exponent

    def get_digit(self, C_index):
        if C_index == 0:
            return 0
        return (C_index - 1) % (self.base - 1) + 1

    def get_value_of(self, C_index):
        exponent = self.get_exponent(C_index)
        digit = self.get_digit(C_index)
        return digit * self.base**exponent

    def get_log_value_of(self, C_index):
        if C_index == 0:
            return -np.inf
        exponent = self.get_exponent(C_index)
        digit = self.get_digit(C_index)
        return np.log2(digit) + exponent * np.log2(self.base)

    def lowest_index_of_same_exponent(self, C_index):
        exponent = self.get_exponent(C_index)
        return self.get_C_index(exponent, 1)

    def lowest_index_reduces_to(self, C_index):
        # assert C_index > 0
        if C_index == 0:
            return 0
        if C_index <= self.base:
            return 1
        else:
            exponent = self.get_exponent(C_index)
            digit = self.get_digit(C_index)
            if digit == 1:
                c = self.get_C_index(exponent - 1, 1)
                return c
            return self.get_C_index(exponent, 1)

    def child_parameters(self, parameters):
        step_type = StepType[self.step_types[parameters]]
        if step_type.is_base():
            return []
        if step_type.is_duality():
            return [tuple(p.optimal_values[parameters] for p in self.parameters)]

        n, l, C_index = parameters
        l_star = self.l_star.optimal_values[parameters]
        right_C_index = self.C_star.optimal_values[parameters]
        left_C_index = None
        if right_C_index == 0:
            left_C_index = C_index
        else:
            low = self.lowest_index_reduces_to(C_index)
            objective_index = right_C_index - low + 1
            left_C_index = C_index - objective_index

        assert self.get_value_of(left_C_index) + self.get_value_of(right_C_index) == self.get_value_of(C_index), f"""
                Sum of C indices left + right {self.get_value_of(left_C_index)} + {self.get_value_of(right_C_index)}
                is not equal to parent {self.get_value_of(C_index)}.
                C_indices are: {left_C_index} {right_C_index} {C_index}
                """

        return [(n - l_star, l, left_C_index), (n, l_star, right_C_index)]


