import numpy as np

from OptimizingRecurrenceWithMutualReducibility import OptimizingRecurrenceWithMutualReducibility
from Parameter import Parameter
from Parameters import Parameters
from RecurrenceTypes import RecurrenceTypes
from StepType import StepType
from abc import ABC


class ExactBasisReductionRecurrence(OptimizingRecurrenceWithMutualReducibility, ABC):

    def __init__(self, k, max_N, max_C, debug=True):
        self.k = k
        self.N = Parameter("N", max_N + 1)
        self.l = Parameter("l", max_N + 1)
        self.C = Parameter("C", max_C + 1)
        self.l_star = Parameter("l_star")
        self.C_star = Parameter("C_star")
        self.debug = debug
        self.recurrence_type = RecurrenceTypes.EXACT
        parameters = Parameters([self.N, self.l, self.C])
        optimizing_parameters = Parameters([self.l_star, self.C_star])
        super().__init__(parameters, optimizing_parameters)

    def indices_in_solve_order(self):
        return [((n, l, C), (n, n - l, C))
                for n in range(self.k + 1, self.N.stop_index)
                for C in range(1, self.C.stop_index)
                for l in range(1, n // 2 + 1)]

    def max_runtime_parameter(self):
        return self.C.stop_index

    def recurrence_function(self, current_indices):
        n, l, C = current_indices
        min_n = max(l, self.k)
        left_term = self.objective_values[n - 1:min_n - 1:-1, l, C:0:-1]
        # left_term.shape = (n - min_n, C)
        # left_term[i, j] = log_approximation[n - 1 - i, l, C - j].
        # i = 0 -> n' = n - 1. i = -1 -> n' = min_n. j = 0 -> C' = C. j = -1 -> C' = 1. Check.
        # Allowed to choose n = l, which is certainly possible, but probably bad.

        if self.debug:
            assert not np.isnan(left_term).any(), f"""
            {str((n, l, C))}
            {str(left_term)}
            {str(self.objective_values)}
                                                   """

        max_l_star, max_c_star = left_term.shape[0], left_term.shape[1]
        l_indices, _ = np.ogrid[:max_l_star, :max_c_star]
        l_stars = l_indices + 1
        # n' = n - (i + 1) -> l* = i + 1
        # C' = C - j -> C* = j

        right_term = l * self.objective_values[n, 1:n - min_n + 1, 0:C] / (n - l_stars)
        # right_term.shape = (n - min_n, C), check.
        # right_term[i, j] = log_approximation[n, i + 1, j]
        # i = 0 -> l* = 1; i = -1 -> l* = n - min_n. j = 0 -> C' = 0; j = -1 -> C' = C - 1. Check.
        # We are NOT allowing l* = 0.

        objective = left_term + right_term

        ##### OPTIMIZE THEM #####

        best_l_index, best_C_star = np.unravel_index(np.argmin(objective), objective.shape)
        # Get l_star, C_star attaining minimum approx factor
        if self.debug:
            assert objective[best_l_index, best_C_star] == np.min(objective), f"""
                    Claimed minimizer is " + {objective[best_l_index, best_C_star]} 
                    Actual minimizer is " + {np.min(objective)}
                    {self.objective_values}
                    """

        best_l_star = best_l_index + 1
        # Since l* = index + 1

        optimal_left_parameters = (n - best_l_star, l, C - best_C_star)
        optimal_right_parameters = (n, best_l_star, best_C_star)
        minimizer = self.objective_values[optimal_left_parameters] + \
                    (l / (n - best_l_star)) * self.objective_values[optimal_right_parameters]

        return minimizer, [best_l_star, best_C_star], StepType.RECURSIVE, \
            [optimal_left_parameters, optimal_right_parameters]

    def child_parameters(self, parameters):
        # print(parameters)
        step_type = StepType[self.step_types[parameters]]
        # print(step_type)
        if step_type.is_base():
            return []
        if step_type.is_duality():
            return [tuple(p.optimal_values[parameters] for p in self.parameters)]

        n, l, C = parameters
        l_star = self.l_star.optimal_values[parameters]
        C_star = self.C_star.optimal_values[parameters]
        return [(n - l_star, l, C - C_star), (n, l_star, C_star)]
