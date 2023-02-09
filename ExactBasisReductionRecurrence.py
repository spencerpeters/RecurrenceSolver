import numpy as np

from BasisReductionRecurrence import BasisReductionRecurrence
from StepType import StepType
from abc import ABC


class ExactBasisReductionRecurrence(BasisReductionRecurrence, ABC):

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

        # optimal_left_parameters = (n - best_l_star, l, C - best_C_star)
        # optimal_right_parameters = (n, best_l_star, best_C_star)
        # minimizer = self.objective_values[optimal_left_parameters] + \
        #             (l / (n - best_l_star)) * self.objective_values[optimal_right_parameters]

        minimizer = np.min(objective)

        return minimizer, [best_l_star, best_C_star], StepType.RECURSIVE
            # , \
            # [optimal_left_parameters, optimal_right_parameters]

    def child_parameters(self, parameters):
        step_type = StepType[self.step_types[parameters]]
        if step_type.is_base():
            return []
        if step_type.is_duality():
            return [tuple(p.optimal_values[parameters] for p in self.parameters)]

        n, l, C = parameters
        l_star = self.l_star.optimal_values[parameters]
        C_star = self.C_star.optimal_values[parameters]
        return [(n - l_star, l, C - C_star), (n, l_star, C_star)]