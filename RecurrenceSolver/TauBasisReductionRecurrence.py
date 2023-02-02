import numpy as np

from OptimizingRecurrenceWithMutualReducibility import OptimizingRecurrenceWithMutualReducibility
from Parameter import Parameter
from Parameters import Parameters
from RecurrenceTypes import RecurrenceTypes
from StepType import StepType
from abc import ABC


class TauBasisReductionRecurrence(OptimizingRecurrenceWithMutualReducibility, ABC):

    def __init__(self, k, max_N, max_tau):
        self.k = k
        self.N = Parameter("N", max_N + 1)
        self.l = Parameter("l", max_N + 1)
        self.tau = Parameter("tau", max_tau + 1)
        self.l_star = Parameter("l_star")
        self.recurrence_type = RecurrenceTypes.TAU
        parameters = Parameters([self.N, self.l, self.tau])
        optimizing_parameters = Parameters([self.l_star])
        super().__init__(parameters, optimizing_parameters)

    def max_runtime_parameter(self):
        return self.tau.stop_index

    def indices_in_solve_order(self):
        return [((n, l, tau), (n, n - l, tau))
                for n in range(self.k + 1, self.N.stop_index)
                for tau in range(1, self.tau.stop_index)
                for l in range(1, n // 2 + 1)]

    def recurrence_function(self, current_indices):
        n, l, tau = current_indices
        min_n = max(l, self.k)
        left_term = self.objective_values[n - 1:min_n - 1:-1, l, tau]

        max_l_star = left_term.shape[0]
        l_stars = np.ogrid[:max_l_star] + 1
        right_term = l * self.objective_values[n, 1:n - min_n + 1, tau - 1] / (n - l_stars)

        objective = left_term + right_term
        best_l_index = np.argmin(objective)
        minimizer = objective[best_l_index]
        best_l_star = best_l_index + 1

        optimal_left_parameters = (n - best_l_star, l, tau)
        optimal_right_parameters = (n, best_l_star, tau - 1)

        return minimizer, [best_l_star], StepType.RECURSIVE, [optimal_left_parameters, optimal_right_parameters]

    def child_parameters(self, parameters):
        # print(parameters)
        step_type = StepType[self.step_types[parameters]]
        if step_type.is_base():
            return []
        if step_type.is_duality():
            return [tuple(p.optimal_values[parameters] for p in self.parameters)]

        n, l, tau = parameters
        l_star = self.l_star.optimal_values[parameters]
        return [(n - l_star, l, tau), (n, l_star, tau - 1)]

