from Recurrence import Recurrence
import numpy as np
from abc import ABC

class OptimizingRecurrence(Recurrence, ABC):
    # Recurrence with backpointers (optimal choice must be made when applying recurrence)

    def __init__(self, parameters, optimizing_parameters=None):
        if optimizing_parameters is None:
            self.optimizing_parameters = []
        else:
            self.optimizing_parameters = optimizing_parameters

        super().__init__(parameters)

    def solve(self):
        for p in self.optimizing_parameters:
            p.optimal_values = np.full(self.shape(), np.nan, dtype="int64")
        super().solve()

    # def solve_step(self, current_indices):
    #     objective_value, optimal_parameter_indices, step_type = self.recurrence_function(current_indices)
    #     self.objective_values[current_indices] = objective_value
    #     self.step_types[current_indices] = step_type
    #     for i in range(len(self.optimizing_parameters)):
    #         self.optimizing_parameters[i].optimal_values[current_indices] = optimal_parameter_indices[i]