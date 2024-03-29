# This idea is dumb. The set doesn't reduce to itself in the way you'd like.
# import numpy as np
#
# from OptimizingRecurrenceWithMutualReducibility import OptimizingRecurrenceWithMutualReducibility
# from Parameter import Parameter
# from Parameters import Parameters
# from RecurrenceTypes import RecurrenceTypes
# from StepType import StepType
# from abc import ABC
#
# class CoarseGrainedBasisReductionRecurrence2(OptimizingRecurrenceWithMutualReducibility, ABC):
#
#     def __init__(self, k, max_N, base, max_exponent):
#         self.k = k
#         self.base = base
#         self.max_exponent = max_exponent
#         self.N = Parameter("N", max_N + 1)
#         self.l = Parameter("l", max_N + 1)
#         self.C = Parameter("C", (max_exponent + 1) * (base - 1) + 1)
#         self.l_star = Parameter("l_star")
#         self.C_star = Parameter("C_star")
#         self.recurrence_type = RecurrenceTypes.COARSE_GRAINED
#         parameters = Parameters([self.N, self.l, self.C])
#         optimizing_parameters = Parameters([self.l_star, self.C_star])
#         super().__init__(parameters, optimizing_parameters)
#
#     def get_C_index(self, row, column):
#         # row and column start at 0.
#
#         # First row is b^2 elements, from 0 to b^2 - 1
#         # Each subsequent row is b * (b - 1) elements,
#         # from b^(i+1) to b^(i+2) - b^i
#
#         # The logic is all the same, just based on the lowest nonzero index you can reduce to.
#         # If you're in the first row, that's 1.
#         # If you're at the start of a row, it's the start of the previous row. Otherwise it's the start of your row.
#         if row == 0:
#             return column
#         else:
#             return self.base ** 2 + (row - 1) * (self.base * (self.base - 1)) + column
#
#
#     def get_row(self, C_index):
#         if C_index < self.base ** 2:
#             return 0
#         else:
#             offset = C_index - self.base ** 2
#             return 1 + offset // (self.base * (self.base - 1))
#
#     def get_column(self, C_index):
#         if C_index < self.base ** 2:
#             return C_index
#         else:
#             offset = C_index - self.base ** 2
#             return offset % (self.base * (self.base - 1))
#
#     def get_value_of(self, C_index):
#         row = self.get_row(C_index)
#         column = self.get_column(C_index)
#         if row == 0:
#             return column
#         else:
#             return self.base ** (row + 1) + column * self.base ** row
#
#     def lowest_index_reduces_to(self, C_index):
#         assert C_index > 0
#         row = self.get_row(C_index)
#         if row == 0:
#             return 1
#         column = self.get_column(C_index)
#         if column == 0:
#             return self.get_C_index(row - 1, 0)
#         else:
#             return self.get_C_index(row, 0)
#
#     def recurrence_function(self, current_indices):
#         n, l, C_index = current_indices
#         # print(n, l, C_index)
#         min_n = max(l, self.k)
#         low = self.lowest_index_reduces_to(C_index)
#         assert low < C_index or C_index == 1
#         left_term = self.objective_values[n - 1:min_n - 1:-1, l, C_index:low-1:-1]
#
#         max_l_star = left_term.shape[0]
#         l_stars = np.ogrid[:max_l_star] + 1
#
#         zero_part = np.expand_dims(self.objective_values[n, 1:n - min_n + 1, 0], 1)
#         rest = self.objective_values[n, 1:n - min_n + 1, low:C_index]
#         right_term = (l * np.concatenate([zero_part, rest], axis=1).T / (n - l_stars)).T
#
#         objective = left_term + right_term
#
#         best_l_index, objective_index = np.unravel_index(np.argmin(objective), objective.shape)
#         best_l_star = best_l_index + 1
#
#         right_C_index = 0 if objective_index == 0 else \
#             objective_index + low - 1
#         assert right_C_index < C_index
#         left_C_index = C_index - objective_index
#         assert left_C_index <= C_index
#         assert self.get_value_of(left_C_index) + self.get_value_of(right_C_index) == self.get_value_of(C_index), f"""
#                 Sum of C indices left + right {self.get_value_of(left_C_index)} + {self.get_value_of(right_C_index)}
#                 is not equal to parent {self.get_value_of(C_index)}.
#                 C_indices are: {left_C_index} {right_C_index} {C_index}
#                 """
#
#         optimal_left_parameters = (n - best_l_star, l, left_C_index)
#         optimal_right_parameters = (n, best_l_star, right_C_index)
#
#         minimizer = self.objective_values[optimal_left_parameters] + \
#                     (l / (n - best_l_star)) * self.objective_values[optimal_right_parameters]
#
#         assert np.isclose(minimizer, np.min(objective)), f"""
#                 Claimed minimizer is {minimizer} but
#                 min of objective is {np.min(objective)}.
#                 Objective is {objective}.
#                 Optimal parameters are {optimal_left_parameters}, {optimal_right_parameters}.
#                 {self.objective_values[optimal_left_parameters]}
#                 {self.objective_values[optimal_right_parameters]}
#                 """
#
#         best_C_star = self.get_value_of(right_C_index)
#         return minimizer, [best_l_star, best_C_star], StepType.RECURSIVE, \
#             [optimal_left_parameters, optimal_right_parameters]
#
#     # just copied these three methods. Improve code reuse
#     def indices_in_solve_order(self):
#         return [((n, l, C), (n, n - l, C))
#                 for n in range(self.k + 1, self.N.stop_index)
#                 for C in range(1, self.C.stop_index)
#                 for l in range(1, n // 2 + 1)]
#
#     def max_runtime_parameter(self):
#         return self.C.stop_index
#
#     def child_parameters(self, parameters):
#         # print(parameters)
#         step_type = StepType[self.step_types[parameters]]
#         # print(step_type)
#         if step_type.is_base():
#             return []
#         if step_type.is_duality():
#             return [tuple(p.optimal_values[parameters] for p in self.parameters)]
#
#         n, l, C = parameters
#         l_star = self.l_star.optimal_values[parameters]
#         C_star = self.C_star.optimal_values[parameters]
#         return [(n - l_star, l, C - C_star), (n, l_star, C_star)]
