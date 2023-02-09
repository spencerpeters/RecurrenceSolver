import numpy as np
from CoarseGrainedBasisReductionRecurrence import CoarseGrainedBasisReductionRecurrence
from StepType import StepType
from TreeNode import TreeNode


class ExplicitOracleBasisReductionRecurrence(CoarseGrainedBasisReductionRecurrence):
    # There is no k now
    def recurrence_function(self, current_indices):
        n, l, C_index = current_indices
        # print(n, l, C_index)
        # min_n = max(l, self.k)
        min_n = l
        low = self.lowest_index_reduces_to(C_index)
        # assert low < C_index or C_index == 1
        assert low < C_index or C_index == 0 or C_index == 1
        # left_term = self.objective_values[n - 1:min_n - 1:-1, l, C_index:low-1:-1]
        left_term = self.objective_values[n - 1:min_n - 1:-1, l, C_index:(None if low == 0 else low - 1):-1]

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
        # assert right_C_index < C_index
        assert C_index == 0 or right_C_index < C_index
        left_C_index = C_index - objective_index
        assert left_C_index <= C_index
        # assert np.isclose(self.get_log_value_of(left_C_index) * self.get_log_value_of(right_C_index),
        #                   self.get_log_value_of(C_index))
        #     f"""
        # Sum of C indices left + right {self.get_value_of(left_C_index)} + {self.get_value_of(right_C_index)}
        # is not equal to parent {self.get_value_of(C_index)}.
        # C_indices are: {left_C_index} {right_C_index} {C_index}
        # """

        # assert self.get_value_of(left_C_index) + self.get_value_of(right_C_index) == self.get_value_of(C_index), f"""
        #  Sum of C indices left + right {self.get_value_of(left_C_index)} + {self.get_value_of(right_C_index)}
        #  is not equal to parent {self.get_value_of(C_index)}.
        #  C_indices are: {left_C_index} {right_C_index} {C_index}
        #  """

        ### SANITY CHECK ###

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


        # Hack to prevent overwriting a base case.
        step_type = StepType.RECURSIVE
        # optimal_parameters = [optimal_left_parameters, optimal_right_parameters]
        if self.objective_values[current_indices] <= minimizer:
            minimizer = self.objective_values[current_indices]
            step_type = StepType[self.step_types[current_indices]]
            # optimal_parameters = []

        best_C_star = right_C_index
        return minimizer, [best_l_star, best_C_star], step_type
            # , \
            # optimal_parameters

    def indices_in_solve_order(self):
        return [((n, l, C), (n, n - l, C))
                for n in range(1, self.N.stop_index)
                # for C in range(1, self.C.stop_index)
                for C in range(0, self.C.stop_index)
                for l in range(1, n // 2 + 1)]

    def make_tree(self, parameters, depth=1e10):
        n, l, C = parameters
        if C == 0:
            t = TreeNode(parameters, self, [])
            t.step_type_value = StepType.SORTALLL
            return t
        else:
            return super().make_tree(parameters,depth)
