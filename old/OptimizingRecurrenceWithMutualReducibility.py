from OptimizingRecurrence import OptimizingRecurrence
import numpy as np
from abc import ABC
from StepType import StepType
from TreeNode import TreeNode

class OptimizingRecurrenceWithMutualReducibility(OptimizingRecurrence, ABC):

    def solve(self):
        for p in self.parameters:
            p.optimal_values = np.full(self.shape(), np.nan, dtype=int)
        super().solve()

    def solve_step(self, current_indices):
        # Now, current_indices is a list, containing tuples of indices that are mutually reducible.
        best_objective_value = np.inf
        best_parameter_indices = None
        best_index = None
        best_step_type = None
        for current_index_tuple in current_indices:
            current_objective_value, current_parameter_indices, current_step_type, _ = self.recurrence_function(
                current_index_tuple)
            if current_objective_value < best_objective_value:
                best_objective_value = current_objective_value
                best_parameter_indices = current_parameter_indices
                best_index = current_index_tuple
                best_step_type = current_step_type

        if best_index is not None:
            self.objective_values[best_index] = best_objective_value
            self.step_types[best_index] = best_step_type.name

            for i in range(len(self.optimizing_parameters)):
                assert not np.isnan(best_parameter_indices[i]).any()
                self.optimizing_parameters[i].optimal_values[best_index] = best_parameter_indices[i]

            for index_tuple in set(current_indices).difference({best_index}):
                self.objective_values[index_tuple] = best_objective_value
                self.step_types[index_tuple] = StepType.DUALITY.name
                for i in range(len(self.parameters)):  # was: self.optimizing_parameters
                    assert not np.isnan(best_index[i]).any()
                    self.parameters[i].optimal_values[index_tuple] = best_index[i]  # :0 stunning. so beautiful.
        else:
            for index_tuple in current_indices:
                self.objective_values[index_tuple] = np.inf
                self.step_types[index_tuple] = StepType.STUCK.name
                # for p in self.optimizing_parameters: p.optimal_values[current_indices] = np.nan

    def make_tree(self, parameters):
        children = [self.make_tree(p) for p in self.child_parameters(parameters)]
        return TreeNode(parameters, self, children)


    # def make_tree(self, parameter_values):
    #     if self.objective_values is None:
    #         print("Need to solve before making tree!")
    #         return
    #
    #     step_type = StepType(solution.step_types[parameter_values])
    #
    #     if step_type == StepType.STUCK:
    #         return NoSolutionNode(k, n, l, C, solution)
    #     if step_type == StepType.LLL:
    #         return LLLLeaf(k, n, l, C, solution)
    #     if step_type == step_type.SVP:
    #         return SVPLeaf(k, n, l, C, solution)
    #     if step_type == step_type.DSP:
    #         return DSPLeaf(k, n, l, C, solution)
    #     if step_type == step_type.HKZ:
    #         return HKZLeaf(k, n, l, C, solution)
    #
    #     next_step_is_dual = step_type == StepType.DUALITY
    #     # TODO!!! This absolutely is NOT generic.
    #
    #     if next_step_is_dual:
    #         working_l = n - l
    #     else:
    #         working_l = l
    #
    #     best_l_star = solution.l_star.optimal_[n, working_l, C])
    #     best_C_star = solution.C_stars[n, working_l, C])
    #
    #     left_n = n - best_l_star
    #     left_l = working_l
    #     left_C = C - best_C_star
    #     left_child = TreeNode.make_tree(solution, k, left_n, left_l, left_C)
    #
    #     right_n = n
    #     right_l = best_l_star
    #     right_C = best_C_star
    #     right_child = TreeNode.make_tree(solution, k, right_n, right_l, right_C)
    #
    #     normal_node = NormalNode(k, n, working_l, C, solution, left_child, right_child)
    #
    #     if next_step_is_dual:
    #         return DualityNode(k, n, l, C, solution, normal_node)
    #     else:
    #     return normal_node

    # What to do:
    # There should not be node classes of different types, since that screws up the genericity of it,
    # and duplicates information between step-types and node types.
    # Instead, the nodes should come only in leaf and base varieties, and have type labels.
    # How will I know how to generically reduce? The recurrence method should return the parameter indices that we are reducing to.
    # (In addition to the parameters that induced those indices--sometimes we might need additional parameters.)