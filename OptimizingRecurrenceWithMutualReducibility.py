import numpy as np
from StepType import StepType
from TreeNode import TreeNode
from abc import ABC, abstractmethod
from tqdm import tqdm

class OptimizingRecurrenceWithMutualReducibility(ABC):

    def __init__(self, parameters, optimizing_parameters=None, debug=False):
        self.debug = debug
        if optimizing_parameters is None:
            self.optimizing_parameters = []
        else:
            self.optimizing_parameters = optimizing_parameters
        self.parameters = parameters
        self.objective_values = None
        self.step_types = None

    def shape(self):
        return tuple(p.stop_index for p in self.parameters)

    def __getitem__(self, sliceobj):
        return self.objective_values[sliceobj]

    def __len__(self):
        return len(self.objective_values)

    def with_base_cases(self, base_case_function):
        self.base_cases_and_types = base_case_function(self)

    @abstractmethod
    def indices_in_solve_order(self):
        pass

    @abstractmethod
    def child_parameters(self, parameters):
        pass

    @abstractmethod
    def recurrence_function(self, current_index_tuple):
        pass

    def solve(self):
        for p in self.parameters:
            p.optimal_values = np.full(self.shape(), np.nan, dtype=int)
        for p in self.optimizing_parameters:
            p.optimal_values = np.full(self.shape(), np.nan, dtype="int64")
        self.objective_values, self.step_types = self.base_cases_and_types()
        assert self.objective_values.shape == self.shape()
        assert self.step_types.shape == self.shape()
        for current_indices in tqdm(self.indices_in_solve_order()):
            # if not self.step_types[current_indices[0]] == 'nan':
            self.solve_step(current_indices)


    def solve_step(self, current_indices):
        # Now, current_indices is a list, containing tuples of indices that are mutually reducible.
        best_objective_value = np.inf
        best_parameter_indices = None
        best_index = None
        best_step_type = None
        for current_index_tuple in current_indices:
            current_objective_value, current_parameter_indices, current_step_type = self.recurrence_function(
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

    def make_tree(self, parameters, depth=1e10):
        if depth == 0:
            return TreeNode(parameters, self, [])
        children = [self.make_tree(p, depth - 1) for p in self.child_parameters(parameters)]
        return TreeNode(parameters, self, children)

    def __hash__(self):
        descriptor_list = [ord(c) for c in list(self.__class__.__name__)] + [p.stop_index for p in self.parameters]
        initial = 37
        modulus = 999331
        for element in descriptor_list:
            initial = initial * 37 ** element % modulus
        return initial
