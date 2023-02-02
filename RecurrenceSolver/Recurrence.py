from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np

class Recurrence(ABC):
    # Generic setup for solving a recurrence numerically

    def __init__(self, parameters):
        self.parameters = parameters
        self.objective_values = None
        self.step_types = None

    def shape(self):
        return tuple(p.stop_index for p in self.parameters)

    def __getitem__(self, sliceobj):
        return self.objective_values[sliceobj]

    def __len__(self):
        return len(self.objective_values)

    @abstractmethod
    def base_cases_and_types(self):
        pass

    @abstractmethod
    def indices_in_solve_order(self):
        pass

    @abstractmethod
    def recurrence_function(self, current_indices):
        pass

    @abstractmethod
    def child_parameters(self, parameters):
        pass

    def solve(self):
        self.objective_values, self.step_types = self.base_cases_and_types()
        assert self.objective_values.shape == self.shape()
        assert self.step_types.shape == self.shape()
        for current_indices in tqdm(self.indices_in_solve_order()):
            # if not self.step_types[current_indices[0]] == 'nan':
            self.solve_step(current_indices)

    @abstractmethod
    def solve_step(self, current_indices):
        pass

    def __hash__(self):
        descriptor_list = [ord(c) for c in list(self.__class__.__name__)] + [p.stop_index for p in self.parameters]
        initial = 37
        modulus = 999331
        for element in descriptor_list:
            initial = initial * 37 ** element % modulus
        return initial



