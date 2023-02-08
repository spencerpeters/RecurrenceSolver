from OptimizingRecurrenceWithMutualReducibility import OptimizingRecurrenceWithMutualReducibility
from StepType import StepType
from abc import ABC
from Parameter import Parameter
from Parameters import Parameters

class BasisReductionRecurrence(OptimizingRecurrenceWithMutualReducibility, ABC):

    def __init__(self, k, max_N, max_C, debug=False):
        self.k = k
        self.N = Parameter("N", max_N + 1)
        self.l = Parameter("l", max_N + 1)
        self.C = Parameter("C", max_C + 1)
        self.l_star = Parameter("l_star")
        self.C_star = Parameter("C_star")
        parameters = Parameters([self.N, self.l, self.C])
        optimizing_parameters = Parameters([self.l_star, self.C_star])
        super().__init__(parameters, optimizing_parameters, debug)

    def indices_in_solve_order(self):
        return [((n, l, C), (n, n - l, C))
                for n in range(self.k + 1, self.N.stop_index)
                for C in range(1, self.C.stop_index)
                for l in range(1, n // 2 + 1)]