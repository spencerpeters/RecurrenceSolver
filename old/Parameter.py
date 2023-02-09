import numpy as np
from collections.abc import Iterable
from scipy.special import binom

class Parameter:

    def __init__(self, name, stop_index=None):
        self.name = name
        self.stop_index = stop_index
        self.optimal_values = None


# class Parameter:
#     # Essentially a mapping of table indices to parameter values.
#
#     def __init__(self, name, stop_index=None):
#         self.name = name
#         self.stop_index = stop_index
#         if self.stop_index is not None:
#             self.values = np.array([self.index_to_value(i) for i in range(self.stop_index)])
#
#         self.vtoi = self.value_to_index
#         self.itov = self.index_to_value
#
#         self.i = self.indices
#
#     def index_to_value(self, index):
#         return index
#
#     def value_to_index(self, index):
#         assert index < self.stop_index
#         return self.values[index]
#
#     def value_slice_to_index_slice(self, sliceobj):
#         return slice(self.vtoi(sliceobj.start), self.vtoi(sliceobj.stop), sliceobj.step)
#
#     def _index(self, obj):
#         if isinstance(obj, int):
#             return self.value_to_index(obj)
#         if isinstance(obj, slice):
#             return self.value_slice_to_index_slice(obj)
#         assert False, "Unexpected type for parameter index."
#
#     def indices(self, obj):
#         if isinstance(obj, Iterable):
#             return tuple(self._index(element) for element in obj)
#         else:
#             return self._index(obj)
#
#
# class TauParameter(Parameter):
#
#     def __init__(self, name, stop_index, k, n):
#         self.k = k
#         super().__init__(name, stop_index)
#
#     def index_to_value(self, tau):
#         return np.binom(self.stop_index - self.k, tau - 1)