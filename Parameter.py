import numpy as np
from collections.abc import Iterable
from scipy.special import binom

class Parameter:

    def __init__(self, name, stop_index=None):
        self.name = name
        self.stop_index = stop_index
        self.optimal_values = None