# Idea: coarse-grained, fixed-dimension oracle recurrence
# Proceeds in 3 phases, using 3 different recursive steps ("gadgets")
# Not an optimizing recurrence, so just write the thing by hand? Use the coarseness code

# Try just splitting oracle calls in two in every call in every phase. Not great but acceptable for now.
# Maybe later try optimizing just over the oracle calls?

import numpy as np
from tqdm import tqdm

from fixed_dimension_base_cases import svp_only_base_cases


class SpencerGadgetReduction:
    MAGIC_FRACTION = 0.38  # solution of x^2 - 3x + 1 = 0

    def __init__(self, maxN, max_exponent, k):
        self.maxN = maxN
        self.max_exponent = max_exponent
        # self.base = base
        self.base = 2
        self.k = k
        self.maxC_index = (max_exponent + 1) * (self.base - 1) + 1
        self.objective_values = self.base_cases()

    def base_cases(self):
        return svp_only_base_cases(self.k, self.maxN, self.maxC_index)
        # result = np.full(self.maxN + 1, self.maxN + 1, self.maxC_index, np.nan)


    def solve(self):
        for n in range(self.maxN + 1):
            for l in range(1, self.maxN):
                for C_index in range(1, self.maxC_index):
                    if self.phase(n, l) == 1 or self.phase(n, l) == 2:
                        l_star = np.ceil(SpencerGadgetReduction.MAGIC_FRACTION * n)
                    elif self.phase(n, l) == 3:
                        l_star = ??
                    new_n = n - l_star
                    self.objective_values[n, l, C_index] = self.objective_values[new_n, min(l, new_n - l), C_index - 1] + \
                                                               (l / (n - l_star)) * self.objective_values[n, l_star, C_index - 1]


    # Try constant factors in all 3 phases.
    # I think it's a bad approximation for the 3rd phase.
    # I guess it's also worth taking one more look at the estimate I consed up
    # Weirdly 1/estimate seems pretty predictive but it doesn't look like I
    # have things upside down.

    # Basically, we're assuming that C_index is large enough that we don't run out
    # of oracle calls when applying our gadgets.
    def phase(self, n, l):
        tolerance = 0.05  # shouldn't matter too much
        if n < 2 * self.k:
            return 3
        elif SpencerGadgetReduction.MAGIC_FRACTION - tolerance < l / n < SpencerGadgetReduction.MAGIC_FRACTION + tolerance:
            return 2
        else:
            return 1


    def __getitem__(self, sliceobj):
        return self.objective_values[sliceobj]

    def __len__(self):
        return len(self.objective_values)

