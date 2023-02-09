from TauBasisReductionRecurrence import TauBasisReductionRecurrence
import base_cases
from BaseCaseTypes import BaseCaseTypes

class TauReductionToSVPInDimKOnlyRecurrence(TauBasisReductionRecurrence):

    def __init__(self, k, maxN, max_tau):
        self.base_case_type = BaseCaseTypes.SVPInDimensionKOnly
        super().__init__(k, maxN, max_tau)

    def base_cases_and_types(self):
        return base_cases.svp_in_dim_k_only_base_cases(self.k, self.N.stop_index, self.tau.stop_index)

class TauReductionToHKZ(TauBasisReductionRecurrence):

    def __init__(self, k, maxN, max_tau):
        self.base_case_type = BaseCaseTypes.HKZ
        super().__init__(k, maxN, max_tau)

    def base_cases_and_types(self):
        return base_cases.hkz_base_cases(self.k, self.N.stop_index, self.tau.stop_index)

class TauReductionToSVPAndLLL(TauBasisReductionRecurrence):

    def __init__(self, k, maxN, max_tau):
        self.base_case_type = BaseCaseTypes.SVPAndLLL
        super().__init__(k, maxN, max_tau)

    def base_cases_and_types(self):
        return base_cases.svp_and_lll_base_cases(self.k, self.N.stop_index, self.tau.stop_index)

class TauReductionToSVPOnly(TauBasisReductionRecurrence):
    def __init__(self, k, maxN, max_tau):
        self.base_case_type = BaseCaseTypes.SVPOnly
        super().__init__(k, maxN, max_tau)

    # Base cases for SVP in dimension n <= k. n <= k, l != 1 not allowed.
    def base_cases_and_types(self):
        return base_cases.svp_only_base_cases(self.k, self.N.stop_index, self.tau.stop_index)

class TauReductionToDSP(TauBasisReductionRecurrence):
    def __init__(self, k, maxN, max_tau):
        self.base_case_type = BaseCaseTypes.DSP
        super().__init__(k, maxN, max_tau)
        
    # Base cases for DSP in dimension k, using our optimistic guess k^(l * (k - l) / (2 * (k - 1))) for the approximation factor.
    def base_cases_and_types(self):
        return base_cases.dsp_base_cases(self.k, self.N.stop_index, self.tau.stop_index)