from ExactBasisReductionRecurrence import ExactBasisReductionRecurrence
import fixed_dimension_base_cases

class ExactReductionToSVPInDimKOnlyRecurrence(ExactBasisReductionRecurrence):

    def base_cases_and_types(self):
        return fixed_dimension_base_cases.svp_in_dim_k_only_base_cases(self.k, self.N.stop_index, self.C.stop_index)

class ExactReductionToHKZ(ExactBasisReductionRecurrence):

    def base_cases_and_types(self):
        return fixed_dimension_base_cases.hkz_base_cases(self.k, self.N.stop_index, self.C.stop_index)

class ExactReductionToSVPAndLLL(ExactBasisReductionRecurrence):

    def base_cases_and_types(self):
        return fixed_dimension_base_cases.svp_and_lll_base_cases(self.k, self.N.stop_index, self.C.stop_index)

class ExactReductionToSVPOnly(ExactBasisReductionRecurrence):

    # Base cases for SVP in dimension n <= k. n <= k, l != 1 not allowed.
    def base_cases_and_types(self):
        return fixed_dimension_base_cases.svp_only_base_cases(self.k, self.N.stop_index, self.C.stop_index)

class ExactReductionToDSP(ExactBasisReductionRecurrence):

    # Base cases for DSP in dimension k, using our optimistic guess
    # k^(l * (k - l) / (2 * (k - 1))) for the approximation factor.
    def base_cases_and_types(self):
        return fixed_dimension_base_cases.dsp_base_cases(self.k, self.N.stop_index, self.C.stop_index)