from CoarseGrainedBasisReductionRecurrence import CoarseGrainedBasisReductionRecurrence
import fixed_dimension_base_cases


class CoarseReductionToSVPInDimKOnlyRecurrence(CoarseGrainedBasisReductionRecurrence):

    def base_cases_and_types(self):
        # return fixed_dimension_base_cases.svp_in_dim_k_only_base_cases(self.k, self.N.stop_index, self.C.stop_index)
        return fixed_dimension_base_cases.svp_in_dim_k_only_base_cases(self)


class CoarseReductionToHKZ(CoarseGrainedBasisReductionRecurrence):

    def base_cases_and_types(self):
        # return fixed_dimension_base_cases.hkz_base_cases(self.k, self.N.stop_index, self.C.stop_index)
        return fixed_dimension_base_cases.hkz_base_cases(self)



class CoarseReductionToSVPAndLLL(CoarseGrainedBasisReductionRecurrence):

    def base_cases_and_types(self):
        # return fixed_dimension_base_cases.svp_and_lll_base_cases(self.k, self.N.stop_index, self.C.stop_index)
        return fixed_dimension_base_cases.svp_and_lll_base_cases(self)



class CoarseReductionToSVPOnly(CoarseGrainedBasisReductionRecurrence):

    # Base cases for SVP in dimension n <= k. n <= k, l != 1 not allowed.
    def base_cases_and_types(self):
        # return fixed_dimension_base_cases.svp_only_base_cases(self.k, self.N.stop_index, self.C.stop_index)
        return fixed_dimension_base_cases.svp_only_base_cases(self)



class CoarseReductionToDSP(CoarseGrainedBasisReductionRecurrence):

    # Base cases for DSP in dimension k, using our optimistic guess k^(l * (k - l) / (2 * (k - 1))) for the approximation factor.
    def base_cases_and_types(self):
        # return fixed_dimension_base_cases.dsp_base_cases(self.k, self.N.stop_index, self.C.stop_index)
        return fixed_dimension_base_cases.dsp_base_cases(self)
