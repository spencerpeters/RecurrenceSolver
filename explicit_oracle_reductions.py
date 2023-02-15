from ExplicitOracleBasisReductionRecurrence import ExplicitOracleBasisReductionRecurrence
import explicit_oracle_base_cases
from utilities import common_base_cases

class ExplicitReductionToHKZ(ExplicitOracleBasisReductionRecurrence):

    def base_cases_and_types(self):
        return explicit_oracle_base_cases.hkz_base_cases(self)

class ExplicitReductionToDSP(ExplicitOracleBasisReductionRecurrence):

    def base_cases_and_types(self):
        return explicit_oracle_base_cases.dsp_base_cases_nl(self)

class ExplicitReductionToDSPCostN(ExplicitOracleBasisReductionRecurrence):

    def base_cases_and_types(self):
        return explicit_oracle_base_cases.dsp_base_cases_n(self)

class ExplicitReductionToSVP(ExplicitOracleBasisReductionRecurrence):

    def base_cases_and_types(self):
        return explicit_oracle_base_cases.svp_only_base_cases(self)

class ExplicitReductionSanityCheck(ExplicitOracleBasisReductionRecurrence):

    def base_cases_and_types(self):
        return common_base_cases(self)