from RecurrenceTypes import RecurrenceTypes
from BaseCaseTypes import BaseCaseTypes
import exact_reductions
import tau_reductions

# jeez, do I want persistence with my sandwich too?

class LabBench:

    def __init__(self):
        self.stored_solutions = []

    def solution(self, k, maxN, max_runtime_parameter, recurrence_type, base_case_type):
        for stored in self.stored_solutions:
            if          stored.k == k and \
                        stored.N.stop_index > maxN and \
                        stored.max_runtime_parameter() >= max_runtime_parameter and \
                        stored.recurrence_type == recurrence_type and \
                        stored.base_case_type == base_case_type:
                return stored

        mapping = {
            RecurrenceTypes.EXACT: {
                BaseCaseTypes.SVPInDimensionKOnly: exact_reductions.ExactReductionToSVPInDimKOnlyRecurrence,
                BaseCaseTypes.SVPOnly: exact_reductions.ExactReductionToSVPOnly,
                BaseCaseTypes.SVPAndLLL: exact_reductions.ExactReductionToSVPAndLLL,
                BaseCaseTypes.HKZ: exact_reductions.ExactReductionToHKZ,
                BaseCaseTypes.DSP: exact_reductions.ExactReductionToDSP,
            },
            RecurrenceTypes.TAU: {
                BaseCaseTypes.SVPInDimensionKOnly: tau_reductions.TauReductionToSVPInDimKOnlyRecurrence,
                BaseCaseTypes.SVPOnly: tau_reductions.TauReductionToSVPOnly,
                BaseCaseTypes.SVPAndLLL: tau_reductions.TauReductionToSVPAndLLL,
                BaseCaseTypes.HKZ: tau_reductions.TauReductionToHKZ,
                BaseCaseTypes.DSP: tau_reductions.TauReductionToDSP,
            }
        }
        reduction = mapping[recurrence_type][base_case_type](k, maxN, max_runtime_parameter)
        reduction.solve()
        self.stored_solutions.append(reduction)
        return reduction

