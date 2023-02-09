# from exact_reductions import ExactReductionToSVPInDimKOnlyRecurrence
# import tau_reductions
#
# # e = ExactReductionToSVPInDimKOnlyRecurrence(2, 5, 3)
# e = tau_reductions.TauReductionToSVPInDimKOnlyRecurrence(2, 5, 3)
#
# e.solve()
#
# print(e.objective_values)
#
# t = e.make_tree((5, 1, 3))
#
# # print(e.make_tree((2, 5, 3)))
#
# print(t)
# print(t.as_dict())

from coarse_reductions import CoarseReductionToDSP

k = 10
max_N = 20
base = 10
max_exponent = 3

recurrence = CoarseReductionToDSP(k, max_N, base, max_exponent)

recurrence.solve()