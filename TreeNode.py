from StepType import StepType
from treelib import Node, Tree
from termcolor import colored

class TreeNode:
    def __init__(self, parameters, solution, children=None):
        self.parameters = parameters
        self.solution = solution
        if children is None:
            self.children = []
        else:
            self.children = children

        self.treelib_key = None
        self.step_type_value = None

    def __iter__(self):
        yield self
        for child in self.children:
            for node in iter(child):
                yield node

    def n(self):
        return self.parameters[0]
    def l(self):
        return self.parameters[1]
    def T(self):
        return self.parameters[2]

    def left(self):
        return self.children[1]
    def right(self):
        return self.children[0]

    def l_star(self):
        if self.is_duality() or self.is_leaf():
            return None
        else:
            return self.right().l()

    def is_leaf(self):
        return len(self.children) == 0

    def is_duality(self):
        # return self.step_type() == StepType.DUALITY
        return self.step_type() == "DUALITY"

    def is_recursive(self):
        return not self.is_leaf() and not self.is_duality()

    def step_type(self):
        if self.step_type_value:
            return self.step_type_value
        return self.solution.step_types[self.parameters]

    def as_dict(self, depth=1e10, to_string=lambda x: x.short_str()):
        if depth <= 0:
            return to_string(self)
        return {to_string(self): [child.as_dict(depth - 1, to_string) for child in self.children]}

    def __str__(self):
        return f"""{self.step_type()}{self.parameters}"""

    def short_str(self):
        step_type = StepType[self.step_type()]
        return f"""{step_type.short_name()}{self.parameters}"""

    def _add_children_treelib(self, tree, to_string=str):
        for i, child in enumerate(self.children):
            child.treelib_key = i
            tree.create_node(to_string(child), hash(child), parent=hash(self), data=child)
            child._add_children_treelib(tree, to_string)

    def as_treelib(self, to_string=str):
        result = Tree()
        result.create_node(to_string(self), hash(self), data=self)
        self._add_children_treelib(result, to_string)
        return result

    @staticmethod
    def to_string(n):
        ell_by_n_minus_ell_star = -1 if n.is_leaf() else n.parameters[1] / (
                    n.parameters[0] - n.children[0].parameters[1])
        # tweaked = -1 if n.is_leaf() else min(n.parameters[1], n.parameters[0] - n.parameters[1]) / min(n.children[0].parameters[1],
        #             n.parameters[0] - n.children[0].parameters[1])
        if not n.is_leaf() and not n.is_duality():
            # print(n.is_duality())
            # print(n.step_type())
            # print(n.step_type() == StepType.DUALITY)
            # print(n)
            # print(n.children)
            right = n.children[0]
            left = n.children[1]
            g = n.solution.get_log_value_of
            times_ratioRL = 2**(g(right.parameters[2]) - g(left.parameters[2]))
            ns_ratioLR = left.parameters[0] / right.parameters[0]
            nL = left.parameters[0]
            lL = left.parameters[1]
            nR = right.parameters[0]
            lR = right.parameters[1]
            # ls_ratioLR = left.parameters[1] / right.parameters[1]
            ls_ratioLR = min(lL, nL - lL) / min(lR, nR - lR)
            # ugh, shouldn't be this. Should not have the 1/.
            # I think I just gotta try with a constant factor.
            ratio_estimate = 1 / (times_ratioRL * ns_ratioLR * ls_ratioLR)
            # ratio_estimate = 1 / times_ratioRL
            # ratio_estimate = 1 / (times_ratioRL * ns_ratioLR ** 2 * ls_ratioLR **2)
            # ratio_estimate = ns_ratioLR * ls_ratioLR / times_ratioRL
        else:
            ratio_estimate = -1
        # raw_string = f"{n.step_type()}({n.parameters[0]}, {n.parameters[1]}, {n.solution.get_log_value_of(n.parameters[2]):.2f}) -> {n.solution[n.parameters] / n.parameters[1]:.2f}"
        # raw_string = f"{n.step_type()}({n.parameters[0]}, {n.parameters[1]}, {n.solution.get_log_value_of(n.parameters[2]):.2f}) -> ratio: {ell_by_n_minus_ell_star:.2f} estimate: {ratio_estimate:.2f}"
        raw_string = f"{n.step_type()}({n.parameters[0]}, {n.parameters[1]}, {n.solution.get_log_value_of(n.parameters[2]):.2f}) -> ratio: {ell_by_n_minus_ell_star:.2f}"

        colored_string = colored(raw_string, StepType[n.step_type()].display_color())
        return colored_string

    def show_coarse(self):
        return self.as_treelib(self.to_string).show(key=lambda n: -n.data.treelib_key)

    def show_coarse_left_first(self):
        return self.as_treelib(self.to_string).show(key=lambda n: n.data.treelib_key)



    # Would be nice to compress duality so that the tree was easier to read.


