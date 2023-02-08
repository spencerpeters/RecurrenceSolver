from utilities import C_of

from StepType import StepType

class TreeNode:
    def __init__(self, parameters, solution, children=None):
        self.parameters = parameters
        self.solution = solution
        if children is None:
            self.children = []
        else:
            self.children = children

    def __iter__(self):
        yield self
        for node in self.children:
            yield node

    def is_leaf(self):
        return self.children is None

    def is_duality(self):
        return self.step_type() == StepType.DUALITY

    def is_recursive(self):
        return not self.is_leaf() and not self.is_duality()

    def step_type(self):
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

# class TauTreeNode(TreeNode):
#     def short_str(self, k):
#         return


