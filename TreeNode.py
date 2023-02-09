from StepType import StepType
from treelib import Node, Tree

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
        for node in self.children:
            yield node

    def is_leaf(self):
        return self.children is None

    def is_duality(self):
        return self.step_type() == StepType.DUALITY

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

    def show_coarse(self):
        def to_string(n):
            return f"{n.step_type()}({n.parameters[0]}, {n.parameters[1]}, {n.solution.get_log_value_of(n.parameters[2]):.2f})"
        return self.as_treelib(to_string).show(key=lambda n: -n.data.treelib_key)


    # Would be nice to compress duality so that the tree was easier to read.


