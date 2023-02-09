from scipy.special import binom

class Parameters:
    # Convenient for looking up a parameter by name or by index

    def __init__(self, parameters):
        self.parameters = parameters
        self.parameters_by_name = {p.name: p for p in self.parameters}

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.parameters[index]
        if isinstance(index, slice):
            return self.parameters[index]
        if isinstance(index, str):
            return self.parameters_by_name[index]
        assert False, "Unexpected type for looking up parameter."

    def __len__(self):
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def values(self, parameter_values):
        assert len(parameter_values) == len(self.parameters)
        for i in range(len(self.parameters)):
            assert 0 <= parameter_values[i] < self.parameters[i].stop_index, f"""
            Invalid index {parameter_values[i]} for parameter with index 
            0 <= index < {self.parameters[i].stop_index}"""
        return parameter_values


