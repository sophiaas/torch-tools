import torch


class Regularizer(torch.nn.Module):
    def __init__(self, function, variables, coefficient):
        super().__init__()
        self.name = "{}_{}".format(function.__name__, str(variables))
        self.function = function
        self.variables = variables if type(variables) == list else [variables]
        self.coefficient = coefficient

    def __str__(self):
        param_dict = self.get_regularizer_param_dict()
        out_string = "Regularizer | " + str(param_dict)
        return out_string

    def get_regularizer_param_dict(self):
        param_dict = {
            key: val
            for (key, val) in vars(self).items()
            if ("_" not in key) and ("training" not in key)
        }
        return param_dict

    def compute_penalty(self, variable_dict):
        return self.function(*(variable_dict[v] for v in self.variables))

    def forward(self, variable_dict):
        penalty = self.compute_penalty(variable_dict)
        return self.coefficient * penalty


class OldMultiRegularizer(torch.nn.Module):
    def __init__(self, regularizers):
        """
        regularizers: list of regularizer objects
        """
        super().__init__()
        self.regularizers = regularizers

    def __str__(self):
        return "".join(["{}\n".format(str(reg)) for reg in self.regularizers])

    def forward_dict(self, variable_dict):
        return {
            reg.name: reg.compute_penalty(variable_dict) for reg in self.regularizers
        }

    def forward(self, variable_dict):
        total_penalty = 0
        for i, reg in enumerate(self.regularizers):
            penalty = reg.forward(variable_dict)
            total_penalty += penalty
        return total_penalty
    

class MultiRegularizer(torch.nn.Module):
    def __init__(self, regularizer_configs):
        """
        regularizer_configs: list of regularizer configs
        """
        super().__init__()
        self.regularizers = [x.build() for x in regularizer_configs]

    def __str__(self):
        return "".join(["{}\n".format(str(reg)) for reg in self.regularizers])

    def forward_dict(self, variable_dict):
        return {
            reg.name: reg.compute_penalty(variable_dict) for reg in self.regularizers
        }

    def forward(self, variable_dict):
        total_penalty = 0
        for i, reg in enumerate(self.regularizers):
            penalty = reg.forward(variable_dict)
            total_penalty += penalty
        return total_penalty
