import geomstats.backend as gs
import torch


class Regularizer(torch.nn.Module):
    def __init__(self, function, variables, coefficient):
        super().__init__()
        self.function = function
        self.variables = variables
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

    def forward(self, variable_dict):
        penalty = self.function(*(variable_dict[v] for v in self.variables))
        return self.coefficient * penalty


class MultiRegularizer(torch.nn.Module):
    def __init__(self, regularizers):
        """
        regularizers: list of regularizer objects
        weights: list of coefficients on the regularizer functions, same length as regularizers
        """
        super().__init__()
        self.regularizers = regularizers
        # self.regularizer_params = [x.__dict__ for x in self.regularizers] # USE IF NEEDED

    def __str__(self):
        return "".join(["{}\n".format(str(reg)) for reg in self.regularizers])

    def forward(self, variable_dict):
        total_penalty = 0
        for i, reg in enumerate(self.regularizers):
            penalty = reg.forward(variable_dict)
            total_penalty += penalty
        return total_penalty
