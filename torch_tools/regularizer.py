import geomstats.backend as gs
import torch


class Regularizer(torch.nn.module):
    def __init__(self, function, variables, coefficient):
        self.function = function
        self.variables = variables
        self.coefficient = coefficient

    def forward(self, variable_dict):
        penalty = self.function(*(variable_dict[v] for v in self.variables))
        return self.coefficient * penalty


class MultiRegularizer(torch.nn.Module):
    def __init__(self, regularizers):
        """
        regularizers: list of regularizer objects
        weights: list of coefficients on the regularizer functions, same length as regularizers
        """
        self.regularizers = regularizers
        self.weights = weights
        # self.regularizer_params = [x.__dict__ for x in self.regularizers] # USE IF NEEDED

    def forward(self, variable_dict):
        total_penalty = 0
        for i, reg in enumerate(self.regularizers):
            penalty = reg.forward(variable_dict)
            total_penalty += penalty
        return total_penalty
