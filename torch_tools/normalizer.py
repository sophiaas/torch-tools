import torch


class Normalizer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.name = "{}_{}".format("normalizer", str(variables))
        self.variables = variables if type(variables) == list else [variables]

    def forward(self, variable_dict):
        with torch.no_grad():
            self.normalize(variable_dict)

    def normalize(self, variable_dict):
        raise NotImplementedError

