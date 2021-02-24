import torch


class SimpleModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.in_dim, out_features=self.out_dim)
        )

    def forward(self, x):
        return self.layers.forward(x)
