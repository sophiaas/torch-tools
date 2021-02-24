import torch
from torch.optim import Adam

from torch_tools.experiment import Experiment
from torch_tools.functional import l1_norm, matrix_2_norm
from torch_tools.generic import ParameterDict
from torch_tools.model import Model
from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer
from transform_dataset.torch.vector import Translation


model_params = ParameterDict({"n_neurons": 64})

data_params = ParameterDict(
    {
        "c_classes": 10,
        "digits": np.arange(10),
        "max_transformation_steps": 10,
        "dim": 32,
        "noise": 0.2,
        "seed": 0,
    }
)


optimizer_params = ParameterDict({"lr": 1e-3})

optimizer = Adam(**optimizer_params)

reg1 = Regularizer(l1_norm, "s", 0.7)
reg2 = Regularizer(matrix_2_norm, "A", 0.1)
regularizer = MultiRegularizer([reg1, reg2])


class TestExperiment(Experiment):
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        regularizer,
        loss_function,
        device="cuda",
    ):
        super().__init__(
            self, experiment_name, model, optimizer, regularizer, device="cuda"
        )
        self.loss_function = loss_function

    def train_step(self, data_loader, grad=True):

        total_L = 0

        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)

            if grad:
                self.optimizer.zero_grad()
                y = self.forward(x)
            else:
                with torch.no_grad():
                    y = self.forward(x)

            L = self.loss_function(y, labels)

            if grad:
                L.backward()
                self.optimizer.step()

            total_L += L

        total_L /= len(data)

        return total_L


model = torch.nn.Sequential([torch.nn.Linear(model_params.n_neurons)])

experiment_name = "test"
loss = torch.nn.CrossEntropyLoss()

experiment = TestExperiment(experiment_name, model, optimizer, regularizer, loss)

experiment.train()
