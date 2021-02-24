import torch
from torch_tools.experiment import Experiment


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

    def train_step(self, data, grad=True, output=False):

        total_L = 0

        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)

            if grad:
                self.optimizer.zero_grad()
                output = self.forward(x)
            else:
                with torch.no_grad():
                    output = self.forward(x)

            L = self.loss_function(output, labels)

            if grad:
                L.backward()
                self.optimizer.step()

            total_L += L

        total_L /= len(data)

        return total_L
