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
        super().__init__(experiment_name, model, optimizer, regularizer, device=device)
        self.loss_function = loss_function

    def train_step(self, data):
        total_L = 0
        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)

            output = self.model.forward(x)
            L = self.loss_function(output, labels)

            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()

            total_L += L
        total_L /= len(data)
        return total_L

    def evaluate(self, data):
        with torch.no_grad():
            total_L = 0
            for i, (x, labels) in enumerate(data):
                x = x.to(self.device)
                labels = labels.to(self.device)
                output = self.model.forward(x)
                L = self.loss_function(output, labels)
                total_L += L
            total_L /= len(data)
        return total_L
