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

    def train_step(self, data, epoch=None):
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

        self.log_model_params(epoch)

        return total_L

    def evaluate(self, data, epoch=None):
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

    # LOGGING
    def on_begin(self, data):
        batch_x, batch_y = next(iter(data.train))
        self.writer.add_graph(self.model, batch_x)

        train_x, train_y = data.train.dataset[data.train.batch_sampler.sampler.indices]
        self.writer.add_embedding(train_x, metadata=train_y)

    def on_end(self, data):
        # log train, val, and hyperparameters
        hparam_dict = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "bsize": data.batch_size,
        }
        train_loss = self.evaluate(data.train)
        metric_dict = {"hparam/train_loss": train_loss}
        if data.val is not None:
            validation_loss = self.evaluate(data.val)
            metric_dict["hparam/val_loss"] = validation_loss

        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_model_params(self, epoch):
        self.writer.add_histogram("linear0.weights", self.model.layers[0].weight, epoch)
        self.writer.add_histogram("linear0.bias", self.model.layers[0].bias, epoch)
