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

    # TRAINING
    def train_step(self, data, epoch=None):

        total_L = 0
        loss_term = 0
        reg_terms = torch.zeros(len(self.regularizer.regularizers))
        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)

            yh = self.model.forward(x)
            loss = self.loss_function(yh, labels)

            batch_dict = {"x": x, "yh": yh}

            variable_dict = self.gen_variable_dict(batch_dict=batch_dict)

            L = loss + self.regularizer(variable_dict)

            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()

            with torch.no_grad():
                total_L += L
                loss_term += loss

                batch_reg_terms = self.regularizer.forward_dict(variable_dict)
                reg_terms += torch.tensor(list(batch_reg_terms.values()))

        total_L /= len(data)
        loss_term /= len(data)
        reg_terms /= len(data)

        self.log_model_params_epoch(epoch)

        loss_terms_dict = {"L": total_L, "loss_term": loss_term}
        reg_term_dict = {
            reg.name: reg_terms[i]
            for i, reg in enumerate(self.regularizer.regularizers)
        }

        results = {**loss_terms_dict, **reg_term_dict}
        return results

    def evaluate(self, data, epoch=None):
        with torch.no_grad():
            total_L = 0
            loss_term = 0
            reg_terms = torch.zeros(len(self.regularizer.regularizers))
            for i, (x, labels) in enumerate(data):
                x = x.to(self.device)
                labels = labels.to(self.device)
                yh = self.model.forward(x)
                batch_dict = {"x": x, "yh": yh}
                variable_dict = self.gen_variable_dict(batch_dict=batch_dict)

                loss = self.loss_function(yh, labels)
                L = loss + self.regularizer(variable_dict)

                total_L += L
                loss_term += loss

                batch_reg_terms = self.regularizer.forward_dict(variable_dict)
                reg_terms += torch.tensor(list(batch_reg_terms.values()))

            total_L /= len(data)
            loss_term /= len(data)
            reg_terms /= len(data)

        loss_terms_dict = {"L": total_L, "loss_term": loss_term}
        reg_term_dict = {
            reg.name: reg_terms[i]
            for i, reg in enumerate(self.regularizer.regularizers)
        }

        results = {**loss_terms_dict, **reg_term_dict}
        return results

    def gen_variable_dict(self, batch_dict):
        return {**dict(self.model.named_parameters()), **batch_dict}

    # Virtual Functions
    def on_begin(self, data):
        self.log_model_graph(data)
        self.log_data_embedding(data)

    # def on_end(self, data):
    # self.log_hyperparameters(data)

    # Logging functions
    def log_model_params_epoch(self, epoch):
        self.writer.add_histogram("linear0.weights", self.model.layers[0].weight, epoch)
        self.writer.add_histogram("linear0.bias", self.model.layers[0].bias, epoch)

    def log_model_graph(self, data):
        batch_x, batch_y = next(iter(data.train))
        self.writer.add_graph(self.model, batch_x)

    def log_data_embedding(self, data):
        train_x, train_y = data.train.dataset[data.train.batch_sampler.sampler.indices]
        self.writer.add_embedding(train_x, metadata=train_y)
