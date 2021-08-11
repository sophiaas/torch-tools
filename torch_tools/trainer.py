import datetime
import os
import pickle
import torch
from torch.nn import ParameterDict
import torch_tools.utils as utils
from inspect import signature


class Trainer(torch.nn.Module):
    def __init__(
        self,
        model,
        loss,
        logger,
        optimizer_config,
        regularizer=None,
        normalizer=None,
        device="cuda",
    ):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.device = device
        self.loss = loss
        self.epoch = 0
        self.n_examples = 0
        self.logger = logger

        optimizer_config["params"]["params"] = model.parameters()
        self.optimizer = optimizer_config.build()

    def step(self, data_loader, grad=True):

        accumulator = self.reset_accumulator()

        for i, (x, labels) in enumerate(data_loader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            if grad:
                self.optimizer.zero_grad()
                out = self.model.forward(x)

            else:
                with torch.no_grad():
                    out = self.model.forward(x)

            L, accumulator = self.accumulate_loss(x, out, labels, accumulator)

            if grad:
                L.backward()
                self.optimizer.step()

            if self.normalizer is not None:
                self.normalizer(self.model.state_dict())

        accumulator = self.average_loss(accumulator, len(data_loader))
        
        return accumulator

#         return (losses, other object) #,  

    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        print_interval=1,
    ):
        self.logger.begin(self.model, data_loader)

        try:
            for i in range(start_epoch, start_epoch + epochs + 1):
                self.epoch = i
                train_results = self.step(data_loader.train, grad=True)
                self.logger.log_step(
                    results=train_results,
                    step_type="train",
                    epoch=self.epoch,
                    n_examples=self.n_examples,
                )

                if data_loader.val is not None:

                    validation_results = self.evaluate(data_loader.val)
                    self.logger.log_step(
                        results=validation_results,
                        step_type="val",
                        epoch=self.epoch,
                        n_examples=self.n_examples,
                    )

                    if i % print_interval == 0 and print_status_updates == True:
                        self.print_update(train_results, validation_results)

                self.logger.save_checkpoint(self.model, self.epoch)

                self.n_examples += len(data_loader.train.dataset)

        except KeyboardInterrupt:
            print("Stopping and saving run at epoch {}".format(i))

        self.logger.end(self.model, data_loader)

    def resume(self, data_loader, epochs):
        self.train(data_loader, epochs, start_epoch=self.epoch)
        

    @torch.no_grad()
    def evaluate(self, data_loader):
        results = self.step(data_loader, grad=False)
        return results

    def print_update(self, training_loss, validation_loss):
        print(
            "Epoch {}  ||  N Examples {} || Training Loss: {:0.5f}  |  Validation Loss: {:0.5f}".format(
                self.epoch,
                self.n_examples,
                training_loss["total_loss"],
                validation_loss["total_loss"],
            )
        )

# below: all accumulator, all in step
    
    def reset_accumulator(self):
        d = {"total_loss": 0}

        if self.regularizer is not None:
            d["regularization_loss"] = 0

        return d

    def accumulate_loss(self, x, out, labels, accumulator):
        L = self.loss(out, labels)
        accumulator["total_loss"] += L
        L_reg, accumulator = self.get_regularizer_loss(x, out, accumulator)
        L += L_reg
        return L, accumulator

    def average_loss(self, accumulator, n_data_points):
        for key in accumulator.keys():
            accumulator[key] /= n_data_points
        return accumulator

    def get_regularizer_loss(self, x, out, accumulator):
        if self.regularizer is not None:
            L_reg = self.regularizer(self.model.state_dict())
            accumulator["regularization_loss"] += L_reg
        else:
            L_reg = 0.0
        return L_reg, accumulator

