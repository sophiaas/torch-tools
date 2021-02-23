import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment(torch.nn.Module):
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        optimizer_params,
        regularizer,
        regularizer_params,
        device="cuda",
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.regularizer = regularizer
        self.regularizer_params = regularizer_params
        self.device = device
        self.logdir = None
        self.dataset_name = None

    # TODO: what should be part of the path?
    def create_logdir(self):
        logdir = os.path.join(
            "logs",
            self.experiment_name,
            self.model.__class__.__name__,
            self.dataset_name,
        )

        os.makedirs(logdir, exist_ok=True)

        logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.mkdir(logdir)
        os.mkdir(os.path.join(logdir, "checkpoints"))
        return logdir

    # TODO: what else should we log?
    # Should we give options or leave that to a subclass?
    def save_checkpoint(self):
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(
                    self.logdir, "checkpoints", "checkpoint_{}.pt".format(epoch)
                ),
            )

    # logstring = "Epoch: {} || Training L: {:.5f}".format(epoch, variable)
    def log(self, epoch, group, name, value):
        self.writer.add_scalar("{}/{}".format(group, name), value, epoch)

    # right now this is an overridden method by the inherited class...
    # do we want models to inherit experiment or for experiments to ingest models?
    #   ingest? -> 1) model provides its own train step, this calls it
    #              2) New experiment subclass for each train step type (meh), idk
    #              3) new class
    def train_step(self, data, grad=True, output=False):

        raise NotImplementedError

    # do we want to pass in data and let the train loading scheme be passed in/parameterized in constructor?
    def train(self, data_loader, epochs, start_epoch=0):
        try:
            self.dataset_name = data_loader.name
            self.logdir = self.create_logdir()
            self.writer = SummaryWriter(self.logdir)
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

        for i in range(start_epoch, epochs):
            training_loss = self.train_step(data_loader.train, grad=True)
            self.log(i, group="loss", name="train", value=training_loss)

            if data_loader.val is not None:
                validation_loss = self.train_step(data_loader.val, grad=False)
                self.log(i, group="loss", name="val", value=validation_loss)

    def load_checkpoint(self, path, checkpoint):
        state_dict = torch.load(
            path + "checkpoints/checkpoint_{}.pt".format(checkpoint)
        )
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def resume(self, path, checkpoint, data_loader, epochs):
        self.load_checkpoint(path, checkpoint)
        self.train(data_loader, epochs, start_epoch=checkpoint)

    ## this may not work if training has access to things not available in test
    def evaluate(self, test_loader, output=True):
        out = self.train_step(test_loader, grad=False, output=True)
        return out
