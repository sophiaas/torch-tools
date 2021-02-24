import datetime
import os
import pickle

import torch
from torch.nn import ParameterDict
from torch.utils.tensorboard import SummaryWriter

# import generic # fails!, also, AttrDict not defined


class Experiment(torch.nn.Module):
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        regularizer,
        device="cuda",
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.device = device
        self.logdir = None
        self.dataset_name = None
        self.loss_function = None

    # TODO: what should be part of the path?
    # We should remove dataset name as a dependency here
    def create_logdir(self, dataset_name):
        self.dataset_name = dataset_name

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

        self.logdir = logdir

    # logstring = "Epoch: {} || Training L: {:.5f}".format(epoch, variable)
    def log(self, epoch, group, name, value):
        self.writer.add_scalar("{}/{}".format(group, name), value, epoch)

    def save_pickle(self, param_dict, path, fname):
        final_path = os.path.join(path, fname)
        with open(final_path, "wb") as f:
            pickle.dump(param_dict, f)

    def pickle_attribute_dicts(self):
        self.save_pickle(self.model.__dict__, self.logdir, "model" + "_dict")
        self.save_pickle(self.optimizer.__dict__, self.logdir, "optimizer" + "_dict")
        self.save_pickle(
            self.regularizer.__dict__, self.logdir, "regularizer" + "_dict"
        )

    def pickle_data_loader_dicts(self, data_loader):
        self.save_pickle(data_loader.__dict__, self.logdir, "data_loader" + "_dict")
        self.save_pickle(
            data_loader.train.__dict__,
            self.logdir,
            "training_data" + "_dict",
        )

    # right now this is an overridden method by the inherited class...
    # do we want models to inherit experiment or for experiments to ingest models?
    #   ingest? -> 1) model provides its own train step, this calls it
    #              2) New experiment subclass for each train step type (meh), idk
    #              3) new class
    def train_step(self, data, grad=True, output=False):

        raise NotImplementedError

    # do we want to pass in data and let the train loading scheme be passed in/parameterized in constructor?
    def train(self, data_loader, epochs, start_epoch=0, checkpoint_interval=10):
        try:
            self.create_logdir(dataset_name=data_loader.name)
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

        self.writer = SummaryWriter(self.logdir)

        self.pickle_attribute_dicts()
        self.pickle_data_loader_dicts(data_loader)

        for i in range(start_epoch, epochs + 1):
            training_loss = self.train_step(data_loader.train, grad=True)
            self.log(i, group="loss", name="train", value=training_loss)

            if data_loader.val is not None:
                validation_loss = self.train_step(data_loader.val, grad=False)
                self.log(i, group="loss", name="val", value=validation_loss)

            if i % checkpoint_interval == 0:
                self.save_checkpoint(epoch=i)

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.logdir, "checkpoints", "checkpoint_{}.pt".format(epoch)),
        )

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
