import datetime
import os
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import torch_tools.generic as generic
from tqdm import tqdm


class Experiment(torch.nn.Module):
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        regularizer,
        device="cuda",
        print_frequency=1,
        save_frequency=10,
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
        self.print_frequency = print_frequency
        self.save_frequency = save_frequency

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
    def save_checkpoint(self, epoch):
        if epoch % self.save_frequency == 0:
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
        self.save_checkpoint(epoch)

    # right now this is an overridden method by the inherited class...
    # do we want models to inherit experiment or for experiments to ingest models?
    #   ingest? -> 1) model provides its own train step, this calls it
    #              2) New experiment subclass for each train step type (meh), idk
    #              3) new class
    def train_step(self, data, grad=True, output=False):
        """
        The output of train_step should be a dictionary containing losses to log.
        Examples:
                loss_dict = {
                    "total_loss": L_mean,
                    "representation_loss": L_rep_mean,
                    "reconstruction_loss": L_rec_mean,
                    "regularization_loss": L_reg_mean,
                }
        """
        raise NotImplementedError

    # do we want to pass in data and let the train loading scheme be passed in/parameterized in constructor?
    def train(self, data_loader, epochs, start_epoch=0):
        try:
            self.dataset_name = data_loader.name
            self.logdir = self.create_logdir()
            self.writer = SummaryWriter(self.logdir)

            generic.save_pickle(self.model.__dict__, self.logdir, "model" + "_dict")
            generic.save_pickle(
                self.optimizer.__dict__, self.logdir, "optimizer" + "_dict"
            )
            generic.save_pickle(
                data_loader.__dict__, self.logdir, "data_loader" + "_dict"
            )
            generic.save_pickle(
                data_loader.train.__dict__,
                self.logdir,
                "training_data" + "_dict",
            )
            if self.regularizer is not None:
                generic.save_pickle(
                    self.regularizer.__dict__, self.logdir, "regularizer" + "_dict"
                )

        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

        for i in tqdm(range(start_epoch, epochs)):
            training_loss = self.train_step(data_loader.train, grad=True)
            for k in training_loss.keys():
                self.log(i, group=k, name="train", value=training_loss[k])

            if data_loader.val is not None:
                validation_loss = self.train_step(data_loader.val, grad=False)
                for k in validation_loss.keys():
                    self.log(i, group=k, name="val", value=validation_loss[k])

            self.print_update(training_loss, validation_loss, epoch)

    def print_update(self, training_loss, validation_loss, epoch):
        if epoch % self.print_frequency == 0:
            print(
                "Epoch {}  ||  Training Loss: {:0.5f}  |  Validation Loss: {:0.5f}".format(
                    i, training_loss["total_loss"], validation_loss["total_loss"]
                )
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
