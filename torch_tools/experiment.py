import datetime
import os
import pickle

import torch
from torch.nn import ParameterDict
from torch.utils.tensorboard import SummaryWriter

# import generic


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
        self.logdir = None  # remove? / make immutable?
        self.dataset_name = None  # remove? / make immutable?
        self.loss_function = None

    # Virtual Functions
    def train_step(self, data, epoch=None):
        raise NotImplementedError

    def evaluate(self, data, epoch=None):
        raise NotImplementedError

    def on_begin(self, data):
        raise NotImplementedError

    def on_end(self, data):
        raise NotImplementedError

    # TRAINING
    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        checkpoint_interval=10,
    ):
        self.begin(data=data_loader)

        for i in range(start_epoch, epochs + 1):
            train_results = self.train_step(data_loader.train, epoch=i)
            self.log_step(epoch=i, results=train_results, step_type="train")

            if data_loader.val is not None:
                validation_results = self.evaluate(data_loader.val, epoch=i)
                self.log_step(epoch=i, results=validation_results, step_type="val")

            if i % checkpoint_interval == 0:
                if print_status_updates == True:
                    self.print_status(
                        epoch=i, name="Train Loss", value=train_results["L"]
                    )
                    self.print_status(
                        epoch=i, name="Val Loss", value=validation_results["L"]
                    )
                self.save_checkpoint(epoch=i)

        self.end(data=data_loader)

    # LOGGING SETUP
    def create_logdir(self, data):
        self.dataset_name = data.name

        logdir = os.path.join(
            "logs",
            self.experiment_name,
            self.model.__class__.__name__,
            self.dataset_name,
            "b_size: {}".format(data.batch_size),
            "lr: {}__".format(self.optimizer.param_groups[0]["lr"]),
        )

        os.makedirs(logdir, exist_ok=True)

        logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.mkdir(logdir)
        os.mkdir(os.path.join(logdir, "checkpoints"))

        self.logdir = logdir

    def begin(self, data):
        try:
            self.create_logdir(data=data)
            self.writer = SummaryWriter(self.logdir)
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

        try:
            self.on_begin(data=data)
        except NotImplementedError:
            pass

        self.pickle_attribute_dicts()
        self.pickle_data_loader_dicts(data)

    def end(self, data):
        self.log_hyperparameters(data)
        try:
            self.on_end(data=data)
        except NotImplementedError:
            pass

    # TB LOGGING
    def log_step(self, epoch, results, step_type):
        for (name, value) in results.items():
            self.log_scalar(
                epoch=epoch,
                group="loss_reg_{}".format(step_type),
                name=name,
                value=value,
            )

    def log_scalar(self, epoch, group, name, value):
        self.writer.add_scalar("{}/{}".format(group, name), value, epoch)

    # TB HPARAMS
    def log_hyperparameters(self, data):
        opt_hparams = self.get_optimizer_hparams()
        data_hparams = self.get_data_hparams(data)
        reg_hparams = self.get_regularizer_hparams()

        hparam_dict = {**opt_hparams, **data_hparams, **reg_hparams}
        metric_dict = self.get_hparam_metrics(data)

        self.writer.add_hparams(hparam_dict, metric_dict)

    def get_hparam_metrics(self, data):
        train_results = self.evaluate(data.train)
        metric_dict = {"hparam/train_loss": train_results["L"]}
        if data.val is not None:
            validation_results = self.evaluate(data.val)
            metric_dict["hparam/val_loss"] = validation_results["L"]

        return metric_dict

    def get_regularizer_hparams(self):
        regs = self.regularizer.regularizers
        reg_hparam_dict = {}
        for reg in regs:
            reg_param_dict = reg.get_regularizer_param_dict()
            name = reg_param_dict["name"]
            coeff = reg_param_dict["coefficient"]
            reg_hparam_dict[name] = coeff

        return reg_hparam_dict

    def get_optimizer_hparams(self):
        return {"lr": self.optimizer.param_groups[0]["lr"]}

    def get_data_hparams(self, data):
        return {"bsize": data.batch_size}

    # ALTERNATIVE LOGGING
    def print_status(self, epoch, name, value):
        status_string = "Epoch: {} || {}: {:.5f}".format(epoch, name, value)
        print(status_string)

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

    # STATE MANAGEMENT
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
