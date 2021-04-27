import datetime
import os
import pickle
import torch
from torch.nn import ParameterDict
from torch.utils.tensorboard import SummaryWriter
import torch_tools.generic as generic
from inspect import signature


class Experiment(torch.nn.Module):
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        loss_function,
        regularizer=None,
        normalizer=None,
        device="cuda",
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.device = device
        self.loss_function = loss_function
        self.epoch = 0

    def on_begin(self, data, writer):
        raise NotImplementedError

    def on_end(self, data, writer):
        raise NotImplementedError
        
    def reset_accumulator(self):
        d =  {'total_loss': 0}
        
        if self.regularizer is not None:
            d['regularization_loss'] = 0
            
        return d
        
    def forward(self, x):
        out = self.model.forward(x)
        return out

    def accumulate_loss(self, x, out, labels, accumulator):
        
        L = self.loss_function(out, labels)
        accumulator['total_loss'] += L

        if self.regularizer is not None:
            L_reg = self.regularizer(self.model.state_dict())
            L += L_reg
            accumulator['regularization_loss'] += L_reg
            
        return L, accumulator
    
    def step(self, data, grad=True):
        
        accumulator = self.reset_accumulator()

        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)

            if grad:
                self.optimizer.zero_grad()
                out = self.forward(x)
            
            else:
                with torch.no_grad():
                    out = self.forward(x)
            
            L, accumulator = self.accumulate_loss(x, out, labels, accumulator)

            if grad:
                L.backward()
                self.optimizer.step()
                
            if self.normalizer is not None:
                self.normalizer(self.model.state_dict())
            
        accumulator = self.average_loss(accumulator, len(data))

        return accumulator
    
    def average_loss(self, accumulator, n_datapoints):
        for key in accumulator.keys():
            accumulator[key] /= n_datapoints
        return accumulator

    # CONVENIENCE WRAPPER
    @torch.no_grad()
    def evaluate(self, data):
        results = self.step(data, grad=False)
        return results

    # TRAINING
    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        checkpoint_interval=10,
        print_interval=1,
    ):
        writer = self.begin(data=data_loader)

        try:
            for i in range(start_epoch, epochs + 1):
                self.epoch = i
                train_results = self.step(data_loader.train, grad=True)
                self.log_step(results=train_results, step_type="train", writer=writer)

                if data_loader.val is not None:
                    validation_results = self.evaluate(data_loader.val)
                    self.log_step(
                        results=validation_results, step_type="val", writer=writer
                    )

                if i % print_interval == 0 and print_status_updates == True:
                    self.print_update(train_results, validation_results)

                if i % checkpoint_interval == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("Stopping and saving run at epoch {}".format(i))

        self.end(data=data_loader, writer=writer)

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
            self.save_data_params(data)
            writer = SummaryWriter(self.logdir)
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

        try:
            self.on_begin(data=data, writer=writer)
        except NotImplementedError:
            pass

        return writer

    def end(self, data, writer):
        # THIS ORDER MATTERS: TB BUG?
        try:
            self.on_end(data=data, writer=writer)
        except NotImplementedError:
            pass
        self.log_hyperparameters(data, writer=writer)

    # TB LOGGING
    def log_step(self, results, step_type, writer):
        for (name, value) in results.items():
            self.log_scalar(
                group="loss_reg_{}".format(step_type),
                name=name,
                value=value,
                writer=writer,
            )

    def log_scalar(self, group, name, value, writer):
        writer.add_scalar("{}/{}".format(group, name), value, self.epoch)

    # TB HPARAMS
    def log_hyperparameters(self, data, writer):
        opt_hparams = self.get_optimizer_hparams()
        data_hparams = self.get_data_hparams(data)

        if self.regularizer is not None:
            reg_hparams = self.get_regularizer_hparams()
            hparam_dict = {**opt_hparams, **data_hparams, **reg_hparams}

        else:
            hparam_dict = {**opt_hparams, **data_hparams}

        metric_dict = self.get_hparam_metrics(data)

        writer.add_hparams(hparam_dict, metric_dict)

    def get_hparam_metrics(self, data):
        train_results = self.evaluate(data.train)
        metric_dict = {"hparam/train_loss": train_results["total_loss"]}
        if data.val is not None:
            validation_results = self.evaluate(data.val)
            metric_dict["hparam/val_loss"] = validation_results["total_loss"]
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

    def save_data_params(self, data_loader):
        # TODO: Find a less brittle method to accomplish the same thing
        data_loader_signature = signature(data_loader.__init__)
        data_loader_dict = {}
        data_loader_dict["args"] = {
            arg.name: data_loader.__dict__[arg.name]
            for arg in data_loader_signature.parameters.values()
        }
        data_loader_dict["type"] = type(data_loader)
        torch.save(data_loader_dict, os.path.join(self.logdir, "data_loader_params.pt"))

        dataset_signature = signature(data_loader.train.dataset.__init__)
        dataset_dict = {}
        dataset_dict["args"] = {
            arg.name: data_loader.train.dataset.__dict__[arg.name]
            for arg in dataset_signature.parameters.values()
        }
        dataset_dict["type"] = type(data_loader.train.dataset)
        torch.save(dataset_dict, os.path.join(self.logdir, "dataset_params.pt"))

    def print_update(self, training_loss, validation_loss):
        print(
            "Epoch {}  ||  Training Loss: {:0.5f}  |  Validation Loss: {:0.5f}".format(
                self.epoch, training_loss["total_loss"], validation_loss["total_loss"]
            )
        )

    def save_checkpoint(self):
        torch.save(
            self,
            os.path.join(
                self.logdir, "checkpoints", "checkpoint_{}.pt".format(self.epoch)
            ),
        )

    def resume(self, data_loader, epochs):
        self.train(data_loader, epochs, start_epoch=checkpoint)
