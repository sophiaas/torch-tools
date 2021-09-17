import wandb
from torch.utils.tensorboard import SummaryWriter
import torch
import os


class WBLogger:
    def __init__(
        self,
        config,
        project=None,
        data_project=None,
        entity=None,
        watch_interval=1,
        log_interval=1,
        plot_interval=1,
        checkpoint_interval=10,
        step_plotter=None,
        end_plotter=None,
    ):
        """
        watch_interval is in number of batches
        log_interval is in number of epochs
        """
        self.project = project
        self.data_project = data_project
        self.entity = entity
#         self.run = wandb.init(
#             config=config,
#             project=self.project,
#             entity=self.entity,
#             resume="allow",
#             reinit=True,
#         )
        self.watch_interval = watch_interval
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.step_plotter = step_plotter
        self.end_plotter = end_plotter
        self.is_finished = False

    def begin(self, model, data_loader):
#         if self.is_finished:
#             run_id = self.run.id
#             self.run = wandb.init(
#                 id=run_id,
#                 project=self.project,
#                 entity=self.entity,
#                 resume="allow",
#                 reinit=True,
#             )
#             self.is_finished = False
        wandb.watch(model, log_freq=self.watch_interval, log_graph=False)
        os.makedirs(os.path.join(wandb.run.dir, "checkpoints"), exist_ok=True)
        

    def log_step(self, log_dict, variable_dict, epoch, val_log_dict=None, n_examples=None):
        full_log_dict = {}
        if epoch % self.log_interval == 0:
            if val_log_dict is not None:
                for k in log_dict:
                    full_log_dict['train_'+k] = log_dict[k]
                    full_log_dict['val_'+k] = val_log_dict[k]
            else:
                full_log_dict = log_dict
                
            full_log_dict["epoch"] = epoch
            if n_examples is not None:
                full_log_dict["n_examples"] = n_examples

            if self.step_plotter is not None and variable_dict is not None and (epoch % self.plot_interval == 0):
                plots = self.step_plotter.plot(variable_dict)
                full_log_dict.update(plots)

            wandb.log(full_log_dict)

    def save_checkpoint(self, model, iter):
        if iter % self.checkpoint_interval == 0:
            torch.save(
                model,
                os.path.join(wandb.run.dir, "checkpoints", "checkpoint_{}.pt".format(iter)),
            )
            wandb.save(os.path.join(wandb.run.dir, "checkpoints", "checkpoint_{}.pt".format(iter)), base_path=wandb.run.dir, policy="now")

    def end(self, variable_dict):
        if self.end_plotter is not None:
            plots = self.end_plotter.plot(variable_dict)
            wandb.log(plots)
                        
        wandb.finish()
        self.is_finished = True


class TBLogger:
    def __init__(self, log_interval, logdir=None):
        self.log_interval = log_interval
        self.logdir = logdir

    def begin(self, model, data_loader):
        try:
            self.create_logdir()
            self.writer = SummaryWriter(self.logdir)
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")

    def end(self, model, data_loader):
        self.log_hyperparameters(data_loader)

    def create_logdir(self):
        if self.logdir is None:
            self.logdir = os.path.join(
                "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        os.makedirs(self.logdir, exist_ok=True)
        os.mkdir(os.path.join(self.logdir, "checkpoints"))

    def log_step(self, results, step_type, writer):
        for (name, value) in results.items():
            self.log_scalar(
                group="loss_reg_{}".format(step_type),
                name=name,
                value=value,
                writer=self.writer,
            )

    def log_scalar(self, group, name, value, iter):
        self.writer.add_scalar("{}/{}".format(group, name), value, iter)

    def save_checkpoint(self, model, iter):
        torch.save(
            model,
            os.path.join(self.logdir, "checkpoints", "checkpoint_{}.pt".format(iter)),
        )

    def log_hyperparameters(self, data):
        opt_hparams = self.get_optimizer_hparams()
        data_hparams = self.get_data_hparams(data)

        if self.regularizer is not None:
            reg_hparams = self.get_regularizer_hparams()
            hparam_dict = {**opt_hparams, **data_hparams, **reg_hparams}

        else:
            hparam_dict = {**opt_hparams, **data_hparams}

        metric_dict = self.get_hparam_metrics(data)

        self.writer.add_hparams(hparam_dict, metric_dict)

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
