import wandb
from torch.utils.tensorboard import SummaryWriter


class WBLogger:
    def __init__(
        self,
        config,
        project=None,
        entity=None,
        watch_interval=1000,
        log_interval=10,
        step_plotter=None,
        end_plotter=None,
    ):
        """
        watch_interval is in number of batches
        log_interval is in number of epochs
        """
        self.project = project
        self.entity = entity
        self.run = wandb.init(
            config=config,
            project=self.project,
            entity=self.entity,
            resume=True,
            reinit=True,
        )
        self.watch_interval = watch_interval
        self.log_interval = log_interval
        self.step_plotter = step_plotter
        self.end_plotter = end_plotter
        self.is_finished = False

    def begin(self, model, data_loader):
        if self.is_finished:
            run_id = self.run.id
            self.run = wandb.init(
                id=run_id,
                project=self.project,
                entity=self.entity,
                resume=True,
                reinit=True,
            )
            self.is_finished = False
        wandb.watch(model, log_freq=self.log_interval)

    def format_dict(self, results, step_type, epoch, n_examples=None):
        log_dict = {}
        for k, v in results.items():
            log_dict["{}_{}".format(step_type, k)] = v
        log_dict["epoch"] = epoch
        if n_examples is not None:
            log_dict["n_examples"] = n_examples
        return log_dict

    def log_step(self, log_dict, variable_dict, step_type, epoch, n_examples=None):
        if epoch % self.log_interval == 0:
            log_dict_formatted = self.format_dict(
                log_dict, step_type, epoch, n_examples
            )
            wandb.log(log_dict_formatted)

            if self.step_plotter is not None:
                plots = self.step_plotter.plot(variable_dict)
                plots_formatted = self.format_dict(plots, step_type, epoch, n_examples)
                wandb.log(plots_formatted)

    def save_checkpoint(self, model, iter):
        return
        # torch.save(
        #     model,
        #     os.path.join(wandb.run.dir, "checkpoints", "checkpoint_{}.pt".format(iter)),
        # )

    def end(self, variable_dict):
        data_loader = variable_dict["data_loader"]
        train_inds = data_loader.train.sampler.indices
        train_data = data_loader.train.dataset.data[train_inds]
        val_inds = data_loader.val.sampler.indices
        val_data = data_loader.val.dataset.data[val_inds]

        variable_dict = {
            "model": variable_dict["model"],
            "train_data": train_data,
            "val_data": val_data,
        }

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
