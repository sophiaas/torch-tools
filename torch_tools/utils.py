import os
import pickle
import inspect
import torch
import copy
import wandb


class ParameterDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_pickle(object, path, fname):
    final_path = os.path.join(path, fname)
    with open(final_path, "wb") as f:
        pickle.dump(object, f)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def get_variables(variable_dict, variables):
    return [variable_dict[v] for v in variables]


def load_checkpoint(logdir):
    from torch_tools.config import Config
    checkpoint = torch.load(logdir)
    if not hasattr(checkpoint, "model"):
        trainer = checkpoint["trainer"]
        model_config = Config(trainer.logger.config["model"])
        optimizer_config = Config(copy.deepcopy(trainer.logger.config["optimizer"]))
        trainer.model = model_config.build()
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_config["params"]["params"] = trainer.model.parameters()
        trainer.optimizer = optimizer_config.build()
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = trainer
    return checkpoint


def load_wandb_checkpoint(entity, project, run_id):
    api = wandb.Api()
    run = api.run("{}/{}/{}".format(entity, project, run_id))
    epoch = run.summary.epoch
    loaded = False
    while not loaded:
        if epoch < 0:
            raise Exception("No saved checkpoints.")
        try:
            checkpoint_path = wandb.restore('checkpoints/checkpoint_{}.pt'.format(epoch), run_path="{}/{}/{}".format(entity, project, run_id)).name
            loaded = True
        except:
            epoch -= 1
    trainer = load_checkpoint(checkpoint_path)
    return trainer