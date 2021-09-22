import os
import pickle
import inspect
import torch
import copy

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