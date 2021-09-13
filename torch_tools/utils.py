import os
import pickle
import inspect


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


def construct_trainer(master_config, logger_config, data_project=None, entity=None):
    """
    master_config has the following format:

    master_config = {
        "dataset": dataset_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "loss": loss_config,
        "data_loader": data_loader_config,
    }
    
    with optional regularizer and normalizer
    """
    if data_project is not None and entity is not None:
        dataset = master_config["dataset"].build()
    else:
        dataset = load_or_create_dataset(master_config["dataset"], 
                                         data_project, 
                                         entity)
    data_loader = master_config["data_loader"].build()
    data_loader.load(dataset)

    master_config["model"]["size_in"] = dataset.dim
    model = master_config["model"].build()
    
    loss = loss_config.build()
    
    logger_config["config"] = master_config
    logger = logger_config.build()
    
    optimizer_config = copy.deepcopy(master_config["optimizer"])
    optimizer_config["params"]["params"] = model.parameters()
    optimizer = optimizer_config.build()

    train_config = Config(
        {
            "type": Trainer,
            "params": {
                "model": model,
                "loss": loss,
                "logger": logger,
                "optimizer": optimizer,
            },
        }
    )

    trainer = train_config.build()
    return trainer, data_loader, master_config