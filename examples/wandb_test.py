from spectral_networks.nn.models import BispectralEmbedding
from transform_datasets.patterns.synthetic import *
from transform_datasets.transforms import *
from transform_datasets.utils.wandb import load_or_create_dataset
from torch_tools.trainer import Trainer
from torch_tools.logger import WBLogger
from torch_tools.config import Config
from torch_tools.data import TrainValLoader
from pytorch_metric_learning import losses, distances
from torch.optim import Adam

PROJECT = "datasets"
ENTITY = "naturalcomputation"
DEVICE = "cuda:0"
SEED = 0

"""
DATASET
"""

dataset_config = Config(
    {
        "type": HarmonicsS1,
        "params": {"dim": 256, "n_classes": 10, "seed": 5},
    }
)

transforms_config = {
    "0": Config(
        {
            "type": CyclicTranslation1D,
            "params": {
                "fraction_transforms": 1.0,
                "sample_method": "linspace",
            },
        }
    ),
    "1": Config(
        {
            "type": UniformNoise,
            "params": {"n_samples": 1, "magnitude": 0.1},
        }
    ),
}


tdataset_config = {"dataset": dataset_config, "transforms": transforms_config}

dataset = load_or_create_dataset(tdataset_config, PROJECT, ENTITY)

"""
DATA_LOADER
"""

data_loader_config = Config(
    {
        "type": TrainValLoader,
        "params": {
            "batch_size": 32,
            "fraction_val": 0.2,
            "num_workers": 1,
            "seed": SEED,
        },
    }
)

data_loader = data_loader_config.build()
data_loader.load(dataset)

"""
MODEL
"""
model_config = Config(
    {
        "type": BispectralEmbedding,
        "params": {
            "size_in": dataset.dim,
            "hdim": 256,
            "seed": SEED,
        },
    }
)
model = model_config.build()

"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, "params": {"lr": 0.001}})
# optimizer = optimizer_config.build()

"""
LOSS
"""
loss_config = Config(
    {
        "type": losses.ContrastiveLoss,
        "params": {
            "pos_margin": 0,
            "neg_margin": 1,
            "distance": distances.LpDistance(),
        },
    }
)
loss = loss_config.build()

"""
MASTER CONFIG
"""

config = {
    "dataset": dataset_config,
    "model": model_config,
    "optimizer": optimizer_config,
    "loss": loss_config,
    "data_loader": data_loader_config,
}

"""
LOGGING
"""
logging_config = Config(
    {
        "type": WBLogger,
        "params": {
            "config": config,
            "project": PROJECT,
            "entity": ENTITY,
            "log_interval": 10,
            "watch_interval": 10 * len(data_loader.train),
        },
    }
)

logger = logging_config.build()

"""
TRAINER
"""

training_config = Config(
    {
        "type": Trainer,
        "params": {
            "model": model,
            "loss": loss,
            "logger": logger,
            "device": DEVICE,
            "optimizer_config": optimizer_config,
        },
    }
)

trainer = training_config.build()
