from torch_tools.generic import ParameterDict
from torch_tools.model import Model
from torch_tools.experiment import Experiment
from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer
from transform_dataset.torch.vector import Translation


model_params = ParameterDict({})

data_params = ParameterDict(
    {
        "c_classes": 10,
        "digits": np.arange(10),
        "max_transformation_steps": 10,
        "dim": 32,
        "noise": 0.2,
        "seed": 0,
    }
)
