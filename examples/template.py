from torch_tools.generic import ParameterDict
from torch_tools.model import Model
from torch_tools.experiment import Experiment
from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer
from transform_dataset.torch.vector import Translation
from torch.optim import Adam
from torch_tools.functional import l1_norm, matrix_2_norm


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


optimizer_params = ParameterDict(
    {
        "lr": 1e-3,
    }
)

optimizer = Adam(**optimizer_params)

reg1 = Regularizer(l1_norm, "s", 0.7)
reg2 = Regularizer(matrix_2_norm, "A", 0.1)
regularizer = MultiRegularizer([reg1, reg2])
