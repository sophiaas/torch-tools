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