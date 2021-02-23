import os
import pickle


class ParameterDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def save_pickle(param_dict, path, fname):
    final_path = os.path.join(path, fname)
    with open(final_path, "wb") as f:
        pickle.dump(f, param_dict)
