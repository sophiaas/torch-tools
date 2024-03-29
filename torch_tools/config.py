from torch_tools.utils import get_default_args


class Config(dict):
    def __init__(self, config):
        """
        Takes in a dictionary config of the following form:

        config = {
            "type": Class,
            "params": {
                "param1": val,
                "param2": val
                }
        }
        """
        config = self.fill_defaults(config)
        super().__init__(**config)
        self.__dict__ = self

    def fill_defaults(self, config):
        defaults = get_default_args(config["type"])
        for k, v in defaults.items():
            if k not in config["params"]:
                config["params"][k] = v
        return config

    def build(self):
        return self["type"](**self["params"])
