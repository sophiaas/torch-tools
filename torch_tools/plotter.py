from torch_tools.utils import get_variables

class Plotter:
    def __init__(self, function, variables, f_params):
        self.function = function
        self.variables = variables
        self.f_params = f_params
        
    def plot(self, variable_dict):
        args_list = get_variables(variable_dict, self.variables)
        return self.function(*args_list, **self.f_params)
    
    
class MultiPlotter:
    def __init__(self, plotter_configs):
        self.plotters = [x.build() for x in plotter_configs]
        
    def plot(self, variable_dict):
        result_dict = {}
        for plotter in self.plotters:
            result_dict.update(plotter.plot(variable_dict))
        return result_dict