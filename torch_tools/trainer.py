import datetime
import os
import pickle
import torch
from torch.nn import ParameterDict
import torch_tools.utils as utils
from inspect import signature


class Trainer(torch.nn.Module):
    def __init__(
        self,
        model,
        loss,
        logger,
        optimizer_config,
        regularizer=None,
        normalizer=None,
        device="cuda",
    ):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.device = device
        self.loss = loss
        self.epoch = 0
        self.n_examples = 0
        self.logger = logger

        optimizer_config["params"]["params"] = model.parameters()
        self.optimizer = optimizer_config.build()

    def step(self, data_loader, grad=True):
        '''Compute a single step of training.
        
        This example is a minimal implementation of the `step` function
        for a classification problem with a simple regularization on 
        the model parameters.
        
        Parameters
        ----------
        data_loader : torch.utils.data.dataloader.DataLoader
        grad : boolean
            required argument to switch between training and evaluation
        
        Returns
        -------
        results_dict : dictionary with losses to be logged by the trainer/logger
            format - {'total_loss': total_loss, 'l1_penalty': l1_penalty, ...}
            Your dictionary must contain a key called `total_loss` 
            
        '''
        results_dict = {'loss': 0, 'reg_loss': 0, 'total_loss': 0}
        for i, (x, labels) in enumerate(data_loader):
            loss = 0
            reg_loss = 0
            total_loss = 0
            
            x = x.to(self.device)
            labels = labels.to(self.device)

            if grad:
                self.optimizer.zero_grad()
                out = self.model.forward(x)

            else:
                with torch.no_grad():
                    out = self.model.forward(x)

            # Compute loss term without regularization terms (e.g. classification loss)
            loss += self.loss(out, labels)
            results_dict['loss'] += loss
            total_loss += loss

            # Compute regularization penalty terms (e.g. sparsity, l2 norm, etc.)
            if self.regularizer:
                variable_dict = {'x': x, 'out': out, 'params': self.model.state_dict()}
                reg_loss += self.regularizer(variable_dict)
                results_dict['reg_loss'] += reg_loss
                total_loss += reg_loss

            if grad:
                total_loss.backward()
                self.optimizer.step()
            
            results_dict['total_loss'] += total_loss
            
        # Normalize loss terms for the number of samples/batches in the epoch (optional)
        n_samples = len(data_loader)
        for key in results_dict.keys():
            results_dict[key] /= n_samples

        return results_dict

    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        print_interval=1,
    ):
        self.logger.begin(self.model, data_loader)

        try:
            for i in range(start_epoch, start_epoch + epochs + 1):
                self.epoch = i
                train_results = self.step(data_loader.train, grad=True)
                self.logger.log_step(
                    results=train_results,
                    step_type="train",
                    epoch=self.epoch,
                    n_examples=self.n_examples,
                )
                
                if data_loader.val is not None:
                    validation_results = self.evaluate(data_loader.val)
                    self.logger.log_step(
                        results=validation_results,
                        step_type="val",
                        epoch=self.epoch,
                        n_examples=self.n_examples,
                    )

                if i % print_interval == 0 and print_status_updates == True:
                    if data_loader.val is not None:
                        self.print_update(train_results, validation_results)
                    else:
                        self.print_update(train_results)

                self.logger.save_checkpoint(self.model, self.epoch)

                self.n_examples += len(data_loader.train.dataset)

        except KeyboardInterrupt:
            print("Stopping and saving run at epoch {}".format(i))

        self.logger.end(self.model, data_loader)

    def resume(self, data_loader, epochs):
        self.train(data_loader, epochs, start_epoch=self.epoch)
        

    @torch.no_grad()
    def evaluate(self, data_loader):
        results = self.step(data_loader, grad=False)
        return results

    def print_update(self, result_dict_train, result_dict_val = None):

        update_string = 'Epoch {} ||  N Examples {} || Train Total Loss {:0.5f}'.format(
            self.epoch,
            self.n_examples,
            result_dict_train["total_loss"]
        )
        if result_dict_val:
            update_string += ' || Validation Total Loss {:0.5f}'.format(result_dict_val["total_loss"])
        print(update_string)


# # below: all accumulator, all in step
    
#     def reset_accumulator(self):
#         d = {"total_loss": 0}

#         if self.regularizer is not None:
#             d["regularization_loss"] = 0

#         return d

#     def accumulate_loss(self, x, out, labels, accumulator):
#         L = self.loss(out, labels)
#         accumulator["total_loss"] += L
#         L_reg, accumulator = self.get_regularizer_loss(x, out, accumulator)
#         L += L_reg
#         return L, accumulator

#     def average_loss(self, accumulator, n_data_points):
#         for key in accumulator.keys():
#             accumulator[key] /= n_data_points
#         return accumulator

#     def get_regularizer_loss(self, x, out, accumulator):
#         if self.regularizer is not None:
#             L_reg = self.regularizer(self.model.state_dict())
#             accumulator["regularization_loss"] += L_reg
#         else:
#             L_reg = 0.0
#         return L_reg, accumulator

