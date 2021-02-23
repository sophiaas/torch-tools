from collections import OrderedDict
import datetime
import os

from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch
from torch.utils.tensorboard import SummaryWriter

# from torch.optim import Adam
 
class Experiment(torch.nn.Module): # should we make this inherit Module?
    def __init__(self, experiment_name, task_name, dataset_name, model, dataset, optimizer, optimizer_params, regularizers, regularizer_params, device = 'cuda'):
        self.experiment_name = experiment_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.regularizers = regularizers
        self.regularizer_params = regularizer_params
        self.device = device
        self.writer = None  # pass this in or create it inside?
        try:
            self.logdir = create_logdir()
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")
    
    def create_logdir(self):
        logdir = os.path.join('logs', 
                              self.experiment_name, 
                              self.model.__class__.__name__,
                              self.dataset_name)
        
        os.makedirs(logdir, exist_ok=True)

        logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.mkdir(logdir)
        os.mkdir(os.path.join(logdir, 'checkpoints'))
        return logdir

    def log(self, epoch, training_loss, validation_loss):
        print('Epoch: {} || Training L: {:.5f} | Validation Loss: {:.5f}'.format(epoch, training_loss, validation_loss))

        self.writer.add_scalar('loss/train', training_loss, epoch)
        self.writer.add_scalar('loss/val', validation_loss, epoch)
        # what else?

        if epoch % 10 == 0: #make the interval a parameter or a function of the total num_epochs?
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, 
                        os.path.join(self.logdir, 'checkpoints', 'checkpoint_{}.pt'.format(epoch)))
    
    # right now this is an overridden method by the inherited class...
    # do we want models to inherit experiment or for experiments to ingest models?
    #   ingest? -> 1) model provides its own train step, this calls it
    #              2) New experiment subclass for each train step type (meh), idk
    #              3) new class
    def train_step(self,
                   data,
                   grad=True,
                   output=False):
        
        raise NotImplementedError

    # do we want to pass in data and let the train loading scheme be passed in/parameterized in constructor?
    def train(self, 
            train_loader, 
            validation_loader,
            epochs,
            start_epoch=0):

        for i in range(start_epoch, epochs):
            training_loss = self.train_step(train_loader, grad=True)
            validation_loss = self.train_step(validation_loader, grad=False)
            self.log(i, training_loss, validation_loss)


#this depends implicitly on self.model, hdim being set by a child
class Model(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.field = None
        self.name = None
        self.writer = None
        
    def create_logdir(self):
        
        logdir = os.path.join('logs', 
                              self.experiment_name, 
                              self.dataset,
                              self.name, 
                              self.field, 
                              str(self.hdim))
        
        os.makedirs(logdir, exist_ok=True)

        logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.mkdir(logdir)
        os.mkdir(os.path.join(logdir, 'checkpoints'))
        self.logdir = logdir
        
    def log(self, epoch, training_loss, validation_loss):
        print('Epoch: {} || Training L: {:.5f} | Validation Loss: {:.5f}'.format(epoch, training_loss, validation_loss))

        self.writer.add_scalar('loss/train', training_loss, epoch)
        self.writer.add_scalar('loss/val', validation_loss, epoch)

        if epoch % 10 == 0:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, 
                        os.path.join(self.logdir, 'checkpoints', 'checkpoint_{}.pt'.format(epoch)))
            
    def train_step(self,
                   data,
                   grad=True,
                   output=False):
        
        raise NotImplementedError
            
    def train(self, 
              train_loader, 
              validation_loader,
              epochs,
              start_epoch=0):

        for i in range(start_epoch, epochs):
            training_loss = self.train_step(train_loader, grad=True)
            validation_loss = self.train_step(validation_loader, grad=False)
            self.log(i, training_loss, validation_loss)
            
    def load_checkpoint(self,
                        path, 
                        checkpoint):
        state_dict = torch.load(path+'checkpoints/checkpoint_{}.pt'.format(checkpoint))
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
    def resume(self,
               path,
               checkpoint,
               train_loader,
               validation_loader, 
               epochs):
        self.load_checkpoint(path, checkpoint)
        self.train(train_loader, validation_loader, epochs, start_epoch=checkpoint)

    def eval(self,
             test_loader,
             output=True):
        out = self.train_step(test_loader, grad=False, output=True)
        return out        
            