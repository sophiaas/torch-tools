import torch
import torch.nn as nn
from torch.optim import Adam
import geomstats.backend as gs

from geometry import MatrixLieGroup, gen_se_n_lie_algebra, compute_As
from regularizers import l1_penalty, matrix_2_norm_penalty, lie_alg_sparsity_reg

import ss_models



class ParameterEstimator(nn.Module):
    '''Module which estimates optimal parameters for the transformer.'''

    def __init__(self, in_dimension, num_est_parameters, num_neurons=32):
        super(ParameterEstimator, self).__init__()
        self.in_dimension = in_dimension
        self.num_est_parameters = num_est_parameters
        self.num_neurons = num_neurons
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.in_dimension, out_features= num_neurons),
            nn.ReLU(),
            nn.Linear(in_features = num_neurons, out_features= num_neurons),
            nn.ReLU(),
            nn.Linear(in_features = num_neurons, out_features=self.num_est_parameters),
        )

    def forward(self, x):
        x = x.view(-1, self.in_dimension)
        s = self.mlp.forward(x)
        return s
    
    
class ExplicitTransformer(nn.Module):
    '''Module which transforms input images, e.g. via translation.'''

    def __init__(self, n_features, lie_algebra_dim=1, device = 'cuda'):
        super(ExplicitTransformer, self).__init__()
        self.n_features = n_features
        self.lie_algebra_dim = lie_algebra_dim
        self.lie_algebra = nn.Parameter(data = 10*(2*torch.rand((self.lie_algebra_dim,self.n_features,self.n_features))-1), requires_grad = True)
#         self.sl_lie_algebra = nn.Parameter(data = 2*torch.rand((self.la_dim,self.n_features,self.n_features))-1, requires_grad = True)
#         self.lie_algebra = gs.linalg.expm(self.sl_lie_algebra[0]).unsqueeze(0) #nn.Parameter(data = 2*torch.rand((self.la_dim,self.n_features,self.n_features))-1, requires_grad = True)
        # torch.Tensor([[0,-1],[1,0]]).unsqueeze(0) #

    def normalize_lie_algebra(self, lie_algebra):
        unit_lie_algebra = lie_algebra / gs.linalg.norm(lie_algebra,axis = (1,2)).reshape(-1,1,1)
        return unit_lie_algebra
    
    def transform(self, x0, s):
#         self.lie_algebra.data = normalize_lie_algebra(self.lie_algebra)
        lie_algebra = self.lie_algebra
        
        grp = MatrixLieGroup(lie_algebra, device)
        x, g = grp.action(x0,s)
        return x

    def forward(self, x0, s):
        x = self.transform(x0,s)
#         print("Inside: input size", x.shape)
        return x

    
class QuotientTransformer(nn.Module):
    def __init__(self, in_dimension, num_est_parameters, transformer = None):
        super(QuotientTransformer, self).__init__()
        self.in_dimension = in_dimension
        self.num_est_parameters = num_est_parameters
        self.parameter_estimator = ParameterEstimator(self.in_dimension, self.num_est_parameters)
        self.transformer = transformer

    def forward(self, x):
        s = self.parameter_estimator(x)
        x0 = self.transformer(x,s)
        return x0, s

##############################
# Inheriting Sophia's Models #
############################$$

class ContrastiveTransformer(ss_models.Model):
    def __init__(self,
                 model,
                 experiment_name=None,
                 dataset=None,
                 regularization_function=None,
                 lambd=0.01,
                 optimizer=Adam,
                 lr=1e-5,
                 device='cuda',
                 seed=0):
        
        super().__init__()
        self.model = model
        self.experiment_name=experiment_name,
        self.dataset = dataset,
        self.regularization_function = regularization_function
        self.lambd = lambd
        self.optimizer = optimizer,
        self.lr = lr
        self.device = device,
        self.seed = seed
        
    def forward(self, x):
        out = self.model.forward(x)
        return out
            
    def train_step(self,
                   data,
                   grad=True,
                   output=False):
        
        total_L = 0
        
        ## OUTPUT ##
        if output:
            out = []
        
        ## BATCH ##
        for i, (x, labels) in enumerate(data):
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            ## TRACKING GRADIENTS ##
            if grad:
                self.optimizer.zero_grad()
                output, sh = self.forward(x)
            ## NOT TRACKING GRADIENTS ##
            else: 
                with torch.no_grad():
                    output, sh = self.forward(x)
            
            if output:
                out.append((output, sh))
                    
            L = self.loss(output, labels)

            # let's generalize this to generic regularization
            if self.regularization is not None: # don't overwrite x, even in local scope
                
                reg_s_l1 = l1_penalty(sh) 
                As = compute_As(self.model.lie_algebra,sh)
                reg_norm_As = matrix_2_norm_penalty(As)
                reg_la_sparsity = lie_alg_sparsity_reg(self.model.lie_algebra) 
                # + l1_penalty(s_rnd)
                #0.001*gs.sum(gs.linalg.norm(s_rnd,ord=0.1,axis=1)) 
                #+ 0.1*gs.sum(gs.linalg.norm(s_rnd,axis = 1)**2
                
                L = L + reg_s_l1 + 0.25*reg_norm_As + 20*reg_la_sparsity
            
            if grad:
                L.backward()
                self.optimizer.step()
                
            total_L += L
            
        total_L /= len(data)
        
        if output:
            return total_L, out
        
        else:
            return total_L
        
        

# class ContrastiveTransformer(snm.Model):
    
#     def __init__(self,
#                  size_in,
#                  hdim,
#                  optimizer=Adam,
#                  lr=1e-5,
#                  device='cuda',
#                  seed=0,
#                  experiment_name=None,
#                  dataset=None,
#                  complex_weights=False,
#                  regularization=2,
#                  lambd=0.01):
        
#         #TODO: Make real weights a possibility
        
#         super().__init__()
#         torch.manual_seed(seed)
#         if type(device) == int:
#             device = torch.device('cuda:{}'.format(device))
#         self.device = device
#         self.name = 'bispectral-embedding'
                
#         self.experiment_name = experiment_name
#         self.dataset = dataset
        
#         self.hdim = hdim
#         self.complex = complex_weights
#         self.field = 'complex' if complex_weights else 'real'

#         self.regularization = regularization
#         self.lambd = lambd
#         self.build_layers(size_in, hdim)

#         self.optimizer = optimizer(self.parameters(), lr=lr)
#         self.loss = losses.ContrastiveLoss()

#         self.create_logdir()
#         self.writer = SummaryWriter(self.logdir)
        
#     def build_layers(self,
#                      size_in,
#                      hdim):
        
#         layers = OrderedDict({
#             '0': CubicLinearConstrained(size_in,
#                              hdim,
#                              complex_weights=self.complex,
#                              activation=None,
#                              projection=True,
#                              device=self.device)})

#         self.model = torch.nn.Sequential(layers).to(self.device)
        
#     def forward(self, x):
#         out = self.model.forward(x)
#         return out
            
#     def train_step(self,
#                    data,
#                    grad=True,
#                    output=False):
        
#         total_L = 0
        
#         if output:
#             out = []
        
#         for i, (x, labels) in enumerate(data):
#             x = x.to(self.device)
#             labels = labels.to(self.device)
            
#             if grad:
#                 self.optimizer.zero_grad()
#                 embedding = self.forward(x)
#             else:
#                 with torch.no_grad():
#                     embedding = self.forward(x)
            
#             if output:
#                 out.append(embedding)
                    
#             L = self.loss(embedding, labels)

#             if self.regularization is not None:
#                 params = torch.cat([x.view(-1) for x in self.model[0].parameters()])
#                 reg = self.lambd * torch.norm(params, self.regularization)
#                 L = L + reg
            
#             if grad:
#                 L.backward()
#                 self.optimizer.step()
                
#             total_L += L
            
#         total_L /= len(data)
        
#         if output:
#             return total_L, out
        
#         else:
#             return total_L