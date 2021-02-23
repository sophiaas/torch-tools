import geomstats
import geomstats.backend as gs
import numpy as np
import matplotlib.pyplot as plt
from geometry import compute_As, compute_g, compute_action, gen_se_n_lie_algebra, MatrixLieGroup
import torch

torch.set_default_dtype(torch.float32)

# TODO: Add subclass for matrix translation group
# TODO: Add subclass for pure rotation group
# TODO: generalize make_classification to nD
# TODO: generalize make_classification to select random groups
# TODO: refactor SEn and translation groups to append and remove trailing 1

# def make_classification(n_samples=100, n_classes=5, n_features=2, grp_type='translation', x_range=10, cov_scale=1):
    
#     A = gen_se_n_lie_algebra(2)
#     grp = MatrixLieGroup(A, device = 'cpu')

#     x_means = x_range*(2*np.random.rand(n_classes,n_features) - 1)

#     # generate s params
#     s_means = 0*(2*np.random.rand(n_classes, grp.lie_algebra_dim) - 1)

#     s_covs = gs.zeros(n_classes, grp.lie_algebra_dim, grp.lie_algebra_dim) # restricting to last two dim
    
#     if grp_type == 'translation':
#         P = 2*torch.rand(n_classes,grp.lie_algebra_dim - 1,grp.lie_algebra_dim - 1)-1
#         rand_diag = gs.stack([torch.diag(2*gs.random.rand(grp.lie_algebra_dim - 1)) for i in range(n_classes)])
#         cov = torch.matmul(torch.matmul(P,rand_diag),P.transpose(1,2))
#         s_covs[:,1:,1:] = cov_scale*cov
#     elif grp_type == 'rotation':
#         spd = geomstats.geometry.spd_matrices.SPDMatrices(1)
#         s_covs[:,0,0] = cov_scale*spd.random_uniform(n_samples = n_classes).reshape(-1)
#     else:
#         P = 2*torch.rand(grp.n_classes,grp.lie_algebra_dim,grp.lie_algebra_dim)-1
#         rand_diag = gs.stack([torch.diag(2*gs.random.rand(grp.lie_algebra_dim)) for i in range(n_classes)])
#         cov = torch.matmul(torch.matmul(P,rand_diag),P.transpose(1,2))
#         s_covs = cov_scale*cov
        
#     # generate samples
#     X = []
#     Y = []
#     for i in range(n_classes):
#         x0 = x_means[[i]]
#         s = grp.random_normal_s(s_means[i], s_covs[i], n_samples=n_samples)
#         x, g = grp.action(x0, s)
#         y = i*gs.ones(n_samples)
#         X.append(x)
#         Y.append(y)

#     X = gs.concatenate(X,axis = 0)
#     Y = gs.concatenate(Y,axis = 0)
    
#     return X,Y


def make_cov(dim):
    P = 2*torch.rand(dim, dim)-1
    rand_diag = torch.diag(gs.random.rand(dim))
    cov = torch.matmul(torch.matmul(P,rand_diag),P.T)
    return cov

def random_normal(mean, cov, n_samples=100):
    distr = torch.distributions.MultivariateNormal(mean[0], cov)
    x = distr.sample((n_samples,))
    return x

def gen_factored_observations(group, x0, s_trans, s_grp):    
    x_grp, g_grp = group.action(x0, s_grp)
    x_trans = s_trans + x_grp
    return x_trans

def make_random_grp_cluster(grp, n_samples = 1000, x0 = None, s_trans = None):
    grp_lie_algebra_dim = grp.lie_algebra_dim
    n_features = grp.lie_group_matrix_dim
    
    trans_lie_algebra_dim = n_features
    
    if x0 is None:
        x0 = 2*gs.random.rand(1,n_features)-1
    
    if s_trans is None:
        s_trans = 5*(2*gs.random.rand(1,trans_lie_algebra_dim)-1)
    
    s_grp_mean = gs.zeros(1, grp_lie_algebra_dim)
    s_grp_cov = 10*make_cov(grp_lie_algebra_dim)
    s_grp = random_normal(s_grp_mean, s_grp_cov, n_samples)
    
    x = gen_factored_observations(grp, x0, s_trans, s_grp)
    return x, s_trans, s_grp


def make_classification(n_samples=100, n_classes=5, n_features=2, affine=True, grp_type='translation', x_range=10, cov_scale=1, device = 'cuda'):
    
    lie_algebra = gs.array([[0,-1],[1,0]]).unsqueeze(0)
    grp = MatrixLieGroup(lie_algebra, device=device)

    if affine == True:
        s_trans = None
    else:
        s_trans = gs.zeros(1, n_features)
        
    # generate samples
    X = []
    Y = []
    for i in range(n_classes):
        x, s_tr, s_grp = make_random_grp_cluster(grp, n_samples = n_samples, s_trans = s_trans)
        y = i*gs.ones(n_samples)
        X.append(x)
        Y.append(y)

    X = gs.concatenate(X,axis = 0)
    Y = gs.concatenate(Y,axis = 0)
    
    return X,Y

##### Torch Dataset Functions

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

def split_dataset(dataset, 
                  batch_size,
                  validation_split=0.2, 
                  seed=0,
                  n_gpus = 0):
    
    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               sampler=train_sampler,
                                               num_workers = 4*n_gpus,)
#                                                pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    num_workers = 4*n_gpus)
                                                  #  pin_memory=True)
    return train_loader, validation_loader


class DatasetWrapper(Dataset):
    def __init__(self,X,Y):
        self.data = X
        self.labels = Y
        
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)