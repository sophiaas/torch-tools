import geomstats.backend as gs


def l1_penalty(x):
    return gs.sum(gs.sum(gs.abs(x),axis = 1))

def matrix_2_norm_penalty(matrix):
    return gs.sum(gs.linalg.norm(matrix, axis=(1,2))**2)

def lie_alg_sparsity_reg(matrices): #huh?
    return gs.sum(gs.abs(gs.linalg.norm(matrices,axis = (1,2))))