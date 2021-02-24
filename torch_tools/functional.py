import geomstats as gs


def l1_norm(x):
    return gs.sum(gs.sum(gs.abs(x), axis=1))


def matrix_2_norm(matrix):
    return gs.sum(gs.linalg.norm(matrix, axis=(1, 2)) ** 2)
