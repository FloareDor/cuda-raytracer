import numpy as np
from math import sqrt
from numba import njit

@njit
def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)

@njit
def magnitude(v):
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

