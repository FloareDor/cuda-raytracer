import numpy as np
from numba import njit

@njit
def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm