import numpy as np
from numba import njit

@njit
def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)