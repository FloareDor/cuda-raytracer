from numba import jit
import numpy as np

def write_color(pixel_color: np.ndarray):
	return (pixel_color * 255.99).astype(np.float32)