from typing import Tuple
from numpy import float32

# origin = ray[0]
# direction = ray[1]
def at(ray: Tuple, t: float32):
    return ray[0] + (t * ray[1])