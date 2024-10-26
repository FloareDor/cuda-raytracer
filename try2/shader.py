from typing import Tuple
from objects import sphere_intersect
from operations import unit_vector, magnitude
from numpy import float32
import numpy as np
import ray
from numba import jit
from math import pow

@jit
def apply_shading(r: Tuple, t: float32, sphere: Tuple, surface_normal: np.ndarray, lights: Tuple):

    ambientCoefficient = 0.1
    diffuseCoefficient = 0.5  # Increased from 0.3 for more visible diffuse effect
    specularCoefficient = 0.5  # Increased from 0.0 to add specular highlights
    shininess = 32  # Adjusted for a more typical specular highlight size
    # sphere_origin = sphere[0]
    intersection_point = ray.at(r, t)
    ray_dir = r[1]
    # unit_direction = unit_vector(ray_dir)
    sum = np.array([0.0, 0.0, 0.0])
    for light in lights:
        light_position = light[0]
        light_intensity = light[1]
        light_color = light[2]
        VL = unit_vector(light_position - intersection_point)
        VE = unit_vector(r[0] - intersection_point)
        ### Shading calculation
        ambientIntensity = ambientCoefficient * light_intensity
        diffuseIntensity = diffuseCoefficient * light_intensity * max(0.0, np.dot(surface_normal,VL))

        VH = unit_vector(VL + VE)
        specularIntensity = specularCoefficient * light_intensity * pow(max(0.0, np.dot(surface_normal,VH)), shininess)

        shade_val = ambientIntensity + diffuseIntensity + specularIntensity

        shadedColor = sphere[2] * light_color * shade_val
        sum += shadedColor

    return np.clip(sum, 0.0, 1.0).astype(np.float32)