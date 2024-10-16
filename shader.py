from typing import Tuple
from objects import sphere_intersect
from operations import unit_vector
from numpy import float32
import numpy as np
import ray


def apply_shading(r: Tuple, t: float32, sphere: Tuple, surface_normal: np.ndarray, lights: Tuple):
	sphere_origin = sphere[0]
	intersection_point = ray.at(r, t)
	ray_dir = r[1]
	unit_direction = unit_vector(ray_dir)
	sum = np.array([0.0, 0.0, 0.0])
	for light in lights:
		light_position = light[0]
		light_intensity = light[1]
		light_color = light[2]
		VL = unit_vector(light_position - intersection_point)
		VE = unit_vector(r[0] - intersection_point)

		### Shading calculation
		ambientCoefficient = 0.1
		diffuseCoefficient = 0.3
		ambientIntensity = ambientCoefficient * light_intensity
		diffuseIntensity = diffuseCoefficient * light_intensity * max(0.0, np.dot(surface_normal,VL))
		shade_val = ambientIntensity + diffuseIntensity

		shadedColor = sphere[2] * shade_val
		sum += shadedColor
	return sum
	
	# # N = unit_vector(ray.at(r, t) - sphere_origin)
	# if t > 0.0:
	# 	# return 0.2*(0.5*np.array([N[0]+1, N[1]+1, N[2]+1])) + np.array([0.5, 0.7, 0.4])
	# return color

# def calculate_shade(light_intensity: float32, surface_normal: np.ndarray, VL):
# 	ambientCoefficient = 0.6
# 	diffuseCoefficient = 1
# 	double ambientIntensity = ambientCoefficient * lightIntensity
# 	double diffuseIntensity = diffuseCoefficient * lightIntensity * max(0.0, surface_normal.dot(VL))