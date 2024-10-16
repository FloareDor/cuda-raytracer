import numpy as np
from typing import Tuple
from math import sqrt
from numpy import float32 as npfloat32
from operations import unit_vector

def hit_anything(ray: Tuple, spheres: Tuple, t_max: npfloat32):
	hit_anything = False
	closest_t = t_max

	for sphere in spheres:
		# sphere_origin = spheres[i]
		# sphere_radius = spheres[i+1]
		# sphere_color = spheres[i+3]
		closest_sphere = sphere
		surface_normal = np.array([0, 0, 0])
		t, hit_normal = sphere_intersect(ray, sphere)
		if t > 0.01 and t < closest_t:
			closest_t = t
			closest_sphere = sphere
			hit_anything = True
			surface_normal = hit_normal

	return hit_anything, closest_t, closest_sphere, surface_normal


def sphere_intersect(ray: Tuple, sphere: Tuple):
	sphere_origin = sphere[0]
	sphere_radius = sphere[1]
	sphere_color = sphere[2]
	ray_dir = ray[1]
	ray_origin = ray[0]
	a = np.dot(ray_dir, ray_dir)
	b = np.dot(-2 * ray_dir, sphere_origin - ray_origin)
	c = np.dot((sphere_origin - ray_origin), (sphere_origin - ray_origin)) - (
		sphere_radius**2
	)

	discriminant = b**2 - (4 * a * c)

	if discriminant < 0:
		return -1.0, np.array([0, 0, 0])

	t1 = (-b + sqrt(discriminant)) / (2 * a)
	normal = unit_vector(ray_origin + (ray_dir*t1) - sphere_origin)
	return t1, normal