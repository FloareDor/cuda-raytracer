import numpy as np
from typing import Tuple
from math import sqrt
from numpy import float32 as npfloat32
from operations import unit_vector
from numba import jit


from numba.experimental import jitclass
from numba import float32, types

@jit
def hit_anything(ray: Tuple, spheres: Tuple, t_max: npfloat32):
    hit_anything = False
    closest_t = t_max
    closest_sphere = spheres[0]
    surface_normal = np.array([0, 0, 0], dtype=np.float32)
    
    for i in range(len(spheres)):
        sphere = spheres[i]
        t, hit_normal = sphere_intersect(ray[0], ray[1], 
                                       sphere[0], sphere[1])
        if t > 0.0001 and t < closest_t:
            closest_t = t
            closest_sphere = sphere
            hit_anything = True
            surface_normal = hit_normal
    
    return hit_anything, closest_t, closest_sphere, surface_normal

@jit
def sphere_intersect(ray_origin, ray_dir, sphere_origin, sphere_radius):
    oc = ray_origin - sphere_origin
    
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(ray_dir.astype(np.float32), oc.astype(np.float32))
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return -1.0, np.array([0, 0, 0], dtype=np.float32)
    
    # Find the nearest intersection
    sqrt_discriminant = np.sqrt(discriminant)
    t0 = (-b - sqrt_discriminant) / (2.0 * a)
    
    if t0 > 0.0001:
        intersection_point = ray_origin + ray_dir * t0
        normal = (intersection_point - sphere_origin) / sphere_radius
        return t0, normal.astype(np.float32)
        
    t1 = (-b + sqrt_discriminant) / (2.0 * a)
    if t1 > 0.0001:
        intersection_point = ray_origin + ray_dir * t1
        normal = (intersection_point - sphere_origin) / sphere_radius
        return t1, normal.astype(np.float32)
    
    return -1.0, np.array([0, 0, 0], dtype=np.float32)