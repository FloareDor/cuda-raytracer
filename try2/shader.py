from typing import Tuple
from objects import sphere_intersect, hit_anything
from operations import unit_vector, magnitude
from numpy import float32
import numpy as np
import ray
from numba import jit
from math import pow

@jit
def apply_shading(r: Tuple, t: float32, sphere: Tuple, surface_normal: np.ndarray, lights: Tuple, spheres: Tuple):
    ambient_strength = np.float32(0.3)
    diffuse_strength = np.float32(2.0)
    specular_strength = np.float32(2.0)
    shadow_bias = np.float32(0.001)
    
    final_color = np.zeros(3, dtype=np.float32)
    reflection_weight = np.float32(1.0)
    max_bounces = 3
    
    ray_origin = r[0]
    ray_dir = r[1]
    
    for bounce in range(max_bounces):
        hit, closest_t, hit_sphere, hit_normal = hit_anything(
            (ray_origin, ray_dir), 
            spheres, 
            np.float32(1000.0)
        )
        hit_normal = hit_normal.astype(np.float32)
        
        if hit:
            hit_point = ray_origin + ray_dir * closest_t
            
            # lighting for this intersection
            local_color = np.zeros(3, dtype=np.float32)
            
            for light in lights:
                light_pos = light[0]
                light_intensity = light[1]
                light_color = light[2]
                
                # Shadow stuff
                to_light = light_pos - hit_point
                distance_to_light = magnitude(to_light)
                light_dir = unit_vector(to_light)
                
                # Checking for shadows using slightly offset origin
                shadow_origin = hit_point + hit_normal * shadow_bias
                shadow_ray = (shadow_origin, light_dir)
                in_shadow = hit_anything(shadow_ray, spheres, distance_to_light)[0]
                
                view_dir = unit_vector(r[0] - hit_point)
                
                # Reflection direction for specular
                dot_normal_light = np.dot(hit_normal.astype(np.float32), light_dir.astype(np.float32))
                reflect_dir = 2.0 * dot_normal_light * hit_normal - light_dir
                
                # all lighting components
                ambient = ambient_strength * light_intensity
                shadow_multiplier = np.float32(0.1) if in_shadow else np.float32(1.0)
                diffuse = diffuse_strength * max(0.0, dot_normal_light) * shadow_multiplier * light_intensity
                
                spec = max(0.0, np.dot(view_dir.astype(np.float32), reflect_dir.astype(np.float32)))
                specular = specular_strength * pow(spec, hit_sphere[3]) * shadow_multiplier * light_intensity
                
                # Combine components with material color
                light_contribution = (ambient + diffuse + specular) * light_color
                local_color += hit_sphere[2] * light_contribution
            
            # local color + final color with current reflection weight
            final_color += local_color * reflection_weight * (1.0 - hit_sphere[4])
            
            # ray update for next bounce
            dot_ray_normal = np.dot(ray_dir.astype(np.float32), hit_normal.astype(np.float32))
            ray_dir = (ray_dir - 2.0 * dot_ray_normal * hit_normal).astype(np.float32)
            ray_origin = (hit_point + hit_normal * shadow_bias).astype(np.float32)
            
            # reflection weight update for next bounce
            reflection_weight *= hit_sphere[4]
            
            if reflection_weight < 0.01:
                break
        else:
            # sky colr
            t = 0.5 * (ray_dir[1] + 1.0)
            sky_color = (1.0 - t) + t * 0.7
            final_color += np.full(3, sky_color, dtype=np.float32) * reflection_weight
            break
    
    return np.clip(final_color, 0.0, 1.0).astype(np.float32)