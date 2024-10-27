import numpy as np
from numba import cuda
import pygame
import math
from timeit import timeit

pygame.init()
WIDTH, HEIGHT = 800, 450
screen = pygame.display.set_mode((WIDTH, HEIGHT))

sphere_centers = np.ascontiguousarray([
    [0.0, 0.0, -1.0],
    [0.0, -100.5, -1.0]
], dtype=np.float32)

sphere_radii = np.ascontiguousarray([
    0.5,
    100.0
], dtype=np.float32)

sphere_colors = np.ascontiguousarray([
    [0.2, 0.5, 0.55],
    [0.0, 0.0, 0.0]
], dtype=np.float32)

light_pos = np.ascontiguousarray([2.0, 4.0, -3.0], dtype=np.float32)
camera_pos = np.ascontiguousarray([0.0, 0.0, 0.0], dtype=np.float32)

@cuda.jit('void(uint8[:,:,:], float32[:], float32[:,:], float32[:], float32[:,:])', fastmath=True)
def render_kernel(output, camera_pos, sphere_centers, sphere_radii, sphere_colors):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    
    if x >= output.shape[0] or y >= output.shape[1]:
        return
        
    u = (y / output.shape[1]) * 2.0 - 1.0
    v = (x / output.shape[0]) * 2.0 - 1.0
    aspect = output.shape[1] / output.shape[0]
    
    ray_dir = cuda.local.array(3, dtype=np.float32)
    ray_dir[0] = u * aspect
    ray_dir[1] = -v
    ray_dir[2] = -1.0
    
    length = math.sqrt(ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2)
    ray_dir[0] /= length
    ray_dir[1] /= length
    ray_dir[2] /= length
    
    closest_t = 1e20
    hit_sphere_idx = -1
    
    for i in range(sphere_centers.shape[0]):
        oc = cuda.local.array(3, dtype=np.float32)
        oc[0] = camera_pos[0] - sphere_centers[i, 0]
        oc[1] = camera_pos[1] - sphere_centers[i, 1]
        oc[2] = camera_pos[2] - sphere_centers[i, 2]
        
        a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
        b = 2.0 * (oc[0]*ray_dir[0] + oc[1]*ray_dir[1] + oc[2]*ray_dir[2])
        c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radii[i]**2
        
        discriminant = b*b - 4*a*c
        
        if discriminant > 0:
            t = (-b - math.sqrt(discriminant)) / (2.0*a)
            if t > 0.001 and t < closest_t:
                closest_t = t
                hit_sphere_idx = i
    
    if hit_sphere_idx >= 0:
        hit_point = cuda.local.array(3, dtype=np.float32)
        for i in range(3):
            hit_point[i] = camera_pos[i] + closest_t * ray_dir[i]
        
        normal = cuda.local.array(3, dtype=np.float32)
        for i in range(3):
            normal[i] = (hit_point[i] - sphere_centers[hit_sphere_idx, i]) / sphere_radii[hit_sphere_idx]
        
        normal_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        normal[0] /= normal_len
        normal[1] /= normal_len
        normal[2] /= normal_len
        
        light_dir = cuda.local.array(3, dtype=np.float32)
        light_dir[0] = light_pos[0] - hit_point[0]
        light_dir[1] = light_pos[1] - hit_point[1]
        light_dir[2] = light_pos[2] - hit_point[2]
        
        light_len = math.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        light_dir[0] /= light_len
        light_dir[1] /= light_len
        light_dir[2] /= light_len
        
        diffuse = max(0.0, normal[0]*light_dir[0] + normal[1]*light_dir[1] + normal[2]*light_dir[2])
        
        for i in range(3):
            color = sphere_colors[hit_sphere_idx, i] * diffuse
            output[x, y, i] = min(255, int(color * 255))
    else:
        t = 0.5 * (ray_dir[1] + 1.0)
        for i in range(3):
            output[x, y, i] = int((1.0-t)*255 + t*180)

def render():
    output = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (WIDTH + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (HEIGHT + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    d_output = cuda.to_device(output)
    d_camera_pos = cuda.to_device(camera_pos)
    d_sphere_centers = cuda.to_device(sphere_centers)
    d_sphere_radii = cuda.to_device(sphere_radii)
    d_sphere_colors = cuda.to_device(sphere_colors)
    
    render_kernel[blockspergrid, threadsperblock](
        d_output, 
        d_camera_pos,
        d_sphere_centers,
        d_sphere_radii,
        d_sphere_colors
    )
    
    output = d_output.copy_to_host()
    pygame.surfarray.blit_array(screen, output)
    pygame.display.flip()

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        painting_time = timeit(lambda: render(), number=1)
        print(f"Rendering window took {painting_time:.6f} seconds")
        
        pygame.time.Clock().tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()