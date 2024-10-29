import numpy as np
from numba import cuda
import pygame
import math
from timeit import timeit

pygame.init()
WIDTH, HEIGHT = 800, 450
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

sphere_centers = np.ascontiguousarray([
    [0.0, 0.0, -5.0],
    [0.0, 2.0, -5.0]
], dtype=np.float32)

sphere_radii = np.ascontiguousarray([
    0.5,
    2.0
], dtype=np.float32)

sphere_materials = np.ascontiguousarray([
    [0.7, 0.3, 0.3, 32.0],
    [0.3, 0.8, 0.3, 8.0]
], dtype=np.float32)

light_pos = np.ascontiguousarray([2.0, 2.0, 0.0], dtype=np.float32)

class Camera:
    def __init__(self):
        self.position = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 0.1
        self.mouse_speed = 0.1
        self.update_vectors()
    
    def update_vectors(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        
        self.front = np.array([
            math.cos(pitch) * math.cos(yaw),
            -math.sin(pitch),
            math.cos(pitch) * math.sin(yaw)
        ], dtype=np.float32)
        
        self.front /= np.linalg.norm(self.front)
        
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)
        
        self.up = np.cross(self.front, self.right)
        self.up /= np.linalg.norm(self.up)

@cuda.jit('void(uint8[:,:,:], float32[:], float32[:], float32[:], float32[:], float32[:,:], float32[:], float32[:,:])', fastmath=True)
def render_kernel(output, camera_pos, camera_front, camera_right, camera_up, sphere_centers, sphere_radii, sphere_materials):
    y = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    x = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    
    if x >= output.shape[1] or y >= output.shape[0]:
        return
    
    fov = math.tan(math.radians(30.0))
    aspect = output.shape[0] / output.shape[1]
    
    u = (2.0 * (x + 0.5) / output.shape[1] - 1.0) * fov
    v = (1.0 - 2.0 * (y + 0.5) / output.shape[0]) * aspect * fov
    
    ray_dir = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        ray_dir[i] = camera_front[i] + u * camera_right[i] + v * camera_up[i]
    
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
        ambient_strength = 0.3
        diffuse_strength = 2.0
        specular_strength = 2.0
        
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
        
        view_dir = cuda.local.array(3, dtype=np.float32)
        view_dir[0] = camera_pos[0] - hit_point[0]
        view_dir[1] = camera_pos[1] - hit_point[1]
        view_dir[2] = camera_pos[2] - hit_point[2]
        
        view_len = math.sqrt(view_dir[0]**2 + view_dir[1]**2 + view_dir[2]**2)
        view_dir[0] /= view_len
        view_dir[1] /= view_len
        view_dir[2] /= view_len
        
        dot_normal_light = normal[0]*light_dir[0] + normal[1]*light_dir[1] + normal[2]*light_dir[2]
        reflect_dir = cuda.local.array(3, dtype=np.float32)
        for i in range(3):
            reflect_dir[i] = 2.0 * dot_normal_light * normal[i] - light_dir[i]
        
        ambient = ambient_strength
        diffuse = diffuse_strength * max(0.0, dot_normal_light)
        
        spec = max(0.0, view_dir[0]*reflect_dir[0] + view_dir[1]*reflect_dir[1] + view_dir[2]*reflect_dir[2])
        specular = specular_strength * math.pow(spec, 32.0)
        
        for i in range(3):
            color = sphere_materials[hit_sphere_idx, i] * (ambient + diffuse + specular)
            output[y, x, i] = min(255, int(color * 255))
    else:
        t = 0.5 * (ray_dir[1] + 1.0)
        for i in range(3):
            output[y, x, i] = int((1.0-t)*255 + t*180)

def render(camera):
    output = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (HEIGHT + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (WIDTH + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    d_output = cuda.to_device(output)
    d_camera_pos = cuda.to_device(camera.position)
    d_camera_front = cuda.to_device(camera.front)
    d_camera_right = cuda.to_device(camera.right)
    d_camera_up = cuda.to_device(camera.up)
    d_sphere_centers = cuda.to_device(sphere_centers)
    d_sphere_radii = cuda.to_device(sphere_radii)
    d_sphere_materials = cuda.to_device(sphere_materials)
    
    render_kernel[blockspergrid, threadsperblock](
        d_output,
        d_camera_pos,
        d_camera_front,
        d_camera_right,
        d_camera_up,
        d_sphere_centers,
        d_sphere_radii,
        d_sphere_materials
    )
    
    output = d_output.copy_to_host()
    output = np.transpose(output, (1, 0, 2))
    pygame.surfarray.blit_array(screen, output)
    pygame.display.flip()

def main():
    camera = Camera()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            camera.position += camera.front * camera.speed
        if keys[pygame.K_s]:
            camera.position -= camera.front * camera.speed
        if keys[pygame.K_a]:
            camera.position -= camera.right * camera.speed
        if keys[pygame.K_d]:
            camera.position += camera.right * camera.speed
        
        mouse_x, mouse_y = pygame.mouse.get_rel()
        camera.yaw += mouse_x * camera.mouse_speed
        camera.pitch -= mouse_y * camera.mouse_speed
        camera.pitch = min(89.0, max(-89.0, camera.pitch))
        
        camera.update_vectors()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                
        painting_time = timeit(lambda: render(camera), number=1)
        print(f"pos: {camera.position} | time: {painting_time:.3f} | FPS: {(1/painting_time):.1f}")
        
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()