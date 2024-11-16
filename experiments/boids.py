import numpy as np
from numba import cuda
import pygame
import math
from timeit import timeit
import matplotlib.pyplot as plt
from collections import deque

pygame.init()
WIDTH, HEIGHT = 800, 450
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Boids simulation parameters
BOID_COUNT = 512
VISUAL_RANGE = 2.0
PROTECTED_RANGE = 0.5
ALIGNMENT_FACTOR = 0.05
COHESION_FACTOR = 0.005
SEPARATION_FACTOR = 0.05
MIN_SPEED = 0.01
MAX_SPEED = 1.0
TURN_FACTOR = 0.2
BOUNDARY_MARGIN = 10.0

# boids initt
boid_positions = np.random.uniform(-5, 5, (BOID_COUNT, 3)).astype(np.float32)
boid_velocities = np.random.uniform(-1, 1, (BOID_COUNT, 3)).astype(np.float32)
boid_radii = np.full(BOID_COUNT, 0.2, dtype=np.float32)
boid_materials = np.array([[0.3, 0.3, 0.8, 32.0, 0.2] for _ in range(BOID_COUNT)], dtype=np.float32)

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

@cuda.jit(device=True)
def length(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

@cuda.jit(device=True)
def normalize(v):
    l = length(v)
    if l > 0:
        v[0] /= l
        v[1] /= l
        v[2] /= l

@cuda.jit('void(float32[:,:], float32[:,:], float32[:], float32[:,:])')
def update_boids_kernel(positions, velocities, radii, materials):
    boid_idx = cuda.grid(1)
    if boid_idx >= positions.shape[0]:
        return
    
    separation = cuda.local.array(3, dtype=np.float32)
    alignment = cuda.local.array(3, dtype=np.float32)
    cohesion = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        separation[i] = 0.0
        alignment[i] = 0.0
        cohesion[i] = 0.0
    
    neighbors = 0
    close_neighbors = 0
    
    for other in range(positions.shape[0]):
        if other != boid_idx:
            dx = positions[boid_idx, 0] - positions[other, 0]
            dy = positions[boid_idx, 1] - positions[other, 1]
            dz = positions[boid_idx, 2] - positions[other, 2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < VISUAL_RANGE:
                if dist < PROTECTED_RANGE:
                    separation[0] += dx / dist
                    separation[1] += dy / dist
                    separation[2] += dz / dist
                    close_neighbors += 1
                
                alignment[0] += velocities[other, 0]
                alignment[1] += velocities[other, 1]
                alignment[2] += velocities[other, 2]
                
                cohesion[0] += positions[other, 0]
                cohesion[1] += positions[other, 1]
                cohesion[2] += positions[other, 2]
                
                neighbors += 1
    
    if neighbors > 0:
        alignment[0] /= neighbors
        alignment[1] /= neighbors
        alignment[2] /= neighbors
        velocities[boid_idx, 0] += (alignment[0] - velocities[boid_idx, 0]) * ALIGNMENT_FACTOR
        velocities[boid_idx, 1] += (alignment[1] - velocities[boid_idx, 1]) * ALIGNMENT_FACTOR
        velocities[boid_idx, 2] += (alignment[2] - velocities[boid_idx, 2]) * ALIGNMENT_FACTOR
        
        cohesion[0] = cohesion[0]/neighbors - positions[boid_idx, 0]
        cohesion[1] = cohesion[1]/neighbors - positions[boid_idx, 1]
        cohesion[2] = cohesion[2]/neighbors - positions[boid_idx, 2]
        velocities[boid_idx, 0] += cohesion[0] * COHESION_FACTOR
        velocities[boid_idx, 1] += cohesion[1] * COHESION_FACTOR
        velocities[boid_idx, 2] += cohesion[2] * COHESION_FACTOR
    
    if close_neighbors > 0:
        velocities[boid_idx, 0] += separation[0] * SEPARATION_FACTOR
        velocities[boid_idx, 1] += separation[1] * SEPARATION_FACTOR
        velocities[boid_idx, 2] += separation[2] * SEPARATION_FACTOR
    
    # Boundary avoidance
    if positions[boid_idx, 0] < -BOUNDARY_MARGIN: velocities[boid_idx, 0] += TURN_FACTOR
    if positions[boid_idx, 0] > BOUNDARY_MARGIN: velocities[boid_idx, 0] -= TURN_FACTOR
    if positions[boid_idx, 1] < -BOUNDARY_MARGIN: velocities[boid_idx, 1] += TURN_FACTOR
    if positions[boid_idx, 1] > BOUNDARY_MARGIN: velocities[boid_idx, 1] -= TURN_FACTOR
    if positions[boid_idx, 2] < -BOUNDARY_MARGIN: velocities[boid_idx, 2] += TURN_FACTOR
    if positions[boid_idx, 2] > BOUNDARY_MARGIN: velocities[boid_idx, 2] -= TURN_FACTOR
    
    # Speed limits
    speed = length(velocities[boid_idx])
    if speed > MAX_SPEED:
        velocities[boid_idx, 0] = (velocities[boid_idx, 0]/speed) * MAX_SPEED
        velocities[boid_idx, 1] = (velocities[boid_idx, 1]/speed) * MAX_SPEED
        velocities[boid_idx, 2] = (velocities[boid_idx, 2]/speed) * MAX_SPEED
    elif speed < MIN_SPEED:
        velocities[boid_idx, 0] = (velocities[boid_idx, 0]/speed) * MIN_SPEED
        velocities[boid_idx, 1] = (velocities[boid_idx, 1]/speed) * MIN_SPEED
        velocities[boid_idx, 2] = (velocities[boid_idx, 2]/speed) * MIN_SPEED
    
    # Update positions
    positions[boid_idx, 0] += velocities[boid_idx, 0]
    positions[boid_idx, 1] += velocities[boid_idx, 1]
    positions[boid_idx, 2] += velocities[boid_idx, 2]

@cuda.jit(device=True)
def intersect_sphere(ray_origin, ray_dir, sphere_centers, sphere_radii, closest_t, hit_sphere_idx):
    for i in range(sphere_centers.shape[0]):
        oc = cuda.local.array(3, dtype=np.float32)
        oc[0] = ray_origin[0] - sphere_centers[i, 0]
        oc[1] = ray_origin[1] - sphere_centers[i, 1]
        oc[2] = ray_origin[2] - sphere_centers[i, 2]
        
        a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
        b = 2.0 * (oc[0]*ray_dir[0] + oc[1]*ray_dir[1] + oc[2]*ray_dir[2])
        c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radii[i]**2
        
        discriminant = b*b - 4*a*c
        
        if discriminant > 0:
            t = (-b - math.sqrt(discriminant)) / (2.0*a)
            if t > 0.001 and t < closest_t[0]:
                closest_t[0] = t
                hit_sphere_idx[0] = i

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
    
    final_color = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        final_color[i] = 0.0
    
    ray_origin = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        ray_origin[i] = camera_pos[i]
    
    reflection_weight = 1.0
    max_bounces = 3
    
    for bounce in range(max_bounces):
        closest_t = cuda.local.array(1, dtype=np.float32)
        hit_sphere_idx = cuda.local.array(1, dtype=np.int32)
        closest_t[0] = 1e20
        hit_sphere_idx[0] = -1
        
        intersect_sphere(ray_origin, ray_dir, sphere_centers, sphere_radii, closest_t, hit_sphere_idx)
        
        if hit_sphere_idx[0] >= 0:
            ambient_strength = 0.3
            diffuse_strength = 2.0
            specular_strength = 2.0
            
            hit_point = cuda.local.array(3, dtype=np.float32)
            for i in range(3):
                hit_point[i] = ray_origin[i] + closest_t[0] * ray_dir[i]
            
            normal = cuda.local.array(3, dtype=np.float32)
            for i in range(3):
                normal[i] = (hit_point[i] - sphere_centers[hit_sphere_idx[0], i]) / sphere_radii[hit_sphere_idx[0]]
            
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
            
            in_shadow = False
            shadow_origin = cuda.local.array(3, dtype=np.float32)
            shadow_bias = 0.001
            for i in range(3):
                shadow_origin[i] = hit_point[i] + normal[i] * shadow_bias
            
            for i in range(sphere_centers.shape[0]):
                if i != hit_sphere_idx[0]:
                    oc_x = shadow_origin[0] - sphere_centers[i, 0]
                    oc_y = shadow_origin[1] - sphere_centers[i, 1]
                    oc_z = shadow_origin[2] - sphere_centers[i, 2]
                    
                    b = 2.0 * (oc_x*light_dir[0] + oc_y*light_dir[1] + oc_z*light_dir[2])
                    c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - sphere_radii[i]*sphere_radii[i]
                    
                    discriminant = b*b - 4.0*c
                    
                    if discriminant > 0:
                        t = (-b - math.sqrt(discriminant)) / 2.0
                        if t > 0.001 and t < light_len:
                            in_shadow = True
                            break
            
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
            shadow_multiplier = 0.1 if in_shadow else 1.0
            diffuse = diffuse_strength * max(0.0, dot_normal_light) * shadow_multiplier
            
            spec = max(0.0, view_dir[0]*reflect_dir[0] + view_dir[1]*reflect_dir[1] + view_dir[2]*reflect_dir[2])
            specular = specular_strength * math.pow(spec, sphere_materials[hit_sphere_idx[0], 3]) * shadow_multiplier

            local_color = cuda.local.array(3, dtype=np.float32)
            for i in range(3):
                local_color[i] = sphere_materials[hit_sphere_idx[0], i] * (ambient + diffuse + specular)
            
            for i in range(3):
                final_color[i] += local_color[i] * reflection_weight * (1.0 - sphere_materials[hit_sphere_idx[0], 4])
            
            dot_ray_normal = ray_dir[0]*normal[0] + ray_dir[1]*normal[1] + ray_dir[2]*normal[2]
            for i in range(3):
                ray_dir[i] = ray_dir[i] - 2.0 * dot_ray_normal * normal[i]
                ray_origin[i] = hit_point[i] + normal[i] * shadow_bias
            
            reflection_weight *= sphere_materials[hit_sphere_idx[0], 4]
            
            if reflection_weight < 0.01:
                break
        else:
            t = 0.5 * (ray_dir[1] + 1.0)
            sky_color = (1.0-t) + t*0.7
            for i in range(3):
                final_color[i] += sky_color * reflection_weight
            break
    
    for i in range(3):
        output[y, x, i] = min(255, int(final_color[i] * 255))

def update_boids():
    threadsperblock = 256
    blockspergrid = (BOID_COUNT + threadsperblock - 1) // threadsperblock
    
    d_positions = cuda.to_device(boid_positions)
    d_velocities = cuda.to_device(boid_velocities)
    d_radii = cuda.to_device(boid_radii)
    d_materials = cuda.to_device(boid_materials)
    
    update_boids_kernel[blockspergrid, threadsperblock](
        d_positions, d_velocities, d_radii, d_materials
    )
    
    d_positions.copy_to_host(boid_positions)
    d_velocities.copy_to_host(boid_velocities)

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
    d_sphere_centers = cuda.to_device(boid_positions) # boid positions
    d_sphere_radii = cuda.to_device(boid_radii)
    d_sphere_materials = cuda.to_device(boid_materials)
    
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
    MAX_SAMPLES = 1000
    fps_data = deque(maxlen=MAX_SAMPLES)
    render_times = deque(maxlen=MAX_SAMPLES)
    frame_numbers = deque(maxlen=MAX_SAMPLES)
    camera_positions = deque(maxlen=MAX_SAMPLES)
    frame_count = 0
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
        
        update_boids()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                
                plt.style.use('dark_background')
                fig = plt.figure(figsize=(15, 10))
                
                ax1 = plt.subplot(3, 1, 1)
                plt.plot(list(frame_numbers), list(render_times), 'cyan', label='CUDA Render Time', linewidth=2)
                plt.title('CUDA Raytracer Performance Metrics', color='white', fontsize=14)
                plt.xlabel('Frame Number', color='white')
                plt.ylabel('Render Time (seconds)', color='white')
                plt.grid(True, alpha=0.3)
                plt.legend()
                ax1.tick_params(colors='white')
                
                ax2 = plt.subplot(3, 1, 2)
                plt.plot(list(frame_numbers), list(fps_data), 'lime', label='CUDA FPS', linewidth=2)
                plt.xlabel('Frame Number', color='white')
                plt.ylabel('Frames Per Second', color='white')
                plt.grid(True, alpha=0.3)
                plt.legend()
                ax2.tick_params(colors='white')
                
                ax3 = plt.subplot(3, 1, 3)
                positions = np.array(list(camera_positions))
                plt.plot(frame_numbers, positions[:, 0], 'r-', label='X', linewidth=2)
                plt.plot(frame_numbers, positions[:, 1], 'g-', label='Y', linewidth=2)
                plt.plot(frame_numbers, positions[:, 2], 'b-', label='Z', linewidth=2)
                plt.xlabel('Frame Number', color='white')
                plt.ylabel('Camera Position', color='white')
                plt.grid(True, alpha=0.3)
                plt.legend()
                ax3.tick_params(colors='white')

                plt.tight_layout()
                plt.savefig('cuda_performance_metrics.png', facecolor='black', edgecolor='black')
                print("\nPerformance plot saved as 'cuda_performance_metrics.png'")
        
        painting_time = timeit(lambda: render(camera), number=1)
        fps = 1/painting_time
        
        frame_count += 1
        frame_numbers.append(frame_count)
        render_times.append(painting_time)
        fps_data.append(fps)
        camera_positions.append(camera.position.copy())
        
        print(f"pos: {camera.position} | time: {painting_time:.3f} | FPS: {fps:.1f}")
        
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()