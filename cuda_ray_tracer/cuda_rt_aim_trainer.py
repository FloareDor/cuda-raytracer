import numpy as np
from numba import cuda
import pygame
import math
from timeit import timeit
import random
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 450
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ray Tracer Aim Trainer - Click to shoot, ESC to quit")
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Scene setup
sphere_centers = np.ascontiguousarray([
    [0.0, 0.0, -5.0],  # Target sphere
    [0.0, 2.0, -5.0]   # Background sphere
], dtype=np.float32)

sphere_radii = np.ascontiguousarray([
    0.2,  # Target sphere
    0.1   # Background sphere
], dtype=np.float32)

sphere_materials = np.ascontiguousarray([
    [1.0, 0.1, 0.1, 32.0],  # Target sphere - red
    [0.3, 0.8, 0.3, 8.0]    # Background sphere - green
], dtype=np.float32)

light_pos = np.ascontiguousarray([2.0, 2.0, 0.0], dtype=np.float32)

class Camera:
    def __init__(self):
        self.position = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.move_speed = 0.1
        self.mouse_sensitivity = 0.1
        self.update_vectors()
    
    def update_vectors(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        self.front = np.array([
            math.cos(pitch_rad) * math.cos(yaw_rad),
            -math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad)
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
        diffuse_strength = 1.5
        specular_strength = 3.0
        shininess = sphere_materials[hit_sphere_idx, 3]
        
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
        
        spec_angle = max(0.0, 
            view_dir[0]*reflect_dir[0] + view_dir[1]*reflect_dir[1] + view_dir[2]*reflect_dir[2])
        specular = specular_strength * math.pow(spec_angle, shininess)
        
        for i in range(3):
            color = sphere_materials[hit_sphere_idx, i] * (ambient + diffuse + specular)
            output[y, x, i] = min(255, int(color * 255))
    else:
        t = 0.5 * (ray_dir[1] + 1.0)
        for i in range(3):
            output[y, x, i] = int((1.0-t)*40 + t*60)  # Darker background

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
    
    # Add crosshair
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    crosshair_size = 8
    crosshair_thickness = 2
    gap = 4
    crosshair_color = (0, 255, 0)
    
    # Draw horizontal lines with gap
    for i in range(crosshair_thickness):
        output[center_x-crosshair_size:center_x-gap, center_y+i, :] = crosshair_color
        output[center_x-crosshair_size:center_x-gap, center_y-i, :] = crosshair_color
        output[center_x+gap:center_x+crosshair_size, center_y+i, :] = crosshair_color
        output[center_x+gap:center_x+crosshair_size, center_y-i, :] = crosshair_color
    
    # Draw vertical lines with gap
    for i in range(crosshair_thickness):
        output[center_x+i, center_y-crosshair_size:center_y-gap, :] = crosshair_color
        output[center_x-i, center_y-crosshair_size:center_y-gap, :] = crosshair_color
        output[center_x+i, center_y+gap:center_y+crosshair_size, :] = crosshair_color
        output[center_x-i, center_y+gap:center_y+crosshair_size, :] = crosshair_color
    
    # Add center dot
    dot_size = 2
    for x in range(-dot_size, dot_size+1):
        for y in range(-dot_size, dot_size+1):
            if x*x + y*y <= dot_size*dot_size:
                px, py = center_x + x, center_y + y
                output[px, py] = (255, 0, 0)  # Red dot
    
    return output

def move_target():
    # Move target to random position within bounds
    x = random.uniform(-3, 3)
    y = random.uniform(-2, 2)
    z = random.uniform(-7, -3)
    return np.array([[x, y, z], sphere_centers[1]], dtype=np.float32)

def check_hit(camera, sphere_centers):
    # Calculate ray from camera center
    ray_dir = camera.front
    
    # Vector from camera to target sphere
    oc = camera.position - sphere_centers[0]
    
    # Quadratic equation coefficients
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radii[0]**2
    
    discriminant = b*b - 4*a*c
    return discriminant > 0

def main():
    camera = Camera()
    score = 0
    start_time = time.time()
    game_duration = 30  # seconds
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    last_time = pygame.time.get_ticks()
    
    # Initial target position
    global sphere_centers
    sphere_centers = move_target()
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0
        last_time = current_time
        
        elapsed_time = time.time() - start_time
        time_left = max(0, game_duration - elapsed_time)
        
        if time_left == 0:
            print(f"Game Over! Final Score: {score}")
            running = False
            break
        
        # Handle input
        keys = pygame.key.get_pressed()
        speed = camera.move_speed * dt * 5.0
        
        if keys[pygame.K_LSHIFT]:
            speed *= 2.0
        
        if keys[pygame.K_w]:
            camera.position += camera.front * speed
        if keys[pygame.K_s]:
            camera.position -= camera.front * speed
        if keys[pygame.K_a]:
            camera.position -= camera.right * speed
        if keys[pygame.K_d]:
            camera.position += camera.right * speed
        
        # Mouse look
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        camera.yaw += mouse_dx * camera.mouse_sensitivity
        camera.pitch -= mouse_dy * camera.mouse_sensitivity
        camera.pitch = np.clip(camera.pitch, -89.0, 89.0)
        camera.update_vectors()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if check_hit(camera, sphere_centers):
                        score += 1
                        sphere_centers = move_target()  # Generate new target position
        
        # Render the scene
        frame = render(camera)
        pygame.surfarray.blit_array(screen, frame)
        
        # Draw HUD
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        timer_text = font.render(f"Time: {int(time_left)}s", True, (255, 255, 255))
        
        screen.blit(fps_text, (10, 10))
        screen.blit(score_text, (10, 40))
        screen.blit(timer_text, (WIDTH - 120, 10))
        
        pygame.display.flip()
        clock.tick(300)  # Cap at 300 FPS
    
    # Show final score and wait
    screen.fill((0, 0, 0))
    final_score_text = font.render(f"Game Over! Final Score: {score}", True, (255, 0, 0))
    text_rect = final_score_text.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(final_score_text, text_rect)
    pygame.display.flip()
    
    pygame.time.wait(2000)
    pygame.quit()

if __name__ == "__main__":
    main()