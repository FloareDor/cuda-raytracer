import numpy as np
from numba import cuda
import pygame
import math
from timeit import timeit
import matplotlib.pyplot as plt
from collections import deque
import librosa
import sounddevice as sd
import threading
import queue
import time
import vidmaker

pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Boids simulation parameters
BOID_COUNT = 256
MIN_SPEED = 0.01
BOUNDARY_MARGIN = 10.0

# Audio processing parameters
SAMPLE_RATE = 44100
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
AUDIO_BUFFER_SIZE = 2048
ONSET_THRESHOLD = 0.5
ONSET_MIN_DISTANCE = 3



# video = vidmaker.Video("vidmaker.mp4", late_export=True)
pygame.display.set_caption("vidmaker test")

class AudioReactiveStuff:
    def __init__(self, audio_file):
        print(f"Loading audio file: {audio_file}")
        self.audio_data, self.sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        self.duration = librosa.get_duration(y=self.audio_data, sr=self.sr)
        
        self.current_features = {
            'rms': 0.0,
            'spectral_centroid': 0.0,
            'onset_strength': 0.0,
            'low_energy': 0.0,
            'high_energy': 0.0
        }
        
        self.smoothing = 0.8 
        self._previous_features = self.current_features.copy()
        
        self.current_frame = 0
        self.playing = False
        print("Audio processor initialized")
    
    def smooth_features(self, new_features):
        for key in new_features:
            self.current_features[key] = (self.smoothing * self._previous_features[key] + 
                                        (1 - self.smoothing) * new_features[key])
            self._previous_features[key] = self.current_features[key]
    
    def start_playback(self):
        def audio_callback(outdata, frames, time, status):
            if self.current_frame + frames > len(self.audio_data):
                self.playing = False
                raise sd.CallbackStop()
            
            data = self.audio_data[self.current_frame:self.current_frame + frames]
            outdata[:len(data), 0] = data
            self.current_frame += frames
            

            if len(data) >= AUDIO_BUFFER_SIZE:
                self.process_audio_frame(data)
        
        self.stream = sd.OutputStream(
            channels=1,
            callback=audio_callback,
            samplerate=self.sr,
            blocksize=AUDIO_BUFFER_SIZE
        )
        self.playing = True
        self.stream.start()
    
    def process_audio_frame(self, frame):

        rms = np.sqrt(np.mean(frame**2))
        spec = np.abs(librosa.stft(frame, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
        # Frequency band energies
        freq_bands = librosa.fft_frequencies(sr=self.sr, n_fft=N_FFT)
        low_mask = freq_bands < 200
        high_mask = freq_bands > 2000
        
        low_energy = np.mean(spec[low_mask])
        high_energy = np.mean(spec[high_mask])
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=frame, sr=self.sr)
        
        new_features = {
            'rms': rms,
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(S=spec)),
            'onset_strength': np.mean(onset_env),
            'low_energy': low_energy,
            'high_energy': high_energy
        }
        
        self.smooth_features(new_features)

class BoidParameters:
    def __init__(self):
        self.base_values = {
            'VISUAL_RANGE': 2.0,
            'PROTECTED_RANGE': 0.5,
            'ALIGNMENT_FACTOR': 0.05,
            'COHESION_FACTOR': 0.005,
            'SEPARATION_FACTOR': 0.05,
            'MAX_SPEED': 2.0,
            'TURN_FACTOR': 0.2
        }
        self.current_values = self.base_values.copy()
    
    def update(self, audio_features):
        # Mapping audio features to boid parameters
        rms = audio_features['rms']
        onset = audio_features['onset_strength']
        low_energy = audio_features['low_energy']
        high_energy = audio_features['high_energy']
        
        # Update parameters
        self.current_values['VISUAL_RANGE'] = self.base_values['VISUAL_RANGE'] * (1*(1 + rms * 2))
        self.current_values['ALIGNMENT_FACTOR'] = self.base_values['ALIGNMENT_FACTOR'] * (1/(1+high_energy))
        self.current_values['COHESION_FACTOR'] = self.base_values['COHESION_FACTOR'] * (1*(1 + low_energy*0.5))
        self.current_values['MAX_SPEED'] = self.base_values['MAX_SPEED'] * (1/(1 + onset * 5))
        self.current_values['SEPARATION_FACTOR'] = self.base_values['SEPARATION_FACTOR'] * (1*(1 + onset))
        
        # Visual chagnes
        return {
            'color_shift': onset,
            'reflectivity': rms,
            'size_multiplier': 1 + (low_energy * 0.5)
        }

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

@cuda.jit
def update_boids_kernel(positions, velocities, radii, materials,
                       visual_range, protected_range, alignment_factor,
                       cohesion_factor, separation_factor, max_speed, turn_factor):
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
            
            if dist < visual_range:
                if dist < protected_range:
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
        # Alignment
        alignment[0] /= neighbors
        alignment[1] /= neighbors
        alignment[2] /= neighbors
        velocities[boid_idx, 0] += (alignment[0] - velocities[boid_idx, 0]) * alignment_factor
        velocities[boid_idx, 1] += (alignment[1] - velocities[boid_idx, 1]) * alignment_factor
        velocities[boid_idx, 2] += (alignment[2] - velocities[boid_idx, 2]) * alignment_factor
        
        # Cohesion
        cohesion[0] = cohesion[0]/neighbors - positions[boid_idx, 0]
        cohesion[1] = cohesion[1]/neighbors - positions[boid_idx, 1]
        cohesion[2] = cohesion[2]/neighbors - positions[boid_idx, 2]
        velocities[boid_idx, 0] += cohesion[0] * cohesion_factor
        velocities[boid_idx, 1] += cohesion[1] * cohesion_factor
        velocities[boid_idx, 2] += cohesion[2] * cohesion_factor
    
    if close_neighbors > 0:
        velocities[boid_idx, 0] += separation[0] * separation_factor
        velocities[boid_idx, 1] += separation[1] * separation_factor
        velocities[boid_idx, 2] += separation[2] * separation_factor
    
    # Boundary
    if positions[boid_idx, 0] < -BOUNDARY_MARGIN: velocities[boid_idx, 0] += turn_factor
    if positions[boid_idx, 0] > BOUNDARY_MARGIN: velocities[boid_idx, 0] -= turn_factor
    if positions[boid_idx, 1] < -BOUNDARY_MARGIN: velocities[boid_idx, 1] += turn_factor
    if positions[boid_idx, 1] > BOUNDARY_MARGIN: velocities[boid_idx, 1] -= turn_factor
    if positions[boid_idx, 2] < -BOUNDARY_MARGIN: velocities[boid_idx, 2] += turn_factor
    if positions[boid_idx, 2] > BOUNDARY_MARGIN: velocities[boid_idx, 2] -= turn_factor
    
    # Speed
    speed = length(velocities[boid_idx])
    if speed > max_speed:
        velocities[boid_idx, 0] = (velocities[boid_idx, 0]/speed) * max_speed
        velocities[boid_idx, 1] = (velocities[boid_idx, 1]/speed) * max_speed
        velocities[boid_idx, 2] = (velocities[boid_idx, 2]/speed) * max_speed
    elif speed < MIN_SPEED:
        velocities[boid_idx, 0] = (velocities[boid_idx, 0]/speed) * MIN_SPEED
        velocities[boid_idx, 1] = (velocities[boid_idx, 1]/speed) * MIN_SPEED
        velocities[boid_idx, 2] = (velocities[boid_idx, 2]/speed) * MIN_SPEED
    
    # positions updating
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

light_pos = np.array([2.0, 2.0, 0.0], dtype=np.float32)
@cuda.jit('void(uint8[:,:,:], float32[:], float32[:], float32[:], float32[:], float32[:,:], float32[:], float32[:,:], float32[:])', fastmath=True)
def render_kernel(output, camera_pos, camera_front, camera_right, camera_up, sphere_centers, sphere_radii, sphere_materials, light_pos):
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
            # sky_color = (1.0-t) + t*0.7
            sky_color = (1.0-t) * 0.0 + t*0.1
            for i in range(3):
                final_color[i] += sky_color * reflection_weight
            break
    
    for i in range(3):
        output[y, x, i] = min(255, int(final_color[i] * 255))


def update_boids(params):
    threadsperblock = 256
    blockspergrid = (BOID_COUNT + threadsperblock - 1) // threadsperblock
    
    d_positions = cuda.to_device(boid_positions)
    d_velocities = cuda.to_device(boid_velocities)
    d_radii = cuda.to_device(boid_radii)
    d_materials = cuda.to_device(boid_materials)
    
    update_boids_kernel[blockspergrid, threadsperblock](
        d_positions, d_velocities, d_radii, d_materials,
        params.current_values['VISUAL_RANGE'],
        params.current_values['PROTECTED_RANGE'],
        params.current_values['ALIGNMENT_FACTOR'],
        params.current_values['COHESION_FACTOR'],
        params.current_values['SEPARATION_FACTOR'],
        params.current_values['MAX_SPEED'],
        params.current_values['TURN_FACTOR']
    )
    
    d_positions.copy_to_host(boid_positions)
    d_velocities.copy_to_host(boid_velocities)

d_light_pos = cuda.to_device(light_pos)
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
    d_sphere_centers = cuda.to_device(boid_positions)
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
        d_sphere_materials,
        d_light_pos
    )
    
    output = d_output.copy_to_host()
    output = np.transpose(output, (1, 0, 2))
    pygame.surfarray.blit_array(screen, output)
    pygame.display.flip()

# Initialize boids
boid_positions = np.random.uniform(-5, 5, (BOID_COUNT, 3)).astype(np.float32)
boid_velocities = np.random.uniform(-1, 1, (BOID_COUNT, 3)).astype(np.float32)
boid_radii = np.full(BOID_COUNT, 0.2, dtype=np.float32)
boid_materials = np.array([[0.3, 0.3, 0.8, 32.0, 0.2] for _ in range(BOID_COUNT)], dtype=np.float32)

light_pos = np.array([2.0, 2.0, 0.0], dtype=np.float32)

def main():
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "../media/greenpeace.wav"
    
    audio_processor = AudioReactiveStuff(audio_file)
    boid_params = BoidParameters()
    camera = Camera()
    clock = pygame.time.Clock()

    MAX_SAMPLES = 1000
    fps_data = deque(maxlen=MAX_SAMPLES)
    render_times = deque(maxlen=MAX_SAMPLES)
    frame_numbers = deque(maxlen=MAX_SAMPLES)
    camera_positions = deque(maxlen=MAX_SAMPLES)
    frame_count = 0
    
    audio_processor.start_playback()
    running = True
    
    while running and audio_processor.playing:
        # cam movements
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
        
        # boid parameters updates based on audio
        visual_effects = boid_params.update(audio_processor.current_features)
        
        for i in range(BOID_COUNT):
            # Color mod
            boid_materials[i, 0:3] = [
                0.3 + visual_effects['color_shift'] * 0.7,  # More red on onset
                0.3,
                0.8 - visual_effects['color_shift'] * 0.4   # Less blue on onset
            ]
            boid_materials[i, 4] = min(0.9, visual_effects['reflectivity'])
            # size <=> bass
            boid_radii[i] = 0.1+ 0.2 * visual_effects['size_multiplier'] * 0.3
        
        update_boids(boid_params)
        
        painting_time = timeit(lambda: render(camera), number=1)
        fps = 1/painting_time
        
        frame_count += 1
        frame_numbers.append(frame_count)
        render_times.append(painting_time)
        fps_data.append(fps)
        camera_positions.append(camera.position.copy())
        
        print(f"Frame: {frame_count} | pos: {camera.position} | time: {painting_time:.3f} | FPS: {fps:.1f}")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                
                plt.style.use('dark_background')
                fig = plt.figure(figsize=(15, 10))
                
                ax1 = plt.subplot(3, 1, 1)
                plt.plot(list(frame_numbers), list(render_times), 'cyan', label='CUDA Render Time', linewidth=2)
                plt.title('Audio-Reactive CUDA Raytracer Performance', color='white', fontsize=14)
                plt.xlabel('Frame Number', color='white')
                plt.ylabel('Render Time (seconds)', color='white')
                plt.grid(True, alpha=0.3)
                plt.legend()
                ax1.tick_params(colors='white')
                
                ax2 = plt.subplot(3, 1, 2)
                plt.plot(list(frame_numbers), list(fps_data), 'lime', label='FPS', linewidth=2)
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
                plt.savefig('audio_reactive_performance.png', facecolor='black', edgecolor='black')
                print("\nPerformance plot saved as 'audio_reactive_performance.png'")
        # video.update(pygame.surfarray.pixels3d(screen).swapaxes(0, 1), inverted=False) # THIS LINE
        clock.tick(60)

    # video.compress(target_size=2048, new_file=True)
    # video.export(verbose=True)
    pygame.quit()
    

if __name__ == "__main__":
    main()