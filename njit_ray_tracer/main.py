import pygame
import numpy as np
from timeit import timeit
from numpy import ndarray
from numpy import float32 as npfloat32
from color import write_color
from objects import hit_anything
from shader import apply_shading
from numba import jit
import matplotlib.pyplot as plt
from collections import deque

# Initialize Pygame
pygame.init()

# Set up some constants
# WIDTH, HEIGHT = 1280, 800
WIDTH, HEIGHT = 400, 225
# WIDTH, HEIGHT = 200, 112
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
X_COLOR = (42, 170, 108)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))


class Camera:
    def __init__(self, e):
        self.e = e


e = np.array([0, 0, 0], dtype=np.float32)  # camera center
d = np.float32(1)  # focal length

U = np.array([1, 0, 0])
W = np.array([0, 1, 0])
V = np.array([0, 0, 1])

viewport_height = np.float32(2.0)
viewport_width = np.float32(viewport_height * (float(WIDTH) / HEIGHT))

# Calculate the vectors across the horizontal and down the vertical viewport edges.
viewport_u = np.array([viewport_width, 0, 0], dtype=np.float32)
viewport_v = np.array([0, -viewport_height, 0], dtype=np.float32)

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / WIDTH
pixel_delta_v = viewport_v / HEIGHT

# Calculate the location of the upper left pixel.
viewport_upper_left = (
    e - np.array([0, 0, d], dtype="f") - viewport_u / 2 - viewport_v / 2
)
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

screen.fill(BLACK)

sphere1 = (
    np.array([-0.5, 0, -1], dtype=np.float32),  # position
    np.float32(0.75),                           # radius
    np.array([0.7, 0.65, 0.4], dtype=np.float32), # color
    np.float32(32.0),                           # shininess
    np.float32(0.8)                             # reflectivity
)

sphere2 = (
    np.array([0.5, 0.5, -1], dtype=np.float32),
    np.float32(0.5),
    np.array([0.3, 0.8, 0.3], dtype=np.float32),
    np.float32(8.0),                            # less shiny
    np.float32(0.2)                             # less reflective
)

spheres = (sphere1,sphere2)

# light_position, light_intensity, light_color
light_1 = (np.array([-9, 2, -1], dtype=np.float32), np.float32(1.2), np.array([1, 1, 1], dtype=np.float32))

lights = (light_1,)

# print(spheres, lights, len(spheres), len(lights))

@jit
def calculate_frame(spheres: tuple):
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel_center = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v)
            ray_direction = pixel_center - e
            ray = (pixel_center, ray_direction)
            
            ray_color = np.array([0, 0, 0], dtype=np.float32)
            did_hit_smtng, closest_t, closest_sphere, surface_normal = hit_anything(ray, spheres, np.float32(1000))
            if did_hit_smtng:
                ray_color = apply_shading(ray, closest_t, closest_sphere, surface_normal, lights, spheres)

            
            pixels[y, x] = write_color(ray_color)

    return pixels


def render_frame():
    pixels = calculate_frame(spheres)
    pygame.surfarray.blit_array(screen, pixels.swapaxes(0, 1))

MAX_SAMPLES = 100  # Keep last 100 frames of data
fps_data = deque(maxlen=MAX_SAMPLES)
render_times = deque(maxlen=MAX_SAMPLES)
frame_numbers = deque(maxlen=MAX_SAMPLES)
frame_count = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            
            # Create performance visualization
            plt.figure(figsize=(12, 6))
            
            # First subplot for render times
            plt.subplot(2, 1, 1)
            plt.plot(list(frame_numbers), list(render_times), 'b-', label='Render Time')
            plt.title('JIT Rendering Performance Metrics')
            plt.xlabel('Frame Number')
            plt.ylabel('Render Time (seconds)')
            plt.grid(True)
            plt.legend()

            # Second subplot for FPS
            plt.subplot(2, 1, 2)
            plt.plot(list(frame_numbers), list(fps_data), 'g-', label='FPS')
            plt.xlabel('Frame Number')
            plt.ylabel('Frames Per Second')
            plt.grid(True)
            plt.legend()

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('JIT_performance_metrics.png')
            print("Performance plot saved as 'performance_metrics.png'")

            running = False

    # Measure and record performance
    painting_time = timeit(lambda: render_frame(), number=1)
    fps = 1/painting_time
    
    # Store the data
    frame_count += 1
    frame_numbers.append(frame_count)
    render_times.append(painting_time)
    fps_data.append(fps)
    
    print(f"Frame {frame_count}: Render time: {painting_time:.6f} seconds | FPS: {fps:.2f}")

    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()