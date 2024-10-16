import pygame
import numpy as np
from timeit import timeit
from numpy import ndarray
from numpy import float32 as npfloat32
from color import write_color
from objects import hit_anything
from shader import apply_shading

# Initialize Pygame
pygame.init()

# Set up some constants
# WIDTH, HEIGHT = 1280, 800
WIDTH, HEIGHT = 400, 225
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
X_COLOR = (42, 170, 108)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))


class Camera:
    def __init__(self, e):
        self.e = e


e = np.array([0, 0, 0], dtype="f")  # camera center
d = np.float32(1)  # focal length

U = np.array([1, 0, 0])
W = np.array([0, 1, 0])
V = np.array([0, 0, 1])

viewport_height = np.float32(2.0)
viewport_width = np.float32(viewport_height * (float(WIDTH) / HEIGHT))

# Calculate the vectors across the horizontal and down the vertical viewport edges.
viewport_u = np.array([viewport_width, 0, 0], dtype="f")
viewport_v = np.array([0, -viewport_height, 0], dtype="f")

# Calculate the horizontal and vertical delta vectors from pixel to pixel.
pixel_delta_u = viewport_u / WIDTH
pixel_delta_v = viewport_v / HEIGHT

# Calculate the location of the upper left pixel.
viewport_upper_left = (
    e - np.array([0, 0, d], dtype="f") - viewport_u / 2 - viewport_v / 2
)
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

screen.fill(BLACK)

origin = np.array([0, 0, -1], dtype="f")
radius = np.float32(0.5)

sphere = (origin, radius, np.array([0.2, 0.5, 0.55]))

spheres = (sphere,)

light_1 = (np.array([0,-2,2]), 4, np.array([1,1,1]))

lights = (light_1,)

# print(spheres, lights, len(spheres), len(lights))


def calculate_frame(spheres: tuple):
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel_center = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v)
            ray_direction = pixel_center - e
            ray = (pixel_center, ray_direction)
            
            ray_color = np.array([0, 0, 0])
            did_hit_smtng, closest_t, closest_sphere, surface_normal = hit_anything(ray, spheres, 1000)
            if did_hit_smtng:
                ray_color = apply_shading(ray, closest_t, closest_sphere, surface_normal, lights)

            
            pixels[y, x] = write_color(ray_color)

    return pixels


def render_frame():
    pixels = calculate_frame(spheres)
    pygame.surfarray.blit_array(screen, pixels.swapaxes(0, 1))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        painting_time = timeit(lambda: render_frame(), number=1)
        print(f"Rendering window took {painting_time:.6f} seconds")

        pygame.display.flip()

    pygame.time.Clock().tick(60)