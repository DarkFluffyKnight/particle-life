import numpy as np
import pygame
from numba import njit, prange

WIDTH, HEIGHT = 1920 / 2, 1200 / 2
N_PARTICLES = 1000
N_TYPES = 4
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple"]

R_MIN = 50
R_MAX = 200
# V_MAX = 1000

TICKS = 60
DT = 1 / TICKS
np.random.seed(42)

# Random attraction matrix, each particle is attracted to its own type
# Attraction is positive, repulsion is negative
ATTRACTION_MATRIX = np.random.uniform(low=-1, high=1, size=(N_TYPES, N_TYPES))
for i in range(N_TYPES):
    ATTRACTION_MATRIX[i, i] = 1

# ATTRACTION_MATRIX = np.full((N_TYPES, N_TYPES), -1)
# for i in range(N_TYPES):
#     ATTRACTION_MATRIX[i][i] = 1

# ATTRACTION_MATRIX = np.array([[1, 1], [1, 1]])


positions = np.random.rand(N_PARTICLES, 2) * np.array([WIDTH, HEIGHT])
# velocities = np.random.uniform(low=-V_MAX, high=V_MAX, size=(N_PARTICLES, 2))
velocities = np.zeros((N_PARTICLES, 2))
types = np.random.choice(N_TYPES, N_PARTICLES)
# types = np.random.randint(0, N_TYPES, N_PARTICLES)


def apply_force(p1: int, p2: int):
    """Apply force from particle 2 onto particle 1. Uses standard linear forces mechanism

    Args:
        p1 (int): index of particle 1
        p2 (int): index of particle 2
    """
    vect = positions[p2] - positions[p1]
    dist = np.linalg.norm(vect)

    if dist < R_MIN:
        force = (dist / R_MIN) - 1  # force < 0
    elif dist < R_MAX:
        force = ATTRACTION_MATRIX[types[p1], types[p2]] * (
            1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))  # -2 <= force <= 2
        )
    else:
        return

    velocities[p1] += vect * force / dist


def apply_smooth_force(p1: int, p2: int):
    """Apply force from particle 2 onto particle 1. Uses polynomial force mechanism for smoother movement

    Args:
        p1 (int): index of particle 1
        p2 (int): index of particle 2
    """
    vect = positions[p2] - positions[p1]
    dist = np.linalg.norm(vect)

    if dist < R_MIN:
        force = (dist / R_MIN) - 1
    elif dist < R_MAX:
        force = (
            ATTRACTION_MATRIX[types[p1], types[p2]]
            * 4
            * ((dist - R_MIN) / (R_MAX - R_MIN))
            * (1 - (dist - R_MIN) / (R_MAX - R_MIN))
        )
    else:
        return

    velocities[p1] += vect * force / dist


@njit(parallel=True, fastmath=True)
def update_velocities(positions, velocities):
    """Update velocities of each particle with parallelisation for efficiency

    Args:
        positions (NumPy array): (N x 2) matrix of particle positions
        velocities (NumPy array): (N x 2) matrix of particle velocities
    """
    for i in prange(N_PARTICLES):
        for j in range(N_PARTICLES):
            if i == j:
                continue

            vect = positions[j] - positions[i]
            dist = np.linalg.norm(vect)

            if dist < R_MIN:
                force = (dist / R_MIN) - 1
            elif dist < R_MAX:
                force = ATTRACTION_MATRIX[types[i], types[j]] * (
                    1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))
                )
            else:
                force = 0

            velocities[i] += force * vect / dist

    velocities *= 0.95


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    update_velocities(positions, velocities)

    # Update position
    positions = np.mod(positions + (velocities * DT), [WIDTH, HEIGHT])

    # Draw particles
    for i in range(N_PARTICLES):
        pygame.draw.circle(screen, COLOURS[types[i]], positions[i], 1)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(TICKS)  # limits FPS to 60

pygame.quit()
