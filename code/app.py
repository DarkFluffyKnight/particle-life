from typing import Annotated, Literal, Tuple

import numpy as np
import pygame
from numba import njit, prange
from numpy.typing import NDArray

ParticleVector = Annotated[np.typing.NDArray[np.float64], Literal["N", 2]]

WIDTH, HEIGHT = 1920 / 2, 1200 / 2
N_PARTICLES = 5000
N_TYPES = 4
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple"]

R_MIN = 50
# R_MIN = 100
R_MAX = 200
FRICTION = 0.9

DT = 0.01
np.random.seed(42)

# Random attraction matrix, each particle is attracted to its own type
# Attraction is positive, repulsion is negative
ATTRACTION_MATRIX = np.random.uniform(low=-1, high=1, size=(N_TYPES, N_TYPES))
for i in range(N_TYPES):
    ATTRACTION_MATRIX[i, i] = abs(ATTRACTION_MATRIX[i, i])

# ATTRACTION_MATRIX = np.array([[1, 1], [-1, 1]])
# ATTRACTION_MATRIX = -np.ones(shape=(N_TYPES, N_TYPES))

# ATTRACTION_MATRIX = np.full((N_TYPES, N_TYPES), -1)
# for i in range(N_TYPES):
#     ATTRACTION_MATRIX[i][i] = 1

# ATTRACTION_MATRIX = np.array([[1, 1], [1, 1]])

# Start with random positions and zero velocities
positions = np.random.rand(N_PARTICLES, 2) * np.array([WIDTH, HEIGHT])
# velocities = np.random.uniform(low=-V_MAX, high=V_MAX, size=(N_PARTICLES, 2))
velocities = np.zeros((N_PARTICLES, 2))
types = np.random.choice(N_TYPES, N_PARTICLES)
# types = np.random.randint(0, N_TYPES, N_PARTICLES)


@njit(fastmath=True)
def build_spatial_hash(
    positions: NDArray[np.float64],
) -> Tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
    int,
]:
    """Builds a spatial hash of particles. Each grid has size (R_MAX * R_MAX). Returns an array of N particle ids ordered by the cell they occupy, a reference array of N cell ids for each particle, an accumulator of cell counts, and the number of rows and columns of cells.

    Args:
        positions (NDArray[np.float64]): (N x 2) matrix of particle positions

    Returns:
        Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], int, int]:
        cell_particles: particle ids arranged by cell

        particle_cells: cell ids arranged by particle

        accumulator: cumulative particle counts in cells (starts at 0, ends at N)

        n_rows: number of rows in grid

        n_cols: number of columns in grid
    """

    # Get number of rows, columns, cells
    n_rows = int(np.ceil(HEIGHT / R_MAX))
    n_cols = int(np.ceil(WIDTH / R_MAX))

    n_cells = n_rows * n_cols

    # Stores cell locations for each particle
    particle_cells = np.zeros(N_PARTICLES, dtype=np.int32)
    # Stores number of particles in each cell
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    # For each particle, store its cell and update that cell's count
    for i in range(N_PARTICLES):
        cx = max(0, min(int(np.floor(positions[i, 0] / R_MAX)), n_cols - 1))
        cy = max(0, min(int(np.floor(positions[i, 1] / R_MAX)), n_rows - 1))
        c = (cy * n_cols) + cx
        particle_cells[i] = c
        cell_counts[c] += 1

    # Stores accumulated cell counts
    accumulator = np.zeros(n_cells + 1, dtype=np.int32)
    for c in range(n_cells):
        accumulator[c + 1] = accumulator[c] + cell_counts[c]

    # Reorder particles by cell, e.g. [cell_1 particle_1, cell_1 particle_2, cell_2 particle_1, etc.]
    cell_particles = np.zeros(N_PARTICLES, dtype=np.int32)
    # Counts number of particles in cell so far for offset
    cell_counter = np.zeros(n_cells, dtype=np.int32)
    # For particle i
    for i in range(N_PARTICLES):
        # Find the cell the particle belongs to
        cell_i = particle_cells[i]
        # Put that particle in the corresponding cell, accounting for what's already there
        cell_particles[accumulator[cell_i] + cell_counter[cell_i]] = i
        cell_counter[cell_i] += 1

    return cell_particles, particle_cells, accumulator, n_rows, n_cols


def apply_force(p1: int, p2: int):
    """Apply force from particle 2 onto particle 1. Uses standard linear forces mechanism (unused)

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
            1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))  # -1 <= force <= 1
        )
    else:
        return

    velocities[p1] += vect * force / dist


def apply_smooth_force(p1: int, p2: int):
    """Apply force from particle 2 onto particle 1. Uses polynomial force mechanism for smoother movement (unused)

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
        )  # 4 * a * -(x^2 - (r_max + r_min)x  + r_max*r_min) / (r_max - r_min)^2
    else:
        return

    velocities[p1] += vect * force / dist


@njit(parallel=True, fastmath=True)
def update_velocities(positions, velocities):
    """Update velocities of each particle with parallelisation for efficiency.
    Uses numpy vectors and methods

    Args:
        positions (NumPy array): (N x 2) matrix of particle positions
        velocities (NumPy array): (N x 2) matrix of particle velocities
    """
    for i in prange(N_PARTICLES):
        for j in range(N_PARTICLES):
            # Skip pcomparing particles to themselves
            if i == j:
                continue

            # Get relative position and distance
            vect = positions[j] - positions[i]
            dist = np.linalg.norm(vect)

            # Calculate force factor
            if dist < R_MIN:
                # If too close, repels
                force = (dist / R_MIN) - 1
            elif dist < R_MAX:
                # If in range, uses standard linear forces
                force = ATTRACTION_MATRIX[types[i], types[j]] * (
                    1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))
                )
            else:
                # If far away, no effect
                force = 0

            # Scale position vector appropriately and add to velocity
            velocities[i] += force * vect / dist

    # Friction factor
    velocities *= FRICTION


@njit(parallel=True, fastmath=True)
def update_velocities_components(positions, velocities):
    """Update velocities of each particle with parallelisation for efficiency.
    Breaks into components instead to avoid RAM and use cache instead.

    Args:
        positions (NumPy array): (N x 2) matrix of particle positions
        velocities (NumPy array): (N x 2) matrix of particle velocities
    """
    for i in prange(N_PARTICLES):
        # Vector components
        # Total force (technically acceleration but nvm)
        fx, fy = 0.0, 0.0
        # Position i, avoids looking in positions[i] for each j
        xi, yi = positions[i, 0], positions[i, 1]

        # Avoid retrieving types[i] repeatedly for each j
        type_i = types[i]

        for j in range(N_PARTICLES):
            # Skip comparing particles to themselves
            if i == j:
                continue

            # Get relative position and distance
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi

            dist = (dx**2 + dy**2) ** 0.5

            # Calculate force factor
            if dist < R_MIN:
                # If too close, repels particle i from j
                force = (dist / R_MIN) - 1
            elif dist < R_MAX:
                # If in range, uses standard linear forces
                force = ATTRACTION_MATRIX[type_i, types[j]] * (
                    1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))
                )
            else:
                # If far away, no effect
                force = 0.0

            # Scale position vector appropriately and add to total force
            fx += force * dx / dist
            fy += force * dy / dist

        # Applying friction at the end saves time instead of doing it for each j
        velocities[i, 0] = (velocities[i, 0] + fx) * FRICTION
        velocities[i, 1] = (velocities[i, 1] + fy) * FRICTION


@njit(parallel=True, fastmath=True)
def update_velocities_components_hash(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    cell_particles: NDArray[np.int32],
    particle_cells: NDArray[np.int32],
    accumulator: NDArray[np.int32],
    n_rows: int,
    n_cols: int,
):
    """Update velocities of each particle with parallelisation.
    Breaks into components instead to avoid RAM and use cache instead.
    Uses spatial hash to avoid n^2 processing implemented with numpy arrays and an accumulator.

    Args:
        positions (NDArray[np.float64]): (N x 2) matrix of particle positions
        velocities (NDArray[np.float64]): (N x 2) matrix of particle velocities
        cell_particles (NDArray[np.int32]): (N) array of particle cell locations (ordered by cell)
        particle_cells (NDArray[np.int32]): (N) array of particle cell locations (ordered by particle)
        accumulator (NDArray[np.int32]): (n_rows * n_cols + 1) array with accumulated cell particle counts, starts at zero, ends at total
        n_rows (int): Number of rows of cells
        n_cols (int): Number of columns of cells
    """

    for i in prange(N_PARTICLES):
        # Total force (technically acceleration but nvm)
        fx, fy = 0.0, 0.0

        # Position i, avoids looking in positions[i] for each j
        xi, yi = positions[i, 0], positions[i, 1]
        # Avoid retrieving types[i] repeatedly for each j
        type_i = types[i]

        cell_i = particle_cells[i]
        cell_iy, cell_ix = divmod(cell_i, n_cols)

        # Check adjacent cells
        for cell_dx in range(-1, 2):
            cell_jx = (cell_ix + cell_dx) % n_cols

            for cell_dy in range(-1, 2):
                cell_jy = (cell_iy + cell_dy) % n_rows

                # Second cell
                cell_j = (cell_jy * n_cols) + cell_jx
                cell_j_start = accumulator[cell_j]
                cell_j_end = accumulator[cell_j + 1]

                for loc in range(cell_j_start, cell_j_end):
                    # Skip comparing particles to themselves
                    j = cell_particles[loc]
                    if i == j:
                        continue

                    # Get relative position and distance
                    dx = positions[j, 0] - xi
                    dy = positions[j, 1] - yi

                    # Toroidal wrapping
                    # If two particles have |dx| > WIDTH/2, then they must be on opposite ends
                    if dx > WIDTH * 0.5:
                        dx -= WIDTH
                    elif dx < -WIDTH * 0.5:
                        dx += WIDTH
                    if dy > HEIGHT * 0.5:
                        dy -= HEIGHT
                    elif dy < -HEIGHT * 0.5:
                        dy += HEIGHT

                    dist = (dx**2 + dy**2) ** 0.5

                    # Calculate force factor
                    if dist < R_MIN:
                        # If too close, repels particle i from j
                        force = (dist / R_MIN) - 1  # force < 0
                    elif dist < R_MAX:
                        # If in range, uses standard linear forces
                        # -a < force < a
                        force = ATTRACTION_MATRIX[type_i, types[j]] * (
                            1 - abs((2 * dist - R_MAX - R_MIN) / (R_MAX - R_MIN))
                        )
                    else:
                        # If far away, no effect
                        force = 0.0

                    # Scale position vector appropriately and add to total force
                    fx += force * dx / dist
                    fy += force * dy / dist

        # Applying friction at the end saves time instead of doing it for each j
        velocities[i, 0] = (velocities[i, 0] + fx) * FRICTION
        velocities[i, 1] = (velocities[i, 1] + fy) * FRICTION


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

    # Build spatial hash
    cell_particles, particle_cells, accumulator, n_rows, n_cols = build_spatial_hash(
        positions
    )

    # update_velocities(positions, velocities)
    # update_velocities_components(positions, velocities)
    update_velocities_components_hash(
        positions=positions,
        velocities=velocities,
        cell_particles=cell_particles,
        particle_cells=particle_cells,
        accumulator=accumulator,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    # Update position
    positions = np.mod(positions + (velocities * DT), [WIDTH, HEIGHT])

    # Draw particles
    for i in range(N_PARTICLES):
        pygame.draw.circle(screen, COLOURS[types[i]], positions[i], 1)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
